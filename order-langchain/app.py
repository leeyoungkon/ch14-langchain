"""LangChain order chat: DuckDB on MinIO Parquet + existing customer MCP."""
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field

import logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("order-chat")

class AgentDebugHandler(BaseCallbackHandler):

    def on_tool_start(
        self, serialized, input_str, **kwargs
    ):
        name = (serialized or {}).get("name", "unknown")
        logger.info(
            "[TOOL START] %s | input=%s",
            name,
            str(input_str)[:500]
        )

    def on_tool_end(self, output, **kwargs):
        logger.info(
            "[TOOL END] %s",
            str(output)[:500]
        )

    def on_tool_error(self, error, **kwargs):
        logger.error(
            "[TOOL ERROR] %s", error
        )

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger('order-chat')

S3_URL = os.getenv('ORDERS_PARQUET_URI', 's3://warehouse/raw/orders/**/*.parquet')
S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'http://aistor-server.aistor-server.svc.cluster.local:9000')
MCP_URL = os.getenv('CUSTOMER_MCP_URL', 'http://customer-mcp:8000/mcp')
MODEL = os.getenv('CHAT_MODEL', 'gemma3:12b')
OLLAMA = os.getenv('OLLAMA_BASE_URL', 'http://ollama.default.svc.cluster.local:11434')
MAX_ROWS = min(int(os.getenv('MAX_ROWS', '100')), 500)
TZ = ZoneInfo('Asia/Seoul')

# Only the app administrator controls this S3 path; the LLM cannot modify it.
if not (S3_URL.startswith('s3://') or S3_URL.startswith('file://') or S3_URL.startswith('/')):
    raise RuntimeError('ORDERS_PARQUET_URI must be a trusted s3:// or local path')


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def db_connection():
    """New DuckDB connection per request; no shared mutable connection."""
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    # 실습 환경에서만 TLS 인증서 검증 비활성화
    con.execute("SET enable_curl_server_cert_verification = false")
    con.execute('SET TimeZone=\'Asia/Seoul\'')
    if S3_URL.startswith('s3://'):
        u = urlparse(S3_ENDPOINT)
        if not u.hostname:
            raise RuntimeError('S3_ENDPOINT invalid')
        con.execute('INSTALL httpfs; LOAD httpfs;')
        # Temporary secret, never persisted on disk.
        secret = f"""CREATE SECRET minio_secret (TYPE s3, PROVIDER config,
          KEY_ID {sql_literal(os.environ['S3_ACCESS_KEY'])},
          SECRET {sql_literal(os.environ['S3_SECRET_KEY'])},
          REGION 'us-east-1',
          ENDPOINT {sql_literal(u.netloc)},
          USE_SSL {'true' if u.scheme == 'https' else 'false'},
          URL_STYLE 'path')"""
        con.execute(secret)
    source = S3_URL.replace('file://', '')
    con.execute(f"CREATE VIEW orders AS SELECT * FROM read_parquet({sql_literal(source)}, union_by_name=true)")
    return con


# Deliberately restrict SQL read access to a single logical relation.
BLOCKED_FUNCTIONS = {'read_parquet', 'read_csv', 'read_csv_auto', 'read_json',
                     'read_text', 'read_blob', 'sqlite_scan', 'postgres_scan',
                     'http_get', 'query', 'query_table', 'current_setting'}

def validate_sql(statement: str) -> str:
    try:
        trees = sqlglot.parse(statement, read='duckdb')
    except Exception as e:
        raise ValueError(f'SQL 파싱 실패: {e}') from e
    if len(trees) != 1 or trees[0] is None:
        raise ValueError('SQL은 SELECT 문 하나만 허용합니다.')
    tree = trees[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        raise ValueError('SELECT 또는 WITH ... SELECT만 허용합니다.')
    # No arbitrary file/table functions, connections, table writes, or external tables.
    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create,
                             exp.Drop, exp.Alter, exp.Command, exp.Copy,
                             exp.Attach, exp.Merge)):
            raise ValueError('읽기 이외의 SQL은 금지됩니다.')
        if isinstance(node, exp.Anonymous) and node.name.lower() in BLOCKED_FUNCTIONS:
            raise ValueError(f'금지된 함수: {node.name}')
    # Allow CTE aliases, but physical tables may only be `orders`.
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.db or table.catalog or table.name.lower() not in ({'orders'} | ctes):
            raise ValueError(f'허용되지 않은 테이블: {table.sql()}')
    if not list(tree.find_all(exp.Table)):
        raise ValueError('orders 테이블 조회가 포함되어야 합니다.')
    return tree.sql(dialect='duckdb')


@tool
def get_order_schema() -> str:
    """Return exact columns/types of orders Parquet. Call before generating SQL."""
    con = db_connection()
    try:
        rows = con.execute('DESCRIBE orders').fetchall()
        return json.dumps([{'column': r[0], 'type': r[1]} for r in rows], ensure_ascii=False)
    finally:
        con.close()


@tool
def query_orders(sql: str) -> str:
    """Execute one read-only DuckDB SELECT against orders Parquet; SQL table is orders. LIMIT results."""
    safe = validate_sql(sql)
    con = db_connection()
    try:
        # Wrap the validated query to enforce output rows regardless of inner LIMIT.
        cur = con.execute(f'SELECT * FROM ({safe}) AS result LIMIT {MAX_ROWS + 1}')
        columns = [d[0] for d in cur.description]
        data = cur.fetchall()
        truncated = len(data) > MAX_ROWS
        out = [dict(zip(columns, (str(v) if isinstance(v, (datetime, bytes)) else v for v in row)))
               for row in data[:MAX_ROWS]]
        return json.dumps({'rows': out, 'returned': len(out), 'truncated': truncated},
                          ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({'error': str(e)}, ensure_ascii=False)
    finally:
        con.close()


async def mcp_tools():
    async with streamable_http_client(MCP_URL) as (reader, writer, *_):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{'name': t.name, 'description': t.description,
                     'inputSchema': t.input_schema} for t in result.tools]


async def mcp_call(tool_name: str, arguments: dict):
    async with streamable_http_client(MCP_URL) as (
        reader, writer, *_
    ):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            result = await session.call_tool(
                name=tool_name,
                arguments=arguments
            )

            return result.model_dump()


@tool
def discover_customer_tools() -> str:
    """Discover customer MCP tools and their JSON schemas before calling customer MCP."""
    return json.dumps(asyncio.run(mcp_tools()), ensure_ascii=False, default=str)


@tool
def call_customer_mcp(
    tool_name: str,
    arguments_json: dict[str, Any] | str
) -> str:
    """고객 MCP 도구를 호출한다.

    Args:
        tool_name: 호출할 MCP 도구 이름
        arguments_json: MCP 도구의 매개변수(JSON 객체 또는 문자열)
    """

    try:
        # 문자열일 때만 JSON 파싱
        if isinstance(arguments_json, str):
            arguments = json.loads(arguments_json)
        else:
            arguments = arguments_json

        if not isinstance(arguments, dict):
            raise ValueError(
                "arguments_json은 JSON 객체여야 합니다."
            )

        result = asyncio.run(
            mcp_call(tool_name, arguments)
        )

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

    except Exception as e:
        log.exception("Customer MCP 호출 실패")
        return json.dumps(
            {"error": str(e)},
            ensure_ascii=False
        )


@tool
def get_customer_info(customer_id: str) -> str:
    """고객 ID로 고객명, 등급, 지역, 주소를 조회한다.

    Args:
        customer_id: 조회할 고객 ID. 예: C001
    """
    try:
        result = asyncio.run(
            mcp_call(
                "get_customer_info",
                {"customer_id": customer_id}
            )
        )

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

    except Exception as e:
        log.exception("Customer MCP 조회 실패")
        return json.dumps(
            {"error": str(e)},
            ensure_ascii=False
        )

@tool
def search_customer(customer_name: str) -> str:
    """고객명으로 고객을 검색하여 customer_id, 고객명, 등급, 지역, 주소를 반환한다.

    고객명만 제공되고 customer_id를 모르는 경우 사용한다.
    """

    try:
        result = asyncio.run(
            mcp_call(
                "search_customer",
                {"customer_name": customer_name}
            )
        )

        return json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

    except Exception as e:
        log.exception("고객 검색 실패")

        return json.dumps(
            {"error": str(e)},
            ensure_ascii=False
        )
    
@tool
def current_korean_date() -> str:
    """Get current date/time in Asia/Seoul for today/yesterday/date-range filters."""
    return datetime.now(TZ).isoformat(timespec='seconds')


SYSTEM = """당신은 주문/고객 데이터 분석 Agent입니다. 한국어로 답합니다.
규칙:
1. 집계나 계산을 추측하지 말고 반드시 Tool 조회 결과로 답하세요.
2. 처음 주문 조회 전 get_order_schema를 호출하여 실제 컬럼을 확인하세요. DuckDB SQL에서 orders라는 테이블만 조회하세요.
3. 고객 이름, 등급, 주소, 지역은 반드시 get_customer_info(customer_id) 도구로 조회하세요.
   이 도구는 내부적으로 Customer MCP Server를 호출합니다.
   get_customer_details 등 존재하지 않는 도구를 만들어 호출하지 마세요.
   주문 데이터에서 customer_id를 획득한 후 그 customer_id를 get_customer_info에 전달하세요.
4. 복합 질문은 (필요한 주문 집계 → customer_id 획득 → 고객 상세 조회 → 동일 ID로 추가 주문 조회) 순서로 수행하세요.
5. 질문의 '그 고객' 같은 지시어는 이전 메시지/결과에서 customer_id를 찾고, 불명확하면 확인하세요.
6. '오늘', '어제'는 current_korean_date 결과를 기준으로 Asia/Seoul의 자정부터 다음 자정 미만으로 필터링하세요. order_time이 한국시간으로 저장되었다는 운영 전제를 적용합니다.
7. '가장 많이 주문'은 기본적으로 quantity 합계, '가장 주문액이 큰'은 total_amount 합계입니다. 공동 1위면 동률을 표시하세요.
8. 주소 필드가 MCP에 없다면 주소를 만들어내지 말고 '주소 정보 없음'으로 답하세요.
9. 조회 데이터에 없는 사실을 만들지 말고, 비어 있거나 실패한 결과를 구분하세요. 실행한 날짜 조건, 사용한 순위 기준을 짧게 명시하세요.
10. 고객정보의 개인정보는 인가된 사용자에게만 제공되어야 합니다. 이 데모는 별도 인증 기능이 없으므로 보호된 환경에서만 사용하세요.
"""

llm = ChatOllama(model=MODEL, base_url=OLLAMA, temperature=0,
                 num_predict=int(os.getenv('NUM_PREDICT', '2048')),
                 client_kwargs={'timeout': float(os.getenv('OLLAMA_TIMEOUT', '180'))})
agent = create_agent(model=llm,
                     tools=[ get_order_schema, query_orders, 
                            get_customer_info, search_customer,current_korean_date],
                     system_prompt=SYSTEM,
                     checkpointer=InMemorySaver())

app = FastAPI(title='Order Chat: LangChain + MinIO + Customer MCP')

class ChatInput(BaseModel):
    session_id: str = Field(default='demo', min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=3000)

@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL, 'parquet': S3_URL, 'mcp': MCP_URL,
            'note': 'Connectivity is not checked by this endpoint'}

@app.post('/chat')
def chat(body: ChatInput):
    try:
        result = agent.invoke(
    {
        "messages": [
            HumanMessage(content=body.message)
        ]
    },
    config={
        "configurable": {
            "thread_id": body.session_id
        },
        "recursion_limit": 40,
        "callbacks": [AgentDebugHandler()]
    }
)
        answer = result['messages'][-1].content
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        return {'session_id': body.session_id, 'answer': answer}
    except Exception as e:
        log.exception('Agent error')
        raise HTTPException(status_code=500, detail=str(e))

HTML = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Order Chat</title>
<style>body{font-family:system-ui,sans-serif;max-width:800px;margin:35px auto;padding:0 18px}#messages{height:65vh;overflow:auto;border:1px solid #ccc;padding:16px;white-space:pre-wrap;border-radius:8px}.msg{margin:12px 0}.me{color:#1257a3}.bot{color:#125d44}form{display:flex;margin-top:12px;gap:8px}input{flex:1;padding:12px}button{padding:10px 18px}</style></head><body>
<h2>Order Chat · LangChain</h2><p>MinIO Parquet + Customer MCP</p><div id="messages"></div><form id="f"><input id="q" placeholder="주문 정보를 질문하세요" required><button id="send">질문</button></form>
<script>const sid=crypto.randomUUID(); const el=document.querySelector('#messages'); document.querySelector('#f').onsubmit=async(e)=>{e.preventDefault(); const q=document.querySelector('#q');const text=q.value.trim();if(!text)return; q.value='';add('사용자',text,'me');add('AI','조회 중...','bot');const last=el.lastElementChild;document.querySelector('#send').disabled=true;try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:text})});const x=await r.json();last.textContent='AI: '+(r.ok?x.answer:JSON.stringify(x));}catch(err){last.textContent='AI 오류: '+err;}finally{document.querySelector('#send').disabled=false;}};function add(name,msg,cls){const p=document.createElement('div');p.className='msg '+cls;p.textContent=name+': '+msg;el.appendChild(p);el.scrollTop=el.scrollHeight;}</script></body></html>'''

@app.get('/', response_class=HTMLResponse)
def home():
    return HTML

