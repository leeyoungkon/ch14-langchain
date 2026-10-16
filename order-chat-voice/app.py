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

HTML = '<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Order Chat · 음성</title>\n<style>\nbody{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 18px;color:#172334;background:#f5f7fb}h1{font-size:24px}#messages{height:52vh;overflow:auto;background:white;border:1px solid #ccd4df;padding:16px;white-space:pre-wrap;border-radius:12px}.msg{margin:12px 0;line-height:1.6;overflow-wrap:anywhere}.me{color:#1257a3}.bot{color:#125d44}form{margin-top:14px}textarea{box-sizing:border-box;width:100%;font:inherit;padding:12px;border-radius:8px;border:1px solid #aab8ca}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}button,select{font:inherit;padding:10px;border:1px solid #bbc8d8;border-radius:8px;background:white}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}#send{background:#1558b0;color:white}#status{min-height:1.5em;color:#354d6c}.note{font-size:13px;color:#52637a;line-height:1.6}\n</style></head><body>\n<h1>Order Chat · 음성 대화</h1><p>MinIO Parquet + DuckDB + Customer MCP</p>\n<div id="messages" role="log" aria-label="대화 내역"></div>\n<form id="f"><label for="q">질문</label><textarea id="q" rows="2" maxlength="3000" placeholder="직접 입력하거나 음성 입력을 누르세요" required></textarea>\n<div class="controls"><button type="button" id="mic">음성 입력</button><button type="submit" id="send">질문 전송</button><label><input type="checkbox" id="autoSend">인식 후 자동 전송</label></div></form>\n<div class="controls"><label><input type="checkbox" id="autoSpeak" checked>답변 자동 읽기</label><button type="button" id="replay" disabled>다시 듣기</button><button type="button" id="stopSpeech">읽기 중지</button><label>음성 <select id="voice" aria-label="읽기 음성"></select></label></div>\n<p id="status" role="status" aria-live="polite">준비되었습니다.</p>\n<p class="note">음성 입력 후 질문을 수정하고 전송할 수 있습니다. 음성 인식은 브라우저 제공 서비스로 처리되며 외부 서버로 음성이 전송될 수 있습니다. Chrome에서 localhost 또는 HTTPS로 접속하고 마이크 사용을 허용하세요. 한국어 읽기 음성은 PC·브라우저에 설치된 음성에 따라 달라집니다.</p>\n<script>\n\'use strict\';\nconst $ = id => document.getElementById(id);\nconst sid = globalThis.crypto?.randomUUID?.() || (\'session-\' + Date.now() + \'-\' + Math.random().toString(36).slice(2));\nconst synth = window.speechSynthesis;\nconst Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;\nlet busy=false, recording=false, recognition=null, recognitionError=false, baseText=\'\', finalText=\'\', lastAnswer=\'\';\nlet speechGeneration=0, currentUtterance=null, voiceList=[];\nfunction status(text){$(\'status\').textContent=text;}\nfunction controls(){\n  $(\'send\').disabled=busy || recording;\n  $(\'mic\').disabled=busy || !recognition;\n  $(\'mic\').textContent=recording ? \'인식 마치기\' : \'음성 입력\';\n  $(\'q\').readOnly=recording;\n  $(\'replay\').disabled=!synth || !lastAnswer || busy || recording;\n}\nfunction add(name,text,cls){const p=document.createElement(\'div\');p.className=\'msg \'+cls;p.textContent=name+\': \'+text;$(\'messages\').appendChild(p);$(\'messages\').scrollTop=$(\'messages\').scrollHeight;return p;}\nfunction stopSpeaking(){speechGeneration++;if(synth)synth.cancel();currentUtterance=null;}\nfunction loadVoices(){\n  if(!synth)return;\n  const previous=$(\'voice\').value;\n  voiceList=synth.getVoices();\n  $(\'voice\').replaceChildren();\n  const automatic=document.createElement(\'option\');automatic.value=\'\';automatic.textContent=\'한국어 자동 선택\';$(\'voice\').appendChild(automatic);\n  for(const v of voiceList){const o=document.createElement(\'option\');o.value=v.voiceURI;o.textContent=v.name+\' (\'+v.lang+\')\';$(\'voice\').appendChild(o);}\n  if(voiceList.some(v=>v.voiceURI===previous))$(\'voice\').value=previous;\n}\nfunction speakAnswer(text){\n  if(!synth){status(\'이 브라우저는 음성 출력을 지원하지 않습니다.\');return;}\n  stopSpeaking();\n  const generation=speechGeneration;\n  // Split long answers so a single long utterance does not stall playback.\n  const clean=text.replace(/```[\\s\\S]*?```/g,\' 코드 내용은 화면에서 확인하세요. \').replace(/https?:\\/\\/\\S+/g,\' 링크 \').replace(/[*#`|]/g,\' \');\n  const chunks=clean.match(/[^.!?。！？\\n]{1,160}[.!?。！？\\n]?/g) || [];\n  let index=0;\n  function next(){\n    if(generation!==speechGeneration)return;\n    if(index>=chunks.length){currentUtterance=null;status(\'답변 읽기를 마쳤습니다.\');return;}\n    const u=new SpeechSynthesisUtterance(chunks[index++]);currentUtterance=u;\n    u.lang=\'ko-KR\';u.rate=1;\n    const voice=voiceList.find(v=>v.voiceURI===$(\'voice\').value) || voiceList.find(v=>/^ko(?:-|_)/i.test(v.lang));\n    if(voice){u.voice=voice;u.lang=voice.lang;}\n    u.onend=next;\n    u.onerror=e=>{if(generation===speechGeneration){currentUtterance=null;status(\'음성 재생 실패: \'+e.error+\'. 다시 듣기를 눌러주세요.\');}};\n    synth.speak(u);\n  }\n  status(\'답변을 읽고 있습니다.\');next();\n}\nasync function sendQuestion(){\n  if(busy || recording)return;\n  const text=$(\'q\').value.trim();if(!text)return;\n  if(text.length>3000){status(\'질문은 3,000자 이내로 입력하세요.\');return;}\n  busy=true;controls();stopSpeaking();$(\'q\').value=\'\';\n  add(\'사용자\',text,\'me\');const reply=add(\'AI\',\'조회 중…\',\'bot\');status(\'주문·고객 데이터를 조회하고 있습니다.\');\n  try{\n    const r=await fetch(\'/chat\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify({session_id:sid,message:text})});\n    const x=await r.json();\n    if(!r.ok)throw new Error(typeof x.detail===\'string\'?x.detail:JSON.stringify(x));\n    lastAnswer=typeof x.answer===\'string\'?x.answer:JSON.stringify(x.answer);\n    reply.textContent=\'AI: \'+lastAnswer;status(\'답변이 도착했습니다.\');\n    if($(\'autoSpeak\').checked)speakAnswer(lastAnswer);\n  }catch(e){reply.textContent=\'AI 오류: \'+e.message;status(\'요청이 실패했습니다. 질문을 확인하고 다시 전송하세요.\');if(!$(\'q\').value)$(\'q\').value=text;}\n  finally{busy=false;controls();$(\'messages\').scrollTop=$(\'messages\').scrollHeight;}\n}\n$(\'f\').addEventListener(\'submit\',e=>{e.preventDefault();sendQuestion();});\n$(\'q\').addEventListener(\'keydown\',e=>{if(e.key===\'Enter\' && !e.shiftKey && !e.isComposing){e.preventDefault();$(\'f\').requestSubmit();}});\n$(\'replay\').onclick=()=>speakAnswer(lastAnswer);\n$(\'stopSpeech\').onclick=()=>{stopSpeaking();status(\'답변 읽기를 중지했습니다.\');};\n$(\'autoSpeak\').onchange=()=>{if(!$(\'autoSpeak\').checked)stopSpeaking();};\nif(synth){loadVoices();synth.addEventListener(\'voiceschanged\',loadVoices);}else{$(\'autoSpeak\').checked=false;$(\'autoSpeak\').disabled=true;$(\'stopSpeech\').disabled=true;$(\'voice\').disabled=true;}\nif(Recognition && window.isSecureContext){\n  recognition=new Recognition();recognition.lang=\'ko-KR\';recognition.continuous=false;recognition.interimResults=true;recognition.maxAlternatives=1;\n  recognition.onstart=()=>status(\'듣고 있습니다. 질문을 말씀하세요.\');\n  recognition.onresult=e=>{\n    let interim=\'\';\n    for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)finalText+=e.results[i][0].transcript+\' \';else interim+=e.results[i][0].transcript;}\n    $(\'q\').value=[baseText,finalText+interim].filter(Boolean).join(\' \').trim();\n  };\n  recognition.onerror=e=>{\n    recognitionError=true;\n    const errors={\'not-allowed\':\'마이크 권한이 거부되었습니다. 브라우저의 사이트 설정에서 마이크를 허용하세요.\',\'service-not-allowed\':\'브라우저 음성 인식 서비스를 사용할 수 없습니다.\',\'audio-capture\':\'마이크를 찾을 수 없습니다.\',\'no-speech\':\'음성이 인식되지 않았습니다. 다시 시도하세요.\',\'network\':\'음성 인식 서비스 연결에 실패했습니다. 인터넷 연결을 확인하세요.\',\'language-not-supported\':\'한국어 음성 인식을 지원하지 않습니다.\',\'aborted\':\'음성 인식을 취소했습니다.\'};\n    status(errors[e.error] || \'음성 인식 오류: \'+e.error);\n  };\n  recognition.onend=()=>{\n    recording=false;controls();\n    if(recognitionError)return;\n    if(!finalText.trim()){status(\'확정된 음성이 없습니다. 다시 입력하거나 텍스트를 확인하세요.\');return;}\n    status(\'음성 인식을 마쳤습니다. 질문을 확인하고 전송하세요.\');\n    if($(\'autoSend\').checked)sendQuestion();else $(\'q\').focus();\n  };\n  $(\'mic\').onclick=()=>{\n    if(recording){recognition.stop();return;}\n    stopSpeaking();baseText=$(\'q\').value.trim();finalText=\'\';recognitionError=false;recording=true;controls();\n    try{recognition.start();}catch(e){recording=false;controls();status(\'음성 인식을 시작할 수 없습니다: \'+e.message);}\n  };\n}else{status(Recognition?\'마이크를 사용하려면 localhost 또는 HTTPS로 접속하세요.\':\'이 브라우저는 음성 인식을 지원하지 않습니다. 텍스트 입력을 사용하세요.\');}\nwindow.addEventListener(\'pagehide\',()=>{if(recognition)recognition.abort();stopSpeaking();});\ncontrols();\n</script></body></html>\n'


@app.get('/', response_class=HTMLResponse)
def home():
    return HTML

