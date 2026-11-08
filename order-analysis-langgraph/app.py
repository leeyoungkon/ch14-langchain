"""LangChain order chat: DuckDB on MinIO Parquet + existing customer MCP."""
import asyncio
import json
import logging
import os
import re
from contextlib import closing
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from typing import Any, TypedDict

import duckdb
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
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

S3_URL = os.getenv('ORDERS_PARQUET_URI', 's3://warehouse/raw/stud12/**/*.parquet')
S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'https://aistor-server.aistor-server.svc.cluster.local:443')
MCP_URL = os.getenv('CUSTOMER_MCP_URL', 'http://customer-mcp:8000/mcp')
MODEL = os.getenv('CHAT_MODEL', 'gemma3:12b')
OLLAMA = os.getenv('OLLAMA_BASE_URL', 'http://ollama.default.svc.cluster.local:11434')  # For local testing; in-cluster use ollama.default.svc.cluster.locall
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
    con.execute('SET TimeZone=\'Asia/Seoul\'')
    if S3_URL.startswith('s3://'):
        u = urlparse(S3_ENDPOINT)
        if not u.hostname:
            raise RuntimeError('S3_ENDPOINT invalid')
        con.execute('INSTALL httpfs; LOAD httpfs;')
        verify_cert = os.getenv('S3_VERIFY_SSL', 'false').lower() != 'false'
        con.execute(f"SET enable_curl_server_cert_verification = {str(verify_cert).lower()}")
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

# This workflow uses a fixed, parameterized query against the trusted orders view.
SALES_SQL = """
    SELECT time_bucket(CAST(? AS INTERVAL), CAST(order_time AS TIMESTAMP)) AS bucket,
           SUM(total_amount) AS sales,
           COUNT(*) AS order_rows,
           MAX(CAST(order_time AS TIMESTAMP)) AS latest_order
    FROM orders
    WHERE CAST(order_time AS TIMESTAMP) >= ?
      AND CAST(order_time AS TIMESTAMP) < ?
    GROUP BY bucket
    ORDER BY bucket
"""

TOP_CUSTOMER_SALES_SQL = """
    WITH scoped AS (
        SELECT CAST(customer_id AS VARCHAR) AS customer_id,
               CAST(order_time AS TIMESTAMP) AS order_time,
               total_amount
        FROM orders
        WHERE CAST(order_time AS TIMESTAMP) >= ?
          AND CAST(order_time AS TIMESTAMP) < ?
          AND customer_id IS NOT NULL
          AND TRIM(CAST(customer_id AS VARCHAR)) <> ''
    ), customer_totals AS (
        SELECT customer_id, SUM(total_amount) AS total
        FROM scoped
        GROUP BY customer_id
        HAVING SUM(total_amount) IS NOT NULL
    ), winner AS (
        SELECT customer_id, COUNT(*) OVER (PARTITION BY total) AS tied_customers
        FROM customer_totals
        ORDER BY total DESC, customer_id ASC
        LIMIT 1
    )
    SELECT time_bucket(CAST(? AS INTERVAL), order_time) AS bucket,
           SUM(total_amount) AS sales,
           COUNT(*) AS order_rows,
           MAX(order_time) AS latest_order,
           winner.customer_id,
           winner.tied_customers
    FROM scoped
    JOIN winner ON winner.customer_id = scoped.customer_id
    GROUP BY bucket, winner.customer_id, winner.tied_customers
    ORDER BY bucket
"""

SALES_TREND = re.compile(r'추이|추세|변동|그래프|차트|시계열|그려')
SALES_AMOUNT = re.compile(r'매출|주문\s*금액')
TOP_CUSTOMER = re.compile(r'가장\s*(?:큰|높은|많은)\s*고객|(?:1\s*위|최고)\s*고객|고객\s*(?:중\s*)?1\s*위')


class SalesState(TypedDict, total=False):
    question: str
    start: datetime
    end: datetime
    bucket_minutes: int
    top_customer: bool
    customer_id: str
    tied_customers: int
    series: list[dict[str, Any]]
    latest_order: str
    metrics: dict[str, Any]
    answer: str


def parse_sales_window(state: SalesState) -> dict:
    question = state['question']
    top_customer = bool(TOP_CUSTOMER.search(question))
    if re.search(r'상품별|고객별|지역별|카테고리별|\bP\d+\b', question, re.IGNORECASE) or (
        '고객' in question and not top_customer
    ):
        return {'answer': '차트는 전체 주문금액 추이와 매출 1위 고객의 주문금액 추이를 지원합니다.'}
    now = datetime.now(TZ).replace(tzinfo=None)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    bucket = re.search(r'(\d+)\s*(분|시간)\s*(?:단위|간격|별|마다)', question)
    minutes = int(bucket[1]) * (60 if bucket[2] == '시간' else 1) if bucket else (
        60 if '시간별' in question or '시간 단위' in question else 5
    )
    if minutes not in {5, 10, 15, 30, 60}:
        return {'answer': '집계 간격은 5·10·15·30분 또는 1시간으로 지정해 주세요.'}

    recent = re.search(r'(?:최근|지난)\s*(\d+)\s*(분|시간|일)', question)
    if recent:
        duration = int(recent[1]) * {'분': 1, '시간': 60, '일': 1440}[recent[2]]
        if not minutes <= duration <= 10080 or duration % minutes:
            return {'answer': '최근 기간은 집계 간격의 배수로 지정해 주세요. 최대 7일입니다.'}
        start = None
        end = now
    elif '어제' in question:
        start, end = today - timedelta(days=1), today
    elif '오늘' in question:
        start, end = today, now
    elif re.search(r'최근|지난|이번|저번|부터|까지|\d+\s*[월일]|\d{4}-\d{2}-\d{2}', question):
        return {'answer': '조회 기간은 “최근 30분”, “최근 2시간”, “오늘”, “어제”처럼 지정해 주세요.'}
    else:
        duration = max(30, minutes)
        start = None
        end = now

    end = end.replace(second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % minutes)
    if start is None:
        start = end - timedelta(minutes=duration)
    if start >= end:
        return {'answer': '요청한 기간에 아직 마감된 집계 구간이 없습니다.'}
    if (end - start) / timedelta(minutes=minutes) > 300:
        return {'answer': '그래프는 최대 300개 구간까지 표시합니다. 조회 기간을 줄이거나 간격을 늘려 주세요.'}
    return {'start': start, 'end': end, 'bucket_minutes': minutes, 'top_customer': top_customer}


def aggregate_sales(state: SalesState) -> dict:
    start, end = state['start'], state['end']
    minutes = state['bucket_minutes']
    top_customer = state['top_customer']
    with closing(db_connection()) as con:
        interval = f'{minutes} minutes'
        if top_customer:
            rows = con.execute(TOP_CUSTOMER_SALES_SQL, [start, end, interval]).fetchall()
        else:
            rows = con.execute(SALES_SQL, [interval, start, end]).fetchall()

    if not rows:
        empty = '순위를 계산할 수 있는 고객 주문이 없습니다.' if top_customer else '조회된 주문이 없습니다.'
        return {
            'series': [],
            'answer': f'{start:%m-%d %H:%M} ~ {end:%m-%d %H:%M} KST에 {empty}',
        }

    if all(row[1] is None for row in rows):
        return {'series': [], 'answer': '주문은 조회됐지만 해당 기간에 집계할 주문금액이 없습니다.'}

    observed = {row[0]: (row[1], row[2]) for row in rows}
    last_bucket = rows[-1][0]
    latest_order = max(row[3] for row in rows)
    series = []
    bucket = start
    while bucket < end:
        # Leave trailing buckets blank because the Parquet feed may still be catching up.
        sales, count = observed.get(
            bucket, (Decimal(0), 0) if rows[0][0] <= bucket <= last_bucket else (None, 0)
        )
        series.append({
            'time': bucket.replace(tzinfo=TZ).isoformat(timespec='minutes'),
            'sales': Decimal(str(sales)) if sales is not None else None,
            'order_rows': count,
        })
        bucket += timedelta(minutes=minutes)
    result = {'series': series, 'latest_order': latest_order.replace(tzinfo=TZ).isoformat(timespec='seconds')}
    if top_customer:
        result.update(customer_id=rows[0][4], tied_customers=rows[0][5])
    return result


def calculate_sales_metrics(state: SalesState) -> dict:
    series = state['series']
    points = [point for point in series if point['sales'] is not None]
    peak = max(points, key=lambda point: point['sales'], default=None)
    changes = []
    for previous, current in zip(series, series[1:]):
        before, after = previous['sales'], current['sales']
        if before is None or after is None or before == after:
            continue
        difference = after - before
        changes.append({
            'time': current['time'],
            'difference': difference,
            'pct': round(difference / before * 100, 1) if before > 0 else None,
        })
    return {'metrics': {
        'total': sum((point['sales'] for point in points), Decimal(0)),
        'peak': peak,
        'largest_increase': max(
            (change for change in changes if change['difference'] > 0),
            key=lambda change: change['difference'], default=None,
        ),
        'largest_decrease': min(
            (change for change in changes if change['difference'] < 0),
            key=lambda change: change['difference'], default=None,
        ),
    }}


def explain_sales(state: SalesState) -> dict:
    metrics = state['metrics']
    peak = metrics['peak']
    facts = []
    if state.get('customer_id'):
        if state['tied_customers'] > 1:
            facts.append(f"공동 1위 {state['tied_customers']}명 중 ID 정렬 기준 첫 고객 {state['customer_id']}입니다.")
        else:
            facts.append(f"주문금액 1위 고객은 {state['customer_id']}입니다.")
    facts.append(f"조회된 주문금액 합계는 {metrics['total']:,.2f}원입니다.")
    def bucket_label(value: str) -> str:
        return value[5:16].replace('T', ' ')

    if peak:
        facts.append(f"최고 구간은 {bucket_label(peak['time'])} 시작 구간으로 {peak['sales']:,.2f}원입니다.")
    for key, label in [('largest_increase', '최대 증가'), ('largest_decrease', '최대 감소')]:
        change = metrics[key]
        if change:
            pct = f" ({change['pct']:+.1f}%)" if change['pct'] is not None else ''
            facts.append(f"{label}는 {bucket_label(change['time'])} 시작 구간에서 "
                         f"직전 구간 대비 {change['difference']:+,.2f}원{pct}입니다.")
    if metrics['largest_increase'] is None and metrics['largest_decrease'] is None:
        facts.append('비교 가능한 연속 구간에서 확인된 금액 변동이 없습니다.')
    summary = ' '.join(facts)
    try:
        response = llm.invoke([
            SystemMessage(content=(
                '주문금액 집계와 계산된 지표를 근거로 변동 추이를 한국어 두세 문장으로 설명하세요. '
                '시간은 KST이고 각 시각은 구간 시작입니다. 수치를 새로 계산하거나 변동 원인을 추정하지 마세요.'
            )),
            HumanMessage(content=json.dumps({
                'bucket_minutes': state['bucket_minutes'],
                'latest_order': state['latest_order'],
                **metrics,
            }, ensure_ascii=False, default=str)),
        ]).content
        if isinstance(response, str) and response.strip():
            return {'answer': summary + '\n\nLLM 해설\n' + response.strip()}
    except Exception:
        log.warning('Sales trend explanation failed', exc_info=True)
    return {'answer': summary + '\n\nLLM 해설을 가져오지 못했습니다.'}


sales_graph = StateGraph(SalesState)
sales_graph.add_node('parse_window', parse_sales_window)
sales_graph.add_node('aggregate', aggregate_sales)
sales_graph.add_node('calculate_metrics', calculate_sales_metrics)
sales_graph.add_node('explain', explain_sales)
sales_graph.add_edge(START, 'parse_window')
sales_graph.add_conditional_edges(
    'parse_window', lambda state: 'aggregate' if 'start' in state else END,
    {'aggregate': 'aggregate', END: END},
)
sales_graph.add_conditional_edges(
    'aggregate', lambda state: 'calculate_metrics' if state['series'] else END,
    {'calculate_metrics': 'calculate_metrics', END: END},
)
sales_graph.add_edge('calculate_metrics', 'explain')
sales_graph.add_edge('explain', END)
sales_workflow = sales_graph.compile()


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
        if SALES_AMOUNT.search(body.message) and SALES_TREND.search(body.message):
            trend = sales_workflow.invoke({'question': body.message})
            response = {'session_id': body.session_id, 'answer': trend['answer']}
            if trend.get('series'):
                response['trend'] = {
                    'start': trend['start'].replace(tzinfo=TZ).isoformat(timespec='minutes'),
                    'end': trend['end'].replace(tzinfo=TZ).isoformat(timespec='minutes'),
                    'bucket_minutes': trend['bucket_minutes'],
                    'latest_order': trend['latest_order'],
                    'customer_id': trend.get('customer_id'),
                    'series': [
                        {**point, 'sales': float(point['sales']) if point['sales'] is not None else None}
                        for point in trend['series']
                    ],
                }
            return response
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
    except duckdb.IOException as e:
        log.exception('Order storage error')
        if S3_URL.startswith('s3://') and 'Could not connect to server' in str(e):
            raise HTTPException(
                status_code=502,
                detail='AIStor S3 API 연결에 실패했습니다. order-sales-stud12 ConfigMap의 S3_ENDPOINT를 확인하세요.',
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        log.exception('Agent error')
        raise HTTPException(status_code=500, detail=str(e))

HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Order Chat · 음성</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 18px;color:#172334;background:#f5f7fb}h1{font-size:24px}#messages{height:52vh;overflow:auto;background:white;border:1px solid #ccd4df;padding:16px;white-space:pre-wrap;border-radius:12px}.msg{margin:12px 0;line-height:1.6;overflow-wrap:anywhere}.me{color:#1257a3}.bot{color:#125d44}form{margin-top:14px}textarea{box-sizing:border-box;width:100%;font:inherit;padding:12px;border-radius:8px;border:1px solid #aab8ca}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}button,select{font:inherit;padding:10px;border:1px solid #bbc8d8;border-radius:8px;background:white}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}#send{background:#1558b0;color:white}#status{min-height:1.5em;color:#354d6c}.note{font-size:13px;color:#52637a;line-height:1.6}

.sales-chart{margin:12px 0 20px;padding:14px;background:#fff;border:1px solid #ccd4df;border-radius:8px;color:#243448}
.sales-chart h2{font-size:17px;margin:0}.sales-chart p{font-size:13px;margin:6px 0}
.sales-chart svg{width:100%;height:auto;display:block}.sales-chart svg text{font-size:12px;fill:currentColor}
.sales-grid{stroke:#dce3ea}.sales-line{fill:none;stroke:#17669b;stroke-width:2.5}.sales-dot{fill:#17669b}
.sales-chart table{border-collapse:collapse;width:100%;margin-top:8px}
.sales-chart td,.sales-chart th{text-align:left;padding:6px;border-bottom:1px solid #dce3ea}
</style></head><body>
<h1>Order Chat · 음성 대화</h1><p>MinIO Parquet + DuckDB + Customer MCP</p>
<div id="messages" role="log" aria-label="대화 내역"></div>
<form id="f"><label for="q">질문</label><textarea id="q" rows="2" maxlength="3000" placeholder="직접 입력하거나 음성 입력을 누르세요" required></textarea>
<div class="controls"><button type="button" id="mic">음성 입력</button><button type="submit" id="send">질문 전송</button><button type="button" id="sales-example">최근 30분 매출</button><label><input type="checkbox" id="autoSend">인식 후 자동 전송</label></div></form>
<div class="controls"><label><input type="checkbox" id="autoSpeak" checked>답변 자동 읽기</label><button type="button" id="replay" disabled>다시 듣기</button><button type="button" id="stopSpeech">읽기 중지</button><label>음성 <select id="voice" aria-label="읽기 음성"></select></label></div>
<p id="status" role="status" aria-live="polite">준비되었습니다.</p>
<p class="note">음성 입력 후 질문을 수정하고 전송할 수 있습니다. 음성 인식은 브라우저 제공 서비스로 처리되며 외부 서버로 음성이 전송될 수 있습니다. Chrome에서 localhost 또는 HTTPS로 접속하고 마이크 사용을 허용하세요. 한국어 읽기 음성은 PC·브라우저에 설치된 음성에 따라 달라집니다.</p>
<script>
'use strict';
const $ = id => document.getElementById(id);
const sid = globalThis.crypto?.randomUUID?.() || ('session-' + Date.now() + '-' + Math.random().toString(36).slice(2));
const synth = window.speechSynthesis;
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let busy=false, recording=false, recognition=null, recognitionError=false, baseText='', finalText='', lastAnswer='';
let speechGeneration=0, currentUtterance=null, voiceList=[];
function status(text){$('status').textContent=text;}
function controls(){
  $('send').disabled=busy || recording;
  $('sales-example').disabled=busy || recording;
  $('mic').disabled=busy || !recognition;
  $('mic').textContent=recording ? '인식 마치기' : '음성 입력';
  $('q').readOnly=recording;
  $('replay').disabled=!synth || !lastAnswer || busy || recording;
}
function add(name,text,cls){const p=document.createElement('div');p.className='msg '+cls;p.textContent=name+': '+text;$('messages').appendChild(p);$('messages').scrollTop=$('messages').scrollHeight;return p;}
function stopSpeaking(){speechGeneration++;if(synth)synth.cancel();currentUtterance=null;}
function loadVoices(){
  if(!synth)return;
  const previous=$('voice').value;
  voiceList=synth.getVoices();
  $('voice').replaceChildren();
  const automatic=document.createElement('option');automatic.value='';automatic.textContent='한국어 자동 선택';$('voice').appendChild(automatic);
  for(const v of voiceList){const o=document.createElement('option');o.value=v.voiceURI;o.textContent=v.name+' ('+v.lang+')';$('voice').appendChild(o);}
  if(voiceList.some(v=>v.voiceURI===previous))$('voice').value=previous;
}
function speakAnswer(text){
  if(!synth){status('이 브라우저는 음성 출력을 지원하지 않습니다.');return;}
  stopSpeaking();
  const generation=speechGeneration;
  // Split long answers so a single long utterance does not stall playback.
  const clean=text.replace(/```[\s\S]*?```/g,' 코드 내용은 화면에서 확인하세요. ').replace(/https?:\/\/\S+/g,' 링크 ').replace(/[*#`|]/g,' ');
  const chunks=clean.match(/[^.!?。！？\n]{1,160}[.!?。！？\n]?/g) || [];
  let index=0;
  function next(){
    if(generation!==speechGeneration)return;
    if(index>=chunks.length){currentUtterance=null;status('답변 읽기를 마쳤습니다.');return;}
    const u=new SpeechSynthesisUtterance(chunks[index++]);currentUtterance=u;
    u.lang='ko-KR';u.rate=1;
    const voice=voiceList.find(v=>v.voiceURI===$('voice').value) || voiceList.find(v=>/^ko(?:-|_)/i.test(v.lang));
    if(voice){u.voice=voice;u.lang=voice.lang;}
    u.onend=next;
    u.onerror=e=>{if(generation===speechGeneration){currentUtterance=null;status('음성 재생 실패: '+e.error+'. 다시 듣기를 눌러주세요.');}};
    synth.speak(u);
  }
  status('답변을 읽고 있습니다.');next();
}
function renderSalesTrend(reply, trend) {
  const panel = document.createElement('section');
  panel.className = 'sales-chart';
  const heading = document.createElement('h2');
  const subject = trend.customer_id ? '1위 고객 ' + trend.customer_id + ' · ' : '';
  heading.textContent = subject + '매출 추이 · ' + trend.bucket_minutes + '분 단위';
  const note = document.createElement('p');
  const formatTime = value => value.slice(5, 16).replace('T', ' ');
  note.textContent = formatTime(trend.start) + ' ~ ' + formatTime(trend.end) +
    ' KST · 조회된 마지막 주문 ' + formatTime(trend.latest_order) + ' KST';
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', '0 0 720 250');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', '시간 구간별 주문금액 선 그래프. 아래 표에서 구간별 금액을 확인할 수 있습니다.');
  const series = trend.series;
  const values = series.filter(p => p.sales !== null).map(p => p.sales);
  const low = Math.min(0, ...values);
  const high = Math.max(1, ...values);
  const x = i => 88 + i * 599 / Math.max(1, series.length - 1);
  const y = value => 205 - (value - low) * 175 / (high - low);
  const money = new Intl.NumberFormat('ko-KR', {maximumFractionDigits: 2});
  const axisMoney = new Intl.NumberFormat('ko-KR', {notation: 'compact', maximumFractionDigits: 2});
  function draw(tag, attributes, label) {
    const el = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
    if (label !== undefined) el.textContent = label;
    svg.append(el);
    return el;
  }
  draw('text', {x: 8, y: 18}, '원');
  for (let tick = 0; tick <= 4; tick++) {
    const value = low + (high - low) * tick / 4;
    draw('line', {x1: 88, x2: 687, y1: y(value), y2: y(value), class: 'sales-grid'});
    draw('text', {x: 80, y: y(value) + 4, 'text-anchor': 'end'}, axisMoney.format(value));
  }
  const labelIndices = new Set([0, Math.floor((series.length - 1) / 2), series.length - 1]);
  for (const i of labelIndices) {
    const anchor = i === 0 ? 'start' : i === series.length - 1 ? 'end' : 'middle';
    draw('text', {x: x(i), y: 227, 'text-anchor': anchor}, formatTime(series[i].time));
  }
  let path = '';
  let penDown = false;
  series.forEach((point, i) => {
    if (point.sales === null) {
      penDown = false;
      return;
    }
    path += (penDown ? ' L' : ' M') + x(i) + ' ' + y(point.sales);
    penDown = true;
  });
  draw('path', {d: path, class: 'sales-line'});
  series.forEach((point, i) => {
    if (point.sales === null) return;
    const mark = draw('circle', {cx: x(i), cy: y(point.sales), r: 3, class: 'sales-dot'});
    const title = document.createElementNS(svgNS, 'title');
    title.textContent = formatTime(point.time) + ' · ' + money.format(point.sales) + '원';
    mark.append(title);
  });
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = '구간별 주문금액 보기';
  const table = document.createElement('table');
  const headers = table.createTHead().insertRow();
  for (const label of ['구간 시작 (KST)', '주문금액', '주문 행 수']) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = label;
    headers.append(th);
  }
  const body = table.createTBody();
  for (const point of series) {
    const row = body.insertRow();
    const values = point.sales === null ? ['미관측', '—'] :
      [money.format(point.sales) + '원', money.format(point.order_rows)];
    for (const value of [formatTime(point.time), ...values]) row.insertCell().textContent = value;
  }
  details.append(summary, table);
  panel.append(heading, note, svg, details);
  reply.after(panel);
  $('messages').scrollTop = panel.offsetTop - $('messages').offsetTop;
}

async function sendQuestion(){
  if(busy || recording)return;
  const text=$('q').value.trim();if(!text)return;
  if(text.length>3000){status('질문은 3,000자 이내로 입력하세요.');return;}
  busy=true;controls();stopSpeaking();$('q').value='';
  add('사용자',text,'me');const reply=add('AI','조회 중…','bot');status('주문·고객 데이터를 조회하고 있습니다.');
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:text})});
    const x=await r.json();
    if(!r.ok)throw new Error(typeof x.detail==='string'?x.detail:JSON.stringify(x));
    lastAnswer=typeof x.answer==='string'?x.answer:JSON.stringify(x.answer);
    reply.textContent='AI: '+lastAnswer;if(x.trend)renderSalesTrend(reply,x.trend);status('답변이 도착했습니다.');
    if($('autoSpeak').checked)speakAnswer(lastAnswer);
  }catch(e){reply.textContent='AI 오류: '+e.message;status('요청이 실패했습니다. 질문을 확인하고 다시 전송하세요.');if(!$('q').value)$('q').value=text;}
  finally{busy=false;controls();$('messages').scrollTop=$('messages').scrollHeight;}
}
$('f').addEventListener('submit',e=>{e.preventDefault();sendQuestion();});
$('q').addEventListener('keydown',e=>{if(e.key==='Enter' && !e.shiftKey && !e.isComposing){e.preventDefault();$('f').requestSubmit();}});
$('sales-example').onclick=()=>{
  $('q').value='최근 30분 매출 추이를 5분 단위로 보여줘';
  $('q').focus();
};
$('replay').onclick=()=>speakAnswer(lastAnswer);
$('stopSpeech').onclick=()=>{stopSpeaking();status('답변 읽기를 중지했습니다.');};
$('autoSpeak').onchange=()=>{if(!$('autoSpeak').checked)stopSpeaking();};
if(synth){loadVoices();synth.addEventListener('voiceschanged',loadVoices);}else{$('autoSpeak').checked=false;$('autoSpeak').disabled=true;$('stopSpeech').disabled=true;$('voice').disabled=true;}
if(Recognition && window.isSecureContext){
  recognition=new Recognition();recognition.lang='ko-KR';recognition.continuous=false;recognition.interimResults=true;recognition.maxAlternatives=1;
  recognition.onstart=()=>status('듣고 있습니다. 질문을 말씀하세요.');
  recognition.onresult=e=>{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)finalText+=e.results[i][0].transcript+' ';else interim+=e.results[i][0].transcript;}
    $('q').value=[baseText,finalText+interim].filter(Boolean).join(' ').trim();
  };
  recognition.onerror=e=>{
    recognitionError=true;
    const errors={'not-allowed':'마이크 권한이 거부되었습니다. 브라우저의 사이트 설정에서 마이크를 허용하세요.','service-not-allowed':'브라우저 음성 인식 서비스를 사용할 수 없습니다.','audio-capture':'마이크를 찾을 수 없습니다.','no-speech':'음성이 인식되지 않았습니다. 다시 시도하세요.','network':'음성 인식 서비스 연결에 실패했습니다. 인터넷 연결을 확인하세요.','language-not-supported':'한국어 음성 인식을 지원하지 않습니다.','aborted':'음성 인식을 취소했습니다.'};
    status(errors[e.error] || '음성 인식 오류: '+e.error);
  };
  recognition.onend=()=>{
    recording=false;controls();
    if(recognitionError)return;
    if(!finalText.trim()){status('확정된 음성이 없습니다. 다시 입력하거나 텍스트를 확인하세요.');return;}
    status('음성 인식을 마쳤습니다. 질문을 확인하고 전송하세요.');
    if($('autoSend').checked)sendQuestion();else $('q').focus();
  };
  $('mic').onclick=()=>{
    if(recording){recognition.stop();return;}
    stopSpeaking();baseText=$('q').value.trim();finalText='';recognitionError=false;recording=true;controls();
    try{recognition.start();}catch(e){recording=false;controls();status('음성 인식을 시작할 수 없습니다: '+e.message);}
  };
}else{status(Recognition?'마이크를 사용하려면 localhost 또는 HTTPS로 접속하세요.':'이 브라우저는 음성 인식을 지원하지 않습니다. 텍스트 입력을 사용하세요.');}
window.addEventListener('pagehide',()=>{if(recognition)recognition.abort();stopSpeaking();});
controls();
</script></body></html>
"""


@app.get('/', response_class=HTMLResponse)
def home():
    return HTML

