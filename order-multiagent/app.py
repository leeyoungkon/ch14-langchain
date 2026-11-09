from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from agents import workflow

app = FastAPI(title='LangGraph Order Multi-Agent')
logger = logging.getLogger('uvicorn.error')

class Query(BaseModel):
    question: str = Field(min_length=2, max_length=1000)

@app.get('/', response_class=HTMLResponse)
def index():
    return Path(__file__).parent.joinpath('templates/index.html').read_text(encoding='utf-8')

@app.get('/health')
def health():
    return {'status': 'ok'}  # no external connectivity claims

@app.post('/api/analyze')
def analyze(payload: Query):
    try:
        result = workflow.invoke({'question': payload.question})
        return {'answer': result['answer'], 'selected_agents': result['selected'],
                'period_days': result['days'], 'metrics': {
                    k: v['totals'] for k, v in result['results'].items()},
                'latest_data_date': {k: v['latest_data_date'] for k, v in result['results'].items()},
                'charts': result['charts'], 'trace': result['trace'],
                'customer_info': result.get('customer_info', {}),
                'mcp_warning': result.get('mcp_warning', '')}
    except Exception as exc:
        logger.exception('Analysis failed')
        raise HTTPException(status_code=502,
                            detail='분석 실패: 데이터 연결/스키마/LLM 서버를 확인하세요. 서버 로그 참조.') from exc
