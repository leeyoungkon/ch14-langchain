"""Lightweight integration logic smoke test with external dependencies mocked."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents


def test_customer_enrichment(monkeypatch):
    result = {'kind': 'customer', 'days': 7, 'latest_data_date': '2026-11-09',
              'totals': {'orders': 1, 'quantity': 2, 'revenue': 500},
              'rows': [{'day': '2026-11-09', 'entity': 'C001', 'orders': 1,
                        'quantity': 2, 'revenue': 500}]}
    monkeypatch.setattr(agents, 'fetch_series', lambda kind, days: result)
    monkeypatch.setattr(agents, 'get_customer_details', lambda ids, name: {
        'top_customers': {'C001': {'customer_id': 'C001', 'customer_name': '테스트',
                                    'grade': 'VIP', 'region': '서울'}},
        'name_matches': []})
    out = agents.customer_agent({'days': 7, 'results': {}, 'charts': [],
                                  'trace': ['manager'], 'customer_name': ''})
    assert '테스트 (C001)' == out['charts'][0]['datasets'][0]['label']
    assert 'customer_mcp' in out['trace']


def test_customer_mcp_failure_does_not_break_charts(monkeypatch):
    result = {'kind': 'customer', 'days': 7, 'latest_data_date': None,
              'totals': {'orders': 1, 'quantity': 2, 'revenue': 500},
              'rows': [{'day': '2026-11-09', 'entity': 'C001', 'orders': 1,
                        'quantity': 2, 'revenue': 500}]}
    monkeypatch.setattr(agents, 'fetch_series', lambda kind, days: result)
    def fail(ids, name):
        raise OSError('offline')
    monkeypatch.setattr(agents, 'get_customer_details', fail)
    out = agents.customer_agent({'days': 7, 'results': {}, 'charts': [],
                                  'trace': ['manager']})
    assert out['charts'][0]['datasets'][0]['label'] == 'C001'
    assert out['mcp_warning']
