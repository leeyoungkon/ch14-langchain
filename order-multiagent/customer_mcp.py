"""Customer MCP integration; only two declared tools are used.

This module never returns addresses to the LLM or web client.
"""
import asyncio
import json
from mcp import Client
from config import CUSTOMER_MCP_URL


def _normalize(response):
    if response.is_error:
        raise RuntimeError('Customer MCP tool returned an error')
    obj = response.structured_content
    if obj is None:
        for item in response.content:
            if getattr(item, 'type', None) == 'text':
                obj = json.loads(item.text)
                break
    if isinstance(obj, dict) and set(obj) == {'result'}:
        obj = obj['result']
    return obj


def _safe(info):
    if not isinstance(info, dict):
        return None
    return {key: info.get(key) for key in ('customer_id', 'customer_name', 'grade', 'region')}


async def _fetch(customer_ids, customer_name=None):
    async with Client(CUSTOMER_MCP_URL) as client:
        # The server exposes exactly get_customer_info and search_customer.
        matches = []
        if customer_name:
            response = await client.call_tool('search_customer', {'customer_name': customer_name})
            rows = _normalize(response)
            if isinstance(rows, dict):
                rows = rows.get('result', rows.get('items', []))
            if not isinstance(rows, list):
                rows = []
            matches = [_safe(x) for x in rows[:20] if isinstance(x, dict)]
        info = {}
        for customer_id in list(dict.fromkeys(str(x) for x in customer_ids if x))[:5]:
            response = await client.call_tool('get_customer_info', {'customer_id': customer_id})
            value = _normalize(response)
            if isinstance(value, dict) and value.get('found'):
                info[customer_id] = _safe(value)
        return {'top_customers': info, 'name_matches': matches}


def get_customer_details(customer_ids, customer_name=None):
    """Run the async MCP client from a synchronous LangGraph node."""
    return asyncio.run(_fetch(customer_ids, customer_name))
