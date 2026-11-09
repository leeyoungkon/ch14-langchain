"""Fixed, read-only aggregate queries against order Parquet in AIStor/S3."""
from urllib.parse import urlparse
import duckdb
from config import (ORDERS_URI, S3_ENDPOINT, S3_KEY, S3_SECRET,
                    S3_VERIFY_SSL, S3_REGION)


def connection():
    conn = duckdb.connect(':memory:')
    conn.execute('LOAD httpfs')
    if ORDERS_URI.startswith('s3://'):
        if not S3_KEY or not S3_SECRET:
            raise RuntimeError('S3_ACCESS_KEY 및 S3_SECRET_KEY 환경변수가 필요합니다.')
        u = urlparse(S3_ENDPOINT)
        endpoint = u.netloc or u.path
        ssl = (u.scheme == 'https')
        def q(value):
            return "'" + str(value).replace("'", "''") + "'"
        # Secret exists only in this transient in-memory DuckDB connection.
        conn.execute(
            "CREATE SECRET s3_local (TYPE S3, PROVIDER config, "
            f"KEY_ID {q(S3_KEY)}, SECRET {q(S3_SECRET)}, "
            f"REGION {q(S3_REGION)}, ENDPOINT {q(endpoint)}, "
            f"URL_STYLE 'path', USE_SSL {str(ssl).lower()}, "
            f"VERIFY_SSL {str(S3_VERIFY_SSL).lower()})"
        )
    return conn


def qualified_source():
    return 'read_parquet(?, union_by_name=true)'


def schema_check(conn):
    cur = conn.execute(
        'SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 0', [ORDERS_URI])
    cols = {c[0].lower(): c[0] for c in cur.description}
    required = ['order_id', 'customer_id', 'product_id', 'quantity', 'total_amount']
    absent = [k for k in required if k not in cols]
    date_col = next((cols[k] for k in ('order_time', 'order_date', 'created_at', 'ordered_at') if k in cols), None)
    if absent or not date_col:
        raise RuntimeError('Parquet 스키마 불일치: 없는 컬럼=' + ','.join(absent)
                           + '; 날짜 컬럼(order_time/order_date/created_at/ordered_at) 필요. 실제 컬럼=' + ','.join(cols))
    return date_col


def fetch_series(kind: str, days: int):
    if kind not in ('product', 'customer'):
        raise ValueError('Invalid analysis kind')
    group_col = 'product_id' if kind == 'product' else 'customer_id'
    conn = connection()
    try:
        date_col = schema_check(conn)
        # Validate identifier from schema lookup before embedding as a SQL identifier.
        if not date_col.replace('_', '').isalnum():
            raise ValueError('Invalid date column')
        # Include the most recent date found in the dataset, not necessarily system today.
        # This permits a 2026 classroom dataset to be analyzed after the workshop.
        sql = f'''
        WITH raw AS (
            SELECT CAST({date_col} AS DATE) AS day,
                   CAST({group_col} AS VARCHAR) AS entity,
                   TRY_CAST(quantity AS DOUBLE) AS quantity,
                   TRY_CAST(total_amount AS DOUBLE) AS revenue,
                   CAST(order_id AS VARCHAR) AS order_id
            FROM read_parquet(?, union_by_name=true)
        ), bounds AS (SELECT MAX(day) AS last_day FROM raw),
        daily AS (
            SELECT r.day, r.entity,
                   COUNT(DISTINCT r.order_id) AS orders,
                   SUM(r.quantity) AS quantity,
                   SUM(r.revenue) AS revenue
            FROM raw r CROSS JOIN bounds b
            WHERE r.day BETWEEN b.last_day - (CAST(? AS INTEGER) - 1) AND b.last_day
              AND r.entity IS NOT NULL
            GROUP BY r.day, r.entity
        ), ranks AS (
            SELECT entity, SUM(revenue) AS total_revenue,
                   ROW_NUMBER() OVER (ORDER BY SUM(revenue) DESC, entity) AS rn
            FROM daily GROUP BY entity
        )
        SELECT d.day, d.entity, d.orders, d.quantity, d.revenue
        FROM daily d JOIN ranks k ON d.entity=k.entity
        WHERE k.rn <= 5
        ORDER BY d.day, k.rn
        '''
        rows = conn.execute(sql, [ORDERS_URI, days]).fetchall()
        boundary = conn.execute(
            f'SELECT MAX(CAST({date_col} AS DATE)) FROM read_parquet(?, union_by_name=true)',
            [ORDERS_URI]).fetchone()[0]
        # Total metrics across ALL entities; top five is visualized separately.
        total_sql = f'''
        WITH src AS (
            SELECT CAST({date_col} AS DATE) AS d, order_id, quantity, total_amount
            FROM read_parquet(?, union_by_name=true)
        ), b AS (SELECT MAX(d) AS newest FROM src)
        SELECT COUNT(DISTINCT order_id), SUM(TRY_CAST(quantity AS DOUBLE)),
               SUM(TRY_CAST(total_amount AS DOUBLE))
        FROM src CROSS JOIN b
        WHERE d BETWEEN b.newest - (CAST(? AS INTEGER) - 1) AND b.newest
        '''
        totals = conn.execute(total_sql, [ORDERS_URI, days]).fetchone()
    finally:
        conn.close()
    return {
        'kind': kind,
        'latest_data_date': str(boundary) if boundary else None,
        'days': days,
        'totals': {'orders': int(totals[0] or 0),
                   'quantity': round(float(totals[1] or 0), 2),
                   'revenue': round(float(totals[2] or 0), 2)},
        'rows': [dict(day=str(day), entity=str(ent), orders=int(orders),
                      quantity=round(float(qty or 0), 2),
                      revenue=round(float(rev or 0), 2))
                 for day, ent, orders, qty, rev in rows],
    }


def make_chart(result: dict):
    rows = result['rows']
    days = sorted({r['day'] for r in rows})
    entities = sorted({r['entity'] for r in rows},
                      key=lambda e: -sum(r['revenue'] for r in rows if r['entity'] == e))
    series = []
    for metric in ('quantity', 'revenue'):
        series.append({
            'type': 'line', 'metric': metric,
            'title': ('제품별 판매' if result['kind'] == 'product' else '고객별 주문')
                     + (' 수량 추이' if metric == 'quantity' else ' 금액 추이'),
            'labels': days,
            'datasets': [
                {'label': entity, 'data': [next((r[metric] for r in rows
                                              if r['day'] == day and r['entity'] == entity), 0)
                                            for day in days]}
                for entity in entities
            ]
        })
    return series
