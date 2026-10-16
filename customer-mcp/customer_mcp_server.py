import os
import pymysql

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("customer-db")


def get_connection():
    return pymysql.connect(
        host=os.getenv("CUSTOMER_DB_HOST", "mysql"),
        port=int(os.getenv("CUSTOMER_DB_PORT", "3306")),
        user=os.getenv("CUSTOMER_DB_USER", "root"),
        password=os.getenv("CUSTOMER_DB_PASSWORD", "root1234"),
        database=os.getenv("CUSTOMER_DB_NAME", "customerdb"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )


@mcp.tool()
def get_customer_info(customer_id: str) -> dict:
    """
    customer_id로 고객 기본정보를 조회한다.
    고객명, 등급, 지역, 주소를 반환한다.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            sql = """
            SELECT
                customer_id,
                customer_name,
                grade,
                region,
                address
            FROM customer
            WHERE customer_id = %s
            """

            cursor.execute(sql, (customer_id,))
            result = cursor.fetchone()

            if result is None:
                return {
                    "found": False,
                    "customer_id": customer_id
                }

            return {
                "found": True,
                **result
            }

    finally:
        conn.close()


@mcp.tool()
def search_customer(customer_name: str) -> list:
    """
    고객명으로 고객을 검색한다.
    검색 결과에 고객명, 등급, 지역, 주소를 포함한다.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            sql = """
            SELECT
                customer_id,
                customer_name,
                grade,
                region,
                address
            FROM customer
            WHERE customer_name LIKE %s
            LIMIT 20
            """

            cursor.execute(
                sql,
                (f"%{customer_name}%",)
            )

            return cursor.fetchall()

    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        stateless_http=True,
        json_response=True
    )