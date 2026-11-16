import os

ORDERS_URI = os.getenv('ORDERS_PARQUET_URI', 's3://warehouse/raw/stud12/**/*.parquet')
S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'https://aistor-server.aistor-server.svc.cluster.local:443')
S3_KEY = os.getenv('S3_ACCESS_KEY', 'admin')
S3_SECRET = os.getenv('S3_SECRET_KEY', 'admin1234')
S3_VERIFY_SSL = os.getenv('S3_VERIFY_SSL', 'false').lower() == 'true'
S3_REGION = os.getenv('S3_REGION', 'us-east-1')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://ollama.default.svc.cluster.local:11434').rstrip('/')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'qwen3:8b')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '180'))
MAX_DAYS = 90

CUSTOMER_MCP_URL = os.getenv('CUSTOMER_MCP_URL', 'http://customer-mcp:8000/mcp')
