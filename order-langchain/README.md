# Order Chat (LangChain + DuckDB + MinIO + Customer MCP)

## What is implemented
- LangChain `create_agent` with Ollama Gemma 12B
- Query MinIO orders Parquet via DuckDB virtual table `orders`
- Discover customer MCP tool schemas and call existing read-only customer tools
- Multi-turn history keyed by `session_id` (in-memory; lost on restart)
- FastAPI web UI (`GET /`) and JSON API (`POST /chat`)

## Required confirmation
- `ORDERS_PARQUET_URI`: actual Parquet bucket/prefix, not necessarily example below
- Customer MCP URL and read-only tools (`get_customer_info`, `search_customer` assumed names); MCP must implement Streamable HTTP
- Existing customer schema does not confirm `address`: cannot answer precise address unless MCP exposes it
- `order_time` is assumed to be stored as Korean local time (Asia/Seoul)

## Local run
```powershell
cd order-langchain
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env for locally reachable Ollama/MinIO/MCP endpoints and real credentials.
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000

## Kubernetes run
```powershell
docker build -t YOUR_DOCKERHUB_USER/order-langchain:1.0 .
docker push YOUR_DOCKERHUB_USER/order-langchain:1.0
# Set URL, Parquet prefix and model in configmap (use actual service URLs):
kubectl create configmap order-langchain-config --from-literal=OLLAMA_BASE_URL=http://ollama.default.svc.cluster.local:11434 --from-literal=CHAT_MODEL=gemma3:12b --from-literal=CUSTOMER_MCP_URL=http://customer-mcp:8000/mcp --from-literal=S3_ENDPOINT=http://aistor-server.aistor-server.svc.cluster.local:9000 --from-literal=ORDERS_PARQUET_URI=s3://warehouse/raw/orders/**/*.parquet
kubectl create secret generic order-langchain-secret --from-literal=S3_ACCESS_KEY=YOUR_ACCESS_KEY --from-literal=S3_SECRET_KEY=YOUR_SECRET_KEY
# replace YOUR_DOCKERHUB_USER in k8s.yaml
kubectl apply -f k8s.yaml
kubectl port-forward svc/order-langchain 18000:8000
```
Then visit http://127.0.0.1:18000

## Example JSON request
```powershell
$body = @{ session_id='s001'; message='오늘 주문액 1위 고객과 그 고객의 오늘 최다 품목, 어제 최다 품목, 고객 주소는?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:18000/chat -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## Operational cautions
- Proof-of-concept only: no authentication or per-user authorization, therefore do not expose outside a trusted network.
- This service enforces one SELECT against logical `orders` and caps returned rows; more controls are needed for production (query timeout, workload quota, extensive SQL whitelisting, audit).
- Tool calling reliability depends on Gemma variant; inspect model behavior with real questions.
- InMemorySaver history is lost on restart and is per process; use persistent LangGraph checkpoint for production.
- MCP endpoint must be Streamable HTTP, NOT legacy SSE or ordinary HTTP JSON. Adapt transport if needed.
- Docker first boot may download the DuckDB `httpfs` extension; an air-gapped environment requires pre-bundling it.
