# LangGraph 멀티 에이전트 주문 분석

기존 주문 Parquet를 분석하는 **독립 배포용 대체 서비스**입니다. 기존 `order-analysis-langgraph`는 삭제하거나 변경하지 않습니다.

## 처리

- Manager: Ollama 기반 질문 분류 → `product`, `customer` 또는 둘 다 선택
- Product Agent: DuckDB SQL로 제품별 일별 판매량 및 매출 TOP 5
- Customer Agent: DuckDB SQL로 고객 ID별 주문수량 및 매출 TOP 5
- Summarizer: Ollama가 집계 설명 생성
- Dashboard: FastAPI + Chart.js. 2가지 지표 × 호출한 에이전트의 선형 차트 출력

**Customer MCP를 연동합니다.** Customer Agent는 Parquet에서 고객 ID별 주문·매출을 SQL로 집계한 후, MCP의 `get_customer_info(customer_id)`로 상위 5명 고객명·등급·지역을 조회해 그래프와 보고서에 반영합니다. 질문에 고객명 검색 의도가 있을 때 `search_customer(customer_name)`도 호출합니다. MCP 응답의 주소(`address`)는 개인정보 보호를 위해 수집 후 즉시 제거하며 API/LLM에 전달하지 않습니다. MCP 실패 시에도 Parquet 추이 분석은 정상 수행되지만 고객 정보는 표시하지 않습니다.

중요: 이름 검색은 후보 고객 목록을 제공하지만 현 버전은 고객명 검색 대상의 주문만 필터링하지 않습니다. 이름별 실제 주문 시계열 검색 기능은 식별자 매핑 후 DuckDB 필터 쿼리를 추가해야 합니다.

## 데이터 스키마

Parquet 필수 컬럼: `order_id`, `customer_id`, `product_id`, `quantity`, `total_amount`.
날짜 컬럼은 `order_time`, `order_date`, `created_at`, `ordered_at` 중 하나.
일자별 TOP5는 기간 내 누적 금액 순. 전체 수치는 모든 제품/고객을 포함합니다.
입력한 기간은 **저장된 가장 최신 날짜**를 포함하는 N일입니다. 오늘 기준과는 다를 수 있습니다.

## 배포

```bash
docker build -t YOUR_DOCKERHUB_ID/order-multiagent:1.0 .
docker push YOUR_DOCKERHUB_ID/order-multiagent:1.0
kubectl -n stud12 create secret generic order-multiagent-s3 \
  --from-literal=S3_ACCESS_KEY='YOUR_ACCESS_KEY' \
  --from-literal=S3_SECRET_KEY='YOUR_SECRET_KEY'
# k8s/deployment.yaml의 YOUR_DOCKERHUB_ID 수정
kubectl apply -f k8s/deployment.yaml
kubectl -n stud12 rollout status deploy/order-multiagent
kubectl -n stud12 port-forward svc/order-multiagent 18000:8000
```

브라우저: http://localhost:18000

Windows PowerShell에서는 `kubectl create secret` 인수를 한 줄로 쓰거나 PowerShell 백틱 줄 연결을 사용하세요.

로컬에서 실행하려면 환경변수 설정 후 `pip install -r requirements.txt`, `uvicorn app:app --reload`를 사용합니다. HTTPS에 자체 서명 인증서를 쓴다면 `S3_VERIFY_SSL=false`는 실습에서만 사용하십시오.

## 헬스 및 문제 확인

```bash
kubectl -n stud12 logs deploy/order-multiagent --tail=100
kubectl -n stud12 exec deploy/order-multiagent -- printenv ORDERS_PARQUET_URI
curl http://localhost:18000/health
```

- HTTPS AIStor 엔드포인트는 `aistor-server...:443`으로 설정했습니다. http/9000으로 변경하지 마세요.
- AIStor 접근 실패(403 등): Secret 키 및 bucket 권한 검사.
- No files found: `ORDERS_PARQUET_URI` 경로와 실제 Parquet 파일 확인.
- schema mismatch: 오류에 표시된 컬럼을 보고 `db.py` 조정.
- Chart.js CDN 차단: chart.umd.min.js를 내려받아 로컬에 서빙하도록 HTML 수정.
- 분석 중 오류 502: Pod 로그 확인. Ollama 응답은 180초 타임아웃.
- `/health`는 프로세스의 응답만 확인합니다. 외부 서비스 연결까지 검사하지 않습니다.

## 보안

S3 credential은 ConfigMap이 아닌 Kubernetes Secret으로 전달합니다. 실습용 관리자 키보다 학생별 최소권한 키를 권장합니다. 관리자용 외부 서비스로 공개하지 말고 port-forward/인증 프록시 뒤에서 사용하십시오. 프롬프트에서 선택할 수 있는 SQL은 없고 코드에 고정된 읽기 전용 집계 SQL만 실행합니다.

## Customer MCP 연동 확인

- 실제 MCP: `http://customer-mcp.default.svc.cluster.local:8000/mcp` (ConfigMap에서 변경)
- Streamable HTTP MCP Python SDK v2 클라이언트로 `get_customer_info`, `search_customer` 호출
- `kubectl -n stud12 logs deploy/order-multiagent --tail=100`에서 실패 확인 가능
- `customer_info`는 `/api/analyze` JSON에 고객 기본정보(주소 제외)를 포함합니다.
- 그래프 범례는 MCP 조회가 성공한 경우 `고객명 (customer_id)` 형태로 표시됩니다.
- 고객 DB를 직접 연결하지 않고 반드시 기존 MCP 서비스만 경유합니다.
