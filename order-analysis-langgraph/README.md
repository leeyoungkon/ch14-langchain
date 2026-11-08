# stud12 주문 매출 추이

기존 Order Chat에 매출 추이 분석용 LangGraph와 선 그래프를 추가한 버전입니다.
FastAPI 앱은 stud12 네임스페이스에서 실행하고, 모델 추론은 기존 Ollama 서비스에 맡깁니다.

## 1. 환경 설정

`k8s.yaml`의 ConfigMap 값을 실제 서비스 주소에 맞춥니다.

| 항목 | 제공된 설정 |
|---|---|
| `ORDERS_PARQUET_URI` | `s3://warehouse/raw/stud12/**/*.parquet`. ETL이 다른 버킷이나 접두사에 저장하면 변경합니다. |
| `S3_ENDPOINT` | 기존 첨부 코드의 주소를 그대로 넣었습니다. AIStor가 `aistor-system`에 설치됐다면 실제 **S3 API** 서비스 이름, 네임스페이스, 포트, HTTP/HTTPS를 확인해 바꿉니다. |
| `S3_VERIFY_SSL` | 기본값은 `true`입니다. |
| `CUSTOMER_MCP_URL` | 기존 고객 MCP가 default 네임스페이스에 있고 8000번 포트의 /mcp를 제공한다고 가정합니다. |
| `OLLAMA_BASE_URL` | 기존 GPU 서버의 `ollama.default.svc.cluster.local:11434`입니다. |
| `CHAT_MODEL` | `qwen3:8b`입니다. 공유 Ollama에 해당 모델이 설치되어 있어야 합니다. |

학생 계정에 공유 서비스 조회 권한이 제한되어 있으면, AIStor 서비스 이름과 포트는 관리자에게 확인합니다.
MinIO 키는 주문 Parquet 경로를 조회할 수 있는 키를 사용합니다.

## 2. Docker 이미지 만들기

Docker가 설치된 PC에서 실행합니다. 아래 Docker Hub ID는 실제 ID로 바꿉니다.
Linux x86-64 GPU 서버에 배포하는 예시입니다.

```bash
docker login
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_ID/order-sales-stud12:1.1 .
docker push YOUR_DOCKERHUB_ID/order-sales-stud12:1.1
```

`k8s.yaml`의 `image:`에도 위에서 푸시한 태그를 넣습니다. 비공개 이미지는 네임스페이스에 이미지 풀 인증 설정이 필요합니다.

## 3. stud12 계정으로 배포

아래 명령은 Bash 기준입니다. kubeconfig 컨텍스트 이름이 다르면 `stud12` 부분을 수정합니다.
MinIO 키는 입력 시 화면에 표시되지 않으며, 터미널의 명령 기록에도 값 자체가 남지 않습니다.

```bash
kubectl config use-context stud12
kubectl auth can-i create deployments -n stud12
kubectl auth can-i create pods --subresource=portforward -n stud12

read -rsp 'MinIO Access Key: ' minio_access; printf '\n'
read -rsp 'MinIO Secret Key: ' minio_secret; printf '\n'
printf 'S3_ACCESS_KEY=%s\nS3_SECRET_KEY=%s\n' "$minio_access" "$minio_secret" |
  kubectl -n stud12 create secret generic order-sales-minio --from-env-file=/dev/stdin
unset minio_access minio_secret

kubectl apply -f k8s.yaml
kubectl -n stud12 rollout status deployment/order-sales-stud12
kubectl -n stud12 logs deployment/order-sales-stud12 --tail=50
kubectl -n stud12 port-forward svc/order-sales-stud12 18012:8000
```

Secret이 이미 있으면 최초 생성 명령은 생략합니다. ConfigMap을 배포 후 수정했다면
`kubectl -n stud12 rollout restart deployment/order-sales-stud12`로 다시 읽게 합니다.

포트포워드를 실행한 컴퓨터의 브라우저에서 `http://127.0.0.1:18012`를 엽니다.
GPU 서버 SSH 터미널에서 포트포워드를 실행하고 PC 브라우저로 보려면, PC의 별도 터미널에서
`ssh -N -L 18012:127.0.0.1:18012 stud12@<GPU_SERVER_SSH_HOST>`로 로컬 포트를 연결합니다.

## 4. 질문 예시

- 최근 30분 매출 추이를 5분 단위로 보여줘
- 최근 2시간 주문금액 변동을 10분 단위로 설명해 줘
- 어제 매출 그래프를 시간별로 보여줘
- 오늘 매출액이 가장 큰 고객의 매출 상황 그려줘

매출 지표는 주문금액의 `SUM(total_amount)`입니다. 1위 고객 추이는 같은 조회 기간에
고객별 주문금액 합계로 순위를 계산합니다. 동률이면 공동 1위 인원과 함께 ID 정렬 기준 첫 고객의 추이를 표시합니다. 원본 `order_time`이 KST로 기록된다는 기존 실습 전제를 따릅니다.
현재 진행 중인 구간은 제외합니다. 조회 범위에서 처음과 마지막으로 관측한 주문 구간 사이에 비어 있는 구간은 0원으로 채우고,
양 끝의 미관측 구간은 공백으로 표시합니다. 차트 아래에 조회 범위와 마지막 관측 주문 시각을 표시합니다.
전체 주문 추이의 기간은 최근 N분·N시간·N일, 오늘, 어제를 지원합니다.
일반 주문·고객 질문은 기존 LangChain 에이전트가 처리합니다.

`/health`는 앱 기동 확인용입니다. MinIO와 Ollama 연결은 브라우저에서 매출 추이 질의를 실행해 확인합니다.

## 5. AIStor 연결 실패 시

관리자 컨텍스트에서 실제 S3 API Service를 확인합니다.

```bash
kubectl -n aistor-system get svc
kubectl -n aistor-server get svc
```

S3 API 주소는 `<http 또는 https>://<서비스명>.<네임스페이스>.svc.cluster.local:<Service PORT>`입니다.
Service 포트가 TLS 443이면 예를 들어 `https://aistor-server.aistor-server.svc.cluster.local:443`입니다.
AIStor가 다른 네임스페이스나 서비스명으로 설치되어 있다면 실제 이름을 넣습니다.

이미 배포했다면 `kubectl -n stud12 edit configmap order-sales-stud12`로 `S3_ENDPOINT`를 수정한 후
`kubectl -n stud12 rollout restart deployment/order-sales-stud12`로 적용할 수 있습니다.
이미지를 1.1로 빌드하고 푸시한 기존 배포는 아래처럼 업데이트하면 ConfigMap 수정값도 새 파드에 적용됩니다.

```bash
kubectl -n stud12 set image deployment/order-sales-stud12 order-sales=YOUR_DOCKERHUB_ID/order-sales-stud12:1.1
kubectl -n stud12 rollout status deployment/order-sales-stud12
```
