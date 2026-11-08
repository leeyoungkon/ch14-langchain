FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY customer_mcp_server.py .

EXPOSE 8000

CMD ["python", "-u", "customer_mcp_server.py"]