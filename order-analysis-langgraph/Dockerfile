FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c 'import duckdb; duckdb.sql("INSTALL httpfs"); duckdb.sql("SET TimeZone = \x27Asia/Seoul\x27")'

COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
