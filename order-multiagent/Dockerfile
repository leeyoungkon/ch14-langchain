FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-install DuckDB httpfs while image has internet; avoids Kubernetes offline extension installs
RUN python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL httpfs'); c.close()"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
