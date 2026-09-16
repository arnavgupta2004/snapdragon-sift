# Backend: FastAPI + the LangGraph agent + retrieval/personalization stack.
FROM python:3.11-slim

# libgomp1 is required by lightgbm at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY eval/ eval/
COPY ui/streamlit_app.py ui/streamlit_app.py
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
