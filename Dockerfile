FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()([])" || true
COPY backend/ ./
COPY frontend/ ./frontend/
RUN mkdir -p ./data/mem0_chroma ./data/formulary_chroma
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300", "--workers", "1"]
