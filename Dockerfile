FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY frontend/ ./frontend/

RUN mkdir -p ./data/mem0_chroma

# Pre-download ChromaDB's ONNX embedding model at build time so it's
# baked into the image and never downloaded on cold start (avoids OOM).
RUN python3 -c "import chromadb; c = chromadb.Client(); c.get_or_create_collection('warmup')"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "75"]
