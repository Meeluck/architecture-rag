FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    OLLAMA_URL=http://host.docker.internal:11434/api/chat \
    OLLAMA_MODEL=gemma3:4b

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY Task4/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

# Cache the embedding model at image build time so the REPL can start without
# downloading it on the first container run.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN mkdir -p /app/Task3/faiss_index /app/Task3/embeddings

COPY Task3/faiss_index/chunks_filtered_v2.index /app/Task3/faiss_index/chunks_filtered_v2.index
COPY Task3/embeddings/chunks_filtered_v2_metadata.json /app/Task3/embeddings/chunks_filtered_v2_metadata.json
COPY Task4 /app/Task4
COPY Task5 /app/Task5

CMD ["python", "Task4/rag_pipeline.py", "--show-sources", "--protection-preset", "all"]
