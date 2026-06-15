# Подготовка образа

## Docker-запуск RAG-бота в REPL

Dockerfile находится в `Task5/Dockerfile`. Его нужно собирать из корня репозитория, потому что образ копирует артефакты из `Task3`, `Task4` и `Task5`.

### Что должно быть готово до сборки

1. Локальный Ollama должен быть установлен и запущен на хосте.
2. Модель генерации должна быть скачана в Ollama:

    ```bash
    ollama pull gemma3:4b
    ```

3. Должны существовать runtime-артефакты RAG:

    ```text
    Task3/faiss_index/chunks_filtered_v2.index
    Task3/embeddings/chunks_filtered_v2_metadata.json
    Task4/rag_pipeline.py
    Task4/requirements.txt
    ```

4. Во время сборки Docker-образу нужен интернет, чтобы установить Python-зависимости и скачать embedding-модель `sentence-transformers/all-MiniLM-L6-v2` в cache образа.

### Сборка образа

Команду нужно выполнять из корня репозитория:

```bash
docker build -f Task5/Dockerfile -t architecture-rag .
```

Если на Apple Silicon возникнут проблемы с wheel-пакетами `faiss-cpu` или `torch`, можно собрать amd64-образ:

```bash
docker build --platform linux/amd64 -f Task5/Dockerfile -t architecture-rag .
```

### Запуск REPL

На macOS и Docker Desktop контейнер может обратиться к Ollama на хосте через `host.docker.internal`:

```bash
docker run --rm -it \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat \
  -e OLLAMA_MODEL=gemma3:4b \
  architecture-rag
```

Если контейнер не может подключиться к Ollama, проверьте, что Ollama API доступен на хосте:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3:4b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

После запуска появится интерактивный режим:

```text
RAG REPL. Type a question or 'exit' to stop.

>
```

Для выхода:

```text
exit
```

### Проверка слоев защиты из контейнера

В образ также попадает Python-скрипт проверки prompt injection:

```bash
docker run --rm -it \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat \
  -e OLLAMA_MODEL=gemma3:4b \
  architecture-rag \
  python Task5/run_prompt_injection_protection_matrix.py --max-output-tokens 220
```

### Настройка для Linux

Если `host.docker.internal` недоступен, можно передать host gateway:

```bash
docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat \
  -e OLLAMA_MODEL=gemma3:4b \
  architecture-rag
```

### Что находится внутри образа

- Python runtime и зависимости из `Task4/requirements.txt`;
- cached embedding-модель `sentence-transformers/all-MiniLM-L6-v2`;
- `Task4/rag_pipeline.py`;
- `Task5/run_prompt_injection_protection_matrix.py`;
- FAISS index `Task3/faiss_index/chunks_filtered_v2.index`;
- metadata `Task3/embeddings/chunks_filtered_v2_metadata.json`.

По умолчанию контейнер запускает:

```bash
python Task4/rag_pipeline.py --show-sources --protection-preset all
```
