from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEXES = [
    {
        "name": "chunks_v1",
        "index_path": REPO_ROOT / "Task3" / "faiss_index" / "chunks_v1.index",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_v1_metadata.json",
    },
    {
        "name": "chunks_v2",
        "index_path": REPO_ROOT / "Task3" / "faiss_index" / "chunks_v2.index",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_v2_metadata.json",
    },
    {
        "name": "chunks_filtered_v1",
        "index_path": REPO_ROOT / "Task3" / "faiss_index" / "chunks_filtered_v1.index",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_filtered_v1_metadata.json",
    },
    {
        "name": "chunks_filtered_v2",
        "index_path": REPO_ROOT / "Task3" / "faiss_index" / "chunks_filtered_v2.index",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_filtered_v2_metadata.json",
    },
]

TEST_QUERIES = [
    "Who restored House Volkonsky after the Anfield Derby?",
    "What happened to Daria Romanova in Wembley?",
    "Where is Maracana located?",
]


def load_metadata(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def search_query(
    model: SentenceTransformer,
    query: str,
    index_path: Path,
    metadata_path: Path,
    k: int = 5,
) -> list[dict]:
    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(query_embedding, k)

    results = []

    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1:
            continue

        chunk = metadata[int(chunk_id)]

        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "source_path": chunk["source_path"],
            "document_title": chunk["document_title"],
            "section_path": chunk["section_path"],
            "text": chunk["text"],
        })

    return results


def print_results(index_name: str, query: str, results: list[dict]) -> None:
    print("=" * 100)
    print(f"Index: {index_name}")
    print(f"Query: {query}")
    print("-" * 100)

    for i, result in enumerate(results, start=1):
        section = " > ".join(result["section_path"])
        snippet = result["text"].replace("\n", " ")[:500]

        print(f"{i}. score={result['score']:.4f}")
        print(f"   chunk_id: {result['chunk_id']}")
        print(f"   source: {result['source_path']}")
        print(f"   title: {result['document_title']}")
        print(f"   section: {section}")
        print(f"   text: {snippet}")
        print()


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    for query in TEST_QUERIES:
        for index_config in INDEXES:
            results = search_query(
                model=model,
                query=query,
                index_path=index_config["index_path"],
                metadata_path=index_config["metadata_path"],
                k=5,
            )

            print_results(index_config["name"], query, results)


if __name__ == "__main__":
    main()