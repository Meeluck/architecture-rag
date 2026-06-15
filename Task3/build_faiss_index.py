from pathlib import Path
import json
import numpy as np
import faiss


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "Task3" / "faiss_index"

INDEX_CONFIGS = [
    {
        "name": "chunks_v2",
        "embeddings_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_v2_embeddings.npy",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_v2_metadata.json",
        "index_path": OUTPUT_DIR / "chunks_v2.index",
    },
    {
        "name": "chunks_filtered_v2",
        "embeddings_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_filtered_v2_embeddings.npy",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_filtered_v2_metadata.json",
        "index_path": OUTPUT_DIR / "chunks_filtered_v2.index",
    },
]


def build_index(config: dict) -> dict:
    embeddings = np.load(config["embeddings_path"]).astype("float32")

    with open(config["metadata_path"], encoding="utf-8") as file:
        metadata = json.load(file)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embeddings count does not match metadata count for {config['name']}"
        )

    dimension = embeddings.shape[1]

    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)

    ids = np.arange(embeddings.shape[0]).astype("int64")
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, str(config["index_path"]))

    return {
        "name": config["name"],
        "index_type": "IndexIDMap(IndexFlatIP)",
        "metric": "inner_product_on_normalized_embeddings",
        "embeddings_path": str(config["embeddings_path"].relative_to(REPO_ROOT)),
        "metadata_path": str(config["metadata_path"].relative_to(REPO_ROOT)),
        "index_path": str(config["index_path"].relative_to(REPO_ROOT)),
        "vectors": int(index.ntotal),
        "dimension": int(dimension),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "vector_db": "FAISS",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "indexes": [],
    }

    for config in INDEX_CONFIGS:
        result = build_index(config)
        manifest["indexes"].append(result)
        print(result)

    manifest_path = OUTPUT_DIR / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()