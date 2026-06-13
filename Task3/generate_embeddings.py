from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INPUTS = [
    # {
    #     "chunks_path": REPO_ROOT / "Task3" / "chunks_v2.json",
    #     "embeddings_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_v2_embeddings.npy",
    #     "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_v2_metadata.json",
    # },
    {
        "chunks_path": REPO_ROOT / "Task3" / "chunks_filtered_v2.json",
        "embeddings_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_filtered_v2_embeddings.npy",
        "metadata_path": REPO_ROOT / "Task3" / "embeddings" / "chunks_filtered_v2_metadata.json",
    },
]


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_metadata(chunks: list[dict]) -> list[dict]:
    metadata = []

    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "source_file": chunk["source_file"],
            "source_path": chunk["source_path"],
            "document_title": chunk["document_title"],
            "section_path": chunk["section_path"],
            "word_count": chunk["word_count"],
            "text": chunk["text"],
        })

    return metadata


def generate_for_file(model: SentenceTransformer, config: dict) -> None:
    chunks = load_chunks(config["chunks_path"])
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    config["embeddings_path"].parent.mkdir(parents=True, exist_ok=True)

    np.save(config["embeddings_path"], embeddings.astype("float32"))

    metadata = prepare_metadata(chunks)
    config["metadata_path"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Chunks file: {config['chunks_path']}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings: {config['embeddings_path']}")
    print(f"Saved metadata: {config['metadata_path']}")


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    for config in INPUTS:
        generate_for_file(model, config)


if __name__ == "__main__":
    main()