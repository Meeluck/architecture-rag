from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # Older LangChain versions expose the splitter from this module.
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "RecursiveCharacterTextSplitter is not installed. "
            "Install it with: python3 -m pip install langchain-text-splitters"
        ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "Task2" / "knowledge_base_filtered"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "Task3" / "chunks_filtered_v2.json"

# The task asks for chunks of roughly 100-300 words. LangChain's splitter allows
# a custom length_function, so the limits below are word-based even though the
# recursive splitting strategy still uses character separators.
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 40

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class MarkdownSection:
    """A section of a Markdown file with source metadata."""

    document_title: str
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str


def count_words(text: str) -> int:
    """Return an approximate word count for reporting and QA."""

    return len(text.split())


def normalize_heading(raw_heading: str) -> str:
    """Normalize Markdown heading text before saving it into metadata."""

    return raw_heading.strip().strip("#").strip()


def parse_markdown_sections(markdown: str) -> list[MarkdownSection]:
    """Split Markdown into logical sections before LangChain chunking.

    RecursiveCharacterTextSplitter is good at producing size-limited chunks, but
    by itself it does not understand that two adjacent Markdown sections may be
    unrelated. This lightweight pre-pass preserves section boundaries and gives
    every generated chunk a precise heading path.
    """

    lines = markdown.splitlines()
    document_title = "Untitled document"
    heading_stack: list[tuple[int, str]] = []
    sections: list[MarkdownSection] = []
    current_lines: list[str] = []
    current_start_line = 1
    current_heading_path: list[str] = []

    def flush_section(end_line: int) -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return

        sections.append(
            MarkdownSection(
                document_title=document_title,
                heading_path=current_heading_path.copy(),
                start_line=current_start_line,
                end_line=end_line,
                text=text,
            )
        )

    for line_number, line in enumerate(lines, start=1):
        heading_match = HEADING_RE.match(line)

        if heading_match:
            heading_level = len(heading_match.group(1))
            heading_text = normalize_heading(heading_match.group(2))

            if heading_level == 1 and document_title == "Untitled document":
                document_title = heading_text

            # Save the previous section before starting a new heading scope.
            flush_section(line_number - 1)
            current_lines = []
            current_start_line = line_number + 1

            # Keep only headings that are parents of the current heading.
            heading_stack = [
                (level, title)
                for level, title in heading_stack
                if level < heading_level
            ]
            heading_stack.append((heading_level, heading_text))
            current_heading_path = [title for _, title in heading_stack]
            continue

        current_lines.append(line)

    flush_section(len(lines))
    return sections


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the LangChain splitter used by this script.

    Separators go from coarse to fine. The splitter first tries to keep Markdown
    blocks and paragraphs intact, then falls back to lines, sentences, words and
    finally characters if a block is still too large.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_WORDS,
        chunk_overlap=CHUNK_OVERLAP_WORDS,
        length_function=count_words,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n- ",
            "\n| ",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def build_chunk_text(document_title: str, heading_path: list[str], body: str) -> str:
    """Add compact document context to the text that will be embedded."""

    context_lines = [f"# {document_title}"]

    if heading_path:
        section_path = " > ".join(
            heading for heading in heading_path if heading != document_title
        )
        if section_path:
            context_lines.append(f"Section: {section_path}")

    return "\n".join(context_lines) + "\n\n" + body.strip()


def chunk_section(
    splitter: RecursiveCharacterTextSplitter,
    section: MarkdownSection,
) -> list[dict]:
    """Apply RecursiveCharacterTextSplitter to one logical Markdown section."""

    documents = splitter.create_documents(
        texts=[section.text],
        metadatas=[
            {
                "document_title": section.document_title,
                "section_path": section.heading_path,
                "section_start_line": section.start_line,
                "section_end_line": section.end_line,
            }
        ],
    )

    chunks: list[dict] = []
    for document in documents:
        body = document.page_content.strip()
        metadata = document.metadata
        start_index = metadata.get("start_index")

        chunks.append(
            {
                "text": build_chunk_text(
                    section.document_title,
                    section.heading_path,
                    body,
                ),
                "body": body,
                "document_title": section.document_title,
                "section_path": section.heading_path,
                "section_start_line": section.start_line,
                "section_end_line": section.end_line,
                # start_index is produced by LangChain when add_start_index=True.
                # It is a character offset inside the current Markdown section.
                "section_start_char": start_index,
                "section_end_char": (
                    start_index + len(body)
                    if isinstance(start_index, int)
                    else None
                ),
                "char_count": len(body),
                "word_count": count_words(body),
            }
        )

    return chunks


def build_chunks(source_dir: Path) -> list[dict]:
    """Read Markdown files and build LangChain-generated chunks."""

    splitter = build_splitter()
    all_chunks: list[dict] = []

    for source_file in sorted(source_dir.glob("*.md")):
        markdown = source_file.read_text(encoding="utf-8")
        sections = parse_markdown_sections(markdown)
        file_chunks: list[dict] = []

        for section in sections:
            file_chunks.extend(chunk_section(splitter, section))

        for chunk_index, chunk in enumerate(file_chunks):
            chunk_id = f"{source_file.stem}:v2:{chunk_index:04d}"
            chunk.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "source_file": source_file.name,
                    "source_path": str(source_file.relative_to(REPO_ROOT)),
                    "splitter": "RecursiveCharacterTextSplitter",
                    "chunk_size_words": CHUNK_SIZE_WORDS,
                    "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
                }
            )
            all_chunks.append(chunk)

    return all_chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save generated chunks as UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split Task2 knowledge base Markdown files with LangChain "
            "RecursiveCharacterTextSplitter."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory with source Markdown files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON with LangChain chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {args.source_dir}")

    chunks = build_chunks(args.source_dir)
    save_chunks(chunks, args.output)

    total_words = sum(chunk["word_count"] for chunk in chunks)
    total_chars = sum(chunk["char_count"] for chunk in chunks)
    print(f"Source directory: {args.source_dir}")
    print(f"Output file: {args.output}")
    print(f"Markdown files: {len(list(args.source_dir.glob('*.md')))}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk words: min={min(chunk['word_count'] for chunk in chunks)}, "
          f"max={max(chunk['word_count'] for chunk in chunks)}, "
          f"total={total_words}")
    print(f"Chunk chars: min={min(chunk['char_count'] for chunk in chunks)}, "
          f"max={max(chunk['char_count'] for chunk in chunks)}, "
          f"total={total_chars}")


if __name__ == "__main__":
    main()
