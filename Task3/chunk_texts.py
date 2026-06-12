from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


# Chunking settings are intentionally word-based because the task asks for
# chunks of roughly 100-300 words. Later, when the embedding model is connected,
# these values can be replaced with token-based limits if needed.
MIN_CHUNK_WORDS = 100
MAX_CHUNK_WORDS = 300
LONG_TEXT_OVERLAP_WORDS = 40

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "Task2" / "knowledge_base_filtered"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "Task3" / "chunks_filtered_v1.json"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class MarkdownSection:
    """A logical Markdown section with its heading path and source position."""

    document_title: str
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str


@dataclass
class TextUnit:
    """A paragraph-like unit inside a section.

    start_word and end_word are positions inside the current section, not inside
    the full document. They make it possible to trace a generated chunk back to
    a stable approximate place in the source section.
    """

    text: str
    start_word: int
    end_word: int


def count_words(text: str) -> int:
    """Count non-space word-like fragments.

    This simple tokenizer is enough for chunk sizing. The embedding model will
    later receive the original text, so we do not need linguistic tokenization
    here.
    """

    return len(text.split())


def normalize_heading(raw_heading: str) -> str:
    """Remove lightweight Markdown markup from a heading."""

    return raw_heading.strip().strip("#").strip()


def parse_markdown_sections(markdown: str) -> list[MarkdownSection]:
    """Split a Markdown file into sections by its heading hierarchy.

    The key rule is that a chunk should not silently mix unrelated topics. For
    that reason, the first split happens on Markdown headings. A body under
    "## History" and a body under "## Relationships" become separate logical
    sections before any word-size chunking is applied.
    """

    lines = markdown.splitlines()
    document_title = "Untitled document"
    heading_stack: list[tuple[int, str]] = []
    sections: list[MarkdownSection] = []
    current_lines: list[str] = []
    current_start_line = 1
    current_heading_path: list[str] = []

    def flush_section(end_line: int) -> None:
        """Persist the current accumulated body as one logical section."""

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

            # Everything collected before this heading belongs to the previous
            # section, so it is flushed before the heading stack is updated.
            flush_section(line_number - 1)
            current_lines = []
            current_start_line = line_number + 1

            # Keep only parent headings with a smaller level, then append the
            # new heading. This gives metadata like:
            # ["House Romanov", "History", "Background"].
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


def split_section_into_units(section_text: str) -> list[TextUnit]:
    """Split section text into paragraph-like units and record word positions.

    Blank lines are treated as paragraph boundaries. This keeps ordinary prose,
    lists, and short reference blocks together where possible. If a unit is too
    large, it is handled later by word-window splitting.
    """

    units: list[TextUnit] = []
    current_lines: list[str] = []
    word_cursor = 0

    def flush_unit() -> None:
        nonlocal current_lines, word_cursor

        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return

        word_count = count_words(text)
        units.append(
            TextUnit(
                text=text,
                start_word=word_cursor,
                end_word=word_cursor + word_count,
            )
        )
        word_cursor += word_count
        current_lines = []

    for line in section_text.splitlines():
        if line.strip():
            current_lines.append(line)
        else:
            flush_unit()

    flush_unit()
    return units


def split_long_unit(unit: TextUnit) -> list[TextUnit]:
    """Split a single oversized paragraph into overlapping word windows.

    This is the fallback splitter. It is used only when a single paragraph or
    list block is larger than MAX_CHUNK_WORDS, because cutting inside a paragraph
    is less desirable than cutting between paragraphs.
    """

    words = unit.text.split()
    result: list[TextUnit] = []
    step = MAX_CHUNK_WORDS - LONG_TEXT_OVERLAP_WORDS

    for start in range(0, len(words), step):
        end = min(start + MAX_CHUNK_WORDS, len(words))
        result.append(
            TextUnit(
                text=" ".join(words[start:end]),
                start_word=unit.start_word + start,
                end_word=unit.start_word + end,
            )
        )

        if end == len(words):
            break

    return result


def build_chunk_text(document_title: str, heading_path: list[str], body: str) -> str:
    """Add lightweight context headers to chunk text before embedding.

    Header context improves retrieval quality: a paragraph saying "she arrived"
    becomes easier to interpret when the embedded text also contains the document
    and section names. The original metadata is still stored separately.
    """

    context_lines = [f"# {document_title}"]

    if heading_path:
        # Avoid duplicating the document title if it is already the first heading.
        section_path = " > ".join(
            heading for heading in heading_path if heading != document_title
        )
        if section_path:
            context_lines.append(f"Section: {section_path}")

    return "\n".join(context_lines) + "\n\n" + body.strip()


def chunk_section(section: MarkdownSection) -> list[dict]:
    """Create size-limited chunks from one Markdown section.

    The splitter first aggregates complete paragraph units until adding another
    unit would exceed MAX_CHUNK_WORDS. Very short sections are kept as one chunk
    because preserving logical meaning is more important than forcing every
    chunk over MIN_CHUNK_WORDS.
    """

    units = split_section_into_units(section.text)
    chunks: list[dict] = []
    current_units: list[TextUnit] = []
    current_word_count = 0

    def flush_chunk() -> None:
        nonlocal current_units, current_word_count

        if not current_units:
            return

        body = "\n\n".join(unit.text for unit in current_units)
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
                "start_word": current_units[0].start_word,
                "end_word": current_units[-1].end_word,
                "word_count": current_word_count,
            }
        )
        current_units = []
        current_word_count = 0

    for unit in units:
        unit_word_count = count_words(unit.text)

        if unit_word_count > MAX_CHUNK_WORDS:
            # Do not mix already collected paragraphs with a large paragraph:
            # first emit the clean paragraph-level chunk, then split the large
            # unit with a word-window fallback.
            flush_chunk()

            for long_piece in split_long_unit(unit):
                long_body = long_piece.text
                chunks.append(
                    {
                        "text": build_chunk_text(
                            section.document_title,
                            section.heading_path,
                            long_body,
                        ),
                        "body": long_body,
                        "document_title": section.document_title,
                        "section_path": section.heading_path,
                        "section_start_line": section.start_line,
                        "section_end_line": section.end_line,
                        "start_word": long_piece.start_word,
                        "end_word": long_piece.end_word,
                        "word_count": count_words(long_body),
                    }
                )
            continue

        if current_units and current_word_count + unit_word_count > MAX_CHUNK_WORDS:
            flush_chunk()

        current_units.append(unit)
        current_word_count += unit_word_count

    flush_chunk()
    return chunks


def build_chunks(source_dir: Path) -> list[dict]:
    """Read every Markdown file from the knowledge base and return chunks."""

    all_chunks: list[dict] = []
    markdown_files = sorted(source_dir.glob("*.md"))

    for source_file in markdown_files:
        markdown = source_file.read_text(encoding="utf-8")
        sections = parse_markdown_sections(markdown)
        file_chunks: list[dict] = []

        for section in sections:
            file_chunks.extend(chunk_section(section))

        for chunk_index, chunk in enumerate(file_chunks):
            chunk_id = f"{source_file.stem}:{chunk_index:04d}"
            chunk.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "source_file": source_file.name,
                    "source_path": str(source_file.relative_to(REPO_ROOT)),
                }
            )
            all_chunks.append(chunk)

    return all_chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save chunks as UTF-8 JSON for the next indexing step."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Task2 knowledge base Markdown files into RAG chunks."
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
        help="Path to output JSON with chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {args.source_dir}")

    chunks = build_chunks(args.source_dir)
    save_chunks(chunks, args.output)

    total_words = sum(chunk["word_count"] for chunk in chunks)
    print(f"Source directory: {args.source_dir}")
    print(f"Output file: {args.output}")
    print(f"Markdown files: {len(list(args.source_dir.glob('*.md')))}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk words: min={min(chunk['word_count'] for chunk in chunks)}, "
          f"max={max(chunk['word_count'] for chunk in chunks)}, "
          f"total={total_words}")


if __name__ == "__main__":
    main()
