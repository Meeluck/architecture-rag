from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "Task2" / "knowledge_base"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Task2" / "knowledge_base_filtered"
DEFAULT_REPORT_PATH = REPO_ROOT / "Task2" / "filter_report.json"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_DIVIDER_RE = re.compile(r"^\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|?\s*$")

# These sections are useful on a wiki page, but they add noise to retrieval:
# galleries contain image captions, references are often empty or technical, and
# external links repeatedly mention page names without adding answerable facts.
REMOVED_SECTION_TITLES = {
    "external links",
    "gallery",
    "references",
}

NOTICE_PHRASES = (
    "this section contains a considerable amount of unverified information",
    "you can help the wiki of greater wembley",
)


def normalize_heading_title(raw_title: str) -> str:
    """Normalize a Markdown heading so it can be compared reliably."""

    title = raw_title.strip()
    title = re.sub(r"[*_`{}[\]]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.casefold()


def contains_wiki_notice(line: str) -> bool:
    """Return True when a line belongs to the wiki warning notice."""

    normalized = re.sub(r"\s+", " ", line.casefold())
    return any(phrase in normalized for phrase in NOTICE_PHRASES)


def filter_markdown(markdown: str) -> tuple[str, dict]:
    """Remove noisy sections and wiki notices from one Markdown document.

    The function preserves the rest of the document as-is. Section removal is
    heading-aware: when "## External links" is removed, all lines under it are
    skipped until the next heading with the same or higher level.
    """

    lines = markdown.splitlines()
    filtered_lines: list[str] = []
    removed_sections: Counter[str] = Counter()
    removed_notice_blocks = 0
    removed_lines = 0
    skip_section_level: int | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        heading_match = HEADING_RE.match(line)

        if skip_section_level is not None:
            if heading_match and len(heading_match.group(1)) <= skip_section_level:
                skip_section_level = None
                continue

            removed_lines += 1
            i += 1
            continue

        if heading_match:
            heading_level = len(heading_match.group(1))
            heading_title = normalize_heading_title(heading_match.group(2))

            if heading_title in REMOVED_SECTION_TITLES:
                removed_sections[heading_title] += 1
                removed_lines += 1
                skip_section_level = heading_level
                i += 1
                continue

        if contains_wiki_notice(line):
            removed_notice_blocks += 1
            removed_lines += 1
            i += 1

            # The warning is stored as a tiny Markdown table. Remove the divider
            # row and one following blank line so the remaining section text is
            # not separated by an empty table shell.
            if i < len(lines) and TABLE_DIVIDER_RE.match(lines[i]):
                removed_lines += 1
                i += 1

            if i < len(lines) and not lines[i].strip():
                removed_lines += 1
                i += 1

            continue

        filtered_lines.append(line)
        i += 1

    filtered_markdown = "\n".join(filtered_lines).strip() + "\n"
    return filtered_markdown, {
        "removed_sections": dict(sorted(removed_sections.items())),
        "removed_notice_blocks": removed_notice_blocks,
        "removed_lines": removed_lines,
        "original_lines": len(lines),
        "filtered_lines": len(filtered_markdown.splitlines()),
    }


def filter_knowledge_base(source_dir: Path, output_dir: Path) -> dict:
    """Filter every Markdown file in the knowledge base."""

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dir": str(source_dir.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "files": [],
        "summary": {
            "files_processed": 0,
            "removed_sections": {},
            "removed_notice_blocks": 0,
            "removed_lines": 0,
        },
    }

    total_removed_sections: Counter[str] = Counter()
    markdown_files = sorted(source_dir.glob("*.md"))

    for source_file in markdown_files:
        filtered_markdown, file_report = filter_markdown(
            source_file.read_text(encoding="utf-8")
        )

        output_file = output_dir / source_file.name
        output_file.write_text(filtered_markdown, encoding="utf-8")

        total_removed_sections.update(file_report["removed_sections"])
        report["summary"]["files_processed"] += 1
        report["summary"]["removed_notice_blocks"] += file_report[
            "removed_notice_blocks"
        ]
        report["summary"]["removed_lines"] += file_report["removed_lines"]

        report["files"].append(
            {
                "source_file": str(source_file.relative_to(REPO_ROOT)),
                "output_file": str(output_file.relative_to(REPO_ROOT)),
                **file_report,
            }
        )

    report["summary"]["removed_sections"] = dict(sorted(total_removed_sections.items()))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered copy of Task2 knowledge_base for cleaner RAG indexing."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory with source Markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where filtered Markdown files will be written.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to JSON report with filtering statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = filter_knowledge_base(args.source_dir, args.output_dir)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Report: {args.report}")
    print(f"Files processed: {report['summary']['files_processed']}")
    print(f"Removed sections: {report['summary']['removed_sections']}")
    print(f"Removed notice blocks: {report['summary']['removed_notice_blocks']}")
    print(f"Removed lines: {report['summary']['removed_lines']}")


if __name__ == "__main__":
    main()
