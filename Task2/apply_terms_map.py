#!/usr/bin/env python3
"""Apply terms_map.json replacements to Markdown files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


WORD_CHAR_PATTERN = r"A-Za-z0-9_А-Яа-яЁё"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace source universe terms in Markdown files using terms_map.json."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Task2/knowledge_base_raw"),
        help="Directory with source Markdown files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Task2/knowledge_base"),
        help="Directory for transformed Markdown files.",
    )
    parser.add_argument(
        "--terms-map",
        type=Path,
        default=Path("Task2/terms_map.json"),
        help="JSON file with replacement mapping.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("Task2/replacement_report.json"),
        help="Path for replacement report JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the output directory if they already exist.",
    )
    return parser.parse_args()


def load_replacements(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    replacements = data.get("flat_map")

    if not isinstance(replacements, dict):
        raise ValueError(f"{path} must contain a flat_map object")

    return {
        str(source): str(target)
        for source, target in replacements.items()
        if str(source).strip() and str(target).strip()
    }


def ordered_replacements(replacements: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(
        replacements.items(),
        key=lambda item: (-len(item[0]), item[0].lower()),
    )


def compile_pattern(source: str) -> re.Pattern[str]:
    escaped = re.escape(source)
    if source.lower().startswith(("season ", "seasons ")):
        return re.compile(rf"{escaped}(?![0-9])")

    if source[-1:].isdigit():
        return re.compile(rf"(?<![{WORD_CHAR_PATTERN}]){escaped}(?![0-9])")

    return re.compile(
        rf"(?<![{WORD_CHAR_PATTERN}]){escaped}(?![{WORD_CHAR_PATTERN}])"
    )


def replace_text(
    text: str,
    replacements: list[tuple[str, str]],
) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    result = text

    for source, target in replacements:
        pattern = compile_pattern(source)
        result, count = pattern.subn(target, result)
        if count:
            counts[source] += count

    return result, counts


def extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9а-яё]+", "-", value, flags=re.I)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "document"


def unique_output_path(out_dir: Path, stem: str, used_stems: set[str]) -> Path:
    base = slugify(stem)
    candidate = base
    index = 2

    while candidate in used_stems:
        candidate = f"{base}-{index}"
        index += 1

    used_stems.add(candidate)
    return out_dir / f"{candidate}.md"


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(out_dir.glob("*.md"))

    if existing_files and not overwrite:
        raise FileExistsError(
            f"{out_dir} already contains Markdown files. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        for path in existing_files:
            path.unlink()


def transform_files(
    input_dir: Path,
    out_dir: Path,
    replacements: dict[str, str],
    overwrite: bool,
) -> dict[str, object]:
    ensure_output_dir(out_dir, overwrite)

    ordered = ordered_replacements(replacements)
    used_stems: set[str] = set()
    total_counts: Counter[str] = Counter()
    files: list[dict[str, object]] = []

    for source_path in sorted(input_dir.glob("*.md")):
        original_text = source_path.read_text(encoding="utf-8")
        replaced_text, counts = replace_text(original_text, ordered)
        title = extract_title(replaced_text, source_path.stem)
        target_path = unique_output_path(out_dir, title, used_stems)
        target_path.write_text(replaced_text, encoding="utf-8")

        total_counts.update(counts)
        files.append(
            {
                "source_file": str(source_path),
                "target_file": str(target_path),
                "title": title,
                "replacements": dict(sorted(counts.items())),
            }
        )

    return {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "files_count": len(files),
        "total_replacements": sum(total_counts.values()),
        "terms_replaced": len(total_counts),
        "replacement_counts": dict(
            sorted(total_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "files": files,
    }


def main() -> int:
    args = parse_args()
    replacements = load_replacements(args.terms_map)
    report = transform_files(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        replacements=replacements,
        overwrite=args.overwrite,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"saved {report['files_count']} files to {args.out_dir}")
    print(f"total replacements: {report['total_replacements']}")
    print(f"saved report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
