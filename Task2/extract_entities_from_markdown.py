#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EVENT_WORDS = {
    "Ambush",
    "Assassination",
    "Assassinations",
    "Battle",
    "Conquest",
    "Council",
    "Crisis",
    "Dance",
    "Doom",
    "Expedition",
    "Fall",
    "Great War",
    "Invasion",
    "Long Night",
    "Massacre",
    "Rebellion",
    "Red Wedding",
    "Revolt",
    "Sack",
    "Siege",
    "Tournament",
    "War",
    "Wedding",
}

PLACE_HINTS = {
    "Bay",
    "Braavos",
    "Castle",
    "City",
    "Dragonstone",
    "Essos",
    "Harrenhal",
    "Kingdom",
    "Kingdoms",
    "King's Landing",
    "Landing",
    "Meereen",
    "North",
    "Oldtown",
    "Riverlands",
    "Slaver's Bay",
    "Tower",
    "Vale",
    "Wall",
    "Westeros",
    "Winterfell",
}

PLACE_TITLES = {
    "Braavos",
    "Dragonstone",
    "King's Landing",
    "Meereen",
    "Oldtown",
    "The North",
    "Wall",
    "Winterfell",
}

CHARACTER_TITLE_WORDS = {
    "Lord",
    "Lady",
    "King",
    "Queen",
    "Prince",
    "Princess",
    "Ser",
    "Maester",
    "Khal",
}

OTHER_PREFIXES = (
    "Army of",
    "Faceless",
    "Free Folk",
    "House ",
    "Night's Watch",
    "White Walkers",
)

OTHER_TERMS = {
    "Dragons",
    "Iron Throne",
    "Valyrian Steel",
    "Valyrian steel",
    "Wildfire",
}

STOP_PHRASES = {
    "And",
    "A",
    "After",
    "Background",
    "Biography",
    "Categories",
    "However",
    "In",
    "In the",
    "Killed",
    "Later",
    "Contents",
    "Current",
    "Episode",
    "Game of Thrones",
    "History",
    "House of the Dragon",
    "Main",
    "References",
    "Season",
    "See Also",
    "The",
    "This",
    "This Page",
    "When",
    "While",
    "You",
}

BAD_CANDIDATE_STARTS = {
    "After",
    "At",
    "Before",
    "During",
    "For",
    "From",
    "In",
    "On",
    "The",
    "To",
    "When",
    "While",
    "With",
}

BAD_CANDIDATE_ENDS = {
    "and",
    "for",
    "from",
    "in",
    "of",
    "the",
    "to",
    "with",
}

TOKEN = r"(?:[A-Z][a-z]+(?:'[a-z]+)?|[A-Z]{2,}|[A-Z][a-z]+-[A-Z][a-z]+)"
CONNECTOR = r"(?:of|the|and|in|to|at|for|from)"
PROPER_NOUN_RE = re.compile(
    rf"\b{TOKEN}(?:\s+(?:{CONNECTOR}|{TOKEN}))*\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract characters, places, events, and other replacement terms from Markdown files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Task2/knowledge_base_raw"),
        help="Directory with cleaned Markdown files.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("Task2/entities.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("Task2/entities.md"),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=4,
        help="Minimum number of mentions for secondary candidates.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum number of secondary candidates to include.",
    )
    return parser.parse_args()


def read_markdown_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.md"))


def extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return clean_name(match.group(1))
    return fallback


def clean_name(value: str) -> str:
    value = re.sub(r"[*_`{}]", "", value)
    value = value.replace("\\-", "-")
    value = value.replace("%27", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -:\t\n")


def strip_markdown(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_>#|]", " ", text)
    text = re.sub(r"\\-", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_title(title: str, text: str) -> str:
    normalized = title.lower()
    first_chunk = text[:2000].lower()

    if is_event(title):
        return "events"

    if title.startswith(OTHER_PREFIXES) or title in OTHER_TERMS:
        return "other_terms"

    if is_place_title(title, first_chunk):
        return "places"

    if (
        re.search(r"\b(?:born|son of|daughter of|wife of|husband of|brother of|sister of)\b", first_chunk)
        or re.search(r"\b(?:king|queen|lord|lady|prince|princess|ser|maester|khal)\b", first_chunk)
        or looks_like_person_name(title)
    ):
        return "characters"

    if normalized.startswith(("the ", "wall", "dragonstone", "winterfell")):
        return "places"

    return "other_terms"


def is_event(name: str) -> bool:
    return any(word in name for word in EVENT_WORDS)


def is_place_title(title: str, context: str) -> bool:
    if title in PLACE_TITLES:
        return True
    if any(hint == title or title.endswith(f" {hint}") for hint in PLACE_HINTS):
        return True

    escaped_title = re.escape(title.lower())
    return bool(
        re.search(
            rf"\b{escaped_title}\b\s+(?:is|was)\s+(?:the\s+)?"
            r"(?:capital|castle|city|continent|island|kingdom|region|seat|settlement|territory)",
            context,
        )
    )


def looks_like_person_name(title: str) -> bool:
    parts = title.split()
    if len(parts) == 1:
        return False
    if parts[0] in CHARACTER_TITLE_WORDS:
        return True
    if title.startswith(("House ", "Battle ", "War ")):
        return False
    return all(part[:1].isupper() for part in parts if part.lower() not in {"of", "the"})


def extract_proper_nouns(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for match in PROPER_NOUN_RE.finditer(text):
        name = clean_name(match.group(0))
        if is_valid_candidate(name):
            counter[name] += 1
    return counter


def is_valid_candidate(name: str) -> bool:
    if len(name) < 3 or name in STOP_PHRASES:
        return False
    if len(name.split()) == 1:
        return False
    if re.search(r"'s\b", name):
        return False
    parts = name.split()
    if parts[0] in BAD_CANDIDATE_STARTS or parts[-1].lower() in BAD_CANDIDATE_ENDS:
        return False
    if re.fullmatch(r"(?:Season|Episode)\s+\d+", name):
        return False
    if re.fullmatch(r"(?:Chapter|Part)\s+\d+", name):
        return False
    if name.startswith(("Game of Thrones Season", "House of the Dragon Season")):
        return False
    if name.lower() in {"he", "she", "it", "they"}:
        return False
    return True


def add_entity(
    grouped: dict[str, dict[str, dict[str, object]]],
    category: str,
    name: str,
    source_file: str,
    count: int,
) -> None:
    bucket = grouped[category]
    if name not in bucket:
        bucket[name] = {"name": name, "count": 0, "source_files": []}
    bucket[name]["count"] = int(bucket[name]["count"]) + count
    source_files = bucket[name]["source_files"]
    if source_file and source_file not in source_files:
        source_files.append(source_file)


def build_entities(input_dir: Path, min_count: int, max_candidates: int) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {
        "characters": {},
        "places": {},
        "events": {},
        "other_terms": {},
        "mentioned_candidates": {},
    }
    global_counter: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}

    for path in read_markdown_files(input_dir):
        markdown = path.read_text(encoding="utf-8")
        plain_text = strip_markdown(markdown)
        title = extract_title(markdown, path.stem.replace("-", " ").title())
        category = classify_title(title, plain_text)

        add_entity(grouped, category, title, path.name, 1)

        file_counter = extract_proper_nouns(plain_text)
        source_counts[path.name] = file_counter
        global_counter.update(file_counter)

    candidate_names = {
        name
        for name, count in global_counter.most_common(max_candidates * 3)
        if count >= min_count
    }

    primary_names = {
        name
        for category in ("characters", "places", "events", "other_terms")
        for name in grouped[category]
    }

    for name, count in global_counter.most_common(max_candidates):
        if count < min_count or name not in candidate_names:
            continue
        if name in primary_names:
            continue

        source_files = [
            source_file
            for source_file, counter in source_counts.items()
            if counter.get(name, 0) > 0
        ]
        add_entity(grouped, "mentioned_candidates", name, "", count)
        grouped["mentioned_candidates"][name]["source_files"] = source_files[:5]

    return {
        category: sorted(values.values(), key=lambda item: (-int(item["count"]), str(item["name"])))
        for category, values in grouped.items()
    }


def write_json(path: Path, entities: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_markdown(path: Path, entities: dict[str, list[dict[str, object]]]) -> None:
    lines = ["# Extracted entities", ""]
    titles = {
        "characters": "Characters",
        "places": "Places",
        "events": "Events",
        "other_terms": "Other terms",
        "mentioned_candidates": "Mentioned candidates",
    }

    for category, title in titles.items():
        lines.extend([f"## {title}", ""])
        for item in entities[category]:
            source_files = ", ".join(item["source_files"])
            lines.append(f"- {item['name']} ({item['count']}) - {source_files}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    entities = build_entities(args.input_dir, args.min_count, args.max_candidates)
    write_json(args.out_json, entities)
    write_markdown(args.out_md, entities)

    total = sum(len(items) for items in entities.values())
    print(f"saved {args.out_json}")
    print(f"saved {args.out_md}")
    print(f"extracted {total} entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
