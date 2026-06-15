#!/usr/bin/env python3
"""Extract Game of Thrones cast/character entries from a Fandom page."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


DEFAULT_URL = "https://gameofthrones.fandom.com/wiki/Game_of_Thrones"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

TITLE_PREFIX_RE = re.compile(
    r"^(?:King|Queen|Prince|Princess|Lord|Lady|Ser|Septa|Maester|Khal|"
    r"Grand Maester|High Septon|Archmaester)\s+"
)
ROLE_ONLY_LINK_TEXTS = {
    "Archmaester",
    "Grand Maester",
    "Khal",
    "King",
    "Lady",
    "Lord",
    "Maester",
    "Prince",
    "Princess",
    "Queen",
    "Septa",
    "Ser",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cast character data from the Game of Thrones Fandom page."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Fandom page URL. Fragments are allowed, but the script extracts the Cast section by default.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("Task2/got_cast_characters.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("Task2/got_cast_characters.md"),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--terms-map",
        type=Path,
        default=Path("Task2/terms_map.json"),
        help="Optional terms_map.json path for mapping validation.",
    )
    parser.add_argument(
        "--section",
        default="Cast",
        help="Top-level section to extract.",
    )
    parser.add_argument(
        "--start-heading",
        help=(
            "Optional subsection heading to start from, for example "
            "'Retainers at Winterfell'."
        ),
    )
    parser.add_argument(
        "--stop-at-same-level",
        action="store_true",
        help="When --start-heading is used, stop at the next heading of the same or higher level.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry count.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for downloads.",
    )
    return parser.parse_args()


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS | {"User-Agent": user_agent})
    return session


def to_render_url(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["action"] = "render"
    return urlunparse(parsed._replace(query=urlencode(query), fragment=""))


def without_fragment(url: str) -> str:
    return urlunparse(urlparse(url)._replace(fragment=""))


def to_api_parse_url(url: str) -> str:
    parsed = urlparse(url)
    page_name = parsed.path.rstrip("/").split("/")[-1]
    query = urlencode(
        {
            "action": "parse",
            "page": page_name,
            "prop": "text",
            "format": "json",
            "redirects": "1",
        }
    )
    return urlunparse(
        parsed._replace(path="/api.php", params="", query=query, fragment="")
    )


def download_html(url: str, user_agent: str, retries: int) -> str:
    session = make_session(user_agent)
    target_urls = [without_fragment(url), to_render_url(url), to_api_parse_url(url)]
    last_error: requests.RequestException | None = None

    for target_url in target_urls:
        for attempt in range(1, retries + 1):
            try:
                response = session.get(target_url, timeout=30)
                response.raise_for_status()
                return response_to_html(response)
            except requests.RequestException as error:
                last_error = error
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


def response_to_html(response: requests.Response) -> str:
    content_type = response.headers.get("Content-Type", "")
    if "json" not in content_type and not response.url.endswith("api.php"):
        return response.text

    try:
        data = response.json()
    except ValueError:
        return response.text

    parsed_text = data.get("parse", {}).get("text", {})
    if isinstance(parsed_text, dict) and "*" in parsed_text:
        return parsed_text["*"]
    if isinstance(parsed_text, str):
        return parsed_text
    return response.text


def extract_article_node(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("div.mw-parser-output") or soup.select_one("body")
    if not article:
        raise ValueError("Cannot find article content")
    return article


def heading_level(node: Tag) -> int | None:
    if re.fullmatch(r"h[1-6]", node.name or ""):
        return int(node.name[1])
    return None


def heading_text(node: Tag) -> str:
    clone = BeautifulSoup(str(node), "html.parser")
    for edit_section in clone.select(".mw-editsection"):
        edit_section.decompose()
    return clean_text(clone.get_text(" ", strip=True))


def clean_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\[\s*(?:edit|src|citation needed|\d+|[a-z])\s*\]", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\n,")


def section_children(
    article: Tag,
    section: str,
    start_heading: str | None,
    stop_at_same_level: bool,
) -> list[Tag]:
    in_section = False
    section_level: int | None = None
    in_start_heading = start_heading is None
    start_level: int | None = None
    children: list[Tag] = []

    for child in article.children:
        if not isinstance(child, Tag):
            continue

        level = heading_level(child)
        if level is not None:
            text = heading_text(child)

            if in_section and section_level is not None and level <= section_level:
                break

            if text == section:
                in_section = True
                section_level = level
                continue

            if not in_section:
                continue

            if start_heading and text == start_heading:
                in_start_heading = True
                start_level = level
                continue

            if (
                start_heading
                and stop_at_same_level
                and in_start_heading
                and start_level is not None
                and level <= start_level
            ):
                break

        if in_section and in_start_heading:
            children.append(child)

    return children


def extract_cast_entries(children: list[Tag], page_url: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current_group = ""
    current_subgroup = ""

    for child in children:
        level = heading_level(child)
        if level is not None:
            text = heading_text(child)
            if level == 3:
                current_group = text
                current_subgroup = ""
            elif level >= 4:
                current_subgroup = text
            continue

        for li in child.find_all("li", recursive=True):
            entry = parse_cast_item(li, current_group, current_subgroup, page_url)
            if entry:
                entries.append(entry)

    return deduplicate_entries(entries)


def parse_cast_item(
    li: Tag,
    group: str,
    subgroup: str,
    page_url: str,
) -> dict[str, object] | None:
    full_text = clean_text(li.get_text(" ", strip=True))
    if " as " not in f" {full_text} ":
        return None

    actors_text, role_text = split_cast_text(full_text)
    character_name = extract_character_name(li, role_text)
    if not character_name:
        return None

    actor_links, character_links, related_links = split_links(li, character_name, page_url)

    return {
        "group": group,
        "subgroup": subgroup,
        "actor": actors_text,
        "actor_links": actor_links,
        "character": character_name,
        "character_links": character_links,
        "related_links": related_links,
        "description": clean_description(role_text, character_name),
        "raw_text": full_text,
    }


def split_cast_text(text: str) -> tuple[str, str]:
    before, after = re.split(r"\bas\b", text, maxsplit=1)
    return clean_text(before), clean_text(after)


def extract_character_name(li: Tag, role_text: str) -> str:
    links_after_as = links_after_word_as(li)
    if links_after_as:
        link_texts = [clean_text(link.get_text(" ", strip=True)) for link in links_after_as]
        if len(link_texts) > 1 and link_texts[0] in ROLE_ONLY_LINK_TEXTS:
            return link_texts[1]
        return link_texts[0]

    before_comma = role_text.split(",", 1)[0]
    return clean_character_name(before_comma)


def links_after_word_as(li: Tag) -> list[Tag]:
    result: list[Tag] = []
    seen_as = False

    for node in li.descendants:
        if isinstance(node, str):
            if re.search(r"\bas\b", node):
                seen_as = True
        elif isinstance(node, Tag) and node.name == "a" and seen_as:
            result.append(node)

    return result


def clean_character_name(value: str) -> str:
    value = clean_text(value)
    value = TITLE_PREFIX_RE.sub("", value)
    return value.strip()


def clean_description(role_text: str, character_name: str) -> str:
    description = role_text
    description = TITLE_PREFIX_RE.sub("", description)
    description = re.sub(rf"^{re.escape(character_name)}\s*,?\s*", "", description)
    description = clean_text(description)
    return description


def split_links(
    li: Tag,
    character_name: str,
    page_url: str,
) -> tuple[list[str], list[str], list[str]]:
    actor_links: list[str] = []
    role_links: list[tuple[str, str]] = []
    seen_as = False

    for node in li.descendants:
        if isinstance(node, str):
            if re.search(r"\bas\b", node):
                seen_as = True
            continue

        if not isinstance(node, Tag) or node.name != "a":
            continue

        href = node.get("href")
        if not href:
            continue
        link = urljoin(page_url, href)
        text = clean_text(node.get_text(" ", strip=True))

        if seen_as or text == character_name:
            role_links.append((text, link))
        else:
            actor_links.append(link)

    role_links = unique_pairs(role_links)
    character_links = [
        link
        for text, link in role_links
        if text == character_name or clean_character_name(text) == character_name
    ][:1]
    if not character_links and role_links:
        character_links = [role_links[0][1]]

    related_links = [
        link
        for _, link in role_links
        if link not in character_links
    ]
    return unique(actor_links), character_links, related_links


def unique_pairs(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def deduplicate_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, object]] = []

    for entry in entries:
        key = (
            str(entry["group"]),
            str(entry["subgroup"]),
            str(entry["character"]),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)

    return result


def load_terms_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    flat_map = data.get("flat_map", {})
    if not isinstance(flat_map, dict):
        return {}

    return {str(source): str(target) for source, target in flat_map.items()}


def add_mapping_validation(
    entries: list[dict[str, object]],
    terms_map: dict[str, str],
) -> dict[str, object]:
    missing: list[str] = []
    covered: list[str] = []

    for entry in entries:
        character = str(entry["character"])
        replacement = terms_map.get(character)
        entry["mapping_status"] = "mapped" if replacement else "missing"
        entry["mapped_to"] = replacement or ""

        if replacement:
            covered.append(character)
        else:
            missing.append(character)

    unique_missing = sorted(set(missing))
    unique_covered = sorted(set(covered))
    return {
        "terms_map_loaded": bool(terms_map),
        "unique_characters": len(set(str(entry["character"]) for entry in entries)),
        "mapped_unique_characters": len(unique_covered),
        "missing_unique_characters": len(unique_missing),
        "missing_characters": unique_missing,
    }


def write_json(
    path: Path,
    page_url: str,
    section: str,
    entries: list[dict[str, object]],
    mapping_validation: dict[str, object],
) -> None:
    payload = {
        "source_url": page_url,
        "section": section,
        "characters_count": len(entries),
        "mapping_validation": mapping_validation,
        "characters": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_markdown(
    path: Path,
    page_url: str,
    section: str,
    entries: list[dict[str, object]],
    mapping_validation: dict[str, object],
) -> None:
    lines = [
        "# Game of Thrones cast characters",
        "",
        f"Source: {page_url}",
        f"Section: {section}",
        f"Characters: {len(entries)}",
        f"Unique characters: {mapping_validation['unique_characters']}",
        f"Mapped unique characters: {mapping_validation['mapped_unique_characters']}",
        f"Missing unique characters: {mapping_validation['missing_unique_characters']}",
        "",
    ]

    current_group = None
    current_subgroup = None
    for entry in entries:
        group = str(entry["group"] or "Ungrouped")
        subgroup = str(entry["subgroup"] or "")

        if group != current_group:
            lines.extend([f"## {group}", ""])
            current_group = group
            current_subgroup = None

        if subgroup and subgroup != current_subgroup:
            lines.extend([f"### {subgroup}", ""])
            current_subgroup = subgroup

        description = str(entry["description"])
        suffix = f" - {description}" if description else ""
        mapped_to = str(entry.get("mapped_to") or "")
        mapping_note = f"; mapped to: {mapped_to}" if mapped_to else "; mapping: missing"
        lines.append(
            f"- **{entry['character']}**; actor: {entry['actor']}{mapping_note}{suffix}"
        )

    missing = mapping_validation.get("missing_characters", [])
    if missing:
        lines.extend(["", "## Missing from terms_map", ""])
        for character in missing:
            lines.append(f"- {character}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    html = download_html(args.url, args.user_agent, args.retries)
    article = extract_article_node(html)
    children = section_children(
        article=article,
        section=args.section,
        start_heading=args.start_heading,
        stop_at_same_level=args.stop_at_same_level,
    )

    if not children:
        print(f"Cannot find section: {args.section}", file=sys.stderr)
        return 1

    entries = extract_cast_entries(children, args.url)
    terms_map = load_terms_map(args.terms_map)
    mapping_validation = add_mapping_validation(entries, terms_map)
    write_json(args.out_json, args.url, args.section, entries, mapping_validation)
    write_markdown(args.out_md, args.url, args.section, entries, mapping_validation)

    print(f"saved {len(entries)} characters to {args.out_json}")
    print(f"saved report to {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
