#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as html_to_markdown


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


UNWANTED_SELECTORS = [
    "aside",
    "nav",
    "footer",
    "header",
    "figure",
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    ".ads",
    ".portable-infobox",
    ".page-header",
    ".page-side-tools",
    ".page-footer",
    ".page__right-rail",
    ".toc",
    ".mw-editsection",
    ".reference",
    ".references",
    ".reflist",
    ".printfooter",
    ".catlinks",
    ".metadata",
    ".navbox",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Game of Thrones Fandom HTML pages to clean Markdown."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Page URL to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Text file with one URL per line. Empty lines and # comments are ignored.",
    )
    parser.add_argument(
        "--html-file",
        action="append",
        type=Path,
        default=[],
        help="Local HTML file to convert. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Task2/knowledge_base_raw"),
        help="Directory for resulting Markdown files.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Delay between HTTP requests in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="HTTP retry count for each URL.",
    )
    parser.add_argument(
        "--fetch-mode",
        choices=["render", "page"],
        default="render",
        help=(
            "Use lightweight MediaWiki rendered HTML or the full Fandom page. "
            "The render mode is less noisy and usually avoids 403 responses."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed URL instead of continuing with the next one.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for downloads.",
    )
    return parser.parse_args()


def read_urls(url_file: Path | None, inline_urls: Iterable[str]) -> list[str]:
    urls = [url.strip() for url in inline_urls if url.strip()]

    if url_file:
        for line in url_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    return urls


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS | {"User-Agent": user_agent})
    return session


def download_html(
    session: requests.Session,
    url: str,
    retries: int,
    fetch_mode: str,
) -> str:
    target_url = to_render_url(url) if fetch_mode == "render" else url
    last_error: requests.RequestException | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(target_url, timeout=30)
            if response.status_code == 403 and fetch_mode == "page":
                response = session.get(to_render_url(url), timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


def to_render_url(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["action"] = "render"
    return urlunparse(parsed._replace(query=urlencode(query)))


def extract_article_node(soup: BeautifulSoup) -> Tag:
    selectors = [
        "div.mw-parser-output",
        "main.page__main",
        "article",
        "main",
        "body",
    ]

    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node

    raise ValueError("Cannot find article content in HTML")


def clean_article_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    article = extract_article_node(soup)

    for selector in UNWANTED_SELECTORS:
        for node in article.select(selector):
            node.decompose()

    unwrap_links(article)
    remove_empty_tables(article)

    return title, str(article)


def extract_title(soup: BeautifulSoup) -> str:
    title_node = soup.select_one("h1#firstHeading") or soup.select_one("h1")
    if title_node:
        return normalize_text(title_node.get_text(" ", strip=True))

    if soup.title and soup.title.string:
        title = soup.title.string.split("|")[0]
        return normalize_text(title)

    return "untitled"


def unwrap_links(node: Tag) -> None:
    for link in node.find_all("a"):
        text = link.get_text(" ", strip=True)
        if text:
            link.replace_with(text)
        else:
            link.decompose()


def remove_empty_tables(node: Tag) -> None:
    for table in node.find_all("table"):
        text = normalize_text(table.get_text(" ", strip=True))
        if not text:
            table.decompose()


def convert_to_markdown(title: str, article_html: str) -> str:
    markdown = html_to_markdown(
        article_html,
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    markdown = remove_markdown_links(markdown)
    markdown = normalize_markdown(markdown)

    if not markdown.startswith("# "):
        markdown = f"# {title}\n\n{markdown}"

    return markdown.strip() + "\n"


def remove_markdown_links(markdown: str) -> str:
    markdown = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", markdown)
    markdown = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    markdown = re.sub(r"<https?://[^>\s]+>", "", markdown)
    return markdown


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\r\n?", "\n", markdown)
    markdown = re.sub(r"\\?\[(?:src|source|citation needed|note \d+)\\?\]", "", markdown, flags=re.I)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"(?m)^\s*\[\s*edit\s*\]\s*$", "", markdown, flags=re.I)
    markdown = re.sub(r"(?m)^\s*Advertisement\s*$", "", markdown, flags=re.I)
    markdown = re.sub(r"(?m)^Categories\s*$.*", "", markdown, flags=re.I | re.S)
    return markdown.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9а-яё]+", "-", value, flags=re.I)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "document"


def filename_from_url(url: str, fallback_title: str) -> str:
    path_name = Path(urlparse(url).path).name
    if path_name:
        return slugify(path_name)
    return slugify(fallback_title)


def title_from_url(url: str) -> str:
    path_name = Path(urlparse(url).path).name
    if not path_name:
        return "untitled"
    return path_name.replace("_", " ").replace("%27", "'")


def save_markdown(markdown: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def convert_html_document(html: str, source_name: str, out_dir: Path) -> Path:
    title, article_html = clean_article_html(html)
    markdown = convert_to_markdown(title, article_html)
    stem = slugify(Path(source_name).stem) if source_name else slugify(title)
    return save_markdown(markdown, out_dir, stem)


def convert_url(
    session: requests.Session,
    url: str,
    out_dir: Path,
    retries: int,
    fetch_mode: str,
) -> Path:
    html = download_html(session, url, retries, fetch_mode)
    title, article_html = clean_article_html(html)
    if title == "untitled":
        title = title_from_url(url)
    markdown = convert_to_markdown(title, article_html)
    stem = filename_from_url(url, title)
    return save_markdown(markdown, out_dir, stem)


def main() -> int:
    args = parse_args()
    urls = read_urls(args.url_file, args.url)
    session = make_session(args.user_agent)
    failed_urls: list[str] = []

    if not urls and not args.html_file:
        print("Pass at least one --url, --url-file, or --html-file", file=sys.stderr)
        return 2

    for html_file in args.html_file:
        html = html_file.read_text(encoding="utf-8")
        result = convert_html_document(html, html_file.name, args.out_dir)
        print(f"saved {result}")

    for index, url in enumerate(urls, start=1):
        try:
            result = convert_url(
                session=session,
                url=url,
                out_dir=args.out_dir,
                retries=args.retries,
                fetch_mode=args.fetch_mode,
            )
            print(f"saved {result}")
        except Exception as error:
            failed_urls.append(url)
            print(f"failed {url}: {error}", file=sys.stderr)
            if args.fail_fast:
                return 1
        if index < len(urls) and args.sleep > 0:
            time.sleep(args.sleep)

    if failed_urls:
        print("\nFailed URLs:", file=sys.stderr)
        for url in failed_urls:
            print(f"- {url}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
