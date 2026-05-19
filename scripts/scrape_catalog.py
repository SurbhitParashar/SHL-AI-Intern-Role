"""Refresh the local SHL Individual Test Solutions catalog.

The runtime API never scrapes. This script is intentionally separate so a
submission can run quickly and reliably from data/shl_catalog.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "shl_catalog.json"
HEADERS = {"User-Agent": "SHL-assessment-recommender/1.0"}
TYPE_MAP = {
    "knowledge": "K",
    "skill": "K",
    "ability": "A",
    "aptitude": "A",
    "personality": "P",
    "behavior": "P",
    "behaviour": "P",
    "situational": "B",
    "simulation": "S",
}


def main() -> None:
    items = scrape_listing()
    if not items:
        raise RuntimeError("No catalog items found; keeping the existing cache is safer.")
    OUTPUT_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} catalog items to {OUTPUT_PATH}")


def scrape_listing() -> list[dict]:
    session = requests.Session()
    session.headers.update(HEADERS)
    seen: set[str] = set()
    items: list[dict] = []

    for page in range(0, 50):
        url = BASE_URL if page == 0 else f"{BASE_URL}?start={page * 12}"
        response = session.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_items = parse_page(soup, session)
        new_items = [item for item in page_items if item["url"] not in seen]
        for item in new_items:
            seen.add(item["url"])
            items.append(item)
        if not new_items:
            break

    return items


def parse_page(soup: BeautifulSoup, session: requests.Session) -> list[dict]:
    links = []
    for anchor in soup.select("a[href*='product-catalog/view'], a[href*='productcatalog/view']"):
        name = anchor.get_text(" ", strip=True)
        href = anchor.get("href")
        if name and href:
            links.append((name, urljoin(BASE_URL, href)))

    items = []
    for name, url in dedupe_links(links):
        detail = scrape_detail(session, url)
        if is_individual_test_solution(detail):
            items.append(
                {
                    "name": clean_name(detail.get("name") or name),
                    "url": url,
                    "test_type": infer_type(detail),
                    "description": detail.get("description", ""),
                    "skills": detail.get("skills", []),
                    "job_family": detail.get("job_family", ""),
                    "duration_minutes": detail.get("duration_minutes"),
                    "remote_testing": detail.get("remote_testing"),
                }
            )
    return items


def scrape_detail(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.select_one("h1")
    description = first_text(
        soup.select_one(".product-catalogue__description"),
        soup.select_one("[class*='description']"),
        soup.select_one("main p"),
    )
    full_text = soup.get_text(" ", strip=True)
    skills = extract_skills(full_text)
    return {
        "name": title.get_text(" ", strip=True) if title else "",
        "description": description,
        "full_text": full_text,
        "skills": skills,
        "job_family": infer_job_family(full_text),
        "duration_minutes": extract_duration(full_text),
        "remote_testing": "remote" in full_text.lower(),
    }


def is_individual_test_solution(detail: dict) -> bool:
    text = detail.get("full_text", "").lower()
    return "pre-packaged job solution" not in text


def infer_type(detail: dict) -> str:
    text = " ".join(
        [
            detail.get("name", ""),
            detail.get("description", ""),
            detail.get("full_text", ""),
            " ".join(detail.get("skills", [])),
        ]
    ).lower()
    for trigger, type_code in TYPE_MAP.items():
        if trigger in text:
            return type_code
    return "K"


def infer_job_family(text: str) -> str:
    lowered = text.lower()
    families = {
        "Technology": ["developer", "software", "programming", "java", "python", "sql"],
        "Sales": ["sales", "account manager", "business development"],
        "Customer Service": ["customer", "contact center", "call centre", "call center"],
        "Management": ["manager", "leadership", "supervisor"],
        "Finance": ["finance", "accounting", "bookkeeping"],
        "Administrative": ["administrative", "office", "data entry"],
    }
    for family, terms in families.items():
        if any(term in lowered for term in terms):
            return family
    return "General"


def extract_skills(text: str) -> list[str]:
    common = [
        "Java",
        "Python",
        "SQL",
        "JavaScript",
        "Excel",
        "Sales",
        "Customer service",
        "Leadership",
        "Personality",
        "Reasoning",
        "Numerical reasoning",
        "Verbal reasoning",
        "Situational judgment",
        "Attention to detail",
        "Communication",
    ]
    lowered = text.lower()
    return [skill for skill in common if skill.lower() in lowered]


def extract_duration(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(minutes|mins|min)\b", text, re.I)
    return int(match.group(1)) if match else None


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def first_text(*nodes) -> str:
    for node in nodes:
        if node:
            return clean_name(node.get_text(" ", strip=True))
    return ""


def dedupe_links(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    unique = []
    for name, url in links:
        if url in seen:
            continue
        seen.add(url)
        unique.append((name, url))
    return unique


if __name__ == "__main__":
    main()
