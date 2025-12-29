# scrape_data.py
from __future__ import annotations

import csv
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://web-scraping.dev"

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36"
}


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def scrape_products(max_pages: int = 5) -> None:
    rows = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/products?page={page}"
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "lxml")
        # product titles are typically in h3 or a card-title; keep it robust
        titles = [t.get_text(strip=True) for t in soup.select("h3") if t.get_text(strip=True)]
        if not titles:
            break

        for t in titles:
            rows.append({"title": t, "page": page})

        time.sleep(0.2)

    out = DATA_DIR / "products.csv"
    save_csv(out, rows, ["title", "page"])
    print("Saved:", out, "rows:", len(rows))


def scrape_testimonials(max_pages: int = 5) -> None:
    rows = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/api/testimonials?page={page}"
        r = requests.get(url, headers={**UA, "Referer": f"{BASE}/testimonials"}, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            break

        soup = BeautifulSoup(r.text, "lxml")
        blocks = soup.select(".testimonial")
        if not blocks:
            break

        for b in blocks:
            text = b.get_text(" ", strip=True)
            if text:
                rows.append({"text": text, "page": page})

        time.sleep(0.2)

    out = DATA_DIR / "testimonials.csv"
    save_csv(out, rows, ["text", "page"])
    print("Saved:", out, "rows:", len(rows))


def scrape_reviews(max_pages: int = 12) -> None:
    """
    Reviews endpoint in this sandbox can be a bit inconsistent.
    This version scrapes /reviews?page=X (HTML) and extracts review text.
    Dates are not reliably provided -> we leave date blank (app will generate placeholder dates).
    """
    rows = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/reviews?page={page}"
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "lxml")

        # Try common patterns
        # 1) cards
        candidates = soup.select(".card, .review, .review-card, article")
        texts = []

        if candidates:
            for c in candidates:
                txt = c.get_text(" ", strip=True)
                if txt and len(txt) > 10:
                    texts.append(txt)

        # 2) fallback: list items / paragraphs
        if not texts:
            for p in soup.select("p"):
                txt = p.get_text(" ", strip=True)
                if txt and len(txt) > 20:
                    texts.append(txt)

        # Stop if nothing found (likely last page)
        if not texts:
            break

        # Keep it reasonable (avoid grabbing nav/footer huge text)
        for t in texts[:25]:
            rows.append({"date": "", "text": t, "page": page})

        time.sleep(0.2)

    out = DATA_DIR / "reviews.csv"
    save_csv(out, rows, ["date", "text", "page"])
    print("Saved:", out, "rows:", len(rows))


if __name__ == "__main__":
    scrape_products(max_pages=5)
    scrape_testimonials(max_pages=5)
    scrape_reviews(max_pages=12)