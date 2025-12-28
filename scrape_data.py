import csv
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://web-scraping.dev"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

UA = {
    "User-Agent": "Mozilla/5.0 (HW3 Scraper)",
    "Accept-Language": "en-US,en;q=0.9",
}


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write CSV (creates file even if rows is empty)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


# -------------------------
# PRODUCTS: HTML pages /products?page=...
# -------------------------
def scrape_products(max_pages: int = 5) -> None:
    rows = []

    for page in range(1, max_pages + 1):
        url = f"{BASE}/products?page={page}"
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            break

        soup = BeautifulSoup(r.text, "lxml")

        titles = soup.select("h3")
        if not titles:
            break

        for h in titles:
            title = h.get_text(strip=True)
            if title:
                rows.append({"title": title, "page": page})

        time.sleep(0.2)

    out = DATA_DIR / "products.csv"
    save_csv(out, rows, ["title", "page"])
    print("Saved:", out, "rows:", len(rows))


# -------------------------
# TESTIMONIALS: HTML fragment API /api/testimonials?page=...
# -------------------------
def scrape_testimonials(max_pages: int = 5) -> None:
    headers = {**UA, "Referer": f"{BASE}/testimonials"}
    rows = []

    for page in range(1, max_pages + 1):
        url = f"{BASE}/api/testimonials?page={page}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            break

        soup = BeautifulSoup(r.text, "lxml")

        cards = soup.select(".testimonial")
        if not cards:
            break

        for c in cards:
            inner = c.select_one("div")
            text = inner.get_text(" ", strip=True) if inner else c.get_text(" ", strip=True)
            if text and len(text) >= 10:
                rows.append({"text": text, "page": page})

        time.sleep(0.2)

    out = DATA_DIR / "testimonials.csv"
    save_csv(out, rows, ["text", "page"])
    print("Saved:", out, "rows:", len(rows))


# -------------------------
# REVIEWS: HTML pages /reviews?page=...
# (no API, no product_id, no Playwright)
# -------------------------
def scrape_reviews_pages(max_pages: int = 12) -> None:
    rows = []

    for page in range(1, max_pages + 1):
        url = f"{BASE}/reviews?page={page}"
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            break

        soup = BeautifulSoup(r.text, "lxml")

        # Most robust approach: gather chunks that contain a <time> tag (reviews usually have dates)
        time_tags = soup.select("time")
        if not time_tags:
            # fallback selector
            blocks = soup.select(".review, article, li, .card, div")
        else:
            # take parents of <time> as candidate blocks
            blocks = []
            for t in time_tags:
                parent = t.parent
                if parent:
                    blocks.append(parent)

        added = 0
        for b in blocks:
            text = b.get_text(" ", strip=True)
            if not text or len(text) < 30:
                continue

            t = b.select_one("time")
            date_str = ""
            if t:
                date_str = t.get("datetime") if t.has_attr("datetime") else t.get_text(strip=True)

            rows.append({"date": date_str, "text": text, "page": page})
            added += 1

        # if nothing added on this page, likely selector mismatch or end of pages
        if added == 0:
            break

        time.sleep(0.2)

    out = DATA_DIR / "reviews.csv"
    save_csv(out, rows, ["date", "text", "page"])
    print("Saved:", out, "rows:", len(rows))


if __name__ == "__main__":
    scrape_products(max_pages=5)
    scrape_testimonials(max_pages=5)
    scrape_reviews_pages(max_pages=12)
