import csv
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL = "https://web-scraping.dev/reviews"


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def scrape_reviews(max_clicks=30):
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle")

        # klikamo "Load More" večkrat (dokler gre)
        for i in range(max_clicks):
            # poberi trenutno število reviewev (da vidimo, če se povečuje)
            before = page.locator("[data-testid='review']").count()

            btn = page.get_by_role("button", name="Load More")
            if btn.count() == 0:
                break

            try:
                btn.click(timeout=2000)
            except Exception:
                break

            page.wait_for_timeout(800)
            after = page.locator("[data-testid='review']").count()

            # če se nič ne poveča, smo na koncu
            if after <= before:
                break

        # zdaj poberemo vse reviewe
        review_cards = page.locator("[data-testid='review']")
        n = review_cards.count()

        for idx in range(n):
            card = review_cards.nth(idx)

            # tekst reviewa
            text = card.inner_text().strip()

            # datum: na strani je pogosto <time> ali podoben element
            date_str = ""
            time_el = card.locator("time")
            if time_el.count() > 0:
                date_str = time_el.first.get_attribute("datetime") or time_el.first.inner_text().strip()

            rows.append({"date": date_str, "text": text})

        browser.close()

    out = DATA_DIR / "reviews.csv"
    save_csv(out, rows, ["date", "text"])
    print("Saved:", out, "rows:", len(rows))


if __name__ == "__main__":
    scrape_reviews(max_clicks=40)
