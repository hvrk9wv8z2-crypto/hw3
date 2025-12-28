from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DATA_DIR = Path("data")

st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")

def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()

def render_records(df: pd.DataFrame, fields: list[str], limit: int = 25):
    if df.empty:
        st.info("No data to show.")
        return
    n = min(limit, len(df))
    st.caption(f"Showing {n} rows (of {len(df)})")
    for i in range(n):
        st.markdown(f"**Row {i+1}**")
        for f in fields:
            if f in df.columns:
                st.markdown(f"- **{f}**: {df.iloc[i][f]}")
        st.markdown("---")

def safe_to_datetime(series: pd.Series) -> pd.Series:
    # handles invalid/empty gracefully
    return pd.to_datetime(series, errors="coerce")

def plot_bar(pos_count: int, neg_count: int, avg_pos: float | None, avg_neg: float | None):
    labels = ["Positive", "Negative"]
    values = [pos_count, neg_count]

    fig = plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")

    # show avg confidence text (no fancy tooltips needed)
    txt = []
    if avg_pos is not None:
        txt.append(f"Avg conf (Positive): {avg_pos:.2f}")
    if avg_neg is not None:
        txt.append(f"Avg conf (Negative): {avg_neg:.2f}")
    if txt:
        plt.xlabel(" | ".join(txt))

    st.pyplot(fig)

def plot_wordcloud(text: str):
    if not text.strip():
        st.info("Not enough text for word cloud.")
        return
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)

    # IMPORTANT: use wc.to_array() to avoid numpy 'copy' issues
    arr = wc.to_array()

    fig = plt.figure()
    plt.imshow(arr)
    plt.axis("off")
    st.pyplot(fig)

st.title("Brand Reputation Monitor (2023)")
st.caption("Scraped from web-scraping.dev (sandbox)")

section = st.sidebar.radio("Select section:", ["Products", "Testimonials", "Reviews"])

products_df = load_csv_safe(DATA_DIR / "products.csv")
testimonials_df = load_csv_safe(DATA_DIR / "testimonials.csv")
reviews_df = load_csv_safe(DATA_DIR / "reviews.csv")

if section == "Products":
    st.header("Products (scraped)")
    if products_df.empty:
        st.error("No products.csv found or it’s empty. Run your scraper and commit data/products.csv to GitHub.")
    else:
        render_records(products_df, fields=["title", "page"], limit=25)

elif section == "Testimonials":
    st.header("Testimonials (scraped)")
    if testimonials_df.empty:
        st.error("No testimonials.csv found or it’s empty. Run your scraper and commit data/testimonials.csv to GitHub.")
    else:
        render_records(testimonials_df, fields=["text", "page"], limit=25)

else:
    st.header("Reviews — Sentiment Analysis")

    if reviews_df.empty:
        st.error("No reviews.csv found or its empty. Run scrape_reviews.py locally and push data/reviews.csv to GitHub.")
        st.stop()

    # make sure required columns exist
    needed = {"date", "text"}
    if not needed.issubset(set(reviews_df.columns)):
        st.error("reviews.csv must contain at least: date, text (and ideally sentiment, confidence, page).")
        st.stop()

    reviews_df["date"] = safe_to_datetime(reviews_df["date"])
    reviews_df = reviews_df.dropna(subset=["date"])
    if reviews_df.empty:
        st.error("All review dates are empty/invalid after parsing. Fix reviews.csv date format (YYYY-MM-DD).")
        st.stop()

    # keep only 2023
    reviews_df = reviews_df[(reviews_df["date"].dt.year == 2023)]
    if reviews_df.empty:
        st.error("No reviews from year 2023 found in reviews.csv.")
        st.stop()

    reviews_df["month"] = reviews_df["date"].dt.to_period("M").astype(str)
    months = sorted(reviews_df["month"].unique().tolist())

    if not months:
        st.error("No months available.")
        st.stop()

    selected = st.select_slider("Select month (2023):", options=months, value=months[0])

    month_df = reviews_df[reviews_df["month"] == selected].copy()
    st.write(f"Number of reviews in **{selected}**: {len(month_df)}")

    # if sentiment missing, show message
    if "sentiment" not in month_df.columns or "confidence" not in month_df.columns:
        st.warning("This reviews.csv has no sentiment/confidence columns. Run scrape_reviews.py locally with transformers installed.")
        render_records(month_df, fields=["date", "text", "page"], limit=10)
        st.stop()

    # normalize sentiment labels
    month_df["sentiment"] = month_df["sentiment"].astype(str)
    month_df["confidence"] = pd.to_numeric(month_df["confidence"], errors="coerce")

    pos = month_df[month_df["sentiment"].str.lower().str.startswith("pos")]
    neg = month_df[month_df["sentiment"].str.lower().str.startswith("neg")]

    pos_count = len(pos)
    neg_count = len(neg)

    avg_pos = float(pos["confidence"].mean()) if pos_count and pos["confidence"].notna().any() else None
    avg_neg = float(neg["confidence"].mean()) if neg_count and neg["confidence"].notna().any() else None

    st.subheader("Sentiment Distribution")
    plot_bar(pos_count, neg_count, avg_pos, avg_neg)

    st.subheader("Word Cloud (selected month)")
    all_text = " ".join(month_df["text"].astype(str).tolist())
    plot_wordcloud(all_text)

    st.subheader("Sample Reviews")
    show_cols = [c for c in ["date", "text", "sentiment", "confidence", "page"] if c in month_df.columns]
    render_records(month_df[show_cols], fields=show_cols, limit=10)