# app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Brand Reputation Monitor (2023)",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
TESTIMONIALS_CSV = DATA_DIR / "testimonials.csv"
REVIEWS_CSV = DATA_DIR / "reviews.csv"

# ----------------------------
# Small UI polish (CSS)
# ----------------------------
st.markdown(
    """
<style>
/* Bigger title vibe */
.big-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtle {
    opacity: 0.85;
    margin-top: 0rem;
}

/* Cards */
.card {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.meta {
    font-size: 0.85rem;
    opacity: 0.8;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 700;
    margin-right: 8px;
}
.badge-pos { background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.35); }
.badge-neg { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
.small-h {
    margin-top: 1.2rem;
    font-size: 1.2rem;
    font-weight: 700;
}
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 12px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback (če je encoding čuden)
        return pd.read_csv(path, encoding="utf-8", errors="ignore")


def render_list_cards(rows, title_key=None, text_key=None, page_key="page", limit=25):
    """
    Render rows as nice cards.
    rows: list[dict]
    """
    if not rows:
        st.info("No data to show.")
        return

    st.caption(f"Showing first {min(limit, len(rows))} rows (of {len(rows)})")
    for i, r in enumerate(rows[:limit], start=1):
        title = (r.get(title_key) if title_key else None)
        text = (r.get(text_key) if text_key else None)
        page = r.get(page_key, "")

        st.markdown(
            f"""
<div class="card">
  <div class="meta"><b>Row {i}</b></div>
  {"<div><b>title:</b> " + str(title) + "</div>" if title is not None else ""}
  {"<div><b>text:</b> " + str(text) + "</div>" if text is not None else ""}
  {"<div class='meta'><b>page:</b> " + str(page) + "</div>" if page != "" else ""}
</div>
""",
            unsafe_allow_html=True,
        )


def ensure_reviews_date_2023(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    If df has no usable date, generate placeholder dates across 2023
    to enable month filtering (as per assignment requirement).
    Returns (df, used_placeholder_dates)
    """
    used_placeholder = False

    if df.empty:
        return df, used_placeholder

    # If no "date" column, create it
    if "date" not in df.columns:
        df["date"] = ""

    # Parse date
    parsed = pd.to_datetime(df["date"], errors="coerce")

    if parsed.notna().sum() == 0:
        used_placeholder = True
        # Spread rows across 2023 (Jan -> Dec)
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-12-31")
        dates = pd.date_range(start, end, periods=len(df))
        df["date"] = dates
    else:
        df["date"] = parsed

    return df, used_placeholder


@st.cache_resource
def get_sentiment_model():
    # Model suggestion from assignment prompt
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
    )


def run_sentiment(texts):
    model = get_sentiment_model()
    preds = model(texts)
    # preds: [{'label': 'POSITIVE', 'score': 0.999}, ...]
    out = []
    for p in preds:
        label = "Positive" if p["label"].upper().startswith("POS") else "Negative"
        out.append((label, float(p["score"])))
    return out


def plot_sentiment_bar(pos_count, neg_count, avg_conf):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(["Positive", "Negative"], [pos_count, neg_count])
    ax.set_ylabel("Count")
    ax.set_title(f"Sentiment Distribution (avg confidence: {avg_conf:.2f})")
    st.pyplot(fig)


def plot_wordcloud_from_text(text: str):
    if not text.strip():
        st.info("Not enough text for word cloud.")
        return

    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
    # Use PIL image directly (avoids numpy/copy issues)
    img = wc.to_image()
    st.image(img, use_container_width=True)


# ----------------------------
# Load data
# ----------------------------
products_df = safe_read_csv(PRODUCTS_CSV)
testimonials_df = safe_read_csv(TESTIMONIALS_CSV)
reviews_df = safe_read_csv(REVIEWS_CSV)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="big-title">Brand Reputation Monitor (2023)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Scraped from web-scraping.dev • Sentiment with Hugging Face Transformers</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.markdown("## Navigation")
section = st.sidebar.radio("Select section:", ["Products", "Testimonials", "Reviews"])

# ----------------------------
# PRODUCTS
# ----------------------------
if section == "Products":
    st.markdown('<div class="small-h">Products (scraped)</div>', unsafe_allow_html=True)

    if products_df.empty:
        st.warning("products.csv not found or empty. Run scrape_data.py first.")
    else:
        # Expect columns: title, page (based on your scrape)
        rows = products_df.to_dict(orient="records")
        render_list_cards(rows, title_key="title", text_key=None, page_key="page", limit=25)

# ----------------------------
# TESTIMONIALS
# ----------------------------
elif section == "Testimonials":
    st.markdown('<div class="small-h">Testimonials (scraped)</div>', unsafe_allow_html=True)

    if testimonials_df.empty:
        st.warning("testimonials.csv not found or empty. Run scrape_data.py first.")
    else:
        rows = testimonials_df.to_dict(orient="records")
        render_list_cards(rows, title_key=None, text_key="text", page_key="page", limit=25)

# ----------------------------
# REVIEWS (Core)
# ----------------------------
else:
    st.markdown('<div class="small-h">Reviews — Sentiment Analysis</div>', unsafe_allow_html=True)

    if reviews_df.empty:
        st.warning("reviews.csv not found or empty. Run scrape_data.py first.")
        st.stop()

    # ensure date usable
    reviews_df, used_placeholder = ensure_reviews_date_2023(reviews_df)

    if used_placeholder:
        st.info(
            "Review dates are missing in the sandbox data. "
            "Placeholder dates across 2023 were generated to enable month-based filtering."
        )

    # Ensure required text column
    if "text" not in reviews_df.columns:
        st.error("reviews.csv must contain a 'text' column.")
        st.stop()

    # Drop empty text
    reviews_df["text"] = reviews_df["text"].fillna("").astype(str)
    reviews_df = reviews_df[reviews_df["text"].str.strip() != ""].copy()

    if reviews_df.empty:
        st.warning("All review texts are empty after cleaning.")
        st.stop()

    # Build month list
    reviews_df["month"] = reviews_df["date"].dt.to_period("M").astype(str)
    months = sorted([m for m in reviews_df["month"].unique() if m.startswith("2023-")])

    if not months:
        st.error("No 2023 dates found (or generated). Can't filter by month.")
        st.stop()

    selected_month = st.select_slider("Select month (2023):", options=months, value=months[0])

    month_df = reviews_df[reviews_df["month"] == selected_month].copy()
    st.write(f"Number of reviews in **{selected_month}**: **{len(month_df)}**")

    if month_df.empty:
        st.info("No reviews in this month. Pick a different month.")
        st.stop()

    # Run sentiment
    texts = month_df["text"].tolist()
    preds = run_sentiment(texts)
    month_df["sentiment"] = [p[0] for p in preds]
    month_df["confidence"] = [p[1] for p in preds]

    pos_count = int((month_df["sentiment"] == "Positive").sum())
    neg_count = int((month_df["sentiment"] == "Negative").sum())
    avg_conf = float(month_df["confidence"].mean()) if len(month_df) else 0.0

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", pos_count)
    c2.metric("Negative", neg_count)
    c3.metric("Avg confidence", f"{avg_conf:.2f}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Sentiment Distribution")
    plot_sentiment_bar(pos_count, neg_count, avg_conf)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Word Cloud (selected month)")
    all_text = " ".join(month_df["text"].tolist())
    plot_wordcloud_from_text(all_text)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Sample Reviews")
    # Show a few as cards
    for i, row in month_df.head(12).iterrows():
        badge_class = "badge-pos" if row["sentiment"] == "Positive" else "badge-neg"
        badge = "Positive" if row["sentiment"] == "Positive" else "Negative"
        conf = row["confidence"]
        date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "N/A"
        page = row["page"] if "page" in month_df.columns else ""

        st.markdown(
            f"""
<div class="card">
  <div class="meta">{date_str} {"• page " + str(page) if page != "" else ""}</div>
  <div style="margin: 8px 0;">
    <span class="badge {badge_class}">{badge}</span>
    <span class="meta">confidence: {conf:.2f}</span>
  </div>
  <div>{row["text"]}</div>
</div>
""",
            unsafe_allow_html=True,
        )
