from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DATA_DIR = Path("data")

st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")

# ---------- CSS (polished look) ----------
st.markdown(
    """
<style>
/* Hide default Streamlit menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px;}
h1 {margin-bottom: 0.2rem;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}

/* Cards */
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 14px 14px;
  margin: 10px 0;
}
.card-title {font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;}
.card-meta {opacity: 0.75; font-size: 0.9rem; margin-bottom: 8px;}
.hr {height:1px; background: rgba(255,255,255,0.08); margin: 10px 0;}

/* Pills */
.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.2px;
  border: 1px solid rgba(255,255,255,0.12);
  margin-right: 8px;
}
.pill-pos { background: rgba(0, 200, 83, 0.18); color: #b9ffcf; border-color: rgba(0,200,83,0.35);}
.pill-neg { background: rgba(255, 59, 48, 0.18); color: #ffd1cf; border-color: rgba(255,59,48,0.35);}
.pill-unk { background: rgba(255, 255, 255, 0.08); color: rgba(255,255,255,0.85); }

/* Sidebar */
section[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,0.08);}

</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def pill(sentiment: str) -> str:
    s = (sentiment or "").strip().lower()
    if s.startswith("pos"):
        return '<span class="pill pill-pos">POSITIVE</span>'
    if s.startswith("neg"):
        return '<span class="pill pill-neg">NEGATIVE</span>'
    return '<span class="pill pill-unk">UNKNOWN</span>'


def render_simple_list(df: pd.DataFrame, fields: list[str], limit: int = 25, title_field: str | None = None):
    if df.empty:
        st.info("No data to show.")
        return

    n = min(limit, len(df))
    st.markdown(f'<div class="small-muted">Showing {n} rows (of {len(df)})</div>', unsafe_allow_html=True)

    for i in range(n):
        row = df.iloc[i]
        title = ""
        if title_field and title_field in df.columns:
            title = str(row.get(title_field, "")).strip()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if title:
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)

        meta_parts = []
        for f in fields:
            if f in df.columns and f != title_field:
                meta_parts.append(f"{f}: {row.get(f, '')}")
        if meta_parts:
            st.markdown(f'<div class="card-meta">{" • ".join(map(str, meta_parts))}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def plot_bar(pos_count: int, neg_count: int, avg_pos: float | None, avg_neg: float | None):
    labels = ["Positive", "Negative"]
    values = [pos_count, neg_count]

    fig = plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")

    info = []
    if avg_pos is not None:
        info.append(f"Avg conf (Positive): {avg_pos:.2f}")
    if avg_neg is not None:
        info.append(f"Avg conf (Negative): {avg_neg:.2f}")
    if info:
        plt.xlabel(" | ".join(info))

    st.pyplot(fig)


def plot_wordcloud(text: str):
    if not text.strip():
        st.info("Not enough text for word cloud.")
        return
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
    arr = wc.to_array()  # avoids numpy copy issues
    fig = plt.figure()
    plt.imshow(arr)
    plt.axis("off")
    st.pyplot(fig)


# ---------- Load data ----------
products_df = load_csv_safe(DATA_DIR / "products.csv")
testimonials_df = load_csv_safe(DATA_DIR / "testimonials.csv")
reviews_df = load_csv_safe(DATA_DIR / "reviews.csv")

# ---------- UI ----------
st.title("Brand Reputation Monitor (2023)")
st.markdown('<div class="small-muted">Scraped from web-scraping.dev (sandbox)</div>', unsafe_allow_html=True)

section = st.sidebar.radio("Select section:", ["Products", "Testimonials", "Reviews"])

if section == "Products":
    st.header("Products (scraped)")
    if products_df.empty:
        st.error("No products.csv found or it’s empty. Run your scraper and commit data/products.csv to GitHub.")
    else:
        render_simple_list(products_df, fields=["page"], title_field="title", limit=25)

elif section == "Testimonials":
    st.header("Testimonials (scraped)")
    if testimonials_df.empty:
        st.error("No testimonials.csv found or it’s empty. Run your scraper and commit data/testimonials.csv to GitHub.")
    else:
        render_simple_list(testimonials_df, fields=["page"], title_field="text", limit=25)

else:
    st.header("Reviews — Sentiment Analysis")

    if reviews_df.empty:
        st.error("No reviews.csv found or it’s empty. Make sure data/reviews.csv exists in GitHub (not just locally).")
        st.stop()

    needed = {"date", "text"}
    if not needed.issubset(set(reviews_df.columns)):
        st.error("reviews.csv must contain at least: date, text (and ideally sentiment, confidence, page).")
        st.stop()

    reviews_df["date"] = safe_to_datetime(reviews_df["date"])
    reviews_df = reviews_df.dropna(subset=["date"])

    if reviews_df.empty:
        st.error("All review dates are empty/invalid. Fix reviews.csv date format to YYYY-MM-DD.")
        st.stop()

    reviews_df = reviews_df[reviews_df["date"].dt.year == 2023]
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

    st.markdown(f"**Number of reviews in {selected}: {len(month_df)}**")

    # If sentiment missing, still show reviews (no model on Render)
    if "sentiment" not in month_df.columns or "confidence" not in month_df.columns:
        st.warning("No sentiment/confidence columns found. (This app doesn’t run the model on Render to save RAM.)")
        # show simple cards
        n = min(15, len(month_df))
        for i in range(n):
            r = month_df.iloc[i]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card-meta">{str(r["date"])[:10]} • page: {r.get("page","")}</div>',
                unsafe_allow_html=True,
            )
            st.write(str(r.get("text", "")))
            st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

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
    n = min(15, len(month_df))

    for i in range(n):
        r = month_df.iloc[i]
        badge = pill(r.get("sentiment", ""))
        conf = r.get("confidence", None)

        meta = f'{str(r["date"])[:10]} • page: {r.get("page","")}'
        if pd.notna(conf):
            meta += f" • conf: {float(conf):.3f}"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"{badge}", unsafe_allow_html=True)
        st.markdown(f'<div class="card-meta">{meta}</div>', unsafe_allow_html=True)
        st.write(str(r.get("text", "")))
        st.markdown("</div>", unsafe_allow_html=True)