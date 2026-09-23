"""Streamlit demo for multimodal product matching.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.preprocessing import load_image  # noqa: E402
from src.retrieval.search import ProductSearchEngine, QueryError, SearchResult  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402

st.set_page_config(page_title="Product Matching", page_icon="🛍️", layout="wide")

MODES = {"Multimodal": "multimodal", "Image": "image", "Text": "text"}
CARDS_PER_ROW = 5

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  .card-title {font-size: 0.86rem; font-weight: 600; line-height: 1.25; min-height: 2.5em;}
  .card-meta {font-size: 0.76rem; color: #6b6b6b;}
  .badge {display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600;}
  .badge-match {background: #e3f4ea; color: #146c3a;}
  .badge-nomatch {background: #f1f1f1; color: #555;}
  .score {font-variant-numeric: tabular-nums; font-size: 0.78rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models and index…")
def get_engine() -> tuple[ProductSearchEngine | None, Config, str | None]:
    """Load config, models and FAISS index once per server process."""
    config = load_config()
    try:
        return ProductSearchEngine.from_artifacts(config), config, None
    except Exception as exc:  # show a helpful message instead of a stack trace
        return None, config, f"{type(exc).__name__}: {exc}"


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def match_badge(is_match: bool) -> str:
    return ('<span class="badge badge-match">✓ MATCH</span>' if is_match
            else '<span class="badge badge-nomatch">NO MATCH</span>')


def render_result(r: SearchResult) -> None:
    img = load_image(r.image_file)
    if img is not None:
        st.image(img, width="stretch")
    else:
        st.caption("no image")
    st.markdown(f'<div class="card-title">{r.title}</div>'
                f'<div class="card-meta">#{r.rank} · {r.product_id} · {r.category}</div>'
                f'<div class="score">image {fmt(r.image_similarity)} · text {fmt(r.text_similarity)}<br>'
                f'<b>combined {r.combined_similarity:.3f}</b></div>{match_badge(r.is_match)}',
                unsafe_allow_html=True)


def search_tab(engine: ProductSearchEngine, config: Config, mode: str, top_k: int) -> None:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Query product")
        source = st.radio("Query source", ["Upload", "Pick from catalog"], horizontal=True,
                          key="query_source",
                          label_visibility="collapsed")
        image: Image.Image | None = None
        title = ""
        if source == "Upload":
            upload = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"])
            if upload is not None:
                image = load_image(upload.getvalue())
                if image is None:
                    st.error("Could not read that image.")
            title = st.text_input("Product title", placeholder="e.g. Nordwave red cotton t-shirt size M")
        else:
            meta = engine.store.metadata
            options = meta["product_id"].tolist()
            pid = st.selectbox("Catalog product", options,
                               format_func=lambda p: f"{p} — {meta.loc[meta.product_id == p, 'title'].iloc[0]}")
            product = engine.product(pid)
            image = load_image(product["image_file"])
            title = str(product["title"])
        if image is not None:
            st.image(image, width=220)
        if title:
            st.caption(title)
        go = st.button("Find Similar Products", type="primary", width="stretch", key="search_btn")

    with right:
        st.subheader("Similar products")
        if not go:
            st.info("Provide an image and/or a title, then click **Find Similar Products**.")
            return
        try:
            with st.spinner("Searching…"):
                results = engine.search(image=image, title=title, mode=mode, top_k=top_k)
        except QueryError as exc:
            st.warning(str(exc))
            return
        if not results:
            st.warning("No results.")
            return
        n_match = sum(r.is_match for r in results)
        st.caption(f"{len(results)} results · {n_match} above the {engine.threshold(mode):.2f} "
                   f"match threshold · mode: {mode}")
        for start in range(0, len(results), CARDS_PER_ROW):
            cols = st.columns(CARDS_PER_ROW)
            for col, r in zip(cols, results[start:start + CARDS_PER_ROW]):
                with col:
                    render_result(r)


def product_input(engine: ProductSearchEngine, label: str) -> dict:
    st.markdown(f"**Product {label}**")
    source = st.radio(f"Source {label}", ["Catalog", "Upload"], horizontal=True,
                      key=f"src_{label}", label_visibility="collapsed")
    if source == "Catalog":
        pid = st.selectbox(f"Product {label}", engine.store.metadata["product_id"].tolist(),
                           key=f"pid_{label}", index=0 if label == "A" else 1)
        product = engine.product(pid)
        img = load_image(product["image_file"])
        if img is not None:
            st.image(img, width=180)
        st.caption(product["title"])
        return {"product_id": pid}
    upload = st.file_uploader(f"Image {label}", type=["jpg", "jpeg", "png", "webp"], key=f"up_{label}")
    img = load_image(upload.getvalue()) if upload else None
    if img is not None:
        st.image(img, width=180)
    title = st.text_input(f"Title {label}", key=f"title_{label}")
    return {"image": img, "title": title}


def match_tab(engine: ProductSearchEngine) -> None:
    a_col, b_col = st.columns(2, gap="large")
    with a_col:
        a = product_input(engine, "A")
    with b_col:
        b = product_input(engine, "B")
    if st.button("Compare products", type="primary", key="match_btn"):
        try:
            res = engine.match(a.get("image"), a.get("title"), b.get("image"), b.get("title"),
                               a.get("product_id"), b.get("product_id"))
        except (QueryError, KeyError) as exc:
            st.warning(str(exc))
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Image similarity", fmt(res.image_similarity))
        c2.metric("Text similarity", fmt(res.text_similarity))
        c3.metric("Combined", f"{res.combined_similarity:.3f}")
        c4.metric("Decision", res.decision, help=f"threshold = {res.threshold:.2f}")


def evaluation_tab(config: Config) -> None:
    results_dir = config.resolve(config.paths.results_dir)
    plots_dir = config.resolve(config.paths.plots_dir)
    table = results_dir / "ablation.csv"
    if not table.is_file():
        st.info("No evaluation results yet. Run `python scripts/evaluate.py`.")
        return
    df = pd.read_csv(table)
    keep = ["label", "recall@1", "recall@5", "recall@10", "map@10", "tuned_threshold",
            "precision", "recall", "f1", "query_f1"]
    st.dataframe(df[[c for c in keep if c in df.columns]].round(3), hide_index=True,
                 width="stretch")
    st.caption("Metrics on held-out label groups of the synthetic sample catalog; the threshold "
               "is tuned on train groups. Not indicative of real marketplace performance.")
    pngs = sorted(plots_dir.glob("*.png"))
    for start in range(0, len(pngs), 2):
        cols = st.columns(2)
        for col, p in zip(cols, pngs[start:start + 2]):
            col.image(str(p), width="stretch")


def main() -> None:
    engine, config, error = get_engine()
    st.title("🛍️ Multimodal Product Matching")
    st.caption("Find duplicate / similar marketplace listings from product images and titles "
               "(CLIP + Sentence Transformers + FAISS).")
    if engine is None:
        st.error(f"Search engine unavailable: {error}")
        st.code("python scripts/build_embeddings.py\npython scripts/build_index.py", language="bash")
        return

    with st.sidebar:
        st.header("Settings")
        mode = MODES[st.radio("Retrieval mode", list(MODES), key="mode")]
        top_k = st.slider("Top-K", 1, 30, config.retrieval.top_k)
        st.divider()
        st.markdown(f"**Catalog:** {len(engine)} products  \n"
                    f"**Weights:** image {config.fusion.image_weight} · text {config.fusion.text_weight}  \n"
                    f"**Match threshold:** {engine.threshold(mode):.2f}  \n"
                    f"**Device:** {engine.encoder.device}")

    search, match, evaluation = st.tabs(["🔎 Search", "⚖️ Match two products", "📊 Evaluation"])
    with search:
        search_tab(engine, config, mode, top_k)
    with match:
        match_tab(engine)
    with evaluation:
        evaluation_tab(config)


main()
