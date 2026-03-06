"""
Competitor Feature Gap Analyzer — PM-focused Streamlit app.
Uses Gemini 2.5 Flash to identify critical features and compare products.
"""

import json
from typing import Optional

import hashlib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import google.generativeai as genai

st.set_page_config(
    page_title="Competitor Feature Gap Analyzer",
    page_icon="📊",
    layout="wide",
)
# Match primary button to corporate navy (in case theme is overridden)
st.markdown(
    "<style>.stButton > button[kind='primary'] { background-color: #003366 !important; }</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")
product_name = st.sidebar.text_input(
    "Your product name",
    placeholder="e.g. Acme CRM",
)
competitor_inputs = [
    st.sidebar.text_input("Competitor 1", placeholder="e.g. Microsoft Loop"),
    st.sidebar.text_input("Competitor 2", placeholder="e.g. Asana"),
    st.sidebar.text_input("Competitor 3", placeholder="e.g. Pipedrive"),
]
competitors = [c.strip() for c in competitor_inputs if c.strip()]

analyze_clicked = st.sidebar.button("Analyze", type="primary")

# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------
def _ensure_genai():
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.")
        return False
    key = st.secrets["GOOGLE_API_KEY"].strip()
    if not key:
        st.error("Set `GOOGLE_API_KEY` in `.streamlit/secrets.toml`. Get a key at https://aistudio.google.com/apikey")
        return False
    genai.configure(api_key=key)
    return True


def analyze_category_and_features(product: str, competitor_names: list) -> Optional[dict]:
    """Use Gemini 2.5 Flash to get 5 critical features and Yes/No/Partial for each product."""
    if not _ensure_genai():
        return None

    all_products = [product] + competitor_names
    products_str = ", ".join(f'"{p}"' for p in all_products)

    prompt = f"""You are a product manager. Respond only with valid JSON, no markdown or extra text.

For the product category implied by "{product}" and its competitors {products_str}:

1. List exactly 5 critical features that matter most in this category (short names, e.g. "Real-time collaboration", "API & integrations").
2. For each feature, rate each product as exactly one of: "Yes", "No", "Partial".

Return a single JSON object with this structure:
{{
  "features": ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"],
  "ratings": {{
    "{product}": ["Yes" or "No" or "Partial", ... 5 values in feature order],
    "<Competitor1>": [...],
    "<Competitor2>": [...],
    "<Competitor3>": [...]
  }}
}}

Use the exact product and competitor names as given. Output only the JSON."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f"Could not parse model response as JSON: {e}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _exec_summary_from_gemini(
    product: str,
    comps: list,
    scores: list,
    gaps: list,
    features: list,
):
    """Ask Gemini for structured insights. Returns dict with positive, neutral, critical or None."""
    if not _ensure_genai():
        return None
    all_products = [product] + comps
    score_lines = " | ".join(f"{p}: {s}%" for p, s in zip(all_products, scores))
    gaps_text = "; ".join(f"{g['feature']} ({g['your_status']})" for g in gaps[:5]) if gaps else "None identified."
    prompt = f"""You are a strategy consultant. Based on this feature gap analysis, return a JSON object with exactly 3 keys:
- "strengths": one short sentence (good news or strength for {product}).
- "neutral_observations": one short sentence (observation or context, e.g. market position).
- "critical_gaps_risks": one short sentence (biggest risk or recommended action).

Context: Product {product}. Competitors: {', '.join(comps)}. Scores: {score_lines}. Gaps: {gaps_text}. Features: {', '.join(features)}.

Output only valid JSON, no markdown. Example: {{"strengths": "...", "neutral_observations": "...", "critical_gaps_risks": "..."}}"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
        return {
            "strengths": data.get("strengths", data.get("positive", "")),
            "neutral_observations": data.get("neutral_observations", data.get("neutral", "")),
            "critical_gaps_risks": data.get("critical_gaps_risks", data.get("critical", "")),
        }
    except Exception:
        return None


def feature_coverage_score(ratings: list, weights: Optional[list] = None) -> float:
    """Score = weighted (Yes*1 + Partial*0.5 + No*0) / total_weight * 100. Equal weights if weights is None."""
    n = len(ratings)
    if n == 0:
        return 0.0
    if weights is None or len(weights) != n:
        weights = [1] * n
    weighted_sum = sum(
        w * (1 if r == "Yes" else 0.5 if r == "Partial" else 0)
        for w, r in zip(weights, ratings)
    )
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return round(100 * weighted_sum / total_weight, 1)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Competitor Feature Gap Analyzer")
st.caption("Identify critical features and compare your product to competitors.")

if not product_name:
    st.info("Enter your product name and up to 3 competitors in the sidebar, then click **Analyze**.")
    st.stop()

if len(competitors) < 1:
    st.warning("Add at least one competitor in the sidebar.")
    st.stop()

# Run analysis on button click and persist result
if analyze_clicked:
    with st.spinner("Analyzing with Gemini 2.5 Flash…"):
        result = analyze_category_and_features(product_name, competitors)
    if result is not None:
        st.session_state["gap_analysis_result"] = result
    if result is None:
        st.stop()

# Use persisted result so sliders and chart stay after first run
result = st.session_state.get("gap_analysis_result")
if result is None:
    st.info("Click **Analyze** in the sidebar to run the analysis.")
    st.stop()

features = result.get("features", [])
ratings = result.get("ratings", {})
comp_display = competitors

# Sidebar: dynamic sliders for feature weights (1–10)
st.sidebar.header("Feature weights")
st.sidebar.caption("Weight each feature from 1 (low) to 10 (high). Higher weight = more impact on score.")
feature_weights = []
for i, feat in enumerate(features):
    w = st.sidebar.slider(
        feat,
        min_value=1,
        max_value=10,
        value=5,
        key=f"feat_weight_{i}",
    )
    feature_weights.append(w)

# Scores (needed for metrics row and charts)
all_products = [product_name] + comp_display
scores = [
    feature_coverage_score(ratings.get(p, []), feature_weights)
    for p in all_products
]
my_score = scores[0] if scores else 0

# Executive Summary metrics row (BLUF) — delta = gap vs your product (e.g. -20% = 20 pts behind)
n_cols = len(all_products)
cols = st.columns(n_cols)
for idx, (prod, sc) in enumerate(zip(all_products, scores)):
    delta = None if idx == 0 else round(sc - my_score, 1)
    delta_str = f"{delta:+.1f}%" if delta is not None else None
    with cols[idx]:
        st.metric(label=prod, value=f"{sc}%", delta=delta_str)
st.caption(f"Delta (Δ) for competitors = gap vs **{product_name}** (e.g. −20% = 20 points behind).")

st.divider()

# Build comparison table
rows = []
for i, feat in enumerate(features):
    row = {"Feature": feat}
    for prod in [product_name] + comp_display:
        r = ratings.get(prod, [])
        row[prod] = r[i] if i < len(r) else "—"
    rows.append(row)

df = pd.DataFrame(rows)
product_cols = [c for c in df.columns if c != "Feature"]

# Conditional formatting: Yes = light green, Partial = light yellow, No = light red
def _status_style(val):
    if val == "Yes":
        return "background-color: #d4edda; color: #155724; font-weight: 500;"
    if val == "Partial":
        return "background-color: #fff3cd; color: #856404;"
    if val == "No":
        return "background-color: #f8d7da; color: #721c24;"
    return ""

styled_df = df.style.apply(
    lambda col: col.map(_status_style) if col.name in product_cols else [""] * len(col),
    axis=0,
)

# column_config for clearer column labels and layout
column_config = {"Feature": st.column_config.TextColumn("Feature", width="medium")}
for c in product_cols:
    column_config[c] = st.column_config.TextColumn(c, width="small")

st.subheader("Feature comparison")
st.dataframe(styled_df, use_container_width=True, hide_index=True, column_config=column_config)
st.download_button(
    label="Export for Deck",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="feature_comparison.csv",
    mime="text/csv",
    key="export_csv",
)

# Theme-aligned colors: deep navy (#003366) to match config.toml
CHART_PRIMARY = "#003366"
NAVY_PALETTE = ["#003366", "#004080", "#004d99", "#0059b3", "#0066cc"]

# 50/50 side-by-side: Bar chart | Radar chart
col1, col2 = st.columns(2)
score_df = pd.DataFrame({
    "Product": all_products,
    "Feature coverage (%)": scores,
})
fig_bar = px.bar(
    score_df,
    x="Product",
    y="Feature coverage (%)",
    title="Feature coverage score (weighted)",
    color="Feature coverage (%)",
    color_continuous_scale=["#B8D4E3", "#003366"],
    range_color=[0, 100],
)
fig_bar.update_layout(
    xaxis_tickangle=-30,
    showlegend=False,
    yaxis_range=[0, 105],
    font=dict(color=CHART_PRIMARY),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="top", y=-0.12),
)
with col1:
    st.plotly_chart(fig_bar, use_container_width=True)

def _rating_to_score(r):
    return 1.0 if r == "Yes" else 0.5 if r == "Partial" else 0.0

fig_radar = go.Figure()
for idx, prod in enumerate(all_products):
    vals = ratings.get(prod, [])
    r = [_rating_to_score(vals[i]) if i < len(vals) else 0 for i in range(len(features))]
    theta = list(features) + [features[0]]
    r_closed = r + [r[0]]
    color = NAVY_PALETTE[idx % len(NAVY_PALETTE)]
    fig_radar.add_trace(
        go.Scatterpolar(
            r=r_closed,
            theta=theta,
            name=prod,
            fill="toself",
            line=dict(width=2, color=color),
        )
    )
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1.05], tickvals=[0, 0.5, 1], ticktext=["No", "Partial", "Yes"]),
        bgcolor="rgba(0,0,0,0)",
    ),
    title="Feature profile (radar)",
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.12),
    font=dict(color=CHART_PRIMARY),
    paper_bgcolor="rgba(0,0,0,0)",
)
with col2:
    st.plotly_chart(fig_radar, use_container_width=True)

# Product roadmap — biggest gaps + executive summary expander
my_ratings = ratings.get(product_name, [])
gaps = []
for i, feat in enumerate(features):
    my_val = my_ratings[i] if i < len(my_ratings) else "No"
    others = [
        ratings.get(c, [])[i] if i < len(ratings.get(c, [])) else "No"
        for c in comp_display
    ]
    if my_val in ("No", "Partial") and any(o == "Yes" for o in others):
        gaps.append({
            "feature": feat,
            "your_status": my_val,
            "competitors_with_yes": [comp_display[j] for j, o in enumerate(others) if o == "Yes"],
        })

st.subheader("Product roadmap — biggest gaps")
if not gaps:
    st.success("No major gaps: your product matches or leads on these critical features.")
else:
    for g in gaps[:5]:
        comp_list = ", ".join(g["competitors_with_yes"])
        st.warning(
            f"**{g['feature']}** — You: {g['your_status']} | Competitors with Yes: {comp_list}"
        )

# Executive summary expander (AI insights) — success / info / warning
result_hash = hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()
if st.session_state.get("exec_summary_hash") != result_hash:
    st.session_state["exec_summary_hash"] = result_hash
    st.session_state["exec_summary_data"] = None
if st.session_state.get("exec_summary_data") is None:
    with st.spinner("Generating strategic insights…"):
        st.session_state["exec_summary_data"] = _exec_summary_from_gemini(
            product_name, comp_display, scores, gaps, features
        )

with st.expander("View AI Strategic Insights"):
    data = st.session_state.get("exec_summary_data")
    if isinstance(data, dict):
        strengths = data.get("strengths") or data.get("positive")
        neutral_obs = data.get("neutral_observations") or data.get("neutral")
        critical = data.get("critical_gaps_risks") or data.get("critical")
        if strengths:
            st.markdown("**Strengths**")
            st.success(strengths)
        if neutral_obs:
            st.markdown("**Neutral Observations**")
            st.info(neutral_obs)
        if critical:
            st.markdown("**Critical Gaps/Risks**")
            st.warning(critical)
    elif data is None:
        st.caption("Insights could not be generated. Check your API key.")
    else:
        st.markdown(data)
