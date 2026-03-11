"""
main.py — Streamlit front-end for LungVision (Keras / TF, 4-class).
Run with: uv run streamlit run main.py
"""

import os
import random
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from feedback_store import build_row, get_store

st.set_page_config(page_title="LungVision", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;800&display=swap');
:root{--bg:#0a0d12;--surface:#111620;--surface2:#161c28;--border:#1e2738;--accent:#00d4ff;--accent2:#00ff9d;--warn:#ff9f43;--danger:#ff4e6a;--text:#e2e8f0;--text2:#8899aa;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Syne',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;}
[data-testid="stFileUploader"]{background:var(--surface2)!important;border:2px dashed var(--border)!important;border-radius:12px!important;}
[data-testid="stMetric"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:1rem!important;}
[data-testid="stMetricValue"]{font-family:'DM Mono',monospace!important;color:var(--accent)!important;font-size:1.4rem!important;}
[data-testid="stMetricLabel"]{color:var(--text2)!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;border-radius:999px!important;}
.stButton>button{background:linear-gradient(135deg,#00d4ff22,#00ff9d11)!important;color:var(--accent)!important;border:1px solid var(--accent)!important;border-radius:8px!important;font-family:'DM Mono',monospace!important;}
.stButton>button:hover{background:var(--accent)!important;color:#000!important;}
button[data-baseweb="tab"]{font-family:'DM Mono',monospace!important;color:var(--text2)!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
hr{border-color:var(--border)!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Class config ──────────────────────────────────────────────────────────────
LABELS = ["bacterial_and_other", "covid19", "normal", "viral_pneumonia"]
DISPLAY = {
    "bacterial_and_other": "Bacterial / Other",
    "covid19":             "COVID-19",
    "normal":              "Normal",
    "viral_pneumonia":     "Viral Pneumonia",
}
CLASS_COLOR = {
    "bacterial_and_other": "#ff9f43",
    "covid19":             "#ff4e6a",
    "normal":              "#00ff9d",
    "viral_pneumonia":     "#a29bfe",
}


# ── Chart helpers ─────────────────────────────────────────────────────────────
def make_bar_chart(probs_dict: dict) -> plt.Figure:
    class_labels = [DISPLAY[k] for k in LABELS]
    colors       = ["#00d4ff", "#00ff9d", "#ff9f43"]
    x     = np.arange(len(LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#111620")
    ax.set_facecolor("#111620")
    for i, (model_name, prob_dict) in enumerate(probs_dict.items()):
        vals = [prob_dict[k] for k in LABELS]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=colors[i], alpha=0.85, edgecolor="#0a0d12", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                        f"{h:.0%}", ha="center", va="bottom",
                        fontsize=7, color="#8899aa", fontfamily="monospace")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_labels, color="#e2e8f0", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Probability", color="#4a5568", fontsize=9)
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2738")
    ax.legend(facecolor="#161c28", edgecolor="#1e2738", labelcolor="#8899aa", fontsize=8)
    ax.set_title("Model Comparison — All Classes", color="#8899aa", fontsize=10, pad=10, loc="left")
    plt.tight_layout()
    return fig


def make_feedback_chart(df) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#111620")

    ax = axes[0]
    ax.set_facecolor("#111620")
    classes  = list(DISPLAY.values())
    agree    = [len(df[(df["predicted_label"] == k) & (df["user_agrees"] == True)])  for k in LABELS]
    disagree = [len(df[(df["predicted_label"] == k) & (df["user_agrees"] == False)]) for k in LABELS]
    x = np.arange(len(classes))
    ax.bar(x, agree,    0.5, label="Correct",   color="#00ff9d", alpha=0.85)
    ax.bar(x, disagree, 0.5, label="Incorrect", color="#ff4e6a", alpha=0.85, bottom=agree)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="#e2e8f0", fontsize=8, rotation=15)
    ax.set_ylabel("Count", color="#4a5568", fontsize=9)
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2738")
    ax.legend(facecolor="#161c28", edgecolor="#1e2738", labelcolor="#8899aa", fontsize=8)
    ax.set_title("Agreement by Predicted Class", color="#8899aa", fontsize=10, loc="left")

    ax2 = axes[1]
    ax2.set_facecolor("#111620")
    wrong = df[df["user_agrees"] == False]
    if len(wrong) > 0:
        confusion = np.zeros((4, 4), dtype=int)
        for _, row in wrong.iterrows():
            try:
                pi = LABELS.index(row["predicted_label"])
                ci = LABELS.index(row["correct_label"])
                confusion[pi][ci] += 1
            except ValueError:
                pass
        ax2.imshow(confusion, cmap="YlOrRd", aspect="auto")
        short = ["Bact.", "COVID", "Normal", "Viral"]
        ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
        ax2.set_xticklabels(short, color="#e2e8f0", fontsize=8)
        ax2.set_yticklabels(short, color="#e2e8f0", fontsize=8)
        ax2.set_xlabel("Correct Label",    color="#4a5568", fontsize=9)
        ax2.set_ylabel("Predicted Label",  color="#4a5568", fontsize=9)
        for i in range(4):
            for j in range(4):
                if confusion[i][j] > 0:
                    ax2.text(j, i, str(confusion[i][j]), ha="center", va="center",
                             color="white", fontsize=9, fontfamily="monospace")
        ax2.set_title("Misprediction Confusion", color="#8899aa", fontsize=10, loc="left")
    else:
        ax2.text(0.5, 0.5, "No incorrect predictions yet",
                 ha="center", va="center", color="#4a5568",
                 fontsize=10, transform=ax2.transAxes)
        ax2.axis("off")

    plt.tight_layout()
    return fig


@st.cache_resource(show_spinner="Loading ensemble models…")
def load_ensemble(resnet_path, densenet_path, meta_path, w_resnet, w_densenet):
    from models.ensemble import PneumoniaEnsemble
    override = (w_resnet, w_densenet) if (w_resnet + w_densenet) > 0 else None
    return PneumoniaEnsemble(resnet_path=resnet_path, densenet_path=densenet_path,
                             meta_path=meta_path, override_weights=override)


def demo_result() -> dict:
    probs   = np.random.dirichlet(np.ones(4)).tolist()
    r_probs = np.random.dirichlet(np.ones(4)).tolist()
    d_probs = np.random.dirichlet(np.ones(4)).tolist()
    idx = int(np.argmax(probs))
    return {
        "label":          LABELS[idx],
        "display_label":  DISPLAY[LABELS[idx]],
        "confidence":     probs[idx],
        "probs":          {k: probs[i]   for i, k in enumerate(LABELS)},
        "resnet_probs":   {k: r_probs[i] for i, k in enumerate(LABELS)},
        "densenet_probs": {k: d_probs[i] for i, k in enumerate(LABELS)},
        "meta":           {},
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 LungVision")
    st.markdown("<p style='color:#4a5568;font-size:0.8rem;font-family:DM Mono,monospace;'>ResNet-50 × DenseNet-121 · 4-Class</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Model Files")
    resnet_path   = st.text_input("ResNet-50 (.h5)",    value="weights/resnet50_best.h5")
    densenet_path = st.text_input("DenseNet-121 (.h5)", value="weights/densenet121_best.h5")
    meta_path     = st.text_input("Metadata JSON",      value="weights/ensemble_weights.json")

    st.divider()
    st.markdown("#### Override Ensemble Weights")
    st.caption("Leave both at 0 to auto-load from JSON")
    col1, col2 = st.columns(2)
    with col1:
        w_resnet   = st.slider("ResNet",   0.0, 1.0, 0.0, 0.05)
    with col2:
        w_densenet = st.slider("DenseNet", 0.0, 1.0, 0.0, 0.05)

    st.divider()
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.01)

    st.divider()
    st.markdown("#### Feedback Storage")
    backend = st.radio(
        "Backend",
        options=["CSV", "SQLite"],
        index=0,
        horizontal=True,
        help="CSV: feedback/feedback_log.csv  |  SQLite: feedback/feedback.db",
    )
    st.caption({
        "CSV":    "📄 `feedback/feedback_log.csv`",
        "SQLite": "🗄️ `feedback/feedback.db`",
    }[backend])

    st.divider()
    st.caption("⚠️ For research/educational use only. Not a medical device.")

# Initialise the chosen store (cached so it isn't reconstructed on every rerun)
@st.cache_resource
def get_feedback_store(backend: str):
    return get_store(backend)

store = get_feedback_store(backend)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.5rem'>
<span style='font-size:2.4rem;font-weight:800;letter-spacing:-0.03em;'>Chest X-Ray Analysis</span><br>
<span style='color:#4a5568;font-family:DM Mono,monospace;font-size:0.85rem;'>
Ensemble · ResNet-50 + DenseNet-121 · Bacterial · COVID-19 · Normal · Viral
</span></div>
""", unsafe_allow_html=True)
st.divider()

uploaded = st.file_uploader("Drop a chest X-ray (JPEG / PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a chest X-ray to begin analysis.")
    st.stop()

image = Image.open(uploaded)

# ── Load models ───────────────────────────────────────────────────────────────
models_exist = all(os.path.isfile(p) for p in [resnet_path, densenet_path, meta_path])

if not models_exist:
    missing = [p for p in [resnet_path, densenet_path, meta_path] if not os.path.isfile(p)]
    st.warning("⚠️ Running in **demo mode** — files not found:\n" + "\n".join(f"- `{p}`" for p in missing))
    result = demo_result()
else:
    ensemble = load_ensemble(resnet_path, densenet_path, meta_path, w_resnet, w_densenet)
    meta = ensemble.meta
    with st.sidebar:
        st.divider()
        st.markdown("#### Training Metrics (from JSON)")
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace;font-size:0.78rem;color:#8899aa;line-height:1.9;'>
        ResNet acc&nbsp;&nbsp;&nbsp;: <span style='color:#00d4ff'>{meta.get('resnet_accuracy', 0):.2%}</span><br>
        DenseNet acc&nbsp;: <span style='color:#00ff9d'>{meta.get('densenet_accuracy', 0):.2%}</span><br>
        Ensemble acc&nbsp;: <span style='color:#ff9f43'>{meta.get('ensemble_accuracy', 0):.2%}</span><br>
        ResNet w&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#00d4ff'>{ensemble.w_resnet:.4f}</span><br>
        DenseNet w&nbsp;&nbsp;: <span style='color:#00ff9d'>{ensemble.w_densenet:.4f}</span>
        </div>""", unsafe_allow_html=True)
    with st.spinner("Analysing image…"):
        result = ensemble.predict(image)

# ── Results layout ────────────────────────────────────────────────────────────
col_img, col_results = st.columns([1, 1.4], gap="large")

with col_img:
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

with col_results:
    label    = result["label"]
    color    = CLASS_COLOR[label]
    icon     = "🟢" if label == "normal" else "🔴"
    low_conf = result["confidence"] < threshold

    st.markdown(f"""
    <div style='background:{color}18;border:1px solid {color}44;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;'>
        <div style='font-size:0.75rem;color:{color};font-family:DM Mono,monospace;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;'>{icon} DIAGNOSIS</div>
        <div style='font-size:2rem;font-weight:800;color:{color};letter-spacing:-0.02em;'>{result['display_label']}</div>
        <div style='font-size:0.8rem;color:#4a5568;font-family:DM Mono,monospace;margin-top:0.3rem;'>Ensemble confidence: {result['confidence']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    if low_conf:
        st.warning(f"⚠️ Low confidence ({result['confidence']:.1%}) — result may be unreliable.")

    st.markdown("**Class Probabilities**")
    for key in LABELS:
        prob   = result["probs"][key]
        c      = CLASS_COLOR[key]
        is_top = key == label
        weight = "800" if is_top else "400"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-family:DM Mono,monospace;"
            f"font-size:0.8rem;color:{'#e2e8f0' if is_top else '#4a5568'};font-weight:{weight};margin-bottom:2px;'>"
            f"<span>{DISPLAY[key]}</span><span style='color:{c};'>{prob:.3f}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(float(prob))

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("ResNet-50",    f"{result['resnet_probs'][label]:.1%}",   delta=f"P({DISPLAY[label]})")
    m2.metric("DenseNet-121", f"{result['densenet_probs'][label]:.1%}", delta=f"P({DISPLAY[label]})")
    m3.metric("Ensemble",     f"{result['probs'][label]:.1%}",          delta=f"P({DISPLAY[label]})")

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.divider()
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison Chart",
    "🔢 Raw Probabilities",
    "💬 Feedback",
    "📋 Feedback Log",
])

with tab1:
    st.pyplot(make_bar_chart({
        "ResNet-50":    result["resnet_probs"],
        "DenseNet-121": result["densenet_probs"],
        "Ensemble":     result["probs"],
    }), use_container_width=True)

with tab2:
    lines = ["Class              ResNet-50   DenseNet-121  Ensemble", "─" * 58]
    for key in LABELS:
        lines.append(
            f"{DISPLAY[key]:<22} {result['resnet_probs'][key]:.6f}   "
            f"{result['densenet_probs'][key]:.6f}    {result['probs'][key]:.6f}"
        )
    lines += ["─" * 58, f"Predicted          {result['display_label']}  (confidence: {result['confidence']:.1%})"]
    st.code("\n".join(lines), language="text")

# ── Feedback tab ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Was this prediction correct?")
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.85rem;color:#8899aa;'>"
        f"Predicted: <span style='color:{CLASS_COLOR[result['label']]};font-weight:800;'>"
        f"{result['display_label']}</span> &nbsp;·&nbsp; "
        f"Confidence: {result['confidence']:.1%} &nbsp;·&nbsp; "
        f"Backend: <span style='color:#00d4ff;'>{backend}</span></p>",
        unsafe_allow_html=True,
    )

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None

    if st.session_state.last_uploaded != uploaded.name:
        st.session_state.feedback_submitted = False
        st.session_state.last_uploaded = uploaded.name

    if st.session_state.feedback_submitted:
        st.success("✅ Feedback recorded. Upload a new image to submit again.")
    else:
        fa, fb = st.columns(2)
        with fa:
            user_agrees = st.radio(
                "Prediction accuracy",
                options=[True, False],
                format_func=lambda x: "✅ Correct" if x else "❌ Incorrect",
                horizontal=True,
            )
        with fb:
            correct_label = st.selectbox(
                "Correct label (if incorrect)",
                options=LABELS,
                format_func=lambda k: DISPLAY[k],
                index=LABELS.index(result["label"]),
                disabled=user_agrees,
            )

        notes = st.text_input(
            "Optional notes",
            placeholder="e.g. image quality issues, unusual presentation…",
            max_chars=200,
        )

        if st.button("Submit Feedback"):
            feedback_row = build_row(
                filename      = uploaded.name,
                result        = result,
                user_agrees   = user_agrees,
                correct_label = result["label"] if user_agrees else correct_label,
                notes         = notes,
            )
            store.save(feedback_row)
            st.session_state.feedback_submitted = True
            st.rerun()

# ── Feedback log tab ──────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"#### Feedback Log &nbsp; <span style='font-family:DM Mono,monospace;font-size:0.8rem;color:#4a5568;'>({backend})</span>", unsafe_allow_html=True)

    try:
        df = store.load()
    except Exception as e:
        st.error(f"Could not load feedback: {e}")
        df = None

    if df is None or len(df) == 0:
        st.info("No feedback submitted yet.")
    else:
        total    = len(df)
        correct  = int(df["user_agrees"].sum())
        accuracy = correct / total if total > 0 else 0

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Submissions", total)
        s2.metric("Marked Correct",    correct)
        s3.metric("Agreement Rate",    f"{accuracy:.1%}")

        st.divider()
        st.pyplot(make_feedback_chart(df), use_container_width=True)
        st.divider()

        st.markdown("**Raw Log**")
        display_cols = ["id", "timestamp", "filename", "predicted_label",
                        "predicted_confidence", "correct_label", "user_agrees", "notes"]
        st.dataframe(
            df[display_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label     = "⬇️ Download as CSV",
            data      = store.export_csv(),
            file_name = "feedback_log.csv",
            mime      = "text/csv",
        )

st.divider()
