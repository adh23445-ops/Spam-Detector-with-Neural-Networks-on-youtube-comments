"""
YouTube Spam Detector — Red Neuronal para Detección de Spam
Proyecto I: Introducción a la IA
Universitat Politècnica de València

Modelo: Red Neuronal Superficial (MLP) + TF-IDF
Datos: YouTube Comments Dataset (1 956 + 45 005 comentarios)
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="YouTube Spam Detector IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — TEMA OSCURO CON NEON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Variables ─────────────────────────────────────────────────────── */
:root {
  --bg0: #080812;
  --bg1: #0f0f23;
  --bg2: #1a1a35;
  --bg3: #1e2040;
  --cyan:   #00d4ff;
  --green:  #00ff88;
  --red:    #ff4455;
  --purple: #a855f7;
  --orange: #ff8c42;
  --txt:    #e2e8f0;
  --txt2:   #94a3b8;
  --border: rgba(0,212,255,0.18);
}

/* ─── Fondo global ───────────────────────────────────────────────────── */
.stApp, .stApp > header {
  background: linear-gradient(135deg, var(--bg0) 0%, var(--bg1) 50%, var(--bg0) 100%) !important;
}
section[data-testid="stSidebar"] {
  background: rgba(8,8,18,0.97) !important;
  border-right: 1px solid var(--border) !important;
}

/* ─── Header hero ────────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #0f0f23, #1a0a2e, #0a1628);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2.5rem 3rem 2rem;
  text-align: center;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,212,255,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.hero-title {
  font-size: 3rem;
  font-weight: 900;
  background: linear-gradient(90deg, #00d4ff 0%, #a855f7 50%, #00ff88 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.5rem;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.hero-sub {
  color: var(--txt2);
  font-size: 1.05rem;
  margin: 0;
}
.hero-badge {
  display: inline-block;
  background: rgba(0,212,255,0.12);
  border: 1px solid rgba(0,212,255,0.35);
  color: var(--cyan);
  padding: 0.3rem 1rem;
  border-radius: 50px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  margin-top: 1rem;
  text-transform: uppercase;
}

/* ─── Cards ──────────────────────────────────────────────────────────── */
.card {
  background: rgba(26,26,53,0.7);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  backdrop-filter: blur(8px);
  transition: border-color 0.25s, transform 0.2s;
  margin-bottom: 1rem;
}
.card:hover { border-color: rgba(0,212,255,0.4); transform: translateY(-1px); }

/* ─── Resultados spam/safe ───────────────────────────────────────────── */
.verdict {
  text-align: center;
  padding: 2rem 1rem 1.5rem;
  border-radius: 20px;
  margin: 1rem 0;
}
.verdict.spam {
  background: rgba(255,68,85,0.08);
  border: 2px solid var(--red);
  box-shadow: 0 0 40px rgba(255,68,85,0.15);
}
.verdict.safe {
  background: rgba(0,255,136,0.08);
  border: 2px solid var(--green);
  box-shadow: 0 0 40px rgba(0,255,136,0.15);
}
.verdict-icon { font-size: 4rem; display: block; margin-bottom: 0.5rem; }
.verdict-label {
  font-size: 2rem;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  display: block;
  margin-bottom: 0.3rem;
}
.verdict.spam  .verdict-label { color: var(--red); }
.verdict.safe  .verdict-label { color: var(--green); }
.verdict-prob {
  font-size: 1rem;
  color: var(--txt2);
  display: block;
}

/* ─── Feature pills ──────────────────────────────────────────────────── */
.pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.pill {
  display: inline-block;
  padding: 0.3rem 0.85rem;
  border-radius: 30px;
  font-size: 0.78rem;
  font-weight: 700;
}
.pill-on  { background:rgba(255,68,85,0.15);   border:1px solid var(--red);   color:var(--red);   }
.pill-off { background:rgba(148,163,184,0.07); border:1px solid rgba(148,163,184,0.2); color:var(--txt2); }

/* ─── Sidebar stats ──────────────────────────────────────────────────── */
.stat-box {
  background: rgba(26,26,53,0.5);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.6rem;
  text-align: center;
}
.stat-val { font-size: 1.5rem; font-weight: 800; color: var(--cyan); }
.stat-lbl { font-size: 0.75rem; color: var(--txt2); text-transform: uppercase; letter-spacing: 0.05em; }

/* ─── Overrides Streamlit ────────────────────────────────────────────── */
.stTextArea textarea {
  background: rgba(26,26,53,0.9) !important;
  border: 1px solid rgba(0,212,255,0.3) !important;
  color: var(--txt) !important;
  border-radius: 12px !important;
  font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stButton > button {
  background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%) !important;
  border: none !important;
  color: white !important;
  font-weight: 800 !important;
  border-radius: 12px !important;
  padding: 0.8rem 2rem !important;
  font-size: 1rem !important;
  letter-spacing: 0.03em !important;
  transition: all 0.25s !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(0,212,255,0.35) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: rgba(26,26,53,0.6) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: var(--txt2) !important; }
.stTabs [aria-selected="true"] { background: rgba(0,212,255,0.15) !important; color: var(--cyan) !important; }
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-size: 1.6rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: var(--txt2) !important; font-size: 0.8rem !important; }
h1,h2,h3 { color: var(--txt) !important; }
p, li { color: var(--txt2) !important; }
hr { border-color: var(--border) !important; }
.stProgress > div > div {
  background: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
  border-radius: 4px !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 12px !important; }
.stAlert { border-radius: 12px !important; }
.uploadedFile { background: rgba(26,26,53,0.8) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES Y HELPERS
# ══════════════════════════════════════════════════════════════════════════════
SPAM_WORDS = re.compile(
    r'subscribe|suscrib|check.?out|my.?channel|free|gratis|click|win|gana|'
    r'giveaway|sorteo|crypto|bitcoin|investment|earn|profit|dm.?me|'
    r'followers|views|likes|watch.?now|promo|discount|descuento|link',
    re.I,
)

EJEMPLOS_SPAM = [
    "Subscribe to my channel NOW! FREE music + giveaway: http://bit.ly/freemusic",
    "WIN $1000 FREE! Click here and subscribe → www.spam.com/win",
    "Check out my channel!! 1000 subs giveaway! Subscribe NOW!!!!!",
]
EJEMPLOS_REAL = [
    "This is one of my favorite songs of all time. Pure nostalgia.",
    "The guitar solo at 2:47 gives me chills every single time I listen to this.",
    "I showed this song to my dad and now he loves it too!",
]


def engineer_features(texts: "list[str] | pd.Series") -> np.ndarray:
    """5 features numéricas derivadas del EDA (Sprint 2)."""
    s = pd.Series(texts).astype(str)
    n_chars  = s.str.len().fillna(0).values
    ratio_up = (s.str.count(r"[A-Z]") / pd.Series(n_chars).clip(1)).fillna(0).values
    has_url  = s.str.contains(r"https?://|www\.|bit\.ly", case=False, na=False).astype(int).values
    log_excl = np.log1p(s.str.count(r"!").fillna(0).values)
    spam_kw  = s.apply(lambda x: 1 if SPAM_WORDS.search(x) else 0).values
    return np.column_stack([n_chars, ratio_up, has_url, log_excl, spam_kw]).astype(np.float32)


def features_activadas(text: str) -> list[tuple[str, bool, str]]:
    """Detalla qué indicadores de spam disparó el texto."""
    s = pd.Series([text]).astype(str)
    return [
        ("🔗 Contiene URL",         bool(s.str.contains(r"https?://|www\.|bit\.ly", case=False, na=False).iloc[0]), "Enlace externo detectado"),
        ("📣 Palabras clave spam",  bool(SPAM_WORDS.search(text)),                                                   "Términos típicos de spam"),
        ("📏 Comentario largo",     s.str.len().iloc[0] > 150,                                                       "Más de 150 caracteres"),
        ("🔠 Exceso de mayúsculas", (s.str.count(r"[A-Z]") / s.str.len().clip(1)).iloc[0] > 0.25,                   "Ratio mayúsculas > 25%"),
        ("❗ Muchas exclamaciones", s.str.count(r"!").iloc[0] > 2,                                                    "Más de 2 signos !"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    dfs = []
    p1 = "Youtube-Spam-Dataset.csv"
    if os.path.exists(p1):
        d = pd.read_csv(p1)[["CONTENT", "CLASS"]].rename(columns={"CONTENT": "text", "CLASS": "label"})
        dfs.append(d)
    p2 = "YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv"
    if os.path.exists(p2):
        d = pd.read_csv(p2)[["comment_text", "label_spam"]].dropna()
        d = d.rename(columns={"comment_text": "text", "label_spam": "label"})
        d["label"] = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(int)
        dfs.append(d)
    if not dfs:
        st.error("❌ No se encontraron archivos CSV. Asegúrate de ejecutar la app desde la carpeta raíz del proyecto.")
        st.stop()
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df[df["text"].str.len() > 2].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO — carga Keras si existe, entrena sklearn MLP si no
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """
    Intenta cargar el modelo Keras pre-entrenado.
    Si no existe, entrena un MLPClassifier de sklearn (también una red neuronal).
    """
    # ── opción 1: modelo Keras (.keras / .h5) ──────────────────────────────
    for path in ["model/spam_model.keras", "model/spam_model.h5"]:
        if os.path.exists(path):
            try:
                import tensorflow as tf
                keras_model = tf.keras.models.load_model(path)
                tfidf = joblib.load("model/tfidf_vectorizer.pkl")
                metrics = {}
                if os.path.exists("model/metrics.json"):
                    with open("model/metrics.json") as f:
                        metrics = json.load(f)
                return keras_model, tfidf, "keras", metrics
            except Exception as e:
                st.warning(f"No se pudo cargar el modelo Keras: {e}. Entrenando modelo de respaldo...")

    # ── opción 2: entrenar MLPClassifier (Red Neuronal sklearn) ───────────
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    tfidf.fit(X_tr)

    def build_X(texts):
        Xt = tfidf.transform(texts).toarray().astype(np.float32)
        Xe = engineer_features(texts.values)
        return np.hstack([Xt, Xe])

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=80,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        verbose=False,
    )
    mlp.fit(build_X(X_tr), y_tr)

    y_pred = mlp.predict(build_X(X_te))
    y_prob = mlp.predict_proba(build_X(X_te))[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_te, y_prob)),
        "model_type": "sklearn MLP (Red Neuronal Superficial)",
        "architecture": "256-128-64-1",
        "y_test":  y_te.tolist(),
        "y_pred":  y_pred.tolist(),
        "y_prob":  y_prob.tolist(),
    }
    return mlp, tfidf, "sklearn", metrics


def predict_proba(text: str, model, tfidf, model_type: str) -> float:
    Xt = tfidf.transform([text]).toarray().astype(np.float32)
    Xe = engineer_features([text])
    X  = np.hstack([Xt, Xe])
    if model_type == "keras":
        return float(model.predict(X, verbose=0)[0][0])
    return float(model.predict_proba(X)[0][1])


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTES VISUALES
# ══════════════════════════════════════════════════════════════════════════════
def gauge_chart(prob: float) -> go.Figure:
    is_spam = prob >= 0.5
    color   = "#ff4455" if is_spam else "#00ff88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#ff4455"}, "decreasing": {"color": "#00ff88"}},
        number={"suffix": "%", "font": {"size": 52, "color": color, "family": "sans-serif"}},
        title={"text": "Probabilidad SPAM", "font": {"size": 15, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#64748b", "size": 11}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(30,32,64,0.6)",
            "borderwidth": 1, "bordercolor": "rgba(0,212,255,0.2)",
            "steps": [
                {"range": [0,  30], "color": "rgba(0,255,136,0.07)"},
                {"range": [30, 70], "color": "rgba(255,140,66,0.05)"},
                {"range": [70,100], "color": "rgba(255,68,85,0.07)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.85, "value": 50},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8", "family": "sans-serif"},
        height=280, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def bar_features_chart(features_info: list) -> go.Figure:
    names   = [f[0] for f in features_info]
    vals    = [1 if f[1] else 0 for f in features_info]
    colors  = ["#ff4455" if v else "rgba(148,163,184,0.3)" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=["DETECTADO" if v else "no detectado" for v in vals],
        textposition="inside",
        textfont={"size": 11, "color": "white"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,32,64,0.4)",
        font={"color": "#94a3b8", "family": "sans-serif"},
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"visible": False, "range": [0, 1.2]},
        yaxis={"tickfont": {"size": 12, "color": "#e2e8f0"}},
    )
    return fig


def confusion_heatmap(cm: np.ndarray) -> go.Figure:
    labels = ["No Spam", "Spam"]
    z_text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=z_text, texttemplate="%{text}",
        textfont={"size": 22, "color": "white"},
        colorscale=[[0, "#1a1a35"], [0.5, "#004d80"], [1, "#00d4ff"]],
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8", "size": 13},
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        xaxis={"title": "Predicción", "titlefont": {"color": "#94a3b8"}},
        yaxis={"title": "Real",       "titlefont": {"color": "#94a3b8"}},
    )
    return fig


def roc_chart(y_test, y_prob, auc: float) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, fill="tozeroy",
        fillcolor="rgba(0,212,255,0.1)",
        line={"color": "#00d4ff", "width": 2.5},
        name=f"Modelo (AUC = {auc:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], line={"color": "#475569", "dash": "dash", "width": 1.5},
        name="Azar (AUC = 0.500)", showlegend=True,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,32,64,0.4)",
        font={"color": "#94a3b8"}, height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,212,255,0.2)", "borderwidth": 1},
        xaxis={"title": "Falsos Positivos", "gridcolor": "rgba(255,255,255,0.06)"},
        yaxis={"title": "Verdaderos Positivos", "gridcolor": "rgba(255,255,255,0.06)"},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(model_type: str, metrics: dict, df: pd.DataFrame):
    with st.sidebar:
        st.markdown("### 🤖 Spam Detector")
        st.markdown("---")

        tipo_lbl = "🧠 Keras (GPU)" if model_type == "keras" else "⚡ MLP sklearn"
        st.markdown(f"""
        <div class="stat-box">
          <div class="stat-lbl">Tipo de Modelo</div>
          <div class="stat-val" style="font-size:1.1rem">{tipo_lbl}</div>
        </div>
        <div class="stat-box">
          <div class="stat-lbl">Arquitectura</div>
          <div class="stat-val" style="font-size:1rem">256 → 128 → 64 → 1</div>
        </div>
        """, unsafe_allow_html=True)

        if metrics:
            acc = metrics.get("accuracy", 0)
            f1  = metrics.get("f1", 0)
            auc = metrics.get("auc_roc", 0)
            st.markdown(f"""
            <div class="stat-box"><div class="stat-lbl">Accuracy</div>
              <div class="stat-val">{acc:.1%}</div></div>
            <div class="stat-box"><div class="stat-lbl">F1-Score</div>
              <div class="stat-val">{f1:.3f}</div></div>
            <div class="stat-box"><div class="stat-lbl">AUC-ROC</div>
              <div class="stat-val">{auc:.3f}</div></div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        n_spam   = int(df["label"].sum())
        n_nospam = len(df) - n_spam
        st.markdown(f"""
        <div class="stat-box"><div class="stat-lbl">Total comentarios</div>
          <div class="stat-val">{len(df):,}</div></div>
        <div class="stat-box"><div class="stat-lbl">Spam / No Spam</div>
          <div class="stat-val" style="font-size:0.95rem">
            <span style="color:#ff4455">{n_spam:,}</span> /
            <span style="color:#00ff88">{n_nospam:,}</span>
          </div></div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#64748b; line-height:1.6;">
        <b style="color:#94a3b8">Datos de entrenamiento:</b><br>
        • Youtube-Spam-Dataset.csv<br>
        • YouTube Comments 45K<br><br>
        <b style="color:#94a3b8">UPV — Proyecto I: Intro IA</b><br>
        Ángel · Samuel · Artur · Pablo
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Carga modelo y datos ────────────────────────────────────────────────
    with st.spinner("🧠 Cargando modelo de red neuronal..."):
        df = load_data()
    with st.spinner("⚡ Inicializando red neuronal..."):
        model, tfidf, model_type, metrics = load_or_train_model()

    render_sidebar(model_type, metrics, df)

    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1 class="hero-title">YouTube Spam Detector</h1>
      <p class="hero-sub">Red Neuronal Superficial para detección de comentarios de spam</p>
      <span class="hero-badge">🎓 Proyecto IA · UPV · 2025</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs principales ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍  Analizar Comentario",
        "📊  Análisis Masivo (CSV)",
        "📈  Rendimiento del Modelo",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: ANÁLISIS INDIVIDUAL
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.markdown("#### ✍️ Introduce un comentario")
            comentario = st.text_area(
                label="",
                placeholder="Escribe o pega aquí un comentario de YouTube...",
                height=160,
                key="texto_input",
            )

            # Ejemplos rápidos
            st.markdown("<p style='font-size:0.82rem;margin-bottom:0.3rem;'>Ejemplos rápidos:</p>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚨 Ejemplo SPAM", use_container_width=True):
                    st.session_state["demo"] = np.random.choice(EJEMPLOS_SPAM)
            with c2:
                if st.button("✅ Ejemplo real", use_container_width=True):
                    st.session_state["demo"] = np.random.choice(EJEMPLOS_REAL)

            if "demo" in st.session_state:
                comentario = st.session_state.pop("demo")
                st.rerun()

            analizar = st.button("🔍 Analizar Comentario", use_container_width=True)

        with col_out:
            if analizar and comentario.strip():
                prob = predict_proba(comentario.strip(), model, tfidf, model_type)
                is_spam = prob >= 0.5

                # ── Veredicto ───────────────────────────────────────────
                clase_css = "spam" if is_spam else "safe"
                icono     = "🚨" if is_spam else "✅"
                label     = "SPAM" if is_spam else "NO ES SPAM"
                st.markdown(f"""
                <div class="verdict {clase_css}">
                  <span class="verdict-icon">{icono}</span>
                  <span class="verdict-label">{label}</span>
                  <span class="verdict-prob">Confianza: {prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Gauge ────────────────────────────────────────────────
                st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})

                # ── Indicadores detectados ───────────────────────────────
                feats = features_activadas(comentario.strip())
                activos = sum(1 for f in feats if f[1])
                st.markdown(f"**Indicadores de spam detectados: {activos}/{len(feats)}**")
                st.plotly_chart(bar_features_chart(feats), use_container_width=True, config={"displayModeBar": False})

                # Pills
                pills_html = '<div class="pills">' + "".join(
                    f'<span class="pill {"pill-on" if f[1] else "pill-off"}">{f[0]}</span>'
                    for f in feats
                ) + "</div>"
                st.markdown(pills_html, unsafe_allow_html=True)

            elif analizar:
                st.warning("⚠️ Por favor, introduce un comentario antes de analizar.")
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:3rem 1rem;">
                  <div style="font-size:3rem;margin-bottom:1rem;">👈</div>
                  <p>Escribe un comentario y pulsa <b>Analizar</b><br>para ver el resultado.</p>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: ANÁLISIS MASIVO
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 📤 Sube un archivo CSV para análisis masivo")
        st.markdown("""
        <p>El CSV debe contener una columna <code>CONTENT</code> o <code>comment_text</code>
        con los comentarios a analizar (columna <code>CLASS</code> opcional para validación).</p>
        """, unsafe_allow_html=True)

        archivo = st.file_uploader("Arrastra o selecciona tu CSV", type=["csv"])

        if archivo:
            try:
                df_up = pd.read_csv(archivo)
                col_txt = next((c for c in df_up.columns if c.upper() in ["CONTENT", "COMMENT_TEXT", "TEXT"]), None)
                if col_txt is None:
                    st.error("❌ No se encontró una columna 'CONTENT', 'comment_text' o 'text'.")
                else:
                    df_up["text"] = df_up[col_txt].astype(str).str.strip()
                    df_up = df_up[df_up["text"].str.len() > 2].reset_index(drop=True)
                    st.info(f"📋 {len(df_up):,} comentarios cargados. Analizando...")

                    barra = st.progress(0, text="Procesando...")
                    probs = []
                    BATCH = 256
                    for i in range(0, len(df_up), BATCH):
                        batch = df_up["text"].iloc[i:i+BATCH].tolist()
                        Xt = tfidf.transform(batch).toarray().astype(np.float32)
                        Xe = engineer_features(batch)
                        X  = np.hstack([Xt, Xe])
                        if model_type == "keras":
                            pb = model.predict(X, verbose=0).flatten().tolist()
                        else:
                            pb = model.predict_proba(X)[:, 1].tolist()
                        probs.extend(pb)
                        barra.progress(min((i + BATCH) / len(df_up), 1.0), text=f"Procesando {min(i+BATCH, len(df_up))}/{len(df_up)}...")

                    df_up["spam_prob"]  = probs
                    df_up["prediccion"] = (df_up["spam_prob"] >= 0.5).map({True: "🚨 SPAM", False: "✅ No Spam"})
                    barra.empty()

                    n_spam = (df_up["spam_prob"] >= 0.5).sum()
                    n_real = len(df_up) - n_spam

                    # KPIs
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total comentarios", f"{len(df_up):,}")
                    k2.metric("🚨 Spam detectado",  f"{n_spam:,}")
                    k3.metric("✅ No spam",          f"{n_real:,}")
                    k4.metric("% Spam",              f"{n_spam/len(df_up):.1%}")

                    c_pie, c_dist = st.columns(2)
                    with c_pie:
                        fig_pie = go.Figure(go.Pie(
                            labels=["🚨 SPAM", "✅ No Spam"],
                            values=[n_spam, n_real],
                            marker_colors=["#ff4455", "#00ff88"],
                            hole=0.5,
                            textfont={"color": "white", "size": 13},
                        ))
                        fig_pie.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", height=260,
                            margin=dict(l=10,r=10,t=30,b=10),
                            legend={"font": {"color": "#94a3b8"}},
                            title={"text": "Distribución", "font": {"color": "white", "size": 13}},
                        )
                        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                    with c_dist:
                        fig_hist = go.Figure(go.Histogram(
                            x=df_up["spam_prob"],
                            nbinsx=40,
                            marker_color="#00d4ff",
                            opacity=0.8,
                        ))
                        fig_hist.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,32,64,0.4)",
                            height=260, margin=dict(l=10,r=10,t=30,b=10),
                            xaxis={"title": "Probabilidad spam", "gridcolor": "rgba(255,255,255,0.05)"},
                            yaxis={"title": "Frecuencia", "gridcolor": "rgba(255,255,255,0.05)"},
                            title={"text": "Distribución de probabilidades", "font": {"color": "white", "size": 13}},
                            font={"color": "#94a3b8"},
                            shapes=[{"type": "line", "x0": 0.5, "x1": 0.5, "y0": 0, "y1": 1,
                                     "yref": "paper", "line": {"color": "#ff4455", "width": 2, "dash": "dash"}}],
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

                    # Tabla de resultados
                    cols_show = ["text", "spam_prob", "prediccion"]
                    if "CLASS" in df_up.columns or "label" in df_up.columns:
                        col_label = "CLASS" if "CLASS" in df_up.columns else "label"
                        cols_show.insert(0, col_label)
                    st.dataframe(
                        df_up[cols_show].rename(columns={"text": "Comentario", "spam_prob": "P(spam)", "prediccion": "Predicción"}),
                        use_container_width=True, height=300,
                    )

                    csv_out = df_up[cols_show].to_csv(index=False)
                    st.download_button("⬇️ Descargar resultados CSV", csv_out, "resultados_spam.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: MÉTRICAS DEL MODELO
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 📈 Rendimiento del modelo en el conjunto de test")

        if not metrics:
            st.info("ℹ️ Métricas no disponibles para el modelo Keras cargado.")
        else:
            acc  = metrics.get("accuracy",  0)
            prec = metrics.get("precision", 0)
            rec  = metrics.get("recall",    0)
            f1   = metrics.get("f1",        0)
            auc  = metrics.get("auc_roc",   0)

            # KPIs
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{acc:.2%}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall",    f"{rec:.3f}")
            m4.metric("F1-Score",  f"{f1:.3f}")
            m5.metric("AUC-ROC",   f"{auc:.3f}")

            st.markdown("---")
            c_cm, c_roc = st.columns(2, gap="large")

            y_test_lst = metrics.get("y_test", [])
            y_pred_lst = metrics.get("y_pred", [])
            y_prob_lst = metrics.get("y_prob", [])

            if y_test_lst:
                y_test_arr = np.array(y_test_lst)
                y_pred_arr = np.array(y_pred_lst)
                y_prob_arr = np.array(y_prob_lst)

                with c_cm:
                    st.markdown("**Matriz de Confusión**")
                    cm = confusion_matrix(y_test_arr, y_pred_arr)
                    st.plotly_chart(confusion_heatmap(cm), use_container_width=True, config={"displayModeBar": False})
                    tn, fp, fn, tp = cm.ravel()
                    c1, c2 = st.columns(2)
                    c1.metric("Verdaderos Positivos (TP)", f"{tp:,}")
                    c1.metric("Verdaderos Negativos (TN)", f"{tn:,}")
                    c2.metric("Falsos Positivos (FP)", f"{fp:,}")
                    c2.metric("Falsos Negativos (FN)", f"{fn:,}")

                with c_roc:
                    st.markdown("**Curva ROC**")
                    st.plotly_chart(roc_chart(y_test_arr, y_prob_arr, auc), use_container_width=True, config={"displayModeBar": False})

                st.markdown("---")
                st.markdown("**Classification Report**")
                report = classification_report(y_test_arr, y_pred_arr, target_names=["No Spam", "Spam"], output_dict=True)
                df_report = pd.DataFrame(report).T.round(4)
                st.dataframe(df_report.style.highlight_max(subset=["precision","recall","f1-score"], color="#0d2d3a"), use_container_width=True)

        st.markdown("---")
        st.markdown("**Información del Modelo**")
        arch_col, info_col = st.columns(2)
        with arch_col:
            st.markdown("""
            <div class="card">
              <b style="color:#00d4ff">Arquitectura Red Neuronal</b><br><br>
              <code style="color:#a855f7">Input (5 005 features)</code><br>
              &emsp;↓<br>
              <code style="color:#00d4ff">Dense(256, ReLU) + BN + Dropout(0.3)</code><br>
              &emsp;↓<br>
              <code style="color:#00d4ff">Dense(128, ReLU) + BN + Dropout(0.2)</code><br>
              &emsp;↓<br>
              <code style="color:#00d4ff">Dense(64, ReLU)</code><br>
              &emsp;↓<br>
              <code style="color:#00ff88">Dense(1, Sigmoid)</code>
            </div>
            """, unsafe_allow_html=True)
        with info_col:
            st.markdown(f"""
            <div class="card">
              <b style="color:#00d4ff">Features de Entrada</b><br><br>
              <span style="color:#e2e8f0">• TF-IDF bigramas:</span>
              <span style="color:#a855f7">5 000 features</span><br>
              <span style="color:#e2e8f0">• Longitud en caracteres:</span>
              <span style="color:#a855f7">1 feature</span><br>
              <span style="color:#e2e8f0">• Ratio de mayúsculas:</span>
              <span style="color:#a855f7">1 feature</span><br>
              <span style="color:#e2e8f0">• Presencia de URL:</span>
              <span style="color:#a855f7">1 feature</span><br>
              <span style="color:#e2e8f0">• Log(exclamaciones):</span>
              <span style="color:#a855f7">1 feature</span><br>
              <span style="color:#e2e8f0">• Palabras clave spam:</span>
              <span style="color:#a855f7">1 feature</span><br><br>
              <b>Total: <span style="color:#00d4ff">5 005 dimensiones</span></b>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
