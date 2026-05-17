"""
IA YouTube Spam Detector — v9.0  "Híbrida: ML Clásico + RoBERTa Deep Learning"

Fuentes de datos
════════════════════════════════════════════════════════════════════
SPAM (modelo de clasificación)
  1. Youtube-Spam-Dataset.csv        1 956 filas  CONTENT / CLASS (0/1)
  2. YouTube Comments Dataset …csv  45 005 filas  comment_text / label_spam
  ──────────────────────────────────────────────
  Total combinado: ~47 000 filas  │  spam:1 821  real:45 135
  Modelo: LR + TF-IDF word bigramas + char (2,5)-grams + 9 features EDA

SENTIMIENTO (modelo de clasificación)
  3. YouTube Comments Dataset …csv  45 005 filas  comment_text / label_sentiment
  ──────────────────────────────────────────────
  Total: ~45 000 filas  │  Positive / Neutral / Negative
  Modelo base: Regresión Logística multiclase (Fallback)
  Modelo Premium: Twitter-RoBERTa Fine-Tuned en HuggingFace Hub
"""

import hashlib
import csv
from datetime import datetime

# ── Modelo BERT fine-tuned (INTEGRADO REPO AD9394I) ──────────────────────────
# Tu modelo entrenado con el dataset de 30k balanceado
HF_SENTIMENT_MODEL = "ad9394i/youtube-sentiment-roberta"  

# Importar transformers solo si hay modelo configurado (evita peso en arranque)
_BERT_AVAILABLE = False
if HF_SENTIMENT_MODEL:
    try:
        from transformers import pipeline as hf_pipeline
        _BERT_AVAILABLE = True
    except ImportError:
        pass
import os
import re
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from googleapiclient.discovery import build          # pip install google-api-python-client
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS, WordCloud


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Spam Detector", page_icon="🎬", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 26px; color: #ff4b4b; }
.stMetric { background:#fff; padding:14px; border-radius:10px; border:1px solid #eee; }
.rgpd-box { background:#f0f4ff; border-left:4px solid #3b82f6;
            padding:10px 14px; border-radius:6px; font-size:.83rem; margin-bottom:1rem; }
.warn-box  { background:#fff7ed; border-left:4px solid #f59e0b;
             padding:10px 14px; border-radius:6px; font-size:.83rem; margin-bottom:1rem; }
.good-box  { background:#f0fdf4; border-left:4px solid #22c55e;
             padding:10px 14px; border-radius:6px; font-size:.83rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# REGEX Y LÉXICO
# ─────────────────────────────────────────────────────────────────
URL_RE        = re.compile(r"https?://\S+|bit\.ly\S*|www\.\S+", re.I)
EXCL_RE       = re.compile(r"!{2,}")
CAPS_WORD_RE  = re.compile(r"\b[A-Z]{4,}\b")
EMOJI_RE      = re.compile(r"[^\x00-\x7F]")
REPEAT_RE     = re.compile(r"(\b\w+\b)(?:\s+\1){2,}", re.I)
TIMESTAMP_RE  = re.compile(r"\b[A-Z\d]{1,2}:\d{2}\b")
MENTION_RE    = re.compile(r"@\w+")
REPEAT_CHR_RE = re.compile(r"(.)\1{3,}")

SPAM_LEXICON = {
    "subscribe", "suscribete", "suscríbete", "free", "gratis", "click",
    "win", "gana", "money", "dinero", "cash", "giveaway", "sorteo",
    "check out", "visit", "my channel", "mi canal", "promo", "discount",
    "descuento", "link in bio", "crypto", "bitcoin", "investment",
    "earn", "profit", "dm me", "escríbeme",
    "followers", "views", "likes", "watch now",
    "link in description", "check my", "visit my", "go to my",
}
CALL_TO_ACTION = ["click here", "click now", "tap here", "visit", "check out", "watch now"]
EMOJI_UNICODE_RE = re.compile(r"[🌀-🿿]")


# ─────────────────────────────────────────────────────────────────
# FEATURES MANUALES PARA SPAM — 14 señales (v8.5, antes 9)
# ─────────────────────────────────────────────────────────────────
class SpamFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        return np.array([self._f(t) for t in X], dtype=float)

    def _f(self, text: str) -> list:
        t  = str(text)
        tl = t.lower()
        ws = tl.split()
        nc = max(len(t), 1)
        nw = max(len(ws), 1)
        return [
            len(URL_RE.findall(t)),                                     # nº URLs
            sum(c.isupper() for c in t) / nc,                           # ratio mayúsculas
            t.count("!"),                                               # exclamaciones
            len(EXCL_RE.findall(t)),                                    # grupos "!!+"
            len(CAPS_WORD_RE.findall(t)),                               # palabras CAPS
            len(set(ws)) / nw,                                          # diversidad léxica
            sum(1 for w in SPAM_LEXICON if w in tl),                   # hits léxico spam
            len(REPEAT_RE.findall(tl)),                                 # palabras repetidas
            np.log1p(float(len(t))),                                   # longitud log
            sum(c.isdigit() for c in t) / nc,                          # ratio dígitos
            sum(not c.isalnum() and not c.isspace() for c in t) / nc, # ratio especiales
            len(EMOJI_UNICODE_RE.findall(t)),                          # nº emojis unicode
            int(any(p in tl for p in CALL_TO_ACTION)),                # call to action
            np.mean([len(w) for w in ws]) if ws else 0,               # longitud media palabra
        ]


# ─────────────────────────────────────────────────────────────────
# FEATURES MANUALES PARA SENTIMIENTO — 15 señales (v8.6)
# ─────────────────────────────────────────────────────────────────
NEGATION_WORDS = {
    "not","no","never","nothing","nobody","nowhere","neither","nor",
    "cannot","can't","won't","wouldn't","don't","doesn't","didn't",
    "isn't","aren't","wasn't","weren't","without","hardly","barely","scarcely",
}
INTENSIFIERS = {
    "very","really","extremely","absolutely","totally","completely",
    "so","too","such","quite","incredibly","insanely","super","highly",
}
POS_LEXICON = {
    "love","amazing","great","awesome","excellent","perfect","best",
    "wonderful","fantastic","good","beautiful","brilliant","outstanding",
    "enjoy","happy","pleased","incredible","superb","impressive","loved",
    "favourite","favorite","nice","brilliant","delightful","charming",
}
NEG_LEXICON = {
    "hate","terrible","awful","horrible","worst","bad","disgusting",
    "boring","disappointing","useless","pathetic","stupid","waste",
    "annoying","frustrating","broken","garbage","trash","ugly","dreadful",
    "disgusted","rubbish","appalling","dull","mediocre","overrated",
}
_POS_EMOTICON = re.compile(r"[:;=]-?[)\]DPpB]|<3|:\*|:-?\)")
_NEG_EMOTICON = re.compile(r"[:;=]-?[(\[/\\|Cc]|>:|>\.>")
_REPEAT_CHR   = re.compile(r"(.)\1{2,}")
_CAPS_WORD    = re.compile(r"\b[A-Z]{3,}\b")
_EMOJI_UNI    = re.compile("[\U0001F300-\U0001FFFF]")

class SentimentFeatures(BaseEstimator, TransformerMixin):
    """15 señales handcrafted de sentimiento."""
    def fit(self, X, y=None): return self
    def transform(self, X): return np.array([self._f(t) for t in X], dtype=float)
    def _f(self, text: str) -> list:
        t = str(text); tl = t.lower(); ws = tl.split()
        nc = max(len(t), 1); nw = max(len(ws), 1)
        n_pos      = sum(1 for w in POS_LEXICON if w in tl)
        n_neg      = sum(1 for w in NEG_LEXICON if w in tl)
        n_neg_w    = sum(1 for w in ws if w in NEGATION_WORDS)
        n_intens   = sum(1 for w in ws if w in INTENSIFIERS)
        n_excl     = t.count("!")
        n_quest    = t.count("?")
        n_ellip    = t.count("...")
        caps_ratio = sum(c.isupper() for c in t) / nc
        n_caps_w   = len(_CAPS_WORD.findall(t))
        n_emoji    = len(_EMOJI_UNI.findall(t))
        n_pos_emo  = len(_POS_EMOTICON.findall(t))
        n_neg_emo  = len(_NEG_EMOTICON.findall(t))
        n_repeat   = len(_REPEAT_CHR.findall(t))
        log_len    = np.log1p(len(t))
        polarity   = (n_pos - n_neg) / nw
        return [n_pos, n_neg, n_neg_w, n_intens, n_excl, n_quest, n_ellip,
                caps_ratio, n_caps_w, n_emoji, n_pos_emo, n_neg_emo,
                n_repeat, log_len, polarity]


# ─────────────────────────────────────────────────────────────────
# RGPD — SEUDONIMIZACIÓN (Art. 25)
# ─────────────────────────────────────────────────────────────────
def seudonimizar(nombre: str) -> str:
    """SHA-256 del nombre real → 'Usr-XXXXXXXX'. Irreversible sin la clave."""
    d = hashlib.sha256(str(nombre).encode("utf-8")).hexdigest()[:8].upper()
    return f"Usr-{d}"


# ─────────────────────────────────────────────────────────────────
# CARGA Y COMBINACIÓN DE LOS DATASETS
# ─────────────────────────────────────────────────────────────────
def _resolver_ruta(*candidatos: str) -> str | None:
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None

RUTAS = {
    "spam_clasico": _resolver_ruta("Youtube-Spam-Dataset.csv", "data/Youtube-Spam-Dataset.csv"),
    "spam_equilibrado": _resolver_ruta("Youtube-Spam-Dataset_equilibrado_csv.csv", "Youtube-Spam-Dataset_equilibrado.csv", "data/Youtube-Spam-Dataset_equilibrado_csv.csv"),
    "spam_45k": _resolver_ruta("YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv", "YouTube_Comments_Dataset_with_Sentiment_Toxicity_and_Spam_Labels__45K_Rows_.csv", "data/YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv", "data/YouTube_Comments_Dataset_with_Sentiment_Toxicity_and_Spam_Labels__45K_Rows_.csv"),
    "export_anterior": _resolver_ruta("2026-05-13T17-45_export.csv"),
}

@st.cache_data(show_spinner=False)
def cargar_datos_spam(ratio_real_spam: int = 1) -> pd.DataFrame:
    partes = []
    if RUTAS["spam_clasico"]:
        df = pd.read_csv(RUTAS["spam_clasico"])[["CONTENT", "CLASS"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = df["spam"].astype(int)
        partes.append(df)

    if RUTAS["spam_equilibrado"]:
        df = pd.read_csv(RUTAS["spam_equilibrado"])[["CONTENT", "CLASS"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = df["spam"].astype(int)
        partes.append(df)

    if RUTAS["spam_45k"]:
        df = pd.read_csv(RUTAS["spam_45k"])[["comment_text", "label_spam"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = (df["spam"].str.strip().str.lower() == "spam").astype(int)
        partes.append(df)

    if RUTAS["export_anterior"]:
        df = pd.read_csv(RUTAS["export_anterior"])
        if "Comentario" in df.columns and "Spam" in df.columns:
            df = df[["Comentario", "Spam"]].dropna()
            df.columns = ["text", "spam"]
            df["spam"] = df["spam"].str.contains("SÍ|1|spam", case=False, na=False).astype(int)
            partes.append(df)

    if not partes:
        raise FileNotFoundError("No se encontró ningún dataset de spam. Coloca al menos uno de los CSV en la carpeta.")

    combined = pd.concat(partes, ignore_index=True).dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]
    combined = combined.drop_duplicates(subset=["text"])

    spam_df = combined[combined["spam"] == 1]
    real_df  = combined[combined["spam"] == 0]
    n_spam    = len(spam_df)
    n_real_target = min(n_spam * ratio_real_spam, len(real_df))

    real_df_sampled = real_df.sample(n=n_real_target, random_state=42)
    balanced = pd.concat([spam_df, real_df_sampled], ignore_index=True).sample(frac=1, random_state=42)
    return balanced


@st.cache_data(show_spinner=False)
def cargar_datos_sentimiento(ratio_por_clase: int = 1) -> pd.DataFrame:
    partes = []
    if RUTAS["spam_45k"]:
        df = pd.read_csv(RUTAS["spam_45k"], usecols=["comment_text", "label_sentiment"])
        df.columns = ["text", "sentiment"]
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        df = df[df["sentiment"].isin(["positive", "neutral", "negative"])].dropna()
        partes.append(df)

    if not partes:
        raise FileNotFoundError("No se encontró el dataset de sentimiento (45K Rows).")

    combined = pd.concat(partes, ignore_index=True).dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]

    neg_df = combined[combined["sentiment"] == "negative"]
    pos_df = combined[combined["sentiment"] == "positive"]
    neu_df = combined[combined["sentiment"] == "neutral"]

    n_neg = len(neg_df)
    n_pos_target = min(n_neg * ratio_por_clase, len(pos_df))
    n_neu_target = min(n_neg * ratio_por_clase, len(neu_df))

    balanced = pd.concat([
        neg_df,
        pos_df.sample(n=n_pos_target, random_state=42),
        neu_df.sample(n=n_neu_target, random_state=42),
    ], ignore_index=True).sample(frac=1, random_state=42)

    return balanced


# ─────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — SPAM (LR + word + char n-grams + handcrafted)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrenar_spam(df: pd.DataFrame):
    X = df["text"].tolist()
    y = df["spam"].tolist()

    def _feat_union():
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=8000, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=15000, sublinear_tf=True, strip_accents="unicode")),
            ("hc", SpamFeatures()),
        ])

    lr_pipe = Pipeline([
        ("f", _feat_union()), ("s", MaxAbsScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, solver="saga", class_weight="balanced", random_state=42)),
    ])
    svc_pipe = Pipeline([
        ("f", _feat_union()), ("s", MaxAbsScaler()),
        ("clf", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", max_iter=3000, random_state=42), cv=3)),
    ])
    pipeline = VotingClassifier([("lr", lr_pipe), ("svc", svc_pipe)], voting="soft", weights=[2, 1])

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)

    metricas = {
        "accuracy":  accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall":    recall_score(y_val, y_pred, zero_division=0),
        "f1":        f1_score(y_val, y_pred, zero_division=0),
        "reporte":   classification_report(y_val, y_pred, target_names=["Real", "Spam"], zero_division=0),
        "cm":        confusion_matrix(y_val, y_pred),
        "y_val":     y_val,
        "y_pred":    y_pred,
        "n_train":   len(X_tr),
        "n_spam":    int(sum(y)),
        "n_real":    int(len(y) - sum(y)),
    }
    return pipeline, metricas


# ─────────────────────────────────────────────────────────────────
# INFERENCIA BERT (RoBERTa Conectado)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cargar_bert_sentimiento():
    if not _BERT_AVAILABLE or not HF_SENTIMENT_MODEL:
        return None
    try:
        return hf_pipeline(
            "text-classification",
            model=HF_SENTIMENT_MODEL,
            truncation=True,
            max_length=128,
            device=-1,
        )
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo BERT: {e}. Usando fallback Sklearn.")
        return None


def predecir_sentimiento(texto: str, sent_pipe_sklearn) -> tuple[str, float]:
    bert = _cargar_bert_sentimiento()
    if bert is not None:
        try:
            res = bert(texto[:512])[0]
            label = res["label"].lower()
            return label, round(res["score"] * 100, 1)
        except Exception:
            pass  # Fallback
    pred  = sent_pipe_sklearn.predict([texto])[0]
    proba = sent_pipe_sklearn.predict_proba([texto])[0].max()
    return pred, round(float(proba) * 100, 1)


# ─────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — SENTIMIENTO (LR multiclase clásico)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrenar_sentimiento(df: pd.DataFrame):
    X = df["text"].tolist()
    y = df["sentiment"].tolist()

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, sublinear_tf=True, min_df=2, max_df=0.95, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20_000, sublinear_tf=True, min_df=3, strip_accents="unicode")),
            ("sf", SentimentFeatures()),
        ])),
        ("scaler", MaxAbsScaler()),
        ("clf", LogisticRegression(C=0.3, max_iter=1000, solver="saga", class_weight="balanced", random_state=42)),
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)

    metricas = {
        "accuracy":    accuracy_score(y_val, y_pred),
        "f1_macro":    f1_score(y_val, y_pred, average="macro", zero_division=0),
        "f1_negative": f1_score(y_val, y_pred, labels=["negative"], average="macro", zero_division=0),
        "reporte":     classification_report(y_val, y_pred, zero_division=0),
        "cm":          confusion_matrix(y_val, y_pred, labels=["positive", "neutral", "negative"]),
        "n_train":     len(X_tr),
    }
    return pipeline, metricas


# ─────────────────────────────────────────────────────────────────
# ANÁLISIS DE TEMAS (NMF sobre TF-IDF)
# ─────────────────────────────────────────────────────────────────
_EXTRA_STOP = {
    "video","channel","like","just","know","good","really","get","one",
    "make","people","think","want","watch","see","time","go","come",
    "youtube","comment","subscribe","share","pleas","please","thank","thanks",
    "hi","hello","hey","great","nice","love","best","sir","also","even",
    "will","can","don","doesn","didn","isn","aren","wasn","weren",
    "say","said","need","got","going","still","way","thing","things",
}
_CLEAN_RE = re.compile(r"https?://\S+|www\.\S+|[^a-z\s']")

def _limpiar_texto_temas(texto: str) -> str:
    t = str(texto).lower()
    t = _CLEAN_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

@st.cache_data(show_spinner=False, ttl=3600)
def extraer_temas(comentarios: list[str], n_temas: int = 4, n_palabras: int = 8, ngram_max: int = 2) -> list[dict]:
    if len(comentarios) < 20: return []
    limpios = [_limpiar_texto_temas(c) for c in comentarios]
    limpios = [t for t in limpios if len(t.split()) >= 3]
    if len(limpios) < 20: return []

    vec = TfidfVectorizer(max_df=0.80, min_df=max(3, len(limpios)//200), max_features=2500, stop_words="english", ngram_range=(1, ngram_max), token_pattern=r"(?u)\b[a-z]{3,}\b")
    try:
        dtm = vec.fit_transform(limpios)
    except ValueError:
        return []

    words = vec.get_feature_names_out()
    keep_idx = [i for i, w in enumerate(words) if not any(s == w or w.startswith(s + " ") or w.endswith(" " + s) for s in _EXTRA_STOP)]
    if len(keep_idx) < n_palabras * 2: keep_idx = list(range(len(words)))

    dtm_f = dtm[:, keep_idx]
    words_f = words[keep_idx]
    n_comp = min(n_temas, dtm_f.shape[1], dtm_f.shape[0] - 1)
    if n_comp < 1: return []

    nmf = NMF(n_components=n_comp, random_state=42, max_iter=300, init="nndsvda", l1_ratio=0.1)
    W = nmf.fit_transform(dtm_f)

    temas = []
    for i, comp in enumerate(nmf.components_):
        top_idx  = comp.argsort()[-n_palabras:][::-1]
        palabras = [words_f[j] for j in top_idx]
        pesos    = [float(comp[j]) for j in top_idx]
        peso_doc = float(W[:, i].mean())
        temas.append({"id": i + 1, "palabras": palabras, "pesos": pesos, "relevancia": peso_doc})

    temas.sort(key=lambda x: x["relevancia"], reverse=True)
    return temas

def _color_tema(i: int) -> str:
    paleta = ["#534AB7","#1D9E75","#D85A30","#378ADD","#D4537E"]
    return paleta[i % len(paleta)]

def mostrar_analisis_temas(df_res: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    st.subheader("🧩 Análisis de temas por sentimiento")
    st.caption("Algoritmo NMF sobre TF-IDF bigramas — extrae los temas latentes.")

    df_clean = df_res[df_res.get("Spam", pd.Series(["No"]*len(df_res))) != "SÍ"].copy()
    if df_clean.empty or "Sentimiento" not in df_clean.columns:
        st.info("No hay suficientes comentarios analizados para extraer temas.")
        return

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    n_temas   = col_ctrl1.slider("Número de temas", 2, 6, 4, key="sl_ntemas_global")
    n_palabras= col_ctrl2.slider("Palabras por tema", 5, 12, 8, key="sl_npal_global")
    solo_ing  = col_ctrl3.checkbox("Solo comentarios en inglés", value=False)

    if solo_ing and "Idioma" in df_clean.columns:
        df_clean = df_clean[df_clean["Idioma"].str.lower().str.startswith("en", na=False)]

    sentimientos = ["Positive", "Negative", "Neutral"]
    emoji_sent   = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
    tabs = st.tabs([f"{emoji_sent[s]} {s}" for s in sentimientos])

    for tab, sent in zip(tabs, sentimientos):
        with tab:
            subset = df_clean[df_clean["Sentimiento"] == sent]["Comentario"].tolist()
            if len(subset) < 20:
                st.info(f"Solo {len(subset)} comentarios — mínimo 20 para extraer temas.")
                continue

            with st.spinner("Extrayendo temas…"):
                temas = extraer_temas(tuple(subset), n_temas=n_temas, n_palabras=n_palabras)

            if not temas:
                st.warning("No se pudieron extraer temas significativos.")
                continue

            st.markdown(f"**{len(subset):,}** comentarios analizados · **{len(temas)}** temas encontrados")
            st.divider()

            for tema in temas:
                i = tema["id"] - 1; color = _color_tema(i)
                st.markdown(f'<span style="display:inline-block;background:{color}18;color:{color};border:1px solid {color}40;border-radius:6px;padding:3px 10px;font-size:13px;font-weight:500;">Tema {tema["id"]}</span><span style="color:var(--color-text-secondary);font-size:12px;margin-left:8px;">relevancia media {tema["relevancia"]:.4f}</span>', unsafe_allow_html=True)
                
                fig = go.Figure(go.Bar(x=tema["pesos"][::-1], y=tema["palabras"][::-1], orientation="h", marker_color=color, marker_opacity=0.85, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
                fig.update_layout(height=max(180, len(tema["palabras"]) * 28), margin=dict(l=0, r=20, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, showticklabels=False, zeroline=False), yaxis=dict(showgrid=False, tickfont=dict(size=13)), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"tema_{sent}_{i}")

                pills_html = " ".join(f'<span style="background:{color}18;color:{color};border:1px solid {color}40;border-radius:20px;padding:2px 10px;font-size:12px;margin:2px;display:inline-block;">{p}</span>' for p in tema["palabras"])
                st.markdown(pills_html, unsafe_allow_html=True)
                st.markdown("---")


# ─────────────────────────────────────────────────────────────────
# DETECCIÓN BOT EN BATCH Y REGLAS DURAS
# ─────────────────────────────────────────────────────────────────
def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detectar_bots(comentarios: list[dict]) -> dict[int, bool]:
    resultado = {i: False for i in range(len(comentarios))}
    por_autor = {}
    for i, c in enumerate(comentarios):
        por_autor.setdefault(c.get("seudónimo", ""), []).append(i)
    for _, idx in por_autor.items():
        if len(idx) < 2: continue
        textos = [comentarios[i]["texto"] for i in idx]
        total = similares = 0
        for ia in range(len(textos)):
            for ib in range(ia + 1, len(textos)):
                total += 1
                if _sim(textos[ia], textos[ib]) >= 0.80: similares += 1
        if total and (similares / total) >= 0.60:
            for i in idx: resultado[i] = True
    return resultado

def reglas_duras(texto: str):
    t, tl = str(texto), str(texto).lower(); nc = max(len(t), 1)
    if URL_RE.search(t): return True, 95.0
    if sum(1 for w in SPAM_LEXICON if w in tl) >= 3: return True, 90.0
    if REPEAT_RE.search(tl): return True, 85.0
    if sum(c.isupper() for c in t) / nc > 0.70 and len(t) > 15: return True, 80.0
    if t.count("!") >= 5: return True, 78.0
    if len(t.split()) <= 2: return False, 80.0
    return None, None

def preprocesar(texto: str) -> str:
    t = URL_RE.sub(" ", str(texto))
    t = TIMESTAMP_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = REPEAT_CHR_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────
# ANÁLISIS DE UN COMENTARIO
# ─────────────────────────────────────────────────────────────────
def analizar(texto: str, spam_pipe, sent_pipe, batch_spam: bool = False) -> dict:
    texto = str(texto or "").strip()
    if not texto: return {"spam": 0, "spam_conf": 0.0, "sentimiento": "Neutral", "sent_conf": 50.0, "motivo": ""}

    if batch_spam:
        spam, spam_conf, motivo = 1, 97.0, "bot (repetición)"
    else:
        rd, rd_c = reglas_duras(texto)
        if rd is not None:
            spam, spam_conf, motivo = int(rd), rd_c, "regla" if rd else ""
        else:
            probas = spam_pipe.predict_proba([texto])[0]
            clases = spam_pipe.named_steps["clf"].classes_
            idx_spam = int(np.where(clases == 1)[0][0]) if 1 in clases else 1
            spam = int(spam_pipe.predict([texto])[0])
            spam_conf = float(probas[idx_spam]) * 100
            motivo = "LR" if spam else ""

    limpio = preprocesar(texto)
    texto_sent = limpio if limpio else texto
    sent_raw, sent_conf = predecir_sentimiento(texto_sent, sent_pipe)
    label_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    sentimiento = label_map.get(sent_raw, sent_raw.capitalize())

    return {"spam": spam, "spam_conf": spam_conf, "sentimiento": sentimiento, "sent_conf": sent_conf, "motivo": motivo}


# ─────────────────────────────────────────────────────────────────
# YOUTUBE DATA API V3 & CSV
# ─────────────────────────────────────────────────────────────────
def extraer_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

def descargar_comentarios(api_key: str, video_id: str, limite: int) -> list[dict]:
    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    comentarios = []; page_token = None
    while len(comentarios) < limite:
        por_pagina = min(100, limite - len(comentarios))
        kw = dict(part="snippet", videoId=video_id, maxResults=por_pagina, order="time", textFormat="plainText")
        if page_token: kw["pageToken"] = page_token
        resp = yt.commentThreads().list(**kw).execute()
        for item in resp.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            texto = snip.get("textDisplay", "").strip()
            if texto: comentarios.append({"seudónimo": seudonimizar(snip.get("authorDisplayName", "")), "texto": texto})
        page_token = resp.get("nextPageToken")
        if not page_token: break
    return comentarios[:limite]

def leer_csv_usuario(upload) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        return None, str(e)
    cols = {c.lower(): c for c in df.columns}
    col = next((cols[k] for k in ("comentario", "content", "text", "comment") if k in cols), None)
    if col is None: return None, f"Columnas no reconocidas: {list(df.columns)}"
    df = df.dropna(subset=[col]).copy()
    df["texto"] = df[col].astype(str).str.strip()
    df = df[df["texto"] != ""]
    a = next((cols[k] for k in ("autor", "author") if k in cols), None)
    df["seudónimo"] = df[a].apply(seudonimizar) if a else [f"Usr-{i:04d}" for i in range(len(df))]
    return df, None

def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title); plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# FEEDBACK HUMANO COMPLETO RESTAURADO
# ─────────────────────────────────────────────────────────────────
def _feedback_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), FEEDBACK_CSV)

def guardar_feedback(texto: str, pred_spam: str, pred_sent: str, correcto_spam: str, correcto_sent: str, nota: str = "") -> None:
    path = _feedback_path(); nuevo = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FEEDBACK_COLS)
        if nuevo: w.writeheader()
        w.writerow({
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "texto_hash": hashlib.sha256(texto.encode()).hexdigest()[:12],
            "texto": texto[:500], "pred_spam": pred_spam, "pred_sentimiento": pred_sent,
            "correcto_spam": correcto_spam, "correcto_sentimiento": correcto_sent, "nota": nota[:200]
        })

def leer_feedback() -> pd.DataFrame:
    path = _feedback_path()
    if not os.path.exists(path): return pd.DataFrame(columns=_FEEDBACK_COLS)
    try: return pd.read_csv(path, encoding="utf-8")
    except Exception: return pd.DataFrame(columns=_FEEDBACK_COLS)

def _feedback_key(idx: int, campo: str) -> str:
    return f"fb_{campo}_{idx}"

def widget_feedback_fila(idx: int, texto: str, pred_spam: str, pred_sent: str) -> None:
    opciones_spam = ["— sin cambio —", "✅ NO es spam", "🚨 SÍ es spam"]
    opciones_sent = ["— sin cambio —", "Positive", "Neutral", "Negative"]
    c1, c2 = st.columns(2)
    spam_correc = c1.selectbox("¿Spam correcto?", opciones_spam, key=_feedback_key(idx, "spam"))
    sent_correc = c2.selectbox("¿Sentimiento correcto?", opciones_sent, key=_feedback_key(idx, "sent"))
    nota = st.text_input("Nota opcional", key=_feedback_key(idx, "nota"), placeholder="sarcasmo, modismo...")
    if st.button("💾 Guardar corrección", key=_feedback_key(idx, "btn")):
        cs = spam_correc if spam_correc != "— sin cambio —" else pred_spam
        cv = sent_correc if sent_correc != "— sin cambio —" else pred_sent
        guardar_feedback(texto, pred_spam, pred_sent, cs, cv, nota)
        st.success("Corrección guardada ✅")
        st.session_state["fb_stats_stale"] = True

def mostrar_panel_feedback() -> None:
    st.header("🗂️ Feedback acumulado")
    df_fb = leer_feedback()
    if df_fb.empty:
        st.info("Todavía no hay correcciones guardadas.")
        return

    n_total = len(df_fb)
    n_spam_wrong = ((df_fb["pred_spam"] != df_fb["correcto_spam"]) & (df_fb["correcto_spam"] != df_fb["pred_spam"])).sum()
    n_sent_wrong = ((df_fb["pred_sentimiento"] != df_fb["correcto_sentimiento"]) & (df_fb["correcto_sentimiento"] != df_fb["pred_sentimiento"])).sum()
    n_errors = ((df_fb["pred_spam"] != df_fb["correcto_spam"]) | (df_fb["pred_sentimiento"] != df_fb["correcto_sentimiento"])).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Feedbacks totales", n_total)
    k2.metric("Errores detectados", n_errors)
    k3.metric("Errores spam", n_spam_wrong)
    k4.metric("Errores sentimiento", n_sent_wrong)

    if n_total > 0:
        tasa = n_errors / n_total * 100; color = "🔴" if tasa > 20 else "🟡" if tasa > 10 else "🟢"
        st.markdown(f"{color} **Tasa de error percibida:** `{tasa:.1f}%` basada en evaluaciones humanas")

    st.divider()
    df_err = df_fb[(df_fb["pred_spam"] != df_fb["correcto_spam"]) | (df_fb["pred_sentimiento"] != df_fb["correcto_sentimiento"])].copy()

    if not df_err.empty:
        st.subheader("Correcciones guardadas")
        df_err["tipo_error"] = df_err.apply(lambda r: "Spam + Sentimiento" if r["pred_spam"] != r["correcto_spam"] and r["pred_sentimiento"] != r["correcto_sentimiento"] else "Solo spam" if r["pred_spam"] != r["correcto_spam"] else "Solo sentimiento", axis=1)
        st.dataframe(df_err["tipo_error"].value_counts().reset_index(), use_container_width=True, hide_index=True)
        st.subheader("Textos corregidos")
        st.dataframe(df_err[["timestamp","texto","pred_spam","correcto_spam","pred_sentimiento","correcto_sentimiento","nota"]], use_container_width=True, hide_index=True)

    st.divider(); st.subheader("Exportar para reentrenar")
    df_export = df_fb.rename(columns={"texto": "text", "correcto_spam": "spam_label", "correcto_sentimiento": "sentiment_label"})[["timestamp","texto_hash","text","spam_label","sentiment_label","nota"]]
    col_dl, col_rst = st.columns([3, 1])
    col_dl.download_button("📥 Descargar dataset de correcciones", data=df_export.to_csv(index=False).encode("utf-8"), file_name="feedback_correcciones_entrenamiento.csv", mime="text/csv", type="primary")

    with col_rst:
        if st.button("🗑️ Resetear feedback"):
            if os.path.exists(_feedback_path()): os.remove(_feedback_path())
            st.success("Feedback borrado."); st.rerun()

    with st.expander("📖 Cómo usar este CSV para mejorar el modelo"):
        st.markdown("**Para spam y sentimiento:** Añade el archivo como fuente en tus scripts de carga de datos.")


# ─────────────────────────────────────────────────────────────────
# COMPONENTES VISUALES RESTAURADOS
# ─────────────────────────────────────────────────────────────────
def grafico_sentimiento(df_res: pd.DataFrame):
    return px.pie(df_res, names="Sentimiento", title="Distribución de sentimiento", hole=0.35, color="Sentimiento", color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral":  "#95a5a6"})

def nube_palabras(df_res: pd.DataFrame):
    txt = " ".join(df_res.loc[df_res["Spam"] == "✅ NO", "Comentario"].astype(str))
    if not txt.strip(): return None
    wc = WordCloud(background_color="white", collocations=False, stopwords=set(STOPWORDS)).generate(txt)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    return fig

def analizar_batch(comentarios: list[dict], spam_pipe, sent_pipe) -> list[dict]:
    flags = detectar_bots(comentarios); filas = []
    for i, c in enumerate(comentarios):
        res = analizar(c["texto"], spam_pipe, sent_pipe, batch_spam=flags[i])
        filas.append({"Seudónimo": c["seudónimo"], "Comentario": c["texto"], "Spam": "🚨 SÍ" if res["spam"] else "✅ NO", "Motivo": res["motivo"], "Sentimiento": res["sentimiento"], "Conf. spam": f"{res['spam_conf']:.0f}%", "Conf. sent.": f"{res['sent_conf']:.0f}%"})
    return filas

def mostrar_resultados(df_res: pd.DataFrame):
    k1, k2, k3 = st.columns(3)
    k1.metric("Analizados", len(df_res))
    k2.metric("Spam", f"{(df_res['Spam']=='🚨 SÍ').mean()*100:.1f}%", delta_color="inverse")
    k3.metric("Positivos", f"{(df_res['Sentimiento']=='Positive').mean()*100:.1f}%")

    st.divider(); g1, g2 = st.columns(2)
    with g1: st.plotly_chart(grafico_sentimiento(df_res), use_container_width=True)
    with g2:
        st.write("**Nube de palabras — comentarios reales**")
        fig_wc = nube_palabras(df_res)
        if fig_wc: st.pyplot(fig_wc, clear_figure=True)
        else: st.info("No hay suficientes comentarios reales.")

    spam_df = df_res[df_res["Spam"] == "🚨 SÍ"]
    if not spam_df.empty:
        m = spam_df["Motivo"].value_counts().reset_index(); m.columns = ["Motivo", "N"]
        st.plotly_chart(px.bar(m, x="Motivo", y="N", title="Motivo de clasificación como spam"), use_container_width=True)

    st.subheader("📋 Tabla detallada")
    mostrar_fb = st.toggle("✏️ Activar feedback por fila", value=False)
    if mostrar_fb:
        n_fb_open = st.session_state.get("_n_fb_open", 5)
        for idx, row in df_res.head(n_fb_open).iterrows():
            with st.expander(f"Fila {idx+1} — {str(row.get('Comentario'))[:80]}…"):
                widget_feedback_fila(idx, str(row.get('Comentario')), row.get('Spam'), row.get('Sentimiento'))
        if len(df_res) > n_fb_open:
            if st.button("Cargar más filas"): st.session_state["_n_fb_open"] = n_fb_open + 20; st.rerun()
    else:
        st.dataframe(df_res, use_container_width=True)
    st.download_button("📥 Descargar CSV anonimizado", df_res.to_csv(index=False).encode("utf-8"), "auditoria_anonimizada.csv", "text/csv")


AVISO_RGPD = "**Aviso RGPD**\n- **Seudonimización inmediata** (Art. 25): SHA-256\n- **Minimización** (Art. 5.1.c) y **Sin persistencia** (Art. 5.1.e)"
FEEDBACK_CSV = "feedback_correcciones.csv"
_FEEDBACK_COLS = ["timestamp", "texto_hash", "texto", "pred_spam", "pred_sentimiento", "correcto_spam", "correcto_sentimiento", "nota"]


# ─────────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL COMPLETAMENTE RESTAURADA (v9.0)
# ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 YouTube Spam & Sentiment Detector")

    ratio_sel      = st.session_state.get("_ratio_sel", 1)
    ratio_sent_sel = st.session_state.get("_ratio_sent_sel", 1)

    with st.spinner("Cargando datasets…"):
        try:
            df_spam = cargar_datos_spam(ratio_real_spam=ratio_sel)
            df_sent = cargar_datos_sentimiento(ratio_por_clase=ratio_sent_sel)
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()

    with st.spinner("Entrenando modelos…"):
        spam_pipe, m_spam = entrenar_spam(df_spam)
        sent_pipe, m_sent = entrenar_sentimiento(df_sent)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
        st.title("Menú")
        opcion = st.radio("", ["🔎 Análisis manual", "📂 Análisis por fichero", "🎬 Auditoría en tiempo real", "📊 Rendimiento de los modelos", "📈 Datasets de entrenamiento", "🧩 Análisis de temas", "📝 Feedback & reentrenamiento"])
        st.divider()

        st.subheader("⚖️ Balance del dataset")
        st.markdown("**🛡️ Spam — Ratio real:spam**")
        ratio_sel = st.radio("Real : Spam", options=[1, 2, 3], format_func=lambda r: {1: "1:1 — Balanceado", 2: "2:1 — Moderado", 3: "3:1 — Conservador"}[r], index=0, key="sb_r_spam")
        if st.session_state.get("_ratio_sel") != ratio_sel:
            st.session_state["_ratio_sel"] = ratio_sel; cargar_datos_spam.clear(); entrenar_spam.clear()
        
        st.markdown("**💬 Sentimiento — Ratio otras:negative**")
        ratio_sent_sel = st.radio("Pos+Neu : Neg", options=[1, 2, 3, 4], format_func=lambda r: {1: "1:1 — Balanceado", 2: "2:1 — Moderado", 3: "3:1 — Conservador", 4: "4:1 — Original"}[r], index=0, key="sb_r_sent")
        if st.session_state.get("_ratio_sent_sel") != ratio_sent_sel:
            st.session_state["_ratio_sent_sel"] = ratio_sent_sel; cargar_datos_sentimiento.clear(); entrenar_sentimiento.clear()

        if HF_SENTIMENT_MODEL and _BERT_AVAILABLE:
            st.success(f"🤖 BERT activo: `{HF_SENTIMENT_MODEL.split('/')[-1]}`")
        else:
            st.info("💡 Modo Sklearn Tradicional activo")

        st.metric("F1 (spam)", f"{m_spam['f1']:.3f}")
        st.metric("F1 macro (sent)", f"{m_sent['f1_macro']:.3f}")
        st.divider()

    # TABS Y PESTAÑAS INTEGRALES RESTAURADAS 
    if opcion == "🔎 Análisis manual":
        st.header("🔎 Análisis manual")
        texto = st.text_area("Comentario a analizar:", height=110)
        if st.button("Analizar", type="primary"):
            if not texto.strip(): st.warning("Escribe algo."); return
            res = analizar(texto, spam_pipe, sent_pipe)
            c1, c2, c3 = st.columns(3)
            c1.metric("Spam", "🚨 SÍ" if res["spam"] else "✅ NO", f"{res['spam_conf']:.0f}% conf")
            c2.metric("Sentimiento", res["sentimiento"], f"{res['sent_conf']:.0f}% conf")
            c3.metric("Detectado por", res["motivo"] or "—")
            with st.expander("🔍 Features del EDA"):
                for n, v in zip(["Contiene URL","Ratio mayús","Exclamaciones","!! múltiple","Palabras CAPS","Diversidad léxica","Hits léxico spam","Palabras repetidas","Longitud log"], SpamFeatures()._f(texto)):
                    st.write(f"**{n}**: {v:.3f}")
            with st.expander("✏️ Feedback"):
                widget_feedback_fila(0, texto, "🚨 SÍ" if res["spam"] else "✅ NO", res["sentimiento"])

    elif opcion == "📂 Análisis por fichero":
        st.header("📂 Análisis por fichero CSV")
        upload = st.file_uploader("Sube tu CSV", type=["csv"])
        if upload and st.button("🚀 Analizar fichero", type="primary"):
            df_up, err = leer_csv_usuario(upload)
            if err: st.error(err); return
            filas = analizar_batch(df_up[["seudónimo", "texto"]].to_dict("records"), spam_pipe, sent_pipe)
            df_filas = pd.DataFrame(filas); st.session_state["df_resultados_temas"] = df_filas
            mostrar_resultados(df_filas)

    elif opcion == "🎬 Auditoría en tiempo real":
        st.header("🎬 Auditoría en tiempo real")
        url = st.text_input("URL del vídeo:")
        if st.button("🚀 Analizar vídeo", type="primary"):
            v_id = extraer_video_id(url)
            if not v_id: st.error("ID Inválido."); return
            comentarios = descargar_comentarios(api_key, v_id, num_api)
            df_filas = pd.DataFrame(analizar_batch(comentarios, spam_pipe, sent_pipe))
            st.session_state["df_resultados_temas"] = df_filas; mostrar_resultados(df_filas)

    elif opcion == "📊 Rendimiento de los modelos":
        st.header("📊 Rendimiento de los modelos")
        tab_spam, tab_sent = st.tabs(["🛡️ Spam (LR)", "💬 Sentimiento (LR)"])
        with tab_spam:
            st.markdown(f"Entrenado con **{m_spam['n_train']:,}** muestras. Holdout 20%.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m_spam['accuracy']:.3f}")
            c2.metric("Precision", f"{m_spam['precision']:.3f}")
            c3.metric("Recall", f"{m_spam['recall']:.3f}")
            c4.metric("F1 spam", f"{m_spam['f1']:.3f}")
            st.pyplot(plot_cm(m_spam["cm"], ["Real", "Spam"], "Confusión spam"))
            st.code(m_spam["reporte"])
        with tab_sent:
            st.markdown(f"Entrenado con **{m_sent['n_train']:,}** muestras. Holdout 15%.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{m_sent['accuracy']:.3f}")
            c2.metric("F1 macro", f"{m_sent['f1_macro']:.3f}")
            c3.metric("F1 negative", f"{m_sent['f1_negative']:.3f}")
            st.pyplot(plot_cm(m_sent["cm"], ["positive", "neutral", "negative"], "Confusión sentimiento"))
            st.code(m_sent["reporte"])

    elif opcion == "📈 Datasets de entrenamiento":
        st.header("📈 Datasets de entrenamiento")
        tab_s, tab_sent2 = st.tabs(["🛡️ Spam", "💬 Sentimiento"])
        with tab_s:
            df_s = df_spam.copy(); df_s["Etiqueta"] = df_s["spam"].map({0: "Real", 1: "Spam"})
            st.plotly_chart(px.pie(df_s, names="Etiqueta", title="Balance Real / Spam", hole=0.4), use_container_width=True)
            df_s["longitud"] = df_s["text"].str.len()
            st.plotly_chart(px.box(df_s, x="Etiqueta", y="longitud", title="Longitud por clase"), use_container_width=True)
        with tab_sent2:
            df_sv = df_sent.copy(); df_sv["Etiqueta"] = df_sv["sentiment"].str.capitalize()
            st.plotly_chart(px.pie(df_sv, names="Etiqueta", title="Distribución sentimiento", hole=0.4), use_container_width=True)
            df_sv["longitud"] = df_sv["text"].str.len()
            st.plotly_chart(px.box(df_sv, x="Etiqueta", y="longitud", title="Longitud por sentimiento"), use_container_width=True)

    elif opcion == "📝 Feedback & reentrenamiento":
        mostrar_panel_feedback()

    elif opcion == "🧩 Análisis de temas":
        st.header("🧩 Análisis de temas")
        fuente = st.radio("Fuente de comentarios", ["Pegar comentarios manualmente", "Usar último análisis de vídeo/CSV"], horizontal=True, key="rb_fuente_temas")
        if fuente == "Pegar comentarios manualmente":
            raw = st.text_area("Pega aquí los comentarios (uno por línea)", height=200)
            if st.button("🔍 Analizar temas", type="primary") and raw.strip():
                lineas = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                with st.spinner("Analizando…"):
                    resultados = [analizar(c, spam_pipe, sent_pipe) for c in lineas]
                
                # BUGFIX COMPLETADO AQUÍ -> Garantiza que no ocurra KeyError
                df_man = pd.DataFrame({
                    "Comentario": lineas,
                    "Sentimiento": [r["sentimiento"] for r in resultados],
                    "Spam": ["🚨 SÍ" if r["spam"] else "✅ NO" for r in resultados],
                })
                st.session_state["df_resultados_temas"] = df_man; st.rerun()

        df_cached = st.session_state.get("df_resultados_temas", None)
        if df_cached is not None and not df_cached.empty: mostrar_analisis_temas(df_cached)
        else: st.warning("No hay resultados previos.")

    st.divider(); st.caption("v9.0 · Ensemble LR+SVC spam · BERT/LR sent · NMF temas · Feedback · RGPD Art. 25")

if __name__ == "__main__":
    main()
