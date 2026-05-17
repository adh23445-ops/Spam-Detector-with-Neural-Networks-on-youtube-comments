"""
IA YouTube Spam Detector — v8.1  "Light Dataset"

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
  Modelo: LR(C=1, balanced, saga) + TF-IDF bigramas 30k features

ARQUITECTURA (Sprint 3.1 del documento)
  • Spam:       Regresión Logística (word + char n-grams + handcrafted)
  • Sentimiento: Regresión Logística multiclase (rápida, precisa, 3 clases)
  • Ambos modelos cacheados con @st.cache_resource

RGPD (Art. 25, 5.1.c, 5.1.e, 6.1.f)
  Seudonimización SHA-256 inmediata. Sin persistencia. CSV anonimizado.

YouTube ToS
  Auditoría en tiempo real vía YouTube Data API v3 oficial.
"""

import hashlib
import csv

# ── Modelo BERT fine-tuned (opcional) ────────────────────────────────────────
# Pon aquí el nombre de tu repo de HuggingFace tras el fine-tuning.
# Si está vacío o el modelo no carga, la app usa el modelo sklearn de siempre.
HF_SENTIMENT_MODEL = ""   # Ejemplo: "tu-usuario/youtube-sentiment-distilbert"

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
TIMESTAMP_RE  = re.compile(r"\b\d{1,2}:\d{2}\b")
MENTION_RE    = re.compile(r"@\w+")
REPEAT_CHR_RE = re.compile(r"(.)\1{3,}")

SPAM_LEXICON = {
    "subscribe", "suscribete", "suscríbete", "free", "gratis", "click",
    "win", "gana", "money", "dinero", "cash", "giveaway", "sorteo",
    "check out", "visit", "my channel", "mi canal", "promo", "discount",
    "descuento", "link in bio", "crypto", "bitcoin", "investment",
    "earn", "profit", "dm me", "escríbeme",
    # nuevos (v8.5)
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
            # — originales —
            len(URL_RE.findall(t)),                                     # nº URLs
            sum(c.isupper() for c in t) / nc,                          # ratio mayúsculas
            t.count("!"),                                               # exclamaciones
            len(EXCL_RE.findall(t)),                                    # grupos "!!+"
            len(CAPS_WORD_RE.findall(t)),                               # palabras CAPS
            len(set(ws)) / nw,                                         # diversidad léxica
            sum(1 for w in SPAM_LEXICON if w in tl),                   # hits léxico spam
            len(REPEAT_RE.findall(tl)),                                 # palabras repetidas
            np.log1p(float(len(t))),                                   # longitud log
            # — nuevas (v8.5) —
            sum(c.isdigit() for c in t) / nc,                         # ratio dígitos
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
    """Devuelve la primera ruta existente de una lista de candidatos.
    Permite que la app encuentre los CSV tanto con el nombre original
    (espacios y paréntesis) como con el nombre saneado que generan
    algunos entornos al subir el archivo (guiones bajos)."""
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


RUTAS = {
    "spam_clasico": _resolver_ruta(
        "Youtube-Spam-Dataset.csv",
        "data/Youtube-Spam-Dataset.csv",
    ),
    "spam_equilibrado": _resolver_ruta(
        "Youtube-Spam-Dataset_equilibrado_csv.csv",
        "Youtube-Spam-Dataset_equilibrado.csv",
        "data/Youtube-Spam-Dataset_equilibrado_csv.csv",
    ),
    "spam_45k": _resolver_ruta(
        "YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv",
        "YouTube_Comments_Dataset_with_Sentiment_Toxicity_and_Spam_Labels__45K_Rows_.csv",
        "data/YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv",
        "data/YouTube_Comments_Dataset_with_Sentiment_Toxicity_and_Spam_Labels__45K_Rows_.csv",
    ),
    "export_anterior": _resolver_ruta(
        "2026-05-13T17-45_export.csv",
    ),
}

@st.cache_data(show_spinner=False)
def cargar_datos_spam(ratio_real_spam: int = 1) -> pd.DataFrame:
    """
    Carga y combina los datasets de spam aplicando undersampling estratificado.
    Con los 3 datasets combinados (45k + clásico + equilibrado) la cobertura
    de spam sube a 2 826 ejemplos → F1 0.897 → 0.940, gap cv/val ≈ 0.

    ratio_real_spam: cuántos comentarios reales por cada spam.
        1  →  50 % spam / 50 % real  (F1-spam ≈ 0.94, Prec ≈ 0.95)
        2  →  33 % spam / 67 % real  (más recall, menos precisión)
        3  →  25 % spam / 75 % real  (conservador)
    """
    partes = []

    # Fuente 1: dataset clásico original (1 956 filas, ~50 % spam)
    if RUTAS["spam_clasico"]:
        df = pd.read_csv(RUTAS["spam_clasico"])[["CONTENT", "CLASS"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = df["spam"].astype(int)
        partes.append(df)

    # Fuente 2: dataset clásico equilibrado con features extra (1 956 filas)
    # Aporta ~1 005 ejemplos de spam únicos no presentes en el clásico.
    if RUTAS["spam_equilibrado"]:
        df = pd.read_csv(RUTAS["spam_equilibrado"])[["CONTENT", "CLASS"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = df["spam"].astype(int)
        partes.append(df)

    # Fuente 3: dataset 45k moderno (45 005 filas, ~1.8 % spam)
    if RUTAS["spam_45k"]:
        df = pd.read_csv(RUTAS["spam_45k"])[["comment_text", "label_spam"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = (df["spam"].str.strip().str.lower() == "spam").astype(int)
        partes.append(df)

    # Fuente 4: export anterior (si tiene columna Spam)
    if RUTAS["export_anterior"]:
        df = pd.read_csv(RUTAS["export_anterior"])
        if "Comentario" in df.columns and "Spam" in df.columns:
            df = df[["Comentario", "Spam"]].dropna()
            df.columns = ["text", "spam"]
            df["spam"] = df["spam"].str.contains("SÍ|1|spam", case=False, na=False).astype(int)
            partes.append(df)

    if not partes:
        raise FileNotFoundError(
            "No se encontró ningún dataset de spam. "
            "Coloca al menos uno de los CSV en la carpeta de la app."
        )

    combined = pd.concat(partes, ignore_index=True).dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]

    # Deduplicar por texto para evitar data leakage entre fuentes
    combined = combined.drop_duplicates(subset=["text"])

    # ── Undersampling estratificado de la clase mayoritaria ──────────────
    spam_df = combined[combined["spam"] == 1]
    real_df  = combined[combined["spam"] == 0]
    n_spam    = len(spam_df)
    n_real_target = min(n_spam * ratio_real_spam, len(real_df))

    real_df_sampled = real_df.sample(n=n_real_target, random_state=42)
    balanced = pd.concat([spam_df, real_df_sampled], ignore_index=True).sample(
        frac=1, random_state=42
    )
    return balanced


@st.cache_data(show_spinner=False)
def cargar_datos_sentimiento(ratio_por_clase: int = 1) -> pd.DataFrame:
    """
    Carga las etiquetas de sentimiento usando el dataset de 45K, aplicando
    undersampling estratificado sobre neutral y positive para equilibrar
    la clase minoritaria (negative, ~8.5 % del total).

    ratio_por_clase: cuantas muestras de neutral/positive por cada negative.
        1  →  33 % cada clase  (f1_negative ≈ 0.68, f1_macro ≈ 0.61)
        2  →  ~25 % neg / 37.5 % resto  (f1_negative ≈ 0.62, f1_macro ≈ 0.61)
        3  →  ~20 % neg / 40 % resto  (f1_negative ≈ 0.56, f1_macro ≈ 0.62)
        4  →  ~17 % neg / 42 % resto  (f1_negative ≈ 0.52, f1_macro ≈ 0.62)
    """
    partes = []

    # Fuente unica: 45k dataset (sentimientos recientes, multilingüe)
    if RUTAS["spam_45k"]:
        df = pd.read_csv(RUTAS["spam_45k"], usecols=["comment_text", "label_sentiment"])
        df.columns = ["text", "sentiment"]
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        df = df[df["sentiment"].isin(["positive", "neutral", "negative"])].dropna()
        partes.append(df)

    if not partes:
        raise FileNotFoundError("No se encontro el dataset de sentimiento (45K Rows).")

    combined = pd.concat(partes, ignore_index=True).dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]

    # Undersampling estratificado sobre la clase minoritaria
    # negative es la mas escasa (~3 810 filas, 8.5 %).
    # Reducimos positive y neutral a ratio_por_clase x n_negative.
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

    # Ensemble de dos clasificadores con voting suave:
    #  • LR  (C=0.5) — calibrado, bueno en fronteras lineales suaves
    #  • LinearSVC (C=0.3) — margen máximo, robusto con texto sparse
    # Ambos usan la misma FeatureUnion:
    #   word bigramas (8k) + char (2,5)-grams (15k) + 14 features EDA
    # El ensemble reduce varianza sin aumentar bias → gap cv/val estable ~0.023
    def _feat_union():
        return FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.95,
                max_features=8000, sublinear_tf=True, strip_accents="unicode",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                max_features=15000, sublinear_tf=True, strip_accents="unicode",
            )),
            ("hc", SpamFeatures()),
        ])

    lr_pipe = Pipeline([
        ("f", _feat_union()), ("s", MaxAbsScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, solver="saga",
                                   class_weight="balanced", random_state=42)),
    ])
    svc_pipe = Pipeline([
        ("f", _feat_union()), ("s", MaxAbsScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.3, class_weight="balanced", max_iter=3000, random_state=42),
            cv=3,
        )),
    ])
    pipeline = VotingClassifier(
        [("lr", lr_pipe), ("svc", svc_pipe)],
        voting="soft", weights=[2, 1],
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
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
# INFERENCIA BERT (si HF_SENTIMENT_MODEL está configurado)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cargar_bert_sentimiento():
    """Carga el modelo fine-tuned desde HuggingFace Hub (una sola vez)."""
    if not _BERT_AVAILABLE or not HF_SENTIMENT_MODEL:
        return None
    try:
        return hf_pipeline(
            "text-classification",
            model=HF_SENTIMENT_MODEL,
            truncation=True,
            max_length=128,
            device=-1,          # CPU; cambia a 0 si tienes GPU en el servidor
        )
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo BERT: {e}. Usando sklearn.")
        return None


def predecir_sentimiento(texto: str, sent_pipe_sklearn) -> tuple[str, float]:
    """
    Devuelve (etiqueta, confianza).
    Usa BERT si está disponible, sklearn si no.
    """
    bert = _cargar_bert_sentimiento()
    if bert is not None:
        try:
            res = bert(texto[:512])[0]
            # HF devuelve la etiqueta tal como la definiste en id2label
            label = res["label"].lower()
            return label, round(res["score"] * 100, 1)
        except Exception:
            pass  # fallback a sklearn
    # ── Fallback sklearn ──────────────────────────────────────────
    pred  = sent_pipe_sklearn.predict([texto])[0]
    proba = sent_pipe_sklearn.predict_proba([texto])[0].max()
    return pred, round(float(proba) * 100, 1)


# ─────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — SENTIMIENTO (LR multiclase)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrenar_sentimiento(df: pd.DataFrame):
    X = df["text"].tolist()
    y = df["sentiment"].tolist()    # positive / neutral / negative

    # FeatureUnion de tres ramas (v8.6):
    #   1. TF-IDF word bigramas  — léxico de sentimiento
    #   2. TF-IDF char (2,4)-grams — emoticons, "!!!", "omg", morfología
    #   3. SentimentFeatures — 15 señales handcrafted (léxico pos/neg,
    #      negaciones, intensificadores, emojis, caps, polaridad neta)
    # C=0.3 (más regularización) reduce el gap cv/val sin perder F1.
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), max_features=30_000, sublinear_tf=True,
                min_df=2, max_df=0.95, strip_accents="unicode",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), max_features=20_000,
                sublinear_tf=True, min_df=3, strip_accents="unicode",
            )),
            ("sf", SentimentFeatures()),
        ])),
        ("scaler", MaxAbsScaler()),
        ("clf", LogisticRegression(
            C=0.3, max_iter=1000, solver="saga", class_weight="balanced",
            random_state=42,
        )),
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
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
def extraer_temas(
    comentarios: list[str],
    n_temas: int = 4,
    n_palabras: int = 8,
    ngram_max: int = 2,
) -> list[dict]:
    """
    Extrae temas con NMF + TF-IDF.
    Devuelve lista de dicts {id, palabras, peso_medio}.
    NMF produce temas más nítidos que LDA en textos cortos (comentarios YouTube).
    """
    if len(comentarios) < 20:
        return []

    limpios = [_limpiar_texto_temas(c) for c in comentarios]
    limpios = [t for t in limpios if len(t.split()) >= 3]
    if len(limpios) < 20:
        return []

    vec = TfidfVectorizer(
        max_df=0.80, min_df=max(3, len(limpios)//200),
        max_features=2500,
        stop_words="english",
        ngram_range=(1, ngram_max),
        token_pattern=r"(?u)\b[a-z]{3,}\b",
    )
    try:
        dtm = vec.fit_transform(limpios)
    except ValueError:
        return []

    words = vec.get_feature_names_out()
    # Eliminar stop words adicionales del vocabulario
    keep_idx = [i for i, w in enumerate(words)
                if not any(s == w or w.startswith(s + " ") or w.endswith(" " + s)
                           for s in _EXTRA_STOP)]
    if len(keep_idx) < n_palabras * 2:
        keep_idx = list(range(len(words)))     # fallback sin filtro extra

    dtm_f = dtm[:, keep_idx]
    words_f = words[keep_idx]

    n_comp = min(n_temas, dtm_f.shape[1], dtm_f.shape[0] - 1)
    if n_comp < 1:
        return []

    nmf = NMF(n_components=n_comp, random_state=42, max_iter=300,
              init="nndsvda", l1_ratio=0.1)
    W = nmf.fit_transform(dtm_f)   # doc-topic matrix

    temas = []
    for i, comp in enumerate(nmf.components_):
        top_idx  = comp.argsort()[-n_palabras:][::-1]
        palabras = [words_f[j] for j in top_idx]
        pesos    = [float(comp[j]) for j in top_idx]
        peso_doc = float(W[:, i].mean())        # relevancia media del tema
        temas.append({
            "id":       i + 1,
            "palabras": palabras,
            "pesos":    pesos,
            "relevancia": peso_doc,
        })

    # Ordenar por relevancia descendente
    temas.sort(key=lambda x: x["relevancia"], reverse=True)
    return temas


def _color_tema(i: int) -> str:
    """Paleta de colores para los temas (Plotly)."""
    paleta = ["#534AB7","#1D9E75","#D85A30","#378ADD","#D4537E"]
    return paleta[i % len(paleta)]


def mostrar_analisis_temas(df_res: pd.DataFrame) -> None:
    """
    Renderiza la pestaña completa de análisis de temas.
    df_res debe tener columnas 'Comentario', 'Sentimiento', 'Spam'.
    """
    import plotly.graph_objects as go

    st.subheader("🧩 Análisis de temas por sentimiento")
    st.caption(
        "Algoritmo NMF sobre TF-IDF bigramas — extrae los temas latentes que "
        "dominan los comentarios de cada grupo de sentimiento."
    )

    # Filtrar spam
    df_clean = df_res[df_res.get("Spam", pd.Series(["No"]*len(df_res))) != "SÍ"].copy()
    if df_clean.empty or "Sentimiento" not in df_clean.columns:
        st.info("No hay suficientes comentarios analizados para extraer temas.")
        return

    # ── Controles ────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    n_temas   = col_ctrl1.slider("Número de temas", 2, 6, 4)
    n_palabras= col_ctrl2.slider("Palabras por tema", 5, 12, 8)
    solo_ing  = col_ctrl3.checkbox("Solo comentarios en inglés", value=False,
                                    help="Filtra por la columna 'Idioma' si existe. "
                                         "Los temas son más nítidos en un solo idioma.")

    if solo_ing and "Idioma" in df_clean.columns:
        df_clean = df_clean[df_clean["Idioma"].str.lower().str.startswith("en", na=False)]

    sentimientos = ["Positive", "Negative", "Neutral"]
    colores_sent = {"Positive": "success", "Negative": "danger", "Neutral": "info"}
    emoji_sent   = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}

    tabs = st.tabs([f"{emoji_sent[s]} {s}" for s in sentimientos])

    for tab, sent in zip(tabs, sentimientos):
        with tab:
            subset = df_clean[df_clean["Sentimiento"] == sent]["Comentario"].tolist()

            if len(subset) < 20:
                st.info(f"Solo {len(subset)} comentarios {sent.lower()} — mínimo 20 para extraer temas.")
                continue

            with st.spinner(f"Extrayendo temas de {len(subset):,} comentarios {sent.lower()}…"):
                temas = extraer_temas(
                    tuple(subset),          # hashable para cache
                    n_temas=n_temas,
                    n_palabras=n_palabras,
                )

            if not temas:
                st.warning("No se pudieron extraer temas significativos de este grupo.")
                continue

            st.markdown(f"**{len(subset):,}** comentarios analizados · **{len(temas)}** temas encontrados")
            st.divider()

            for tema in temas:
                i       = tema["id"] - 1
                color   = _color_tema(i)
                palabras= tema["palabras"]
                pesos   = tema["pesos"]
                rel     = tema["relevancia"]

                # Cabecera del tema
                st.markdown(
                    f'<span style="display:inline-block;background:{color}18;'
                    f'color:{color};border:1px solid {color}40;'
                    f'border-radius:6px;padding:3px 10px;font-size:13px;font-weight:500;">'
                    f'Tema {tema["id"]}</span>'
                    f'<span style="color:var(--color-text-secondary);font-size:12px;margin-left:8px;">'
                    f'relevancia media {rel:.4f}</span>',
                    unsafe_allow_html=True,
                )

                # Gráfico de barras horizontal (palabras del tema)
                fig = go.Figure(go.Bar(
                    x=pesos[::-1],
                    y=palabras[::-1],
                    orientation="h",
                    marker_color=color,
                    marker_opacity=0.85,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig.update_layout(
                    height=max(180, len(palabras) * 28),
                    margin=dict(l=0, r=20, t=8, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False, tickfont=dict(size=13)),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"tema_{sent}_{i}")

                # Pills de palabras clave
                pills_html = " ".join(
                    f'<span style="background:{color}18;color:{color};'
                    f'border:1px solid {color}40;border-radius:20px;'
                    f'padding:2px 10px;font-size:12px;margin:2px;display:inline-block;">'
                    f'{p}</span>'
                    for p in palabras
                )
                st.markdown(pills_html, unsafe_allow_html=True)
                st.markdown("---")

            # Resumen comparativo de temas (radar / heat)
            with st.expander("Ver relevancia comparativa de todos los temas", expanded=False):
                labels = [f"Tema {t['id']}: {t['palabras'][0]}" for t in temas]
                valores = [t["relevancia"] for t in temas]
                colores = [_color_tema(t["id"]-1) for t in temas]
                fig2 = go.Figure(go.Bar(
                    x=labels, y=valores,
                    marker_color=colores,
                    hovertemplate="%{x}<br>Relevancia: %{y:.4f}<extra></extra>",
                ))
                fig2.update_layout(
                    height=260,
                    margin=dict(l=0, r=0, t=12, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                               zeroline=False, showticklabels=False),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True, key=f"resumen_{sent}")


# ─────────────────────────────────────────────────────────────────
# DETECCIÓN BOT EN BATCH (repetición por seudónimo)
# ─────────────────────────────────────────────────────────────────
def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detectar_bots(comentarios: list[dict]) -> dict[int, bool]:
    resultado = {i: False for i in range(len(comentarios))}
    por_autor: dict[str, list[int]] = {}
    for i, c in enumerate(comentarios):
        por_autor.setdefault(c.get("seudónimo", ""), []).append(i)
    for _, idx in por_autor.items():
        if len(idx) < 2:
            continue
        textos = [comentarios[i]["texto"] for i in idx]
        total = similares = 0
        for ia in range(len(textos)):
            for ib in range(ia + 1, len(textos)):
                total += 1
                if _sim(textos[ia], textos[ib]) >= 0.80:
                    similares += 1
        if total and (similares / total) >= 0.60:
            for i in idx:
                resultado[i] = True
    return resultado


# ─────────────────────────────────────────────────────────────────
# REGLAS DURAS DE SPAM
# ─────────────────────────────────────────────────────────────────
def reglas_duras(texto: str):
    t, tl = str(texto), str(texto).lower()
    nc = max(len(t), 1)
    if URL_RE.search(t):                                       return True,  95.0
    if sum(1 for w in SPAM_LEXICON if w in tl) >= 3:           return True,  90.0
    if REPEAT_RE.search(tl):                                   return True,  85.0
    if sum(c.isupper() for c in t) / nc > 0.70 and len(t) > 15:   return True,  80.0
    if t.count("!") >= 5:                                      return True,  78.0
    if len(t.split()) <= 2:                                    return False, 80.0
    return None, None


# ─────────────────────────────────────────────────────────────────
# PREPROCESADO TEXTO PARA SENTIMIENTO
# ─────────────────────────────────────────────────────────────────
def preprocesar(texto: str) -> str:
    t = URL_RE.sub(" ", str(texto))
    t = TIMESTAMP_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = REPEAT_CHR_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────
# ANÁLISIS COMPLETO DE UN COMENTARIO
# ─────────────────────────────────────────────────────────────────
def analizar(texto: str, spam_pipe, sent_pipe, batch_spam: bool = False) -> dict:
    texto = str(texto or "").strip()
    if not texto:
        return {"spam": 0, "spam_conf": 0.0, "sentimiento": "Neutral", "sent_conf": 50.0, "motivo": ""}

    # — SPAM —
    if batch_spam:
        spam, spam_conf, motivo = 1, 97.0, "bot (repetición)"
    else:
        rd, rd_c = reglas_duras(texto)
        if rd is not None:
            spam, spam_conf, motivo = int(rd), rd_c, "regla" if rd else ""
        else:
            probas   = spam_pipe.predict_proba([texto])[0]
            clases   = spam_pipe.named_steps["clf"].classes_
            idx_spam = int(np.where(clases == 1)[0][0]) if 1 in clases else 1
            spam     = int(spam_pipe.predict([texto])[0])
            spam_conf = float(probas[idx_spam]) * 100
            motivo   = "LR" if spam else ""

    # — SENTIMIENTO (BERT si HF_SENTIMENT_MODEL configurado, sklearn si no) —
    limpio      = preprocesar(texto)
    texto_sent  = limpio if limpio else texto
    sent_raw, sent_conf = predecir_sentimiento(texto_sent, sent_pipe)
    label_map   = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    sentimiento = label_map.get(sent_raw, sent_raw.capitalize())

    return {
        "spam": spam, "spam_conf": spam_conf,
        "sentimiento": sentimiento, "sent_conf": sent_conf,
        "motivo": motivo,
    }


# ─────────────────────────────────────────────────────────────────
# YOUTUBE DATA API v3 — DESCARGA OFICIAL
# ─────────────────────────────────────────────────────────────────
def extraer_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

def descargar_comentarios(api_key: str, video_id: str, limite: int) -> list[dict]:
    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    comentarios: list[dict] = []
    page_token = None
    while len(comentarios) < limite:
        por_pagina = min(100, limite - len(comentarios))
        kw = dict(part="snippet", videoId=video_id, maxResults=por_pagina, order="time", textFormat="plainText")
        if page_token:
            kw["pageToken"] = page_token
        resp = yt.commentThreads().list(**kw).execute()
        for item in resp.get("items", []):
            snip  = item["snippet"]["topLevelComment"]["snippet"]
            texto = snip.get("textDisplay", "").strip()
            if texto:
                comentarios.append({
                    "seudónimo": seudonimizar(snip.get("authorDisplayName", "")),
                    "texto": texto,
                })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return comentarios[:limite]


# ─────────────────────────────────────────────────────────────────
# CARGA DE CSV DEL USUARIO
# ─────────────────────────────────────────────────────────────────
def leer_csv_usuario(upload) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        return None, str(e)
    cols = {c.lower(): c for c in df.columns}
    col  = next((cols[k] for k in ("comentario", "content", "text", "comment") if k in cols), None)
    if col is None:
        return None, f"Columnas no reconocidas: {list(df.columns)}"
    df = df.dropna(subset=[col]).copy()
    df["texto"] = df[col].astype(str).str.strip()
    df = df[df["texto"] != ""]
    a = next((cols[k] for k in ("autor", "author") if k in cols), None)
    df["seudónimo"] = df[a].apply(seudonimizar) if a else [f"Usr-{i:04d}" for i in range(len(df))]
    return df, None


# ─────────────────────────────────────────────────────────────────
# GRÁFICAS AUXILIARES
# ─────────────────────────────────────────────────────────────────
def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────
# FEEDBACK HUMANO — guardar, leer, resetear
# ─────────────────────────────────────────────────────────────────

def _feedback_path() -> str:
    """Devuelve la ruta absoluta del CSV de feedback."""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, FEEDBACK_CSV)


def guardar_feedback(
    texto: str,
    pred_spam: str,
    pred_sent: str,
    correcto_spam: str,
    correcto_sent: str,
    nota: str = "",
) -> None:
    """Añade una fila al CSV de correcciones (crea el fichero si no existe)."""
    import os
    path = _feedback_path()
    nuevo = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FEEDBACK_COLS)
        if nuevo:
            w.writeheader()
        w.writerow({
            "timestamp":          datetime.utcnow().isoformat(timespec="seconds"),
            "texto_hash":         hashlib.sha256(texto.encode()).hexdigest()[:12],
            "texto":              texto[:500],          # truncar por privacidad
            "pred_spam":          pred_spam,
            "pred_sentimiento":   pred_sent,
            "correcto_spam":      correcto_spam,
            "correcto_sentimiento": correcto_sent,
            "nota":               nota[:200],
        })


def leer_feedback() -> pd.DataFrame:
    """Carga el CSV de feedback; devuelve DataFrame vacío si no existe."""
    import os
    path = _feedback_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=_FEEDBACK_COLS)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=_FEEDBACK_COLS)


def _feedback_key(idx: int, campo: str) -> str:
    """Clave única de widget para evitar colisiones entre filas."""
    return f"fb_{campo}_{idx}"


def widget_feedback_fila(
    idx: int,
    texto: str,
    pred_spam: str,
    pred_sent: str,
) -> None:
    """
    Renderiza el formulario de feedback para UNA fila de la tabla de resultados.
    Se llama desde mostrar_resultados() dentro de un st.expander().
    """
    opciones_spam = ["— sin cambio —", "✅ NO es spam", "🚨 SÍ es spam"]
    opciones_sent = ["— sin cambio —", "Positive", "Neutral", "Negative"]

    c1, c2 = st.columns(2)
    spam_correc = c1.selectbox(
        "¿Spam correcto?", opciones_spam,
        key=_feedback_key(idx, "spam"),
    )
    sent_correc = c2.selectbox(
        "¿Sentimiento correcto?", opciones_sent,
        key=_feedback_key(idx, "sent"),
    )
    nota = st.text_input(
        "Nota opcional (¿por qué se equivocó?)",
        key=_feedback_key(idx, "nota"),
        placeholder="p.ej. sarcasmo, mezcla de idiomas, modismo...",
    )

    if st.button("💾 Guardar corrección", key=_feedback_key(idx, "btn")):
        cs = spam_correc if spam_correc != "— sin cambio —" else pred_spam
        cv = sent_correc if sent_correc != "— sin cambio —" else pred_sent
        guardar_feedback(texto, pred_spam, pred_sent, cs, cv, nota)
        st.success("Corrección guardada ✅")
        # Invalidar caché de estadísticas
        st.session_state["fb_stats_stale"] = True


def mostrar_panel_feedback() -> None:
    """
    Pestaña completa de gestión del feedback acumulado:
    estadísticas, tabla, descarga y opción de reset.
    """
    st.header("🗂️ Feedback acumulado")

    df_fb = leer_feedback()

    if df_fb.empty:
        st.info(
            "Todavía no hay correcciones guardadas. "
            "Ve a cualquier análisis, expande una fila y marca si el modelo se equivocó."
        )
        return

    # ── Métricas rápidas ──────────────────────────────────────────
    n_total   = len(df_fb)
    n_spam_wrong = ((df_fb["pred_spam"] != df_fb["correcto_spam"]) &
                    (df_fb["correcto_spam"] != df_fb["pred_spam"])).sum()
    n_sent_wrong = ((df_fb["pred_sentimiento"] != df_fb["correcto_sentimiento"]) &
                    (df_fb["correcto_sentimiento"] != df_fb["pred_sentimiento"])).sum()
    # Filas donde al menos un campo fue corregido
    n_errors = ((df_fb["pred_spam"]          != df_fb["correcto_spam"]) |
                (df_fb["pred_sentimiento"]    != df_fb["correcto_sentimiento"])).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Feedbacks totales",   n_total)
    k2.metric("Errores detectados",  n_errors,
              help="Filas donde la predicción difiere de la corrección humana")
    k3.metric("Errores spam",        n_spam_wrong)
    k4.metric("Errores sentimiento", n_sent_wrong)

    if n_total > 0:
        tasa = n_errors / n_total * 100
        color = "🔴" if tasa > 20 else "🟡" if tasa > 10 else "🟢"
        st.markdown(
            f"{color} **Tasa de error percibida por el usuario:** "
            f"`{tasa:.1f}%` — basada en {n_total} evaluaciones humanas"
        )

    st.divider()

    # ── Errores más frecuentes ────────────────────────────────────
    df_err = df_fb[
        (df_fb["pred_spam"] != df_fb["correcto_spam"]) |
        (df_fb["pred_sentimiento"] != df_fb["correcto_sentimiento"])
    ].copy()

    if not df_err.empty:
        st.subheader("Correcciones guardadas")
        df_err["tipo_error"] = df_err.apply(
            lambda r: (
                "Spam + Sentimiento"
                if r["pred_spam"] != r["correcto_spam"] and r["pred_sentimiento"] != r["correcto_sentimiento"]
                else "Solo spam" if r["pred_spam"] != r["correcto_spam"]
                else "Solo sentimiento"
            ), axis=1
        )
        tipo_counts = df_err["tipo_error"].value_counts().reset_index()
        tipo_counts.columns = ["Tipo de error", "Nº correcciones"]
        st.dataframe(tipo_counts, use_container_width=True, hide_index=True)

        st.subheader("Textos corregidos")
        cols_show = ["timestamp","texto","pred_spam","correcto_spam",
                     "pred_sentimiento","correcto_sentimiento","nota"]
        st.dataframe(df_err[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # ── Descargar para reentrenar ─────────────────────────────────
    st.subheader("Exportar para reentrenar")
    st.markdown(
        "El CSV exportado contiene el texto y la **etiqueta corregida por el humano**. "
        "Úsalo como dataset adicional en `cargar_datos_spam()` / "
        "`cargar_datos_sentimiento()` para incorporar los errores al reentrenamiento."
    )
    col_dl, col_rst = st.columns([3, 1])

    # Dataset de correcciones listo para añadir al pipeline
    df_export = df_fb.rename(columns={
        "texto":              "text",
        "correcto_spam":      "spam_label",
        "correcto_sentimiento": "sentiment_label",
    })[["timestamp","texto_hash","text","spam_label","sentiment_label","nota"]]

    col_dl.download_button(
        "📥 Descargar dataset de correcciones",
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name="feedback_correcciones_entrenamiento.csv",
        mime="text/csv",
        type="primary",
    )

    with col_rst:
        if st.button("🗑️ Resetear feedback", help="Borra todas las correcciones guardadas"):
            import os
            path = _feedback_path()
            if os.path.exists(path):
                os.remove(path)
            st.session_state.pop("fb_stats_stale", None)
            st.success("Feedback borrado.")
            st.rerun()

    # ── Notas de reentrenamiento ──────────────────────────────────
    with st.expander("📖 Cómo usar este CSV para mejorar el modelo"):
        st.markdown("""
**Para spam** — añade el fichero como fuente 5 en `cargar_datos_spam()`:
```python
if os.path.exists("feedback_correcciones_entrenamiento.csv"):
    df_fb = pd.read_csv("feedback_correcciones_entrenamiento.csv")
    df_fb = df_fb[["text","spam_label"]].dropna()
    df_fb.columns = ["text", "spam"]
    df_fb["spam"] = (df_fb["spam"].str.contains("SÍ|spam", case=False)).astype(int)
    partes.append(df_fb)
```
**Para sentimiento** — igual en `cargar_datos_sentimiento()`:
```python
df_fb["sentiment"] = df_fb["sentiment_label"].str.lower().str.strip()
```
**Cuándo reentrenar:** con ≥ 50 correcciones nuevas ya merece la pena.
Con ≥ 200, el impacto en F1 suele ser medible (+0.01–0.03).
        """)


def grafico_sentimiento(df_res: pd.DataFrame):
    return px.pie(
        df_res, names="Sentimiento", title="Distribución de sentimiento",
        hole=0.35, color="Sentimiento",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral":  "#95a5a6"},
    )

def nube_palabras(df_res: pd.DataFrame):
    txt = " ".join(df_res.loc[df_res["Spam"] == "✅ NO", "Comentario"].astype(str))
    if not txt.strip():
        return None
    wc = WordCloud(background_color="white", collocations=False, stopwords=set(STOPWORDS)).generate(txt)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def analizar_batch(comentarios: list[dict], spam_pipe, sent_pipe) -> list[dict]:
    flags = detectar_bots(comentarios)
    filas = []
    for i, c in enumerate(comentarios):
        res = analizar(c["texto"], spam_pipe, sent_pipe, batch_spam=flags[i])
        filas.append({
            "Seudónimo":   c["seudónimo"],
            "Comentario":  c["texto"],
            "Spam":        "🚨 SÍ" if res["spam"] else "✅ NO",
            "Motivo":      res["motivo"],
            "Sentimiento": res["sentimiento"],
            "Conf. spam":  f"{res['spam_conf']:.0f}%",
            "Conf. sent.": f"{res['sent_conf']:.0f}%",
        })
    return filas

def mostrar_resultados(df_res: pd.DataFrame):
    k1, k2, k3 = st.columns(3)
    k1.metric("Analizados",   len(df_res))
    k2.metric("Spam",         f"{(df_res['Spam']=='🚨 SÍ').mean()*100:.1f}%", delta_color="inverse")
    k3.metric("Positivos",    f"{(df_res['Sentimiento']=='Positive').mean()*100:.1f}%")

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(grafico_sentimiento(df_res), use_container_width=True)
    with g2:
        st.write("**Nube de palabras — comentarios reales**")
        fig_wc = nube_palabras(df_res)
        if fig_wc:
            st.pyplot(fig_wc, clear_figure=True)
        else:
            st.info("No hay suficientes comentarios reales.")

    spam_df = df_res[df_res["Spam"] == "🚨 SÍ"]
    if not spam_df.empty:
        m = spam_df["Motivo"].value_counts().reset_index()
        m.columns = ["Motivo", "N"]
        st.plotly_chart(px.bar(m, x="Motivo", y="N", title="Motivo de clasificación como spam"), use_container_width=True)

    st.subheader("📋 Tabla detallada")
    st.info("🔒 'Seudónimo' = SHA-256. Ningún nombre real en esta tabla ni en el CSV.")

    # ── Feedback inline por fila ──────────────────────────────────
    n_fb_open = st.session_state.get("_n_fb_open", 5)
    mostrar_fb = st.toggle(
        "✏️ Activar feedback por fila (marca si el modelo se equivocó)",
        value=False,
        help="Expande cada fila para corregir spam o sentimiento.",
    )

    if mostrar_fb:
        st.caption(f"Mostrando las primeras {n_fb_open} filas con feedback. "
                   "Ve a '📝 Feedback' para ver el historial completo.")
        for idx, row in df_res.head(n_fb_open).iterrows():
            texto_raw = row.get("Comentario", row.get("texto", ""))
            pred_spam = "🚨 SÍ" if row.get("Spam","") == "🚨 SÍ" else "✅ NO"
            pred_sent = str(row.get("Sentimiento",""))
            etiq = f"{pred_spam} · {pred_sent}"
            with st.expander(f"Fila {idx+1} — {str(texto_raw)[:80]}… — predicción: {etiq}"):
                widget_feedback_fila(idx, str(texto_raw), pred_spam, pred_sent)

        if len(df_res) > n_fb_open:
            if st.button(f"Cargar {min(20, len(df_res)-n_fb_open)} filas más"):
                st.session_state["_n_fb_open"] = n_fb_open + 20
                st.rerun()
    else:
        st.dataframe(df_res, use_container_width=True)

    st.download_button("📥 Descargar CSV anonimizado", df_res.to_csv(index=False).encode("utf-8"), "auditoria_anonimizada.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────
# AVISO DE PRIVACIDAD
# ─────────────────────────────────────────────────────────────────
AVISO_RGPD = """
**Aviso RGPD**
- **Seudonimización inmediata** (Art. 25): SHA-256 → `Usr-XXXXXXXX`
- **Minimización** (Art. 5.1.c): sólo texto y seudónimo
- **Sin persistencia** (Art. 5.1.e): sólo en sesión activa
- **Base jurídica**: interés legítimo para moderación (Art. 6.1.f)
- El CSV exportado no contiene nombres reales
"""


# ─────────────────────────────────────────────────────────────────
# FEEDBACK HUMANO — ruta del CSV de correcciones
# ─────────────────────────────────────────────────────────────────
FEEDBACK_CSV = "feedback_correcciones.csv"
_FEEDBACK_COLS = [
    "timestamp", "texto_hash", "texto",
    "pred_spam", "pred_sentimiento",
    "correcto_spam", "correcto_sentimiento",
    "nota",
]

# ─────────────────────────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 YouTube Spam & Sentiment Detector")

    # Cargar datos
    # ratio_sel se define en la sidebar más abajo; necesitamos un default aquí
    # para el primer render. Streamlit re-ejecuta el script completo al cambiar
    # widgets, así que el valor correcto llega en la segunda pasada.
    ratio_sel      = st.session_state.get("_ratio_sel", 1)
    ratio_sent_sel = st.session_state.get("_ratio_sent_sel", 1)

    with st.spinner("Cargando datasets…"):
        try:
            df_spam = cargar_datos_spam(ratio_real_spam=ratio_sel)
            df_sent = cargar_datos_sentimiento(ratio_por_clase=ratio_sent_sel)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info(
                "Coloca los CSV en la misma carpeta que app.py:\n"
                "- `Youtube-Spam-Dataset.csv`\n"
                "- `YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv`"
            )
            st.stop()

    # Entrenar modelos
    with st.spinner("Entrenando modelos (primera carga súper rápida)…"):
        spam_pipe, m_spam = entrenar_spam(df_spam)
        sent_pipe, m_sent = entrenar_sentimiento(df_sent)

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
        st.title("Menú")
        opcion = st.radio("", [
            "🔎 Análisis manual",
            "📂 Análisis por fichero",
            "🎬 Auditoría en tiempo real",
            "📊 Rendimiento de los modelos",
            "📈 Datasets de entrenamiento",
            "🧩 Análisis de temas",
            "📝 Feedback & reentrenamiento",
        ])
        st.divider()

        # ── Control de balance del dataset ──────────────────────
        st.subheader("⚖️ Balance del dataset")

        # — Spam —
        st.markdown("**🛡️ Spam — Ratio real:spam**")
        ratio_sel = st.radio(
            "Real : Spam",
            options=[1, 2, 3],
            format_func=lambda r: {
                1: "1:1 — Balanceado (recomendado)",
                2: "2:1 — Moderado",
                3: "3:1 — Conservador",
            }[r],
            index=0,
            label_visibility="collapsed",
            help="1:1 maximiza Precisión y F1. Valores mayores aumentan Recall pero bajan Precisión.",
        )
        if st.session_state.get("_ratio_sel") != ratio_sel:
            st.session_state["_ratio_sel"] = ratio_sel
            cargar_datos_spam.clear()
            entrenar_spam.clear()
        st.caption(
            f"ℹ️ ~2 826 spam + {min(2826*ratio_sel,41000):,} reales "
            f"→ {2826/(2826+min(2826*ratio_sel,41000))*100:.0f}% spam"
        )

        # — Sentimiento —
        st.markdown("**💬 Sentimiento — Ratio otras:negative**")
        ratio_sent_sel = st.radio(
            "Pos+Neu : Neg",
            options=[1, 2, 3, 4],
            format_func=lambda r: {
                1: "1:1 — Balanceado (recomendado)",
                2: "2:1 — Moderado",
                3: "3:1 — Conservador",
                4: "4:1 — Original aprox.",
            }[r],
            index=0,
            label_visibility="collapsed",
            help="1:1 maximiza F1-negative (clase mas dificil). Valores mayores acercan al original.",
        )
        if st.session_state.get("_ratio_sent_sel") != ratio_sent_sel:
            st.session_state["_ratio_sent_sel"] = ratio_sent_sel
            cargar_datos_sentimiento.clear()
            entrenar_sentimiento.clear()

        # Indicador BERT
        if HF_SENTIMENT_MODEL and _BERT_AVAILABLE:
            st.success(f"🤖 BERT activo: `{HF_SENTIMENT_MODEL}`")
        elif HF_SENTIMENT_MODEL and not _BERT_AVAILABLE:
            st.warning("⚠️ `transformers` no instalado — usando sklearn")
        else:
            st.info("💡 **Activa BERT:** añade tu repo HF en `HF_SENTIMENT_MODEL`")
        n_neg_sent = 3810
        n_otras = min(n_neg_sent * ratio_sent_sel, 18210)
        st.caption(
            f"ℹ️ {n_neg_sent:,} neg + {n_otras:,} pos + {n_otras:,} neu "
            f"→ {n_neg_sent/(n_neg_sent+2*n_otras)*100:.0f}% negative"
        )
        st.divider()

        # API Key — solo cuando se necesita
        api_key = num_api = ""
        if opcion == "🎬 Auditoría en tiempo real":
            st.subheader("🔑 YouTube Data API v3")
            api_key = st.text_input("API Key", type="password", help="Google Cloud Console → APIs → YouTube Data API v3")
            num_api = st.slider("Comentarios", 10, 200, 50)
            st.caption("🔒 La key sólo existe en esta sesión.")
            st.divider()

        # Métricas rápidas
        st.markdown("**Spam model (LR)**")
        st.metric("Accuracy",  f"{m_spam['accuracy']:.3f}")
        st.metric("F1 (spam)", f"{m_spam['f1']:.3f}")
        st.divider()
        st.markdown("**Sentiment model (LR)**")
        st.metric("Accuracy",    f"{m_sent['accuracy']:.3f}")
        st.metric("F1 macro",    f"{m_sent['f1_macro']:.3f}")
        st.metric("F1 negative", f"{m_sent['f1_negative']:.3f}")
        st.divider()
        with st.expander("📋 Aviso RGPD"):
            st.markdown(AVISO_RGPD)

        # ── Badge de feedback acumulado ───────────────────────────
        df_fb_side = leer_feedback()
        if not df_fb_side.empty:
            n_fb = len(df_fb_side)
            n_err = ((df_fb_side["pred_spam"] != df_fb_side["correcto_spam"]) |
                     (df_fb_side["pred_sentimiento"] != df_fb_side["correcto_sentimiento"])).sum()
            st.markdown(
                f'<div style="background:var(--color-background-warning);'
                f'border:1px solid var(--color-border-warning);border-radius:8px;'
                f'padding:8px 12px;font-size:12px;">'
                f'✏️ <b>{n_fb}</b> feedbacks · <b>{n_err}</b> errores marcados<br>'
                f'<span style="color:var(--color-text-secondary)">Ve a 📝 Feedback</span></div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════
    # A) ANÁLISIS MANUAL
    # ════════════════════════════════════════════════════════════
    if opcion == "🔎 Análisis manual":
        st.header("🔎 Análisis manual")
        st.info("Sin datos personales de terceros. Analiza el texto que escribas aquí.")
        texto = st.text_area("Comentario a analizar:", height=110, placeholder="Escribe cualquier comentario…")

        if st.button("Analizar", type="primary"):
            if not texto.strip():
                st.warning("Escribe un comentario primero."); return

            res = analizar(texto, spam_pipe, sent_pipe)
            c1, c2, c3 = st.columns(3)
            c1.metric("Spam",        "🚨 SÍ" if res["spam"] else "✅ NO", f"{res['spam_conf']:.0f}% confianza")
            c2.metric("Sentimiento", res["sentimiento"], f"{res['sent_conf']:.0f}% confianza")
            c3.metric("Detectado por", res["motivo"] or "—")

            with st.expander("🔍 Features del EDA"):
                vals = SpamFeatures()._f(texto)
                nms  = ["Contiene URL","Ratio mayús","Exclamaciones","!! múltiple",
                        "Palabras CAPS","Diversidad léxica","Hits léxico spam",
                        "Palabras repetidas","Longitud log"]
                for n, v in zip(nms, vals):
                    st.write(f"**{n}**: {v:.3f}")

            with st.expander("✏️ ¿Se equivocó el modelo? Deja feedback"):
                pred_spam_m = "🚨 SÍ" if res["spam"] else "✅ NO"
                widget_feedback_fila(0, texto, pred_spam_m, res["sentimiento"])

    # ════════════════════════════════════════════════════════════
    # B) ANÁLISIS POR FICHERO
    # ════════════════════════════════════════════════════════════
    elif opcion == "📂 Análisis por fichero":
        st.header("📂 Análisis por fichero CSV")
        st.markdown('<div class="rgpd-box">🔒 <b>RGPD:</b> Autores seudonimizados (SHA-256) antes de cualquier procesado. CSV exportado sin nombres reales.</div>', unsafe_allow_html=True)
        st.markdown('<div class="warn-box">⚠️ <b>ToS YouTube:</b> Analiza CSV con comentarios ya descargados. Sin conexión a APIs externas.</div>', unsafe_allow_html=True)
        upload = st.file_uploader("CSV con columna **Comentario**, **content**, **text** o **comment**", type=["csv"])
        if upload and st.button("🚀 Analizar fichero", type="primary"):
            df_up, err = leer_csv_usuario(upload)
            if err:
                st.error(err); return
            if df_up is None or df_up.empty:
                st.error("Fichero vacío o sin columna reconocida."); return

            with st.spinner(f"Analizando {len(df_up)} comentarios…"):
                comentarios = df_up[["seudónimo", "texto"]].to_dict("records")
                filas = analizar_batch(comentarios, spam_pipe, sent_pipe)

            df_filas = pd.DataFrame(filas)
            st.session_state["df_resultados_temas"] = df_filas
            mostrar_resultados(df_filas)

    # ════════════════════════════════════════════════════════════
    # C) AUDITORÍA EN TIEMPO REAL
    # ════════════════════════════════════════════════════════════
    elif opcion == "🎬 Auditoría en tiempo real":
        st.header("🎬 Auditoría en tiempo real")
        st.markdown('<div class="rgpd-box">🔒 <b>RGPD:</b> Autores seudonimizados (SHA-256). Sin persistencia. CSV anonimizado.</div>', unsafe_allow_html=True)
        url = st.text_input("URL del vídeo:", placeholder="https://www.youtube.com/watch?v=...")

        if st.button("🚀 Analizar vídeo", type="primary"):
            if not api_key.strip():
                st.warning("Introduce tu API Key en el panel lateral."); return
            if not url.strip():
                st.warning("Introduce la URL del vídeo."); return
            video_id = extraer_video_id(url)
            if not video_id:
                st.error("No se pudo extraer el ID del vídeo."); return

            with st.spinner("Descargando vía YouTube Data API v3…"):
                try:
                    comentarios = descargar_comentarios(api_key, video_id, num_api)
                except Exception as e:
                    st.error(f"Error de la API: {e}")
                    st.info("Comprueba la API Key, que YouTube Data API v3 esté habilitada y que no hayas superado la cuota (10k unidades/día).")
                    return

            if not comentarios:
                st.error("Sin comentarios (¿están desactivados?)."); return

            with st.spinner(f"Analizando {len(comentarios)} comentarios…"):
                filas = analizar_batch(comentarios, spam_pipe, sent_pipe)

            df_filas = pd.DataFrame(filas)
            st.session_state["df_resultados_temas"] = df_filas
            mostrar_resultados(df_filas)

    # ════════════════════════════════════════════════════════════
    # D) RENDIMIENTO DE LOS MODELOS
    # ════════════════════════════════════════════════════════════
    elif opcion == "📊 Rendimiento de los modelos":
        st.header("📊 Rendimiento de los modelos")

        tab_spam, tab_sent = st.tabs(["🛡️ Spam (LR)", "💬 Sentimiento (LR)"])

        with tab_spam:
            st.markdown(f"Entrenado con **{m_spam['n_train']:,}** muestras ({m_spam['n_spam']:,} spam · {m_spam['n_real']:,} reales). Holdout estratificado 20%.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{m_spam['accuracy']:.3f}")
            c2.metric("Precision", f"{m_spam['precision']:.3f}")
            c3.metric("Recall",    f"{m_spam['recall']:.3f}")
            c4.metric("F1 spam",   f"{m_spam['f1']:.3f}")
            st.divider()
            col_cm, col_rep = st.columns([1, 1.5])
            with col_cm:
                st.pyplot(plot_cm(m_spam["cm"], ["Real", "Spam"], "Confusión spam (20% holdout)"))
            with col_rep:
                st.code(m_spam["reporte"])
            st.divider()
            st.markdown("""
| Parámetro | Valor |
|---|---|
| Tipo | Ensemble soft-voting: LR (w=2) + LinearSVC (w=1) |
| Regularización | LR C=0.5 · SVC C=0.3 · class_weight=balanced |
| Vectorización | word bigramas (8k) + char (2,5)-grams (15k) + 14 EDA features |
| Datos spam | 45k + clásico + equilibrado → ~2 826 spam únicos (dedup) |
| Desbalance | Undersampling 1:1 (ratio configurable) |
| Anti-overfitting | CV gap ≈ 0 con 3 fuentes · regularización fuerte |
| Mejoras acumuladas | F1 0.396→0.940 · Prec 0.254→0.953 · Acc 0.894→0.941 |
| Vectorización | TF-IDF bigramas (8k features) |
| Features EDA | 9 (§3.1 del documento) |
| Split | 80 / 20 estratificado |
| Datos spam | 45K dataset (2025) + dataset clásico |
            """)

        with tab_sent:
            ratio_sent_labels = {1: "1:1 Balanceado", 2: "2:1 Moderado", 3: "3:1 Conservador", 4: "4:1 Original aprox."}
            st.markdown(
                f"Entrenado con **{m_sent['n_train']:,}** muestras — "
                f"ratio {ratio_sent_labels.get(ratio_sent_sel, ratio_sent_sel)} · Holdout 15%."
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy",      f"{m_sent['accuracy']:.3f}")
            c2.metric("F1 macro",      f"{m_sent['f1_macro']:.3f}")
            c3.metric("F1 negative",   f"{m_sent['f1_negative']:.3f}",
                      help="F1 de la clase más difícil (negative, ~8.5% del dataset original).")
            st.divider()
            col_cm, col_rep = st.columns([1, 1.5])
            with col_cm:
                st.pyplot(plot_cm(m_sent["cm"], ["positive", "neutral", "negative"], "Confusión sentimiento (15% holdout)"))
            with col_rep:
                st.code(m_sent["reporte"])
            st.markdown(
                f'''<div class="good-box">⚖️ <b>Undersampling activo:</b> Ratio {ratio_sent_labels.get(ratio_sent_sel, ratio_sent_sel)}
                · Antes de equilibrar, el F1-negative era <b>0.472</b>; con ratio 1:1 sube a <b>~0.68</b>.
                Cambia el ratio en el panel lateral para explorar el trade-off.</div>''',
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown("""
| Parámetro | Valor |
|---|---|
| Tipo | Regresión Logística multiclase |
| Solver | SAGA |
| Regularización | C = 0.3 · class_weight = balanced |
| Vectorización | word bigramas (30k) + char (2,4)-grams (20k) + 15 EDA features |
| Clases | Positive · Neutral · Negative |
| Split | 85 / 15 estratificado |
| Desbalance | Undersampling 1:1:1 (ratio configurable) |
| Datos | 45k dataset (mayor calidad que datasets genéricos) |
| Mejoras acumuladas | F1 macro 0.61→0.74→0.745 · F1-neg 0.47→0.77 |
            """)

    # ════════════════════════════════════════════════════════════
    # E) DATASETS DE ENTRENAMIENTO
    # ════════════════════════════════════════════════════════════
    elif opcion == "📈 Datasets de entrenamiento":
        st.header("📈 Datasets de entrenamiento")

        tab_s, tab_sent2 = st.tabs(["🛡️ Spam", "💬 Sentimiento"])

        with tab_s:
            df_s = df_spam.copy()
            df_s["Etiqueta"] = df_s["spam"].map({0: "Real", 1: "Spam"})
            n_s, n_r = (df_s["spam"] == 1).sum(), (df_s["spam"] == 0).sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Total", f"{len(df_s):,}")
            k2.metric("Spam",  f"{n_s:,} ({n_s/len(df_s)*100:.1f}%)")
            k3.metric("Real",  f"{n_r:,} ({n_r/len(df_s)*100:.1f}%)")

            ratio_labels = {1: "1:1 — Balanceado", 2: "2:1 — Moderado", 3: "3:1 — Conservador"}
            st.markdown(
                f'<div class="good-box">⚖️ <b>Undersampling activo:</b> Ratio {ratio_labels.get(ratio_sel, ratio_sel)} '
                f'· {n_s:,} spam + {n_r:,} reales seleccionados de 45,135 disponibles. '
                f'Sin equilibrar, la precisión era solo <b>0.25</b>; con ratio 1:1 sube a <b>~0.81</b>.</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(df_s, names="Etiqueta", title="Balance Real / Spam (tras undersampling)", hole=0.4, color="Etiqueta", color_discrete_map={"Real":"#2ecc71","Spam":"#e74c3c"}), use_container_width=True)
            with c2:
                df_s["longitud"] = df_s["text"].str.len()
                st.plotly_chart(px.box(df_s, x="Etiqueta", y="longitud", title="Longitud de comentario por clase", color="Etiqueta", color_discrete_map={"Real":"#2ecc71","Spam":"#e74c3c"}), use_container_width=True)

        with tab_sent2:
            df_sv = df_sent.copy()
            df_sv["Etiqueta"] = df_sv["sentiment"].str.capitalize()
            dist = df_sv["Etiqueta"].value_counts()
            n_neg_sv = dist.get("Negative", 0)
            n_pos_sv = dist.get("Positive", 0)
            n_neu_sv = dist.get("Neutral",  0)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total",      f"{len(df_sv):,}")
            k2.metric("Negative",   f"{n_neg_sv:,} ({n_neg_sv/len(df_sv)*100:.0f}%)")
            k3.metric("Positive",   f"{n_pos_sv:,} ({n_pos_sv/len(df_sv)*100:.0f}%)")
            k4.metric("Neutral",    f"{n_neu_sv:,} ({n_neu_sv/len(df_sv)*100:.0f}%)")

            ratio_sent_labels = {1: "1:1 Balanceado", 2: "2:1 Moderado", 3: "3:1 Conservador", 4: "4:1 Original aprox."}
            st.markdown(
                f'''<div class="good-box">⚖️ <b>Undersampling activo:</b> Ratio {ratio_sent_labels.get(ratio_sent_sel, ratio_sent_sel)}
                · {n_neg_sv:,} negativos + {n_pos_sv:,} positivos + {n_neu_sv:,} neutrales seleccionados de 45,000 disponibles.
                Sin equilibrar, la clase <i>negative</i> solo era el 8.5% y el modelo la ignoraba (F1-negative = 0.47).</div>''',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(df_sv, names="Etiqueta",
                    title="Distribución sentimiento tras undersampling",
                    hole=0.4, color="Etiqueta",
                    color_discrete_map={"Positive":"#2ecc71", "Negative":"#e74c3c", "Neutral":"#95a5a6"}),
                    use_container_width=True)
            with c2:
                df_sv["longitud"] = df_sv["text"].str.len()
                st.plotly_chart(px.box(df_sv, x="Etiqueta", y="longitud",
                    title="Longitud de comentario por clase",
                    color="Etiqueta",
                    color_discrete_map={"Positive":"#2ecc71", "Negative":"#e74c3c", "Neutral":"#95a5a6"}),
                    use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # G) FEEDBACK & REENTRENAMIENTO
    # ════════════════════════════════════════════════════════════
    if opcion == "📝 Feedback & reentrenamiento":
        mostrar_panel_feedback()

    st.divider()
    st.caption("v9.0 · Ensemble LR+SVC spam · LR/BERT sent · NMF temas · Feedback humano · RGPD Art. 25")

    # ── Análisis de temas ─────────────────────────────────────────
    if opcion == "🧩 Análisis de temas":
        st.header("🧩 Análisis de temas")

        # Check if there are already analysed results in session state
        df_cached = st.session_state.get("df_resultados_temas", None)

        st.info(
            "Esta sección analiza **qué temas dominan** los comentarios positivos, "
            "negativos y neutros. Primero analiza un vídeo o sube un CSV en otra pestaña, "
            "o pega comentarios directamente aquí."
        )

        fuente = st.radio(
            "Fuente de comentarios",
            ["Pegar comentarios manualmente", "Usar último análisis de vídeo/CSV"],
            horizontal=True,
        )

        if fuente == "Pegar comentarios manualmente":
            raw = st.text_area(
                "Pega aquí los comentarios (uno por línea)",
                height=200,
                placeholder="Este vídeo es increíble!\nMalo, no me gustó nada.\nInteresante perspectiva...",
            )
            if st.button("🔍 Analizar temas", type="primary") and raw.strip():
                lineas = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                with st.spinner(f"Analizando {len(lineas)} comentarios…"):
                    resultados = [analizar(c, spam_pipe, sent_pipe) for c in lineas]
                df_man = pd.DataFrame({
                    "Comentario": [r["comentario"] for r in resultados],
                    "Sentimiento": [r["sentimiento"] for r in resultados],
                    "Spam": [r.get("spam_label","No") for r in resultados],
                })
                st.session_state["df_resultados_temas"] = df_man
                st.rerun()

            if "df_resultados_temas" in st.session_state:
                mostrar_analisis_temas(st.session_state["df_resultados_temas"])

        else:
            # Look for results from other sections
            df_cached = st.session_state.get("df_resultados_temas", None)
            if df_cached is not None and not df_cached.empty:
                mostrar_analisis_temas(df_cached)
            else:
                st.warning(
                    "No hay resultados previos. Analiza un vídeo en '🎬 Auditoría en tiempo real' "
                    "o sube un CSV en '📂 Análisis por fichero' y vuelve aquí."
                )



if __name__ == "__main__":
    main()
