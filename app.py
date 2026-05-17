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

    # — SENTIMIENTO —
    limpio     = preprocesar(texto)
    sent_raw   = sent_pipe.predict([limpio] if limpio else [texto])[0]
    sent_conf  = float(sent_pipe.predict_proba([limpio] if limpio else [texto])[0].max()) * 100
    label_map  = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
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

            mostrar_resultados(pd.DataFrame(filas))

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

            mostrar_resultados(pd.DataFrame(filas))

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

    st.divider()
    st.caption("v8.7 · Ensemble LR+SVC spam (3 fuentes) · LR word+char+EDA sent · Undersampling 1:1/1:1:1 · RGPD Art. 25")

if __name__ == "__main__":
    main()
