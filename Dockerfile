# ── Imagen base ligera ────────────────────────────────────────────────────────
FROM python:3.10-slim

# Metadatos
LABEL maintainer="Proyecto IA UPV"
LABEL description="YouTube Spam Detector — Red Neuronal Superficial"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Dependencias del sistema ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Código fuente y datos ─────────────────────────────────────────────────────
COPY app.py           .
COPY *.csv            ./
COPY model/           ./model/

# Crear directorio model si no existe (para montaje de volumen)
RUN mkdir -p model

# ── Puerto Streamlit ──────────────────────────────────────────────────────────
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Comando de inicio ─────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
