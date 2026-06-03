# ── Stage 1: Python base image ──────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="EV Fleet Analytics Team"
LABEL description="EV Fleet Telemetry Analytics & Battery Health Predictive Dashboard"
LABEL version="1.0.0"

# ── Environment variables ────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── System dependencies ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (layer-cached) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copy application source ──────────────────────────────────────────────────────
COPY . .

# ── Create output directories ────────────────────────────────────────────────────
RUN mkdir -p /app/models /app/.streamlit

# ── Streamlit config ─────────────────────────────────────────────────────────────
RUN echo '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n\n[theme]\nbase = "dark"\nprimaryColor = "#00d4ff"\nbackgroundColor = "#0a0e1a"\nsecondaryBackgroundColor = "#0d1b2a"\ntextColor = "#ffffff"\nfont = "sans serif"' \
    > /app/.streamlit/config.toml

# ── Setup entrypoint script ───────────────────────────────────────────────────────
RUN printf '#!/bin/bash\nset -e\necho "⚡ EV Fleet Telemetry Dashboard — Initializing..."\n\n# Step 1: Generate + ingest telemetry data (if DB does not exist)\nif [ ! -f /app/fleet_telemetry.db ]; then\n  echo "📊 Generating fleet telemetry data..."\n  python telemetry_generator.py\nelse\n  echo "✅ Database found, skipping data generation."\nfi\n\n# Step 2: Train ML models (if models do not exist)\nif [ ! -f /app/models/soh_regressor.pkl ]; then\n  echo "🤖 Training ML models..."\n  python ml_model.py\nelse\n  echo "✅ ML models found, skipping training."\nfi\n\n# Step 3: Launch dashboard\necho "🚀 Launching Streamlit dashboard on port 8501..."\nexec streamlit run app.py\n' \
    > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# ── Expose port ──────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run ──────────────────────────────────────────────────────────────────────────
ENTRYPOINT ["/app/entrypoint.sh"]
