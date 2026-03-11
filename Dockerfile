# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.13 slim — matches the pyproject.toml constraint
FROM python:3.13-slim

# ── System deps ───────────────────────────────────────────────────────────────
# libgl1 + libglib2.0 are required by OpenCV (pulled in transitively by TF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy only dependency files first to leverage Docker layer caching
COPY pyproject.toml ./
RUN uv sync --no-dev

# ── Copy application code ─────────────────────────────────────────────────────
COPY main.py feedback_store.py ./
COPY models/ ./models/

# ── Streamlit config ──────────────────────────────────────────────────────────
RUN mkdir -p /app/.streamlit
RUN echo '\
[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uv", "run", "streamlit", "run", "main.py"]
