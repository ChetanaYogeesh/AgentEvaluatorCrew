# ─────────────────────────────────────────────────────────────────
# Stage 1 — dependency builder
# Resolves and wheels all packages so the final stage stays lean.
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ─────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Build-time labels (populated by docker/build-push-action)
ARG BUILD_DATE
ARG GIT_SHA
LABEL org.opencontainers.image.title="Agent Evaluator Crew Dashboard" \
      org.opencontainers.image.description="Streamlit evaluation dashboard for CrewAI multi-agent workflows" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install wheels from builder (no internet needed)
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
 && rm -rf /wheels

# Copy application source
COPY dashboard.py          .
COPY tools.py              .
COPY eval_simple.py        .
COPY eval_crew.py          .
COPY agents.yaml           config/agents.yaml
COPY tasks.yaml            config/tasks.yaml
COPY .streamlit/           .streamlit/

# Placeholder result files so the dashboard starts without errors
RUN echo "[]" > evaluation_history.json \
 && echo "{}" > evaluation_results.json

# Hand off to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Healthcheck — Streamlit exposes a /_stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
