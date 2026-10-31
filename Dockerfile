FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Run as non-root user so bind-mounted host volumes are not polluted with root-owned files
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/raw /app/data/processed /app/data/failed \
                /app/output/predictions /app/output/analytics \
                /app/checkpoints /app/logs \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

CMD ["python", "main.py"]
