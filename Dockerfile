# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN uv sync --frozen --no-dev \
    && addgroup --system gateway \
    && adduser --system --ingroup gateway --home /app gateway \
    && mkdir -p /app/artifacts \
    && chown -R gateway:gateway /app \
    && chmod +x /app/docker/entrypoint.sh

USER gateway

EXPOSE 8000
VOLUME ["/app/artifacts"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["inference-gateway", "serve", "--config", "configs/default.yaml"]
