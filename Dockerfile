# Multi-stage build: resolve deps against the lockfile in a builder image, then
# copy the resolved environment into a slim runtime. The runtime image carries
# only the Python interpreter, the project and its locked dependencies.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

ADD --chmod=755 https://astral.sh/uv/install.sh /uv-install.sh
RUN /uv-install.sh && mv /root/.local/bin/uv /usr/local/bin/uv && rm /uv-install.sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY prompts ./prompts
COPY configs ./configs

RUN uv sync --frozen --extra asr --extra diar --extra nlp --extra service --extra report


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "speechinsight.service.pipeline:build_default_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
