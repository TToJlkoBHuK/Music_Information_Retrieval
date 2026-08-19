# Окружение для запуска и проверки модулей проекта.
# FFmpeg внутри — единственная системная зависимость этапа 2.

FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MIR_INGEST_CACHE_DIR=/cache

WORKDIR /app

# Зависимости ставятся до копирования кода: слой переиспользуется,
# пока pyproject.toml не изменился
COPY pyproject.toml README.md ./
COPY mir/__init__.py mir/
RUN pip install -e ".[dev,docs]"

COPY . .

VOLUME ["/cache", "/app/data"]

CMD ["python", "-m", "scripts.ingest_cli", "--help"]
