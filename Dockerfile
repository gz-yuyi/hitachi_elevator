FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml uv.lock .

RUN pip install --upgrade pip && \
    pip install uv && \
    uv sync --frozen

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=info

EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "src.app:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
