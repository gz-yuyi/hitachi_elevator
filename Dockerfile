FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml uv.lock .

RUN pip install --upgrade pip && \
    pip install uv && \
    uv sync --frozen

# Prefer the project venv binaries (python, uvicorn, etc.)
ENV PATH="/app/.venv/bin:${PATH}"

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
