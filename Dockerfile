FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install with proxy + safety extras
RUN uv pip install --system ".[proxy,safety]"

EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8899/aep/api/state')" || exit 1

ENTRYPOINT ["aceteam-aep"]
CMD ["proxy", "--port", "8899"]
