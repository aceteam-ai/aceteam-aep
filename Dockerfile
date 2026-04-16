FROM python:3.12-slim AS base

WORKDIR /app

# programasweights -> llama-cpp-python may build from source when no wheel matches;
# CMake needs a C/C++ toolchain (slim images omit it by default).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install with proxy
RUN uv pip install --system ".[proxy]"

EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/dashboard/')" || exit 1

ENTRYPOINT ["aceteam-aep"]
CMD ["proxy", "--port", "8899", "--host", "0.0.0.0"]
