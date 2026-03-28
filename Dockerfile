FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install with proxy + safety extras
RUN uv pip install --system ".[proxy,safety]"

# Pre-download safety models so first call isn't slow (~235MB)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('iiiorg/piiranha-v1-detect-personal-information'); \
AutoModelForTokenClassification.from_pretrained('iiiorg/piiranha-v1-detect-personal-information'); \
AutoTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier'); \
AutoModelForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier'); \
print('Models cached')" 2>/dev/null || echo "Model pre-download skipped (no torch)"

EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/aep/')" || exit 1

ENTRYPOINT ["aceteam-aep"]
CMD ["proxy", "--port", "8899", "--host", "0.0.0.0"]
