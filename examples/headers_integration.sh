#!/usr/bin/env bash
# AEP Headers Integration — inject governance context from any language
#
# This example shows how to use X-AEP-* headers with the proxy
# to get Layer 2 features (governance, provenance) without the Python SDK.
#
# Works with: curl, Node.js, Go, Ruby, Java — any HTTP client.
#
# Prerequisites:
#   pip install aceteam-aep[all]
#   aceteam-aep proxy --port 8080   (in another terminal)

set -e

PROXY="http://localhost:8080"
API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

echo "=== Example 1: Basic call (Layer 1 only) ==="
echo "No AEP headers — just cost tracking + safety"
curl -s "$PROXY/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }' | python -m json.tool
echo ""

echo "=== Example 2: With governance headers (Layer 2) ==="
echo "Entity + classification + consent + budget"
curl -sv "$PROXY/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-AEP-Entity: org:acme" \
  -H "X-AEP-Classification: confidential" \
  -H "X-AEP-Consent: training=no,sharing=org-only" \
  -H "X-AEP-Budget: 1.00" \
  -H "X-AEP-Sources: doc:financial-report-2026-q1" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a financial analyst."},
      {"role": "user", "content": "Summarize the quarterly revenue trends."}
    ]
  }' 2>&1 | grep -E "^< X-AEP-|^{" | head -10
echo ""

echo "=== Example 3: Budget enforcement ==="
echo "Setting a $0.001 budget — should block expensive calls"
curl -s "$PROXY/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-AEP-Entity: org:cheapskate" \
  -H "X-AEP-Budget: 0.001" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a 2000 word essay about AI safety."}]
  }' | python -m json.tool
echo ""

echo "=== Dashboard ==="
echo "Open http://localhost:8080/aep/ to see all calls in real-time"
