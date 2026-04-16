"""AEP SDK — governance and provenance example.

Shows how to use the Python SDK for full Layer 2 features:
- Provenance tracking (what sources informed the output)
- Governance enforcement (data classification, consent, redaction)

Prerequisites:
    pip install aceteam-aep[safety] openai
    export OPENAI_API_KEY=sk-...
"""

import openai

from aceteam_aep import wrap
from aceteam_aep.governance_engine import (
    DataClassifier,
    GovernancePolicyEngine,
    OrgPolicy,
    Redactor,
)
from aceteam_aep.provenance import ProvenanceTracker, extract_sources_from_messages

# --- Setup ---

client = wrap(openai.OpenAI(), entity="org:acme")

# Governance: define org policies
policy_engine = GovernancePolicyEngine()
policy_engine.add_policy(
    OrgPolicy(
        entity="org:acme",
        max_classification="confidential",  # can access up to confidential
        consent={"training": False, "sharing": True},
    )
)

classifier = DataClassifier()
redactor = Redactor()
provenance = ProvenanceTracker()

# --- Simulate an agent workflow ---

messages = [
    {"role": "system", "content": "You are a financial analyst. " + "x" * 500},
    {"role": "user", "content": "What was the Q3 revenue? My SSN is 123-45-6789"},
]

# Step 1: Extract provenance (what sources are in context?)
sources = extract_sources_from_messages(messages)
print(f"Sources in context: {len(sources)}")
for src in sources:
    print(f"  [{src.source_type}] {src.source_id}")

# Step 2: Classify data sensitivity
classification = classifier.classify_text(messages[-1]["content"])
print(f"\nData classification: {classification}")

# Step 3: Check governance policy
decision = policy_engine.evaluate("org:acme", classification)
print(f"Governance decision: {decision.action}")
if decision.action == "deny":
    print(f"  Reason: {decision.reason}")
    print("  BLOCKED by governance policy")
    exit(1)

# Step 4: Make the LLM call (with AEP wrapping)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

output = response.choices[0].message.content
print(f"\nRaw output: {output[:100]}...")

# Step 5: Redact if classification requires it
redacted_output, redaction_log = redactor.redact_if_needed(output, classification)
if redaction_log:
    print("\nRedactions applied:")
    for r in redaction_log:
        print(f"  {r}")
    print(f"\nRedacted output: {redacted_output[:100]}...")

# Step 6: Record provenance
provenance.record_sources("call-1", sources)

# Step 7: Print AEP summary
client.aep.print_summary()

print(f"\nProvenance: {provenance.source_count} sources tracked")
print(f"Citations: {len(provenance.get_all_citations())}")
