"""AEP wrapper quickstart — run with: python examples/quickstart.py

Requires: pip install aceteam-aep[safety] openai
Set OPENAI_API_KEY in your environment.
"""

import openai

from aceteam_aep import wrap

# One line to add safety + cost tracking to any OpenAI client
client = wrap(openai.OpenAI(), entity="demo")

# Use exactly as before — AEP intercepts transparently
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.choices[0].message.content)

# See what AEP tracked
client.aep.print_summary()

# To launch the web dashboard:
# client.aep.serve_dashboard()  # opens http://localhost:8899
