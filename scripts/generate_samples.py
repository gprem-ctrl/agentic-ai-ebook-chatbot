"""
Runs the sample questions against the live API and writes SAMPLE_QUERIES.md
with the real answers and scores.

Start the server first:   uvicorn app.api:app --port 8000
Then run:                 python scripts/generate_samples.py
"""
import requests

API_URL = "http://localhost:8000/chat"

QUERIES = [
    ("In-scope: definition", "What is Agentic AI?"),
    ("In-scope: comparison", "How is Agentic AI different from a regular LLM?"),
    ("In-scope: structure", "What are the core components of an Agentic AI system?"),
    ("In-scope: use case", "What are some real-world use cases of Agentic AI in manufacturing?"),
    ("In-scope: challenges", "What challenges do multi-agent systems face, and how can they be mitigated?"),
    ("Out-of-scope (should refuse)", "What is the capital of France?"),
]

lines = [
    "# Sample Queries",
    "",
    "Real outputs from the running API. Wording may vary slightly between runs "
    "because generation uses an LLM.",
    "",
]

for i, (label, question) in enumerate(QUERIES, 1):
    try:
        resp = requests.post(API_URL, json={"question": question}, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Request failed for {question!r}: {e}\n"
                         "Is the server running? (uvicorn app.api:app --port 8000)")
    data = resp.json()
    context = data.get("retrieved_context", [])

    lines += [
        f"## {i}. {label}",
        "",
        f"**Q:** {question}",
        "",
        f"**Answer:** {data.get('answer', '')}",
        "",
        f"**Confidence:** {data.get('confidence')}",
        "",
        f"**Retrieved chunks:** {len(context)}",
        "",
    ]
    if context:
        top = context[0]
        snippet = " ".join(top["text"].split())[:400]
        lines += [
            "<details><summary>Top chunk "
            f"(page {top.get('page')}, score {top.get('score', 0):.3f})</summary>",
            "",
            f"> {snippet}...",
            "",
            "</details>",
            "",
        ]

lines += [
    "---",
    "",
    "Reproduce any of these:",
    "",
    "```bash",
    "curl -X POST http://localhost:8000/chat \\",
    '  -H "Content-Type: application/json" \\',
    """  -d '{"question": "What are the core components of an Agentic AI system?"}'""",
    "```",
    "",
]

with open("SAMPLE_QUERIES.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Wrote SAMPLE_QUERIES.md")
