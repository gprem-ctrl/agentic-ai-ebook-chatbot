# Sample Queries

These illustrate expected behavior once the index is populated (`python -m app.ingest`)
and `PINECONE_API_KEY` / `GROQ_API_KEY` are set. Actual wording will vary slightly
run to run since generation uses an LLM, but grounding and confidence should hold.

### 1. In-scope, direct definition
**Q:** What is Agentic AI?
**Expected:** A grounded definition describing autonomous, goal-driven systems that
understand context, break down goals, and take initiative — citing the "Introduction
to Agentic AI" chapter (page ~2-3). High confidence (~0.6-0.8+).

### 2. In-scope, comparison
**Q:** How is Agentic AI different from a regular LLM?
**Expected:** Answer drawing on the LLM-vs-Agent comparison table (interactivity,
decision-making, dependency, primary function). High confidence.

### 3. In-scope, structural detail
**Q:** What are the core components of an Agentic AI system?
**Expected:** Perception, Reasoning, Planning, Learning, Verification, Execution
(the six "core pillars"), each with its one-line description. High confidence.

### 4. In-scope, use-case retrieval
**Q:** What are some real-world use cases of Agentic AI in manufacturing?
**Expected:** Predictive maintenance and supply chain optimization details, plus
the Factory 4.0 case study (40 assembly lines, Snowflake data extraction). High
confidence.

### 5. In-scope, challenges
**Q:** What challenges do multi-agent systems face, and how can they be mitigated?
**Expected:** Table of challenges (complex system design, interoperability, data
security, conflict resolution, scalability, cost, slow development, orchestration)
paired with their mitigation strategies. High confidence.

### 6. Out-of-scope (should refuse, not hallucinate)
**Q:** What is the capital of France?
**Expected:** The bot should NOT answer from general knowledge. It should return
the out-of-scope message (`OUT_OF_SCOPE_MESSAGE` in `config.py`) with low
confidence (below `MIN_CONFIDENCE_THRESHOLD`, default 0.35), because nothing in
the eBook is relevant to this question.

---

To try these yourself once the server is running:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the core components of an Agentic AI system?"}'
```
