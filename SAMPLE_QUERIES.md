# Sample Queries

Real outputs from the running API. Wording may vary slightly between runs because generation uses an LLM.

## 1. In-scope: definition

**Q:** What is Agentic AI?

**Answer:** Agentic AI refers to systems capable of autonomous decision‑making and action in pursuit of specific objectives. (Page 18)

**Confidence:** 0.8169

**Retrieved chunks:** 8

<details><summary>Top chunk (page 3, score 0.817)</summary>

> Agentic AI An Executive's Guide to In-depth Understanding of Agentic AI...

</details>

## 2. In-scope: comparison

**Q:** How is Agentic AI different from a regular LLM?

**Answer:** Agentic AI differs from a regular LLM in that it is **goal‑driven and autonomous** rather than merely reactive. While an LLM processes and generates text, an agentic system can perceive its environment, reason, plan, learn, and execute actions to achieve a specified goal, often integrating multiple tools and learning from experience (Page 9, Page 10).

**Confidence:** 0.7772

**Retrieved chunks:** 8

<details><summary>Top chunk (page 9, score 0.777)</summary>

> with structured data, like following a strict recipe. Agentic AI adapts to different situations, handling unstructured inputs, much like a chef improvising with available ingredients. LLMs vs. Agentic AI: Distinct Roles, Complementary Strengths Agents are more than LLMs. While LLMs are powerful tools for processing and generating human-like text, agents are goal-driven systems capable of performin...

</details>

## 3. In-scope: structure

**Q:** What are the core components of an Agentic AI system?

**Answer:** The core components of an Agentic AI system are:

- Perception  
- Reasoning  
- Planning  
- Learning  
- Execution  

(See Page 17)

**Confidence:** 0.7481

**Retrieved chunks:** 8

<details><summary>Top chunk (page 3, score 0.748)</summary>

> Agentic AI An Executive's Guide to In-depth Understanding of Agentic AI...

</details>

## 4. In-scope: use case

**Q:** What are some real-world use cases of Agentic AI in manufacturing?

**Answer:** Real‑world manufacturing use cases include:

- **Predictive Maintenance** – AI agents monitor equipment health, predict potential failures, and schedule maintenance to reduce downtime and repair costs. (Page 14)  
- **Supply Chain Optimization** – AI manages inventory, tracks shipments, and adjusts delivery routes in real‑time, improving operational efficiency and reducing costs. (Page 14)

**Confidence:** 0.7471

**Retrieved chunks:** 8

<details><summary>Top chunk (page 54, score 0.747)</summary>

> PRACTICAL APPLICATIONS OF AGENTIC AI Agentic AI is transforming industries by solving critical challenges with tailored solutions and delivering real, measurable results. From manufacturing and retail to healthcare, construction, and pharmaceuticals, these use cases illustrate what was needed, what was done, and the impact it created. In this section, we: � Highlight real-world examples across var...

</details>

## 5. In-scope: challenges

**Q:** What challenges do multi-agent systems face, and how can they be mitigated?

**Answer:** Multi‑agent systems struggle mainly with coordination and reliability issues.  
- **Lack of coordination** leads to high failure rates (70‑80 %) and redundant or conflicted actions (Page 38).  
- **Communication overload** can interfere with decision‑making, while too little information can have similar effects (Page 31).  
- **Partial failures** in one agent can jeopardise the whole system if fault tolerance is absent (Page 39).  

Mitigation strategies include:  
- **Effective orchestration** to synchronize agent activities, manage inter‑agent communication, and optimise resource allocation (Page 37).  
- **Agent registries** that catalogue capabilities and enable dynamic discovery, improving task matchmaking and coordination (Page 40).  
- **Robust fault‑tolerance mechanisms** that isolate failures and maintain overall system performance (Page 39).  
- **Balanced technical and collaborative approaches** to address the complexity of orchestrating MAS (Page 39).

**Confidence:** 0.7111

**Retrieved chunks:** 8

<details><summary>Top chunk (page 29, score 0.711)</summary>

> MULTI-AGENT SYSTEMS In this section, we explore Multi-Agent Systems (MAS) and their capabilities in handling complex, dynamic tasks. MAS are increasingly utilized for their ability to collaborate and adapt in real-time, making them suitable for a variety of applications. Here are the topics that we’ll cover in this section : � What Are Multi-Agent Systems? � Structural Layers in Multi-Agent System...

</details>

## 6. Out-of-scope (should refuse)

**Q:** What is the capital of France?

**Answer:** I couldn't find grounded information for that in the Agentic AI eBook, so I won't guess. Please ask something covered by the document.

**Confidence:** 0.0

**Retrieved chunks:** 8

<details><summary>Top chunk (page 51, score 0.098)</summary>

> and determine the necessary strategies for advancement. The readiness levels are divided into four stages: Initial: Basic infrastructure, limited expertise, and early framework development. Emerging: Building core capabilities and developing talent, with early AI projects in progress. Developing: Systematic integration of AI, skilled talent pool, and centralized AI governance. Advanced: Full integ...

</details>

---

Reproduce any of these:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the core components of an Agentic AI system?"}'
```
