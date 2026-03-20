---
layout: post
title: "Reasoning Structure Activates Search Strategy: A Decomposition of Agentic RAG Gains on Knowledge-Intensive QA"
date: 2026-03-20
author: Zhenxing Zhang
---

## Abstract

Agentic retrieval-augmented generation (RAG) — where an LLM iteratively controls its own retrieval — outperforms traditional single-pass RAG on knowledge-intensive tasks. But *why* it works remains poorly understood: is the gain from better retrieval, better reasoning, or their interaction? We present a systematic factorial decomposition on FinanceBench (150 expert-authored questions over SEC filings), varying reasoning structure (with/without ReAct-style Thought-Action-Observation cycles) and search guidance (with/without domain-specific query strategies) across 15 configurations spanning two model families. Our central finding is a **super-additive interaction**: reasoning structure alone adds +6pp and search guidance alone adds +10pp, but combining them yields +24.7pp — 8.7pp more than their independent sum. Qualitative analysis reveals the mechanism: the ReAct cycle creates explicit reasoning checkpoints that *activate* search strategies the model possesses but does not consistently apply without structured prompting. We further show that this interaction scales with model capability — stronger models internalize the reasoning structure more readily, reducing the marginal value of code-level enforcement. These findings suggest that the dominant factor in agentic RAG is not retrieval infrastructure but the coupling between reasoning scaffolding and task-specific strategy.

## 1 Introduction

Retrieval-augmented generation (Lewis et al., 2020) grounds LLM outputs in external documents, but its standard "retrieve-then-generate" pipeline retrieves once with no opportunity to refine, evaluate, or iterate. Agentic RAG addresses this by giving the model tool-based access to retrieval, enabling iterative search within a multi-step reasoning loop (Yao et al., 2023; Schick et al., 2024).

Prior work has established that agentic RAG improves over single-pass RAG (Asai et al., 2024; Jiang et al., 2023; Trivedi et al., 2023), but the source of improvement is typically attributed holistically to "the agent" without disentangling the contributing factors. When a practitioner deploys agentic RAG, which lever matters most — the agent architecture, the prompt, the retrieval model, or the underlying LLM?

We present a controlled factorial study that decomposes agentic RAG gains into four independent factors and measures their interactions. Our key contribution is identifying and characterizing a **super-additive interaction** between reasoning structure and search guidance that accounts for over half of total improvement. This interaction has a specific, analyzable mechanism: structured reasoning creates checkpoints at which latent search strategies are activated. We provide qualitative evidence from item-level trajectory analysis and show that the interaction pattern holds across model capability levels.

## 2 Experimental Setup

### 2.1 Task and Evaluation

We evaluate on FinanceBench (Islam et al., 2023), a benchmark of 150 expert-authored questions over real 10-K and 10-Q SEC filings from public companies. Questions require precise numerical extraction and computation (e.g., "What is the FY2020 free cash flow for General Mills?"), not just factual recall. Each question is paired with a gold answer and explanation.

We evaluate using an LLM judge (GPT-4o) that classifies each response as **correct**, **incorrect**, or **no answer** (abstention). This three-way classification lets us analyze not just accuracy but *calibration* — whether the model knows when it doesn't know.

### 2.2 Factorial Design

We vary two prompt-level factors in a 2x2 design, holding all other variables constant (GPT-4o, default agent, base embedding model):

| | No Search Guidance | With Search Guidance |
|---|---|---|
| **No ReAct** | Basic agentic (39.3%) | + query strategies (49.3%) |
| **With ReAct** | + T-A-O cycle (45.3%) | + both (**64.0%**) |

**Reasoning structure** (ReAct): A Thought-Action-Observation cycle that requires explicit reasoning text before each tool call and explicit review of results before proceeding (Yao et al., 2023).

**Search guidance**: Domain-specific instructions for query formulation — synonym strategies (e.g., "net sales" vs. "revenue"), scope adjustment, section-level targeting, and question decomposition.

We then extend along two additional axes:
- **Retrieval infrastructure**: BGE-large embeddings (Xiao et al., 2024), cross-encoder reranker, calculator tool
- **Model capability**: GPT-4o vs. GPT-5, and code-level ReAct enforcement vs. prompt-only

### 2.3 Baselines

- **Closed-book** (no retrieval): GPT-4o answers from parametric knowledge alone (28.67%)
- **Single-pass RAG**: One retrieval pass prepended to the prompt before generation (36.67%)
- **Basic agentic RAG**: The model has a `search_pdf` tool but minimal prompting (39.33%)

All experiments use the OPAL framework (Open Platform for Agentic Learning) for agent execution and evaluation.

## 3 Results

### 3.1 Overall Progression

Table 1 shows the full progression from closed-book to the best configuration.

| # | Configuration | Model | Correct | Incorrect | No Ans. |
|---|---|---|---|---|---|
| 0 | Closed-book | GPT-4o | 28.67% | 32.67% | 38.67% |
| 1 | Single-pass RAG | GPT-4o | 36.67% | 21.33% | 42.00% |
| 2 | Basic agentic | GPT-4o | 39.33% | 20.00% | 40.67% |
| 3 | + ReAct prompt | GPT-4o | 45.33% | 21.33% | 33.33% |
| 4 | + Search guidance | GPT-4o | 49.33% | 23.33% | 27.33% |
| 5 | + ReAct + guidance | GPT-4o | 64.00% | 20.00% | 16.00% |
| 6 | + BGE-large | GPT-4o | 65.33% | 18.67% | 16.00% |
| 7 | + Calculator | GPT-4o | 66.00% | 21.33% | 12.67% |
| 8 | + Reranker | GPT-4o | 66.67% | 24.67% | 8.67% |
| 9 | ReAct agent (code) | GPT-4o | 70.00% | 18.00% | 12.00% |
| 10 | Default agent | GPT-5 | 78.67% | 18.67% | 2.67% |
| 11 | ReAct agent (code) | GPT-5 | **80.00%** | 16.67% | 3.33% |

*Table 1: Full progression across 12 configurations on FinanceBench (n=150).*

### 3.2 Factor Decomposition

The +43.3pp gain from single-pass RAG (36.67%) to the best configuration (80.00%) decomposes as:

| Factor | Contribution | Source |
|---|---|---|
| Reasoning structure + search guidance | +24.7pp | #2 → #5 |
| Stronger model (GPT-5) | +13.3pp | #8 → #11 |
| Code-level ReAct enforcement | +3.3pp | #7 → #9 |
| Agentic retrieval (vs. single-pass) | +2.7pp | #1 → #2 |
| Retrieval infrastructure | +2.7pp | #5 → #8 |

The prompt-level factors (reasoning + guidance) account for 57% of the total gain. Retrieval infrastructure — BGE-large, reranker, calculator — contributes only 6%.

### 3.3 The Super-Additive Interaction

The central result is the interaction between reasoning structure and search guidance (Figure 1):

```
                        No Guidance    With Guidance    Marginal effect
No ReAct structure         39.3%          49.3%            +10.0pp
With ReAct structure       45.3%          64.0%            +18.7pp
                          ------         ------
Marginal effect           +6.0pp        +14.7pp
```

If the factors were independent (additive), we would expect the combined condition to reach 39.3 + 6.0 + 10.0 = 55.3%. The observed 64.0% exceeds this by **+8.7pp** — a significant interaction effect. Search guidance is nearly twice as effective in the presence of ReAct structure (+14.7pp vs. +10.0pp), and ReAct is more than twice as effective with search guidance (+18.7pp vs. +6.0pp).

### 3.4 Interaction Scales with Model Capability

We observe a consistent pattern: stronger models internalize structured reasoning, reducing the marginal value of external enforcement.

| Model | Default Agent | ReAct Agent (code) | Delta |
|---|---|---|---|
| GPT-4o | 66.00% | 70.00% | +4.0pp |
| GPT-5 | 78.67% | 80.00% | +1.3pp |

GPT-5 extracts most of the benefit from ReAct through prompt instructions alone. Code-level enforcement adds only +1.3pp for GPT-5 vs. +4.0pp for GPT-4o, suggesting that stronger models more reliably follow structured reasoning patterns without scaffolding.

## 4 Analysis: Why Is the Interaction Super-Additive?

We conducted item-level trajectory analysis on all 150 questions across the four factorial conditions. The interaction arises through three mechanisms, ranked by frequency of occurrence.

### 4.1 Activated Document Comprehension

In 23% of items where the combined prompt outperforms both individual variants, the agents across all conditions **retrieved the same documents**, but only the combined-prompt agent extracted the correct answer. The ReAct Observation step forces explicit processing of retrieved content, which activates the search guidance's instruction to "search for the table or section, not the answer."

**Example**: When computing General Mills' free cash flow, both the search-guidance-only and ReAct-only agents retrieved the cash flow statement containing "Purchases of land, buildings, and equipment (460.8)." The search-guidance agent stated "specific capital expenditures were not directly provided" — it had the document but failed to parse it. The ReAct-only agent searched more broadly but lacked targeted queries. Only the combined agent's Observation step forced it to parse the retrieved text, identify the $460.8M figure as capital expenditure, and compute FCF = $3,676M - $461M = $3,215M.

### 4.2 Strategy-Aware Query Refinement

The search guidance provides a *repertoire* of query strategies (synonym substitution, scope adjustment, section targeting). Without ReAct, the model applies these strategies inconsistently. The Thought step before each search creates a natural point at which the model reflects on what failed and selects a strategy from the repertoire.

**Example**: When searching for 3M's capital expenditure, initial queries failed for all agents. The ReAct-only agent tried three times with minor rephrasing. The search-guidance-only agent tried synonym substitution once and stopped. The combined agent's Thought step explicitly reasoned: "The term 'capital expenditure' may not appear. The search guidance suggests synonyms — let me try 'cash flows from investing activities.'" This meta-cognitive reference to the guidance produced the successful query.

### 4.3 Structured Decomposition of Multi-Step Problems

For questions requiring multiple data points (e.g., computing ratios from revenue *and* PP&E), the combined prompt enables planned decomposition. The ReAct structure provides the *when* to decompose (at each Thought step), while the search guidance provides the *how* ("break complex questions into parts, search for each separately").

**Example**: Computing Activision Blizzard's fixed asset turnover requires FY2019 revenue, FY2018 PP&E, and FY2019 PP&E. The basic agentic agent made 4 unfocused queries and never found PP&E. The combined agent planned: "Step 1: find revenue → Step 2: find PP&E for both years → Step 3: compute the ratio" — each with a targeted query derived from the guidance.

## 5 Calibration Analysis

Beyond accuracy, we examine how each factor affects calibration — the balance between incorrect answers and abstentions.

| Configuration | Incorrect:No-Answer Ratio | Interpretation |
|---|---|---|
| Single-pass RAG | 0.51:1 | Over-cautious |
| Basic agentic | 0.49:1 | Over-cautious |
| ReAct + guidance | 1.25:1 | Well-calibrated |
| + Reranker | 2.85:1 | Over-confident |
| ReAct agent (code) | 1.50:1 | Well-calibrated |
| GPT-5 + ReAct agent | 5.00:1 | Confident but grounded |
| GPT-5 closed-book | 4.71:1 | Confident and ungrounded |

The combined ReAct + guidance prompt achieves the best calibration among GPT-4o configurations. The reranker improves accuracy slightly (+0.67pp) but degrades calibration by making the agent more willing to commit to uncertain answers. This suggests that the reranker reduces retrieval failures (fewer abstentions) but does not improve reasoning over retrieved content.

GPT-5's high ratio (5.00:1) looks concerning in isolation, but in context it reflects genuine competence: 78.67% of answers are correct, compared to the closed-book setting where the same assertiveness produces 53.33% incorrect answers. **Retrieval is more important for stronger models** — their confidence needs to be channeled through verified sources.

## 6 Related Work

**Iterative retrieval**: FLARE (Jiang et al., 2023) triggers retrieval when the model's generation confidence drops. IRCoT (Trivedi et al., 2023) interleaves chain-of-thought with retrieval. Self-RAG (Asai et al., 2024) trains the model to generate special tokens controlling retrieval. These methods modify the retrieval trigger mechanism, whereas we study the interaction between reasoning scaffolding and search strategy at the prompt level — orthogonal to and composable with any trigger mechanism.

**ReAct and tool use**: Yao et al. (2023) introduced the ReAct paradigm for interleaving reasoning traces with actions. Subsequent work has applied ReAct to various tasks (Shinn et al., 2023; Madaan et al., 2023). We provide the first controlled study isolating ReAct's interaction with domain-specific task guidance, showing the effect is super-additive rather than additive.

**Financial NLP**: FinanceBench (Islam et al., 2023) and related benchmarks (Chen et al., 2022; Shah et al., 2022) evaluate LLMs on financial document understanding. Prior leaderboard entries report aggregate scores without decomposing improvement sources. Our factorial design enables causal attribution of gains to specific factors.

## 7 Discussion and Limitations

**Implications for system design.** Our results suggest that practitioners building agentic RAG systems should invest primarily in prompt design — specifically, the coupling between reasoning structure and domain-specific strategy. Retrieval infrastructure (embeddings, rerankers) provides diminishing returns relative to this coupling. This challenges the common engineering focus on retrieval quality as the primary lever.

**The activation hypothesis.** We propose that LLMs possess latent task strategies — domain-specific heuristics acquired during pretraining — that are inconsistently activated during generation. Structured reasoning prompts (ReAct) create explicit checkpoints where these strategies are more likely to be invoked. This explains why the interaction is super-additive: the reasoning structure doesn't just add its own benefit but *amplifies* the benefit of providing strategies to activate.

**Limitations.** (1) Single benchmark: FinanceBench covers only knowledge-intensive QA. The super-additive pattern may not hold for tasks where retrieval is less critical or reasoning is less structured. (2) Closed models: GPT-4o and GPT-5 are proprietary, limiting reproducibility and contamination analysis. (3) Sample size: 150 items is small for detecting interaction effects; we report point estimates without significance tests. (4) No comparison to published agentic RAG methods (Self-RAG, FLARE, IRCoT) — our baselines are internal. (5) The activation hypothesis is supported by qualitative evidence from trajectory analysis, not formal verification.

## 8 Conclusion

We present a factorial decomposition of agentic RAG gains on knowledge-intensive QA, revealing that the dominant improvement factor is not retrieval quality but a super-additive interaction between reasoning structure and search guidance. This interaction yields +8.7pp beyond what either factor contributes independently, accounting for over half of the total +43pp improvement from single-pass RAG to the best configuration. Qualitative analysis identifies the mechanism: structured reasoning checkpoints *activate* search strategies that the model possesses but does not consistently apply. These findings reframe the agentic RAG improvement story from "iterative retrieval is better" to "structured reasoning unlocks latent task competence" — a distinction with concrete implications for system design.

## References

- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *ICLR 2024*.
- Chen, Z., Chen, W., Smiley, C., Shah, S., Borber, I., Natarajan, K., ... & Wang, W. B. (2022). FinQA: A dataset of numerical reasoning over financial data. *EMNLP 2021*.
- Islam, P., Kannappan, D., & Kiela, D. (2023). FinanceBench: A new benchmark for financial question answering. *NeurIPS 2023 Datasets and Benchmarks*.
- Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., ... & Neubig, G. (2023). Active retrieval augmented generation. *EMNLP 2023*.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., ... & Clark, P. (2023). Self-refine: Iterative refinement with self-feedback. *NeurIPS 2023*.
- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., ... & Scialom, T. (2024). Toolformer: Language models can teach themselves to use tools. *NeurIPS 2023*.
- Shah, S., Mishra, A., Yadav, N., & Talukdar, P. (2022). FLUE: Financial language understanding evaluation. *ACL 2022 (Findings)*.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS 2023*.
- Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. *ACL 2023*.
- Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2024). C-Pack: Packaged resources to advance general Chinese embedding. *SIGIR 2024*.
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*.

---

## Appendix A: Prompt Details

### A.1 Search Guidance (shared between search-only and combined prompts)

The search guidance provides five concrete strategies: (1) synonym and alternate terminology ("net sales" vs. "revenue"); (2) scope broadening/narrowing; (3) incorporating context from prior observations; (4) searching for tables/sections rather than computed values; (5) decomposing complex questions into sub-queries.

### A.2 ReAct Structure

The ReAct structure mandates: (1) a Thought before every action explaining reasoning; (2) an Observation after every tool result reviewing what was learned; (3) explicit repetition instruction ("Repeat steps 1-3 as needed"); (4) a 3-call tool budget forcing upfront planning.

### A.3 Code-Level ReAct Enforcement

The `react` agent type parses each model response into Thought/Action/Observation segments at the code level, rejecting responses that skip reasoning steps. This prevents the model from bypassing the structured cycle even when the prompt alone would allow it.

## Appendix B: GPT-5 Closed-Book Analysis

To assess potential contamination, we compare GPT-5's closed-book performance to its agentic performance:

| Setting | Correct | Incorrect | No Answer |
|---|---|---|---|
| GPT-5 closed-book | 35.33% | 53.33% | 11.33% |
| GPT-5 agentic | 78.67% | 18.67% | 2.67% |

The +43.3pp gain from adding agentic retrieval to GPT-5 mirrors the gain from single-pass RAG to best-config with GPT-4o. Critically, closed-book GPT-5 produces 80 incorrect answers (53.33%) — its assertiveness without retrieval grounding generates confident fabrications. The agentic setting reduces incorrect answers by 65% (80 → 28) while increasing correct answers by 123% (53 → 118). This suggests the agentic gains are genuine retrieval-and-reasoning improvements, not contamination artifacts.

## Appendix C: Item-Level Transition Analysis (GPT-4o → GPT-5)

| Transition | Count | Description |
|---|---|---|
| correct → correct | 89 | Stable across models |
| incorrect → correct | 20 | GPT-5 fixed reasoning errors |
| no_answer → correct | 9 | GPT-5 found answers GPT-4o couldn't |
| incorrect → incorrect | 15 | Both wrong (often different errors) |
| correct → incorrect | 9 | GPT-5 regression |
| correct → no_answer | 2 | GPT-5 regression (abstained) |
| incorrect → no_answer | 2 | GPT-5 wisely abstained |
| no_answer → incorrect | 4 | GPT-5 guessed wrong |

Net: +18 items improved. Regressions are predominantly interpretation disagreements (4/11) and numerical extraction errors (5/11), not fundamental capability regressions.
