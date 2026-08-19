# Epistemic Filtering Agent

## Problem Statement

Contradictory clinical literature on semaglutide's effect on MASH/NASH liver fibrosis cannot be
ranked by an LLM's direct judgment of "quality" or "relevance": language-model confidence in
tone is not evidence quality, and a small retrospective study written assertively will
outscore a cautious meta-analysis under free-form LLM ranking. This system resolves that failure
mode structurally, by decoupling *qualitative extraction* (what a paper claims, and under what
design) from *quantitative scoring* (whether that claim is warranted by that design) — the LLM
extracts typed telemetry; a deterministic Bayesian update, not the LLM, produces the ranking.

## Pipeline Topology

```
PubMed (Bio.Entrez)
       |
       v
[INGESTION] --paced fetch--> raw metadata + abstracts
       |
       v
[LLM PYDANTIC EXTRACTION]   Gemini -> PaperTelemetry (design, N, hyperbole, CI, prereg)
       |
       v
[RELEVANCE GATE]            Python -- drop off-topic papers pre-scoring
       |
       v
[DETERMINISTIC SCORING]     Beta(a,b) prior x likelihood decay x precision -> S_posterior
       |
       v
[TRI-STATE ANOMALY GATE]    surrogate-resolved | fixed fail-safe -> async queue (non-blocking)
       |
       v
[ARBITER]                   Gemini -- defends the given top-3 ranking, does not re-rank
       |
       v
[MEMORY]                    SQLite (epistemic_memory.db) + networkx.DiGraph, signed edges
```

## Mathematical State Formulation

$$S_{\text{posterior}} = \text{clip}_{[0,1]}\Big(\big[P(E)_{\text{eff}} \cdot L(D_{\text{adj}}) \cdot \Pi\big] \cdot w_M + V_{\text{norm}} \cdot w_V\Big), \quad w_M = 0.75,\ w_V = 0.25$$

$$D = \max(0,\ H - R_{\text{tier}}), \quad D_{\text{adj}} = D \cdot (2 - W_N), \quad L(D_{\text{adj}}) = \exp\big(-0.5 \cdot \max(0,\ D_{\text{adj}} - 0.5)\big)$$

$$\text{SE} = \frac{\text{CI}_{\text{upper}} - \text{CI}_{\text{lower}}}{3.92}, \quad \Pi = \exp\big(-0.5 \cdot \max(0,\ \text{SE} - 0.5)\big)$$

$$V_{\text{norm}} = \frac{V - V_{\min}}{V_{\max} - V_{\min}}, \quad V = \frac{\text{Citations}}{\Delta\text{Years} + 1}$$

where $H$ is claim hyperbole $\in [1,5]$, $R_{\text{tier}} = P(E)_{\text{eff}} \times 5$ is the design rigor ceiling, and $W_N$ is $\log_{10}(N+1)$ min-max normalized across the batch.

| Beta tier | $\alpha$ | $\beta$ | $E[P(E)]$ | $R_{\text{tier}}$ |
|---|---|---|---|---|
| Meta-Analysis | 19 | 1 | 0.95 | 4.75 |
| Phase III RCT | 18 | 2 | 0.90 | 4.50 |
| Phase II RCT | 14 | 6 | 0.70 | 3.50 |
| Prospective Cohort | 11 | 9 | 0.55 | 2.75 |
| Retrospective | 8 | 12 | 0.40 | 2.00 |
| Review/In Vitro | 3 | 17 | 0.15 | 0.75 |

Out-of-distribution designs register on first sight with an uninformative Jeffreys prior,
$\text{Beta}(0.5, 0.5)$, rather than a `KeyError` or a borrowed tier.

## Core Architectural Invariants

1. **Extraction/scoring decoupling.** `PaperTelemetry` (Gemini, structured JSON) carries only
   observable facts; `score_one()` (pure Python) is the sole author of every ranking decision —
   fully deterministic and auditable to a single function.
2. **Signed knowledge graph & contradiction hubs.** `epistemic_memory.db` persists every paper as
   a node and every citation/contradiction as a directionally-signed edge
   (`SUPPORTING`/`MENTION`/`REFUTING`); contradiction edges are detected via outcome-direction
   inference and are unconditionally `REFUTING`.
3. **Tri-state non-blocking anomaly gate.** A flagged paper ($D \geq 2.0$ or interventional
   $N < 30$) is resolved without ever blocking on `input()`: a trained `LogisticRegression`
   surrogate acts when confidence $\geq 0.85$, otherwise a fixed conservative bound applies and
   the paper is queued to `unresolved_audits` for asynchronous human review.
4. **Empirical Bayes active learning via adversarial LLM critic.** `backtest_calibration.py`
   stress-tests already-scored papers (sample power, selective reporting, endpoint validity) and
   feeds ROBUST/VULNERABLE verdicts into the same `Beta(alpha, beta)` update path a human HITL
   decision uses — one function moves every prior, regardless of trigger source.

## Quickstart & Test Suite

```bash
python -m venv .venv && .venv/Scripts/activate     # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env        # set ENTREZ_EMAIL, GEMINI_API_KEY

python agent.py --limit 10                         # CLI pipeline run
python -m pytest test_matrix.py -v                  # 79/79 offline unit tests
streamlit run ui.py                                  # dual-mode dashboard (live search / cached run)
python backtest_calibration.py --iterations 5        # adversarial calibration pass
```

## Repository Layout

```
agent.py                 run_pipeline(): ingestion -> extraction -> gate -> scoring -> anomaly
                          gate -> arbiter. Shared by the CLI and the dashboard's live-search mode.
graph_memory.py          SQLite + networkx persistence, Beta active learning, OOD/Jeffreys
                          priors, signed graph topology, ML SurrogateOperator, calibration log
backtest_calibration.py  adversarial calibration engine (standalone entry point)
ui.py                    dual-mode Streamlit dashboard (Live PubMed Search / Load Cached Run)
test_matrix.py           79 offline unit tests, network-free, deterministic
requirements.txt / .env.example
sample_run_output.json / epistemic_memory.db   committed reference run + its knowledge graph
```

Extended design rationale, verified runtime behavior, edge cases, and known limitations:
[ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md).
