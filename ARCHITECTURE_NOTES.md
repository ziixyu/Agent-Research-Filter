# Architecture Notes

Design rationale, verified runtime behavior, edge cases, and known limitations for the
Epistemic Filtering Agent. `README.md` is the compact system specification; this document is
the extended engineering log behind it — read it when the "why" behind a constant, a schema
choice, or a defended tradeoff isn't obvious from the code alone.

## 1. Design Rationale & Judgment Calls

**Extraction/scoring decoupling.** An LLM asked "which of these papers is most trustworthy?"
reliably confuses *confidence of tone* with *quality of evidence* — a small retrospective study
written with swagger will out-rank a cautious meta-analysis if an LLM is allowed to freely judge
"relevance" or "quality." The extraction step (`PaperTelemetry`) returns only facts: study
design, sample size, claim hyperbole 1-5, CI bounds, preregistration status. `score_one()` is a
plain, deterministic Python function over those facts — every ranking decision is auditable to
a specific line of code, not a model's latent judgment.

**Posterior weight ratio $w_M=0.75$, $w_V=0.25$.** Citation velocity measures *attention*, not
*validity* — a controversial or clinically urgent paper accumulates citations quickly whether or
not it holds up, so weighting it evenly (or higher) would let a hyped-but-flimsy paper outrank a
boring, solid one. Velocity isn't zeroed out entirely: a rigorous paper the field is actively
building on has survived more scrutiny than one being ignored, so 0.25 lets velocity break
near-ties between similar-rigor papers without overriding a prior-tier gap. Velocity (citations
÷ age) was chosen over raw recency for the same reason — a heavily-cited 2020 meta-analysis
represents more *accumulated, tested* evidence than an unchallenged 2026 single-center study;
"newest" and "most trusted" are not the same axis.

**Smooth exponential decay, not a flat cutoff.** An earlier version applied a flat penalty once
hyperbole crossed a fixed threshold. That treats "barely over the line" and "wildly over the
line" identically, and can't express that a large trial and a tiny one making the identical
overclaim aren't equally suspect. `exp(-k·D_adj)` degrades smoothly and composes with the Sample
Power Weight multiplier; `k=0.5` and `threshold=0.5` were tuned so a single-tier overreach is
barely felt while a two-tier overreach with a small $N$ collapses credence toward zero — see
`test_hyped_observational_study_demoted_below_modest_rct`.

**Beta-tier mapping heuristics (`graph_memory.py`).** The extraction schema's 6
`study_design` categories don't map 1:1 onto the 6 seeded Beta tiers: `RCT` has no phase field,
so it is split by sample size (`N >= 300` assumes Phase III, else Phase II —
`RCT_PHASE_III_MIN_N`, a disclosed heuristic, not a measurement); `In-Vitro/Animal` and
`Review/Opinion` both map to the single `Review/In Vitro` tier because the spec names exactly 6
tiers and this is the pairing that reconciles a 6-category extraction schema against them — they
share one learned `Beta(alpha, beta)`, so feedback on one shifts the other.

**Preregistration bonus.** A flat `+0.05` (capped at 1.0) applied to a preregistered trial's
tier prior, not a 7th evidence tier — a trial registry ID makes after-the-fact outcome-switching
harder to get away with, and everything downstream (rigor baseline, $D$, the posterior) uses
this boosted `effective_prior_credence`.

**Jeffreys prior for OOD tiers.** `Beta(0.5, 0.5)` gives $E[P(E)]=0.5$ and is the
maximum-variance member of the `Beta(a,a)` family on $[0,1]$ — maximally noncommittal rather
than falsely confident in either direction for a design the system has never seen.

## 2. Verified Runtime Behavior

- A live citation-edge lookup against real PubMed data (`fetch_citation_edges_from_pubmed`)
  found real edges in every live run — one run found 9 citation edges (8 `SUPPORTING`, 1
  `MENTION`), confirming the Entrez `elink`/`pubmed_pubmed_refs` integration is not a stub.
- The adversarial calibration engine (`backtest_calibration.py`), run live, produced specific,
  grounded critiques (e.g. flagging a `Review/Opinion` paper's `N=0` as making "statistical
  power... entirely inapplicable") and moved real priors in a single pass:
  `Review/In Vitro` mean 0.150 → 0.120, `Phase III RCT` mean 0.900 → 0.783.
- One real dashboard "Execute Pipeline" click ran the full pipeline end-to-end against real
  PubMed + Gemini data (30 papers) with zero Streamlit runtime errors; one real calibration
  console click rendered real multi-tier convergence curves.
- The tri-state fail-safe's surrogate-confident and fixed-fallback paths have both been
  exercised directly against synthetic anomaly data, not just unit-tested — the anomaly gate
  itself has never fired on real PubMed data in a live run to date (see Known Limitations).

## 3. Runtime Edge Cases

**1. Gemini's flagship free tier: 20 requests/day.** `gemini-flash-latest` (the flagship alias)
enforces a 20-request/day free quota; this pipeline needs ~11 calls minimum per run (10
extraction + 1 arbiter), so a single day of iteration burned the quota and the arbiter step
failed with `RESOURCE_EXHAUSTED`. Fix: default model switched to `gemini-flash-lite-latest`
(burst-tested at 8 rapid calls, zero throttling); `call_gemini_with_retry()` parses the server's
`retryDelay` from 429/503 payloads and backs off accordingly, with an explicit `max_delay=60.0`
ceiling so a single retry chain can never stall a run past a minute.

**2. A real PubMed title crashed the console on Windows.** A real paper title contained a
Unicode thin space (U+2009); Windows' legacy per-codepage console (`cp1252`) cannot encode it,
and Rich's Windows renderer raised `UnicodeEncodeError`, killing the run mid-output after the
expensive PubMed + Gemini calls had already succeeded. Fix: `agent.py` reconfigures
`sys.stdout`/`sys.stderr` to UTF-8 with `errors="replace"` at startup — any unusual character
degrades to a placeholder glyph instead of taking down a run that had already done its expensive
work.

**3. pyvis's embedded graph silently failed inside Streamlit's iframe.** The browser console
showed `SyntaxError: Unexpected identifier 'Streamlit'` — pyvis's default HTML
(`cdn_resources="local"`) emits a `<script src="lib/bindings/utils.js">` path relative to
wherever the HTML is served from, which resolves incorrectly inside Streamlit's own iframe base
URL. Fix: `Network(..., cdn_resources="in_line")` inlines all JS/CSS directly into the generated
HTML. Caught only by opening the dashboard in a real browser and reading the console — not
visible from code review.

**4. A running dashboard silently locked the database file.** `rm epistemic_memory.db` failed
with `Device or resource busy` — a stray `streamlit` process from an earlier session was still
holding the SQLite file open (Windows locks files more aggressively than POSIX). Fix at the time:
`taskkill` on the stray process. Durable takeaway: `graph_memory.init_db()`'s migration path
(`_ensure_column()`, `PRAGMA table_info` + `ALTER TABLE ADD COLUMN`) exists specifically so an
already-committed `epistemic_memory.db` upgrades in place rather than requiring delete-and-
recreate.

**5. `predict_proba()`'s columns don't reliably mean `[PASS, CLAMP, REJECT]`.** The first
`SurrogateOperator.predict()` did `label = int(proba.argmax())` — treating the argmax *index* as
the label. That is wrong whenever training data doesn't yet contain all 3 action types: with
only `confirm`(0) and `reject`(2) seen, `LogisticRegression.classes_` becomes `[0, 2]`, so index
`1` of `predict_proba()`'s output means label `2`, not `1`. Fixed by mapping the argmax index
back through `model.classes_` explicitly; locked in by
`test_surrogate_predict_proba_label_matches_classes_order`.

**6. A synthetic test's own prior mismatch silently understated surrogate confidence.** A query
paper constructed via the normal `score_papers()` pipeline used the tier's *default* seed prior,
while training rows had been seeded with a *different, drifted* prior for the same tier — the
"prior" feature for the query paper sat between two training clusters instead of matching
either, so the model was correctly uncertain given genuinely ambiguous input (a test-harness bug,
not a model bug — confirmed by reproducing the same features directly against bare
`sklearn.LogisticRegression`, which returned 0.97 confidence on the clean case). Fixed by making
the test's `prior_lookup` match what was trained on.

**7. `st.components.v1.html` was already past its own stated removal date.** A live click of the
"Live PubMed Search" sidebar mode surfaced a deprecation notice on the exact line rendering the
knowledge graph, with a removal date that had already passed relative to the run's own clock —
not yet enforced by the installed Streamlit version, but not something to leave pending. Fixed
by switching to `st.iframe(html, height=660)`. Confirmed via
`document.querySelectorAll('iframe').length === 1` and an empty error console post-swap.

## 4. Known Limitations

- **Citation counts are mocked** (`random.randint(0, 200)`), disclosed at runtime and visible in
  `run_output.json`. PubMed's base `efetch`/`esummary` endpoints don't return citation counts;
  the free [NIH iCite API](https://icite.od.nih.gov/api) is a scoped, ~20-line swap into
  `fetch_pubmed()` if real counts are required.
- **The anomaly gate has never fired on real PubMed data in a live run** — published,
  peer-reviewed literature on this topic is generally cautious enough that no paper crossed
  $D \geq 2.0$ or $N < 30$ (interventional). Both resolution paths are exercised directly against
  synthetic anomaly data and covered by dedicated unit tests; an intentionally-adversarial query
  (a preprint server, a known-retracted paper) would be needed to exercise the gate on genuinely
  real data.
- **The ML surrogate has no train/holdout split, cross-validation, or overfitting guard.**
  `SurrogateOperator._fit()` trains on 100% of history and predicts immediately, gated only by
  `MIN_TRAINING_ROWS=10` and `CONFIDENCE_THRESHOLD=0.85`. A rolling holdout-accuracy check would
  be a materially stronger guarantee than row count plus a confidence bar.
- **`FAILSAFE_PRIOR=0.20` / `FAILSAFE_LIKELIHOOD=0.05` are fixed, not data-derived** — chosen to
  be deliberately harsh rather than empirically tuned.
- **Live-write dashboard controls have no scratch-DB default.** The Live Calibration Console,
  "Execute Pipeline" (Live PubMed Search), and the prior import/reset controls all write directly
  to the same `epistemic_memory.db` a CLI run would use next, with no confirmation step (Reset
  Priors to Default is the one exception, gated behind a checkbox). The committed
  `epistemic_memory.db`/`sample_run_output.json` pair is always regenerated from a clean CLI run
  before commit, specifically to stay consistent with each other.
- **`p_value` is extracted but unused in scoring** — only the CI-derived Standard Error feeds the
  Precision Penalty; a p-value-only fallback penalty is a natural, scoped addition.
- **The preregistration bonus trusts the LLM's read of the abstract**, not an independent
  ClinicalTrials.gov lookup — a hallucinated or misread registry mention would incorrectly earn
  the bonus.
- **`backtest_calibration.py`'s self-optimizing loop has no convergence guarantee or dampening**
  — repeated `--iterations` passes can drift `alpha`/`beta` arbitrarily far from seed values
  against a small, non-independent sample. Disclosed in the tool's own `--help`; it is a
  deliberately-invoked calibration tool, not part of the automatic pipeline.
- **`audit_status` has only 3 values** (`PASSED`/`FLAGGED`/`OVERRIDDEN`, per spec), so a manual
  prior clamp and an outright reject are indistinguishable from the node table alone — the finer
  distinction is recoverable from `feedback_log.action`.
- **The LLM extraction step can misread an abstract** (e.g. sample size). `run_output.json`
  persists every paper's raw telemetry so a reviewer can spot-check it against the source
  abstract; no second-LLM cross-check exists yet.
- **A malformed LLM response (fails Pydantic validation) drops the paper rather than
  re-prompting.** Transient 429/503 errors retry with backoff; a validation failure does not.
- **Single topic, single query per run** — no multi-topic/multi-drug comparison mode.
