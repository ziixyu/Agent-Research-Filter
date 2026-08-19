# Epistemic Filtering Agent
/////////////////////////
 
SETUP:

Run in cmd:

cd "[Folder directory]"
 
streamlit run ui.py
 
///////////////////////////////////

Ranks conflicted clinical-literature abstracts on **Semaglutide's effect on NASH/MASH liver
fibrosis** — a topic where papers disagree not because someone is lying, but because they
were produced by fundamentally different kinds of evidence (a 6-patient in-vitro study and a
3,000-patient meta-analysis can both be "positive" and still not deserve equal trust).

The agent retrieves real papers from PubMed, extracts structured metadata about *how* each
claim was made (not *whether* it's true), and ranks them with a transparent, inspectable
formula — then asks an LLM to write up the *defense* of a ranking it did not get to choose.

## The core design decision

**The LLM never decides who wins.** It only extracts facts (study design, sample size, how
hyperbolic the language is). A plain Python function turns those facts into a score. This
split exists because an LLM asked "which of these 10 papers is most trustworthy?" will
reliably confuse *confidence of tone* with *quality of evidence* — a small retrospective study
written with swagger will out-rank a cautious meta-analysis if you let an LLM freely judge
"relevance" or "quality." Decoupling extraction from scoring removes that failure mode and
makes every ranking decision auditable: you can point at the exact line of Python that put
paper X above paper Y.

## Pipeline

```
[FETCH]      Bio.Entrez -> PubMed              real metadata + abstracts, paced for batch size
[EXTRACT]    Gemini (structured JSON)          LLM Step 1: telemetry only, no verdicts
[FILTER]     Python                            Reasoning Step 1: relevance gate
[SCORE]      Deterministic Bayesian update     Reasoning Step 2: the judgment call
[GATE]       Python fail-safe (non-blocking)   ML surrogate resolves, or async-queues, anomalies
[ARBITER]    Gemini (structured JSON)          LLM Step 2: defends the ranking it was given
[MEMORY]     SQLite + networkx                 persists the graph and learns from feedback
```

Four modules beyond the core pipeline:

- **`graph_memory.py`** — a persistent knowledge graph (SQLite + `networkx.DiGraph`), Empirical
  Bayesian active learning over the priors (see "Active learning" below), out-of-distribution
  study designs (see "Out-of-distribution designs" below), and an ML `SurrogateOperator` that
  learns to predict HITL-style decisions from telemetry (see "ML surrogate" below).
- **The non-blocking fail-safe in `agent.py`** — every flagged paper is resolved automatically:
  either the trained surrogate predicts confidently, or a fixed conservative bound applies and
  the paper is queued for async human review. Nothing in this pipeline calls `input()` anymore
  (see "Non-blocking tri-state fail-safe" below).
- **`backtest_calibration.py`** — `python backtest_calibration.py --iterations N` runs an
  adversarial LLM "red team" critic over already-scored papers and automatically feeds its
  verdicts into the same Beta-update mechanism the surrogate/manual overrides use (see
  "Autonomous calibration engine" below).
- **`ui.py`** — `streamlit run ui.py`, a two-tab live dashboard: Tab 1 is the ranking table,
  physics-rendered signed knowledge graph, and the Async Audit Queue / Surrogate Inspector; Tab 2
  is a per-tier convergence chart plus a console that runs the calibration engine live (see
  "Dashboard" below).

### Reasoning Step 1 — the relevance gate

The LLM reads each abstract and returns `is_relevant: bool`. Anything not directly about
semaglutide's effect on liver fibrosis (e.g. a general MASLD guideline that only mentions
semaglutide in passing) is dropped before scoring even starts. This is keyword-independent —
a paper can mention "semaglutide" and "fibrosis" in the same abstract and still get filtered
if it isn't actually reporting on that relationship.

### Reasoning Step 2 — the judgment call, formalized as a Bayesian update

This started as a flat weighted-sum ("methodology score + velocity − flat penalty") and was
reformulated into an explicit epistemic-state update, because a flat sum can't express two
things that actually matter: that overclaiming should be judged *relative to* what a paper's
own evidence tier can support (not against one global hyperbole cutoff), and that the same
overclaim is more forgivable from a well-powered study than a tiny one. The update below is
still 100% deterministic Python — "Bayesian" describes the *shape* of the reasoning (prior →
likelihood → posterior), not that anything is literally sampled or fit.

**1. Prior credence P(E) — a belief in the paper's trustworthiness *before* looking at what it
claims**, set by structural evidence tier. This used to be a fixed number per tier; it is now
the *mean of a Beta(alpha, beta) distribution* that a human operator can shift over time (see
"Active learning" below) — starting from these seed hyperparameters:

| Beta tier | alpha | beta | Mean P(E) |
|---|---|---|---|
| Meta-Analysis | 19 | 1 | 0.95 |
| Phase III RCT | 18 | 2 | 0.90 |
| Phase II RCT | 14 | 6 | 0.70 |
| Prospective Cohort | 11 | 9 | 0.55 |
| Retrospective | 8 | 12 | 0.40 |
| Review/In Vitro | 3 | 17 | 0.15 |

Our extraction schema (`PaperTelemetry.study_design`) has 6 categories that don't map 1:1 onto
these 6 tiers — it splits cohort studies into prospective/retrospective (the tiers don't) and
has no phase field for RCTs (the tiers do). Every category is mapped explicitly in
`graph_memory.py`, not left to guesswork:

- `Meta-Analysis` → Meta-Analysis tier directly.
- `RCT` → split by sample size: N ≥ 300 assumes Phase III, else Phase II
  (`RCT_PHASE_III_MIN_N`, a disclosed heuristic — our schema doesn't extract trial phase, but it
  does extract N, and confirmatory Phase III trials are conventionally powered far larger than
  Phase II ones). This is what actually makes the "Phase II RCT" tier reachable at all.
- `Prospective Cohort` → Prospective Cohort tier directly.
- `Retrospective/Observational` → Retrospective tier directly.
- `In-Vitro/Animal` and `Review/Opinion` → **both** map to the single "Review/In Vitro" tier,
  because the spec gives exactly 6 named Beta tiers and this is the one that merges what our
  schema keeps as two separate categories. This means they share ONE learned Beta(alpha, beta)
  — feedback confirming/overriding/rejecting an In-Vitro/Animal paper *does* shift the prior
  Review/Opinion papers get next, and vice versa. I judged this an acceptable, disclosed
  simplification rather than inventing a 7th tier the spec didn't ask for; splitting them into
  independent trackers with more time would just mean keying `design_priors` by the original
  6-category schema instead of the spec's 6 tier names, with `RCT` still split by the
  phase heuristic above.

**1b. Preregistration bonus.** A trial registry ID (NCT/ISRCTN/...) makes after-the-fact
outcome-switching and cherry-picking harder to get away with, so a preregistered trial gets a
flat **+0.05** added to its tier's prior, capped at 1.0: `effective_prior = clip01(base_prior +
0.05)`. This is a modifier on top of the design-tier prior, not a 7th evidence tier of its own
— everything downstream (rigor_baseline, D, the posterior) uses `effective_prior`.

**2. Likelihood penalty L(Absurdity) — the epistemic gate.** A Discrepancy Index
`D = claim_hyperbole − rigor_baseline` measures how far a paper's claim strength (1–5) overshoots
what its own tier could justify, where `rigor_baseline = effective_prior × 5` (the same 1–5
scale, anchored to the prior). An observational study (P(E)=0.40, baseline≈2.0) claiming a
definitive "reversal/cure" (hyperbole=5) gets D=3.0; the same claim from a meta-analysis
(P(E)=0.95, baseline≈4.75) gets D≈0.25 — the exact "observational study claiming causal
cure/reversal yields a high D" example this was built around. **Underclaiming is never
punished** — D is clamped to 0 when the claim is more cautious than the tier's baseline;
modesty is free.

D is then moderated by a **Sample Power Weight** `W_N = log10(N+1)`, min-max normalized across
the batch to [0,1] — the same overclaim from N=5,000 is treated as less absurd than the
identical overclaim from N=10:

```
D_adjusted = D × (2.0 − W_N)          # multiplier in [1.0, 2.0]: best-powered paper in the
                                       # batch gets no amplification, worst-powered gets 2×
L(Absurdity) = 1.0                          if D_adjusted <= threshold (0.5)
             = exp(−k × (D_adjusted − threshold))     otherwise, k = 0.5
```

**2b. Statistical Precision Penalty.** If the abstract reports a 95% CI for the primary
endpoint, the Standard Error is recovered via the standard normal approximation (a 95% CI spans
~1.96 SE on each side of the point estimate): `SE = (ci_upper − ci_lower) / 3.92`. A wide CI
means the underlying estimate is noisier than the claim's confidence lets on, so it discounts
credence with the exact same shape as the likelihood penalty — smoothly, past a tolerance
threshold, and **only** when a CI was actually reported (no CI -> no penalty; this discounts
*reported-but-noisy* precision, it does not punish a paper for omitting a CI, which is a
separate, unmodeled concern):

```
PrecisionPenalty = 1.0                              if SE is None or SE <= 0.5
                  = exp(−k2 × (SE − 0.5))            otherwise, k2 = 0.5
```

**3. Posterior score — the final ranking metric:**

```
S_posterior = clip01( [effective_prior × L(Absurdity) × PrecisionPenalty] × 0.75
                     + [Velocity_norm] × 0.25 )
```

`Velocity_norm` is unchanged from the original design: citations ÷ years-since-publication,
min-max normalized across the batch. `clip01` is a defensive floor/ceiling — every input is
already bounded to [0,1] by construction, so this is a safety net against a future weight
change quietly breaking that invariant, not something that fires in practice today.

Every intermediate quantity is preserved on the output, not just the final score:
`base_prior_credence`, `preregistration_bonus`, `prior_credence` (the effective value),
`standard_error`, `precision_penalty` — so a reviewer can audit exactly which factor moved a
paper's rank, not just trust the final number.

**Why 0.75/0.25 and not 0.5/0.5, or the reverse?** This is the judgment call the assignment
asks for, and it's defensible rather than "correct":

- A GLP-1/fibrosis result that hasn't been replicated in an RCT is not made more true by going
  viral. Citation velocity measures *attention*, not *validity* — a controversial or clinically
  urgent paper accumulates citations quickly whether or not it holds up. Weighting it at 75%
  would let a hyped-but-flimsy paper outrank a boring, solid one.
- Velocity isn't thrown out entirely (weight 0), because a rigorous paper the field is
  ignoring and a rigorous paper the field is actively building on aren't quite equivalent —
  the second has survived more scrutiny. 0.25 lets velocity break near-ties between
  similar-rigour papers without letting it override a prior-tier gap.
- **The alternative I rejected:** weighting recency instead of/alongside velocity. I chose
  velocity (citations/age) over raw recency because a 2020 meta-analysis that's still being
  cited heavily represents more *accumulated, tested* evidence than a 2026 single-center
  retrospective study that hasn't had time to be challenged yet — "newest" and "most trusted"
  are not the same axis, and this task is about resolving conflicts in trust, not surfacing
  news.
- **Why a smooth exponential decay for the penalty, not a flat cutoff?** The original version
  used a flat −0.5 penalty once hyperbole crossed a fixed threshold — defensible, but it treats
  "barely over the line" and "wildly over the line" identically, and it couldn't express that a
  huge trial and a tiny one making the identical overclaim aren't equally suspect. `exp(−k×D)`
  degrades smoothly and composes naturally with the Sample Power Weight multiplier; `k=0.5` and
  `threshold=0.5` were chosen so a single-tier overreach (e.g. an RCT claiming what only a
  meta-analysis could) is barely felt, while a two-tier overreach with a small N (the
  "observational study claims a cure" case) collapses credence to near-zero — see
  `test_hyped_observational_study_demoted_below_modest_rct` in `test_matrix.py` for the concrete
  numbers.
- **A limitation I've since narrowed, not fully closed:** the extraction schema
  (`PaperTelemetry.study_design`) still has only one `"RCT"` category — no blinding/phase field
  — because touching the LLM extraction schema was explicitly out of scope for this pass (see
  "Keep ... Pydantic telemetry schemas ... intact" in the task that drove this change). What
  changed is that RCT's prior is no longer stuck at a single static number: the sample-size
  heuristic above splits it into Phase II/III *for prior-lookup purposes* without touching
  extraction. What's still true: this is inference from N, not a measurement of actual blinding
  or phase, and the Discrepancy Index remains the backstop that catches an RCT whose *claims*
  overreach its actual design regardless of which phase bucket it landed in.

Verify the logic yourself: `python -m pytest test_matrix.py -v` (or plain `python test_matrix.py`)
runs 43 network-free unit tests: the original 10 against the posterior formula, 14 more added
for the fail-safe/active-learning pass (the anomaly gate's trigger conditions, Empirical
Bayesian updates, the contradiction heuristic), and 19 more added for the extended-telemetry
pass covering the precision penalty, the preregistration bonus, out-of-distribution Jeffreys
priors, signed citation sentiment, `backtest_calibration.py`'s verdict-to-feedback mapping, and
URL/DOI metadata preservation. Every SQLite test uses a private `:memory:` database — none of
them touch a real `epistemic_memory.db` on disk.

### LLM Step 2 — the arbiter

The top 3 papers, their extracted telemetry, and their computed scores are sent back to the
LLM with an explicit instruction: **explain this ranking, don't re-decide it.** It writes a
3-sentence, telemetry-grounded justification per paper and a short note on which factor (the
prior P(E), the likelihood/absurdity penalty, or citation velocity) actually separated rank 1
from rank 2. This is the part of the output you read out loud in the demo.

## Configurable batch sizing & adaptive rate pacing (`agent.py`)

`--max-results` (aliased as `--limit` and `--batch-size` — all three set the same value) is
validated to `[5, 50]` by a custom argparse type (`_batch_size()`); anything outside that range
fails fast with a clear CLI error rather than silently clamping or crashing mid-run.

Batches larger than `RATE_PACING_BATCH_THRESHOLD` (10) trigger a proactive `RatePacer`: a
minimum `4.2`s gap enforced between successive Gemini extraction calls, to stay under the free
tier's ~15 RPM ceiling *before* hitting it, rather than only reacting to 429s after the fact.
This is complementary to, not a replacement for, `call_gemini_with_retry()`'s existing reactive
backoff — that backoff now also has an explicit `max_delay=60.0` ceiling (previously
unbounded exponential growth), so a single retry can never stall the run past a minute
regardless of how many attempts the server's `retryDelay` or the doubling schedule implies.

## Active learning: priors that update from feedback (`graph_memory.py`)

Everything above describes priors as fixed numbers. They aren't, anymore. Each Beta tier's
P(E) is `alpha / (alpha + beta)`, persisted in `epistemic_memory.db`, and a human operator can
shift it three ways — every shift is logged to a `feedback_log` table with a before/after
snapshot, so the whole history is auditable, not just the current number:

- **Confirm** (operator accepts the automated handling of a flagged paper): `alpha += 1` — one
  more piece of evidence the tier's default credence is fine.
- **Reject** (operator quarantines the paper): `beta += 1` — one piece of evidence against the
  tier's default credence, at least for instances like this one.
- **Override** (operator manually clamps P(E) to some value `p`): the distribution is nudged
  toward `p` with `pseudo_weight=2.0` pseudo-observations (`alpha += p*2`, `beta += (1-p)*2`)
  rather than being reset outright — one human judgment shifts the distribution without erasing
  everything learned before it. `test_override_feedback_shifts_mean_toward_manual_value` checks
  this moves monotonically toward the target without ever fully reaching or exceeding it.

`agent.py`'s `main()` loads `graph_memory.get_current_prior_means(conn)` at the start of every
run and passes it into `score_papers()` as `prior_lookup` — so feedback from one run's HITL
review (or from the dashboard's override panel) genuinely changes the next run's rankings, not
just a log message. `score_papers()` itself still defaults to `graph_memory.seed_prior_means()`
(pure, I/O-free) when no `prior_lookup` is given, which is what keeps `test_matrix.py`
deterministic regardless of what's in any real `epistemic_memory.db` on disk.

## Out-of-distribution designs & Jeffreys priors (`graph_memory.py`)

`PaperTelemetry.study_design` is a free-text `str`, not a fixed enum — the extraction prompt
lists the 6 known categories as *preferred*, but explicitly permits the LLM to report something
more specific ("Mendelian Randomization", "Organ-on-a-Chip") when a study genuinely doesn't fit
any of them, rather than forcing a bad fit. That freedom is worthless without a scoring layer
that can actually handle it without crashing.

`graph_memory.beta_tier_for()` resolves an unrecognized design to a tier **named after the
design itself** (not silently folded into some unrelated known tier — an earlier version of
this function did exactly that, defaulting anything unrecognized to "Retrospective", which
would have quietly mis-priced every novel design as observational-grade evidence).
`get_prior_credence(conn, study_design, sample_size)` then does the actual work: if that tier
doesn't exist in the store yet, it's registered on the spot with an **uninformative Jeffreys
prior, Beta(0.5, 0.5)** — `E[P(E)]=0.5`, and also the *maximum-variance* member of the
Beta(a,a) family on [0,1] (`Var=0.125`), i.e. maximally noncommittal rather than falsely
confident in either direction. `agent.py`'s `main()` calls this once per paper before scoring,
so any novel design this batch introduces is registered (and included in `prior_lookup`) before
`score_papers()` runs — and `score_papers()` itself still has its own fallback
(`prior_lookup.get(tier, JEFFREYS_MEAN)`) so it can never `KeyError` even if a caller skips that
step, matching "do not fail on unknown categories" literally.

Registration is idempotent and one-way once feedback exists: calling `get_prior_credence()`
again for a tier a human (or `backtest_calibration.py`) has since given feedback on does **not**
reset it back to Beta(0.5, 0.5) — `test_ood_design_registration_is_idempotent_after_feedback`
checks this explicitly. Every `ScoredPaper` carries `is_ood_design: bool` so the dashboard and
CLI table can flag it (🆕/OOD badge) without a reviewer having to know the tier list by heart.

## Autonomous calibration engine (`backtest_calibration.py`)

A separate, deliberately-invoked tool: an adversarial LLM "red team" critic re-judges
already-scored papers against three methodological stress tests — sample power adequacy,
selective-reporting risk, and endpoint validity (hard clinical outcome vs. oversold surrogate)
— and automatically calls `graph_memory.record_feedback()` for each one: a **ROBUST** verdict
maps to `action="confirm"` (`alpha += 1`), a **VULNERABLE** verdict maps to `action="reject"`
(`beta += 1`) — the exact same feedback vocabulary a human HITL decision uses, via the exact
same function, so there is only ever one place in the codebase that knows how to move a Beta
distribution.

```bash
python backtest_calibration.py                          # 1 pass over sample_run_output.json
python backtest_calibration.py --iterations 5 --db scratch_memory.db   # self-optimizing loop
```

This is explicitly a calibration tool, not part of the normal `agent.py` pipeline — repeated
passes keep shifting the same tiers' priors (that's the point of `--iterations`), so a live run
against real data (see "Edge cases" below) genuinely produced adversarial, specific critiques —
e.g. correctly flagging a Review/Opinion paper's `N=0` as making "statistical power... entirely
inapplicable," and multiple papers for lacking preregistration — and visibly moved
`Review/In Vitro`'s mean from 0.150 to 0.120 and `Phase III RCT`'s from 0.900 to 0.783 in a
single pass. Point `--db` at a scratch copy if you want to experiment without touching your main
learned priors; the mapping logic itself (`apply_verdict`) is unit-tested independently of any
live Gemini call.

### Calibration persistence & convergence (`graph_memory.py`)

Every single call to `record_feedback()` — regardless of whether the trigger was a human, the
ML surrogate, or the calibration engine above — and every *first-time* call to
`register_novel_tier()` appends one row to a `calibration_history` table:
`(timestamp, tier, alpha, beta, expected_credence, iteration, trigger_source)`. `iteration` is
just that tier's own update count so far (`COUNT(*) WHERE tier=?`) — simple, and exactly what a
convergence line chart wants as its x-axis, so a tier that's received more feedback naturally
has more points than a quiet one. `get_calibration_history_df()` returns the whole table as a
pandas DataFrame, ordered `(tier, iteration)`, which `ui.py`'s Tab 2 pivots into wide format
(`index=iteration, columns=tier, values=expected_credence`) for `st.line_chart` — a genuine live
run of this (see "Edge cases" below) produced 10 real snapshots across 3 tiers from a single
calibration pass, rendering as three real convergence curves.

## Fail-safe: ML surrogate + non-blocking tri-state resolution (`agent.py`, `graph_memory.py`)

This is a **separate, coarser gate** from the smooth likelihood penalty in Reasoning Step 2.
The likelihood penalty discounts credence continuously for *any* paper whose claim mildly
outruns its tier (`DISCREPANCY_THRESHOLD=0.5`) — silent, automatic, no human involved. The
anomaly gate fires on a much higher bar and exists to put a *specific* paper in front of
scrutiny, not just discount it:

1. **Discrepancy Index D ≥ 2.0** — the claim doesn't just mildly overreach, it's badly out of
   proportion to what the design tier can support.
2. **N < 30 for an interventional design** — currently just `"RCT"` in our schema (a
   meta-analysis synthesizes other work rather than running an intervention itself; cohort,
   observational, in-vitro, and review designs aren't controlled interventions either).

An earlier version of this gate had an `--interactive` mode that used `console.input()` to ask
a human operator to choose per flagged paper — that blocked the entire pipeline on a single
terminal prompt and doesn't scale past a demo. **It has been removed entirely.** Nothing in
`agent.py` calls `input()`/`console.input()` anymore. Every flagged paper is resolved
automatically, one of two ways:

**1. The ML surrogate is trained and confident.** `graph_memory.SurrogateOperator` is a
`LogisticRegression` trained on the accumulated history in `feedback_log` joined against
`nodes` — once at least `MIN_TRAINING_ROWS=10` past decisions exist. Its feature vector is
exactly the 5 dimensions specified: `[log10(N+1), Discrepancy_D, Standard_Error_SE,
is_preregistered, study_design_prior]`, and its target label is the 3 actions
`record_feedback()` already accepts (0=PASS/"confirm", 1=CLAMP/"override", 2=REJECT/"reject") —
so training labels come straight from history with no separate mapping table to keep in sync.
If the predicted class's probability is `>= CONFIDENCE_THRESHOLD (0.85)`, that action is applied
automatically, `audit_status` becomes `AUTO_RESOLVED_BY_SURROGATE`, and the decision is recorded
as feedback (`trigger_source='surrogate'`) so it also shapes future runs' priors. CLAMP resets
the paper's prior to its tier's *original seed baseline* (not the current learned mean) — a
normalizing action, not a punitive one.

**2. Otherwise (untrained, insufficient history, or not confident enough):** a fixed,
deliberately harsh fail-safe bound applies — `P(E)=0.20`, `L(Absurdity)=0.05` — and the paper is
inserted into a new `unresolved_audits` table (`id, pmid, timestamp, reason, telemetry_json`)
with `audit_status=ASYNC_QUARANTINED`. This **never stalls the run**: insertion is one SQL
statement, execution continues immediately. A human reviews the queue later, on their own time,
via `ui.py`'s Async Quarantine Queue accordion (one-click "confirm & release").

Both a surrogate `REJECT` verdict and any `ASYNC_QUARANTINED` paper are excluded from the top-3
sent to the arbiter — a real regression test
(`test_resolve_anomaly_auto_resolves_when_surrogate_confident`) exercises this against
synthetic, deliberately-separable training data (a clean RCT/preregistered cluster vs. a
tiny/overclaiming Retrospective cluster) and checks the exact predicted label, not just "some
label"; a companion regression test (`test_surrogate_predict_proba_label_matches_classes_order`)
locks in a real bug caught during development — see "Edge cases" below.

## Persistent knowledge graph (`graph_memory.py`)

SQLite (`epistemic_memory.db`) is the durable store; `networkx.DiGraph` is an in-memory view
built from it for graph algorithms and the dashboard. Every scored paper becomes a node
(pmid, title, study_design, sample_size, prior_credence, discrepancy_index,
likelihood_penalty, posterior_score, audit_status, **url, doi, standard_error,
is_preregistered**). The last two exist specifically to feed the `SurrogateOperator`'s feature
vector without a second query path — training data is a straight join of `feedback_log` (the
label) against `nodes` (the features) on `pmid`. Two kinds of edges, both **signed** with a
sentiment tag (`SUPPORTING` / `MENTION` / `REFUTING`), not just present/absent:

- **Citation edges — real data**, not mocked. `fetch_citation_edges_from_pubmed()` calls
  `Bio.Entrez.elink` (`pubmed_pubmed_refs`) and keeps only links where *both* papers are in our
  own fetched batch (we only have telemetry for our own batch, so a citation to a paper outside
  it isn't graphable anyway). This found real edges in every live test run — see
  `sample_run_output.json`'s companion `epistemic_memory.db` (a live run found 9 citation edges,
  8 tagged `SUPPORTING` and 1 `MENTION`). Sentiment is computed by `citation_sentiment()`,
  reusing the exact same outcome-direction heuristic as contradiction detection below: if both
  papers' abstracts report the same non-neutral direction, `SUPPORTING`; opposite, `REFUTING`;
  either side ambiguous or its text unavailable, `MENTION`.
- **Contradiction edges — a disclosed keyword heuristic, not semantic NLI.**
  `infer_outcome_direction()` scans title+abstract for a small set of positive-outcome phrases
  ("significant reduction", "improvement", ...) vs. negative/null-outcome phrases ("no
  significant", "did not improve", ...), checking negative phrasing *first* and letting it win
  outright — a negated result like "no significant improvement" contains the substring
  "improvement" (a positive keyword), and naive substring matching can't tell negated language
  from an affirmative claim otherwise. Two relevant papers with opposite inferred directions get
  a contradiction edge, unconditionally tagged `REFUTING` — a detected contradiction IS a
  refuting relationship by definition. This is deliberately cheap (zero extra LLM calls — this
  repo already documents real free-tier quota pain) rather than accurate; it's meant to surface
  *candidates* for a human to actually read, not to assert a contradiction is real.

`nodes.url`/`nodes.doi`/`nodes.standard_error`/`nodes.is_preregistered` and `edges.sentiment`
were all added to an already-committed schema, not designed in from scratch —
`graph_memory.init_db()` runs a small migration (`_ensure_column()`, `PRAGMA table_info` +
`ALTER TABLE ADD COLUMN`) on every open, so an already-committed `epistemic_memory.db` from an
earlier version of this repo upgrades in place instead of needing to be deleted and
regenerated. Two brand-new tables this pass — `calibration_history` and `unresolved_audits` —
use plain `CREATE TABLE IF NOT EXISTS` instead, since a fresh table needs no column migration.

## Dashboard (`ui.py`) — two tabs

```bash
streamlit run ui.py
```

Institutional typography throughout — no emoji anywhere in the UI. A sidebar "Ingestion
Console" drives two ingestion modes:

- **Live PubMed Search** — a query box (default the Semaglutide/MASH-NASH/fibrosis query), a
  Sample Size (N) slider (5-50), and an "Execute Pipeline" button that calls `agent.run_pipeline()`
  **in-process** — the exact same function `agent.py`'s CLI (`main()`) calls, not a
  reimplementation — with a `progress_cb` wired to `st.progress()` so the fetch/extract/score/
  arbitrate/persist stages report live. Completion updates the active session state (and
  `run_output.json` on disk) without a page reload.
- **Load Cached Run** — the original read-only mode: a checkpoint path (default
  `run_output.json`) and a "Reload Checkpoint" button.

Either mode feeds the same Methodology ($w_M$) / Velocity ($w_V$=1-$w_M$) sliders, which re-sort
every paper's `S_posterior` **entirely in client-side memory** by calling `agent.score_one()`
against already-extracted telemetry — no re-extraction, no API calls.

### Tab 1 — Paper Ranking & Knowledge Graph

- **Mathematical Formulation & State Estimation** — a collapsed-by-default `st.expander` above
  the ranking table rendering the four governing equations (posterior blending, discrepancy/
  likelihood decay, precision/standard error, normalized citation velocity) as `st.latex`, so the
  scoring isn't just described in prose.
- **Paper Ranking** (formerly "Dynamic Ranking Matrix") — PMID and `Link` (renamed from `DOI`)
  columns render as clickable links (`st.column_config.LinkColumn`, with a regex `display_text`
  so the cell shows the bare PMID/DOI rather than the full URL); `[OOD]` flags an
  out-of-distribution design.
- **Paper Inspector** — pick any paper from a dropdown to see its full metadata, "View on
  PubMed"/"View Link" buttons (`st.link_button`), the reported 95% CI/p-value if extracted, and
  the base-prior → bonus → effective-prior → posterior breakdown for that specific paper.
- **Knowledge Graph** — rendered via `st.iframe` over embedded pyvis HTML (`cdn_resources=
  "in_line"` — see the edge case below for why that flag matters; `st.iframe` replaced the
  deprecated `st.components.v1.html` this pass), physics solver `forceAtlas2Based` with
  central-gravity damping (tuned so a 30-50 node batch doesn't collapse into an overlapping
  cluster the way `barnesHut` alone tends to at that scale). All node/edge color and size math
  lives in `graph_memory.py` (`node_radius`, `node_fill_color`, `node_border_style`,
  `edge_style_for`, `EDGE_STYLE`) — one source of truth the dashboard renders from and
  `test_matrix.py` verifies directly, instead of a value dict hand-mirrored into each:
  - Node **diameter** is strictly proportional to `S_posterior`: `12 + 28 * S_posterior`.
  - Node **fill** is a continuous luminance gradient over `log10(N+1)`: light silver/slate
    `#CBD5E1` at N≈0 up to deep electric blue `#0284C7` at N≥2000 (clamped past that ceiling).
  - Node **border**: solid emerald `#10B981` (preregistered), solid violet `#8B5CF6`
    (out-of-distribution/Jeffreys prior), dashed crimson `#EF4444` (quarantined/anomaly) — checked
    in that priority order when more than one flag applies to the same node.
  - **Edges**: `SUPPORTING` solid emerald `#10B981` width 2, `MENTION` slate gray `#64748B`
    width 1, `REFUTING`/contradiction bold red dashed `#EF4444` width 3.
  - The legend above the canvas is a plain Markdown table, not prose bullets.
- **Audit & Exceptions Queue** (formerly "Async Audit Queue & Surrogate Inspector") — cards for
  every `AUTO_RESOLVED_BY_SURROGATE` paper show the predicted action and confidence score; an
  "Asynchronous Quarantine Queue" accordion lists every `unresolved_audits` row with a one-click
  "Resolve (confirm & release)" button that records feedback, flips `audit_status` to
  `OVERRIDDEN`, and clears the queue entry.
- **LLM Synthesis & Arbitration** (formerly "Arbiter Synthesis & Counterfactual Console") —
  renders the persisted arbiter justification, plus a free-text box that calls Gemini live
  (`agent.counterfactual_arbiter()`, reusing `call_gemini_with_retry`) with a user-supplied
  hypothetical and the same top-3 telemetry, explicitly instructed to say so if the scenario
  *wouldn't* plausibly change the ranking rather than manufacturing a change.

### Tab 2 — Empirical Prior Convergence & State Calibration

- **Convergence chart** — `graph_memory.get_calibration_history_df()`, pivoted wide
  (`index=iteration, columns=tier`) and forward-filled so a quiet tier draws a flat line instead
  of appearing to "stop", rendered via `st.line_chart`. Empty-state message when no feedback has
  been recorded yet, rather than an empty/broken chart.
- **Live Calibration Console** — a 1-10 iteration slider and an "Execute Autonomous Adversarial
  Calibration" button that imports `backtest_calibration.py` directly and calls its
  `run_calibration_pass()` in a loop against the loaded run's `full_ranking`, live, streaming a
  running log of each verdict and redrawing the convergence chart when done (`st.rerun()`).
- **Prior Calibration State Management** — an `st.expander` with three controls, all writing
  directly to `epistemic_memory.db` and snapshotting `calibration_history`
  (`graph_memory.set_prior_hyperparams()` is the one function all three funnel through):
  - **Export Priors to JSON** — `graph_memory.export_priors()` behind an `st.download_button`;
    every tier's `(alpha, beta, expected_credence)`.
  - **Import Priors from JSON** — `st.file_uploader` + `graph_memory.import_priors()`; accepts
    either the exporter's own `{"tiers": {...}}` shape or a bare `{tier: {"alpha":.., "beta":..}}`
    mapping, and silently skips any entry with a non-positive or malformed alpha/beta rather than
    corrupting a Beta distribution's support.
  - **Reset Priors to Default** — `graph_memory.reset_priors_to_default()`, gated behind an
    explicit confirmation checkbox since it overwrites every seeded tier's learned history; OOD
    tiers registered later are left untouched (they have no seed value to reset to).

Verified live during development: one real "Execute Pipeline" click ran the full pipeline
end-to-end against real PubMed+Gemini data with zero Streamlit runtime errors, and one real
calibration-console click ran the adversarial judge and rendered real convergence curves — see
"Edge cases" and "known limitations" below for why that state isn't what's committed.

## Edge cases I hit for real, not hypothetically

### 1. Gemini's flagship free tier: 20 requests/day

The first live run used `gemini-flash-latest`, Google's alias for its flagship flash model
(currently `gemini-3.7-flash`). It has a **free-tier quota of 20 requests/day**. This pipeline
needs ~11 calls per run minimum (10 extraction + 1 arbiter), so the very first real run burned
most of the day's quota just on extraction retries and the arbiter step failed outright with
`RESOURCE_EXHAUSTED`. That's not a rare failure — it's the default failure mode of running this
pipeline against the "best" free model more than once a day.

Fix: switched the default to `gemini-flash-lite-latest`, which handled a burst-tested 8 rapid
calls with zero throttling. Also added `call_gemini_with_retry()` in `agent.py`, which parses
the server's `retryDelay` out of 429/503 error payloads and backs off accordingly instead of
hammering a rate-limited endpoint — this matters for demo reliability, since Gemini's free tier
also returns transient 503s ("high demand") independent of your own quota. Every paper that
still fails after retries is dropped with a logged reason, not silently — you can see in
`sample_run_output.json` that 4 of 10 fetched papers were dropped, all for the same reason: the
LLM's relevance gate judged them off-topic (e.g. a general MASLD guideline that only mentions
semaglutide in passing), not extraction failures.

### 2. A real PubMed title crashed the console on Windows

While probing a broader query for this pass, a real paper title ("Semaglutide 2.4 mg...")
contained a Unicode **thin space (U+2009)** — not a hypothetical or crafted edge case, an actual
character in actual PubMed metadata. Windows' legacy per-codepage console (`cp1252` in an
ordinary terminal) can't encode it, and Rich's Windows renderer doesn't degrade gracefully: it
raised `UnicodeEncodeError` and killed the run mid-arbiter-output, after the expensive PubMed +
Gemini calls had already succeeded. An emoji I'd added to the HITL warning card (⚠) had the same
problem for the same reason. Chasing individual characters isn't a real fix — any title,
abstract, or LLM-generated justification could contain something outside cp1252. The actual fix
is at the stream level: `agent.py` now reconfigures `sys.stdout`/`sys.stderr` to UTF-8 with
`errors="replace"` at startup, so an unusual character degrades to a placeholder glyph instead
of taking down a run that had already done all its expensive work.

### 3. pyvis's embedded graph silently failed inside Streamlit's iframe

The knowledge graph rendered structurally in the dashboard (JSON side panel data was correct)
but the browser console showed `SyntaxError: Unexpected identifier 'Streamlit'` — pyvis's
default HTML (`cdn_resources="local"`) emits `<script src="lib/bindings/utils.js">` with a path
relative to wherever the HTML is served from. That's fine for a standalone file opened directly,
but this HTML is embedded via `st.components.v1.html()` inside an iframe whose base URL is
Streamlit's own dev server — the relative path resolved to Streamlit's *own index page* instead
of pyvis's JS bundle, which the browser then tried (and failed) to parse as a script. Fix:
`Network(..., cdn_resources="in_line")` inlines all of pyvis's JS/CSS directly into the
generated HTML, sidestepping the base-URL mismatch entirely. Caught by actually opening the
dashboard in a browser and reading the console — not by code review, which wouldn't have
surfaced a runtime-only, iframe-context-dependent path resolution bug.

### 4. A running dashboard silently locked the database file

Mid-development, `rm epistemic_memory.db` failed with `Device or resource busy` — a `streamlit`
process from an earlier verification pass was still holding the SQLite file open (Windows locks
files more aggressively than POSIX does; a still-running reader is enough). This isn't a code
bug so much as an operational one worth naming: SQLite connections opened by a long-running
Streamlit session don't release the file just because the browser tab closed. Fix at the time
was `taskkill` on the stray process; the durable takeaway (and why `graph_memory.init_db()`'s
migration path matters — see "Persistent knowledge graph" above) is that this repo is built to
tolerate *upgrading* an existing `epistemic_memory.db` in place rather than assuming you can
always delete and recreate it on demand. It recurred verbatim during this pass and was fixed
the same way — worth building a real "is anything holding this file open" check into a future
`Makefile`/dev script rather than re-discovering it by hand each time.

### 5. `predict_proba()`'s columns don't reliably mean `[PASS, CLAMP, REJECT]`

While building `SurrogateOperator.predict()`, the first version did
`label = int(proba.argmax())` — treating the argmax *index* as the label directly. That's wrong
whenever the training data so far doesn't contain all 3 action types: with only `confirm`(0) and
`reject`(2) seen, `LogisticRegression.classes_` becomes `[0, 2]` (only 2 columns), so index `1`
of `predict_proba()`'s output means label `2`, not label `1`. Early in training — exactly when a
real deployment is most likely to have an incomplete action mix — this would have silently
mispredicted CLAMP instead of REJECT. Fixed by mapping the argmax index back through
`model.classes_` explicitly; `test_surrogate_predict_proba_label_matches_classes_order` locks
this in with a training set that deliberately never sees a CLAMP example, asserting the
predicted label is always a real observed one.

### 6. A synthetic test's own prior mismatch silently understated surrogate confidence

Debugging a low-confidence surrogate prediction (0.51-0.79 instead of the expected >0.9) on
clean, hand-separated training data turned out to be a **test-harness bug, not a code bug**: the
test constructed a query paper via the normal `score_papers()` pipeline using the *default*
prior (the tier's static seed mean), while training rows had been seeded with a
*different, drifted* prior value for the same tier — so the "prior" feature for the query paper
sat between the two training clusters instead of matching either, and the model was correctly
uncertain given genuinely ambiguous input. Verified by reproducing the exact same features
directly against bare `sklearn.LogisticRegression` (0.97 confidence on the clean case) before
concluding the model itself was fine. The fixed test
(`test_resolve_anomaly_auto_resolves_when_surrogate_confident`) now explicitly passes a matching
`prior_lookup` so the query paper's features actually resemble what was trained on — a small but
real lesson about synthetic-data test design: a feature vector's parts have to agree with each
other, not just individually look "clearly PASS-like" or "clearly REJECT-like".

### 7. `st.components.v1.html` is already past its own removal date

Verifying the "Live PubMed Search" sidebar mode end-to-end (a real click, real PubMed+Gemini
calls) surfaced a runtime deprecation notice on the exact line rendering the knowledge graph:
`st.components.v1.html` is slated for removal, and the removal date named in the notice had
already passed relative to the run's own clock — it just hadn't been enforced yet in the
installed Streamlit version. Rather than leave a fix pending on when a point release finally
drops it, switched to `st.iframe(html, height=660)`, Streamlit's documented replacement (`st.iframe`
embeds a raw HTML string directly when the `src` argument doesn't match a URL/path pattern, same
as the old call's behavior). Confirmed via `document.querySelectorAll('iframe').length === 1` and
an empty error-console after the swap — this is also why the two ingestion modes needed a real
click each, not just code review, to catch: a soon-to-be-removed API call doesn't fail
`py_compile` or the unit suite.

## What I'd flag as a known limitation (and defend anyway)

- **Citation counts are mocked** (`random.randint(0, 200)`), logged loudly at runtime, and
  visible in `run_output.json`. PubMed's base `efetch`/`esummary` endpoints don't return
  citation counts — that requires NCBI's iCite API or Scopus/Crossref, both of which need
  separate registration and rate-limit handling. I chose to mock-and-disclose rather than
  silently fake it or burn the limited build time on a second external integration for a
  secondary (25%-weighted) signal. **With more time**, swapping in the free
  [NIH iCite API](https://icite.od.nih.gov/api) (`GET /api/pubs?pmids=...`) is a ~20-line
  change to `fetch_pubmed()` and nothing downstream needs to change, since the posterior update
  only cares about the final `citations` integer.
- **The fail-safe anomaly gate never fired on real PubMed data in any live run.** Published,
  peer-reviewed clinical literature on this topic is, unsurprisingly, generally cautious — no
  paper in any live run made a claim absurd enough (D≥2.0) or ran an interventional trial small
  enough (N<30) to trip the gate, so `AUTO_RESOLVED_BY_SURROGATE`/`ASYNC_QUARANTINED` don't
  appear in the committed `sample_run_output.json`. That's a property of the corpus, not
  evidence the mechanism doesn't work: both the surrogate-confident path and the
  fixed-fail-safe-fallback path were exercised directly (not just unit-tested) against synthetic
  anomaly data, and the automated suite has dedicated coverage for the anomaly gate, the
  surrogate (training, feature extraction, confidence routing), and the non-blocking resolution
  itself. **With more time**, I'd want at least one intentionally-adversarial query in the demo
  corpus (a preprint server or a known-retracted paper) so the gate fires on genuinely real
  data. (The adversarial calibration engine, `backtest_calibration.py`, DID fire real VULNERABLE
  verdicts against real data in a live run — see "Autonomous calibration engine" above — but
  that's a separate mechanism that judges papers directly, not through this anomaly gate.)
- **The ML surrogate has no train/holdout split, cross-validation, or overfitting guard.**
  `SurrogateOperator._fit()` trains on 100% of `feedback_log ⋈ nodes` history and immediately
  predicts on new data — with `MIN_TRAINING_ROWS=10` as the only floor, an early model can be
  confidently wrong on a small, non-representative history (this is exactly why the
  `CONFIDENCE_THRESHOLD=0.85` bar and the conservative fixed fail-safe below it exist: a
  low-data model is expected to often fall back rather than act). **With more time:** a rolling
  holdout accuracy check, and refusing to trust the model at all until holdout accuracy clears
  some bar (not just row count), would be a meaningfully stronger guarantee than "10 rows and a
  confidence threshold."
- **`FAILSAFE_PRIOR=0.20` and `FAILSAFE_LIKELIHOOD=0.05` are fixed constants, not derived from
  anything in the data** — chosen to be deliberately harsh (a paper that trips the gate and
  can't be confidently resolved should rank low, not "average") but not empirically tuned.
  **With more time:** these could themselves be learned (e.g. the tier's own low-percentile
  historical credence) rather than hand-picked.
- **The dashboard's Live Calibration Console, "Execute Pipeline" (Live PubMed Search), and the
  Prior Calibration State Management import/reset controls all write directly to the same
  `epistemic_memory.db` a real `agent.py` CLI run would use next**, with no confirmation step
  before committing (Reset Priors to Default is the one exception — it's gated behind an explicit
  checkbox). This is why the committed `epistemic_memory.db`/`sample_run_output.json` pair always
  reflects a plain, freshly-regenerated `agent.py` CLI run rather than whatever state a live
  dashboard click last left behind (see "Edge cases" above) — every dashboard write path was
  verified live during development, then the committed DB was regenerated clean before committing
  so it stays consistent with the committed JSON. **With more time:** "Live PubMed Search" and the
  calibration console should default to a scratch DB path (mirroring `backtest_calibration.py
  --db`'s own advice) rather than the live one, and prior import should preview the diff before
  applying it.
- **`p_value` is extracted but never used in scoring.** The telemetry schema captures it because
  the spec asked for it, and it's visible in the Paper Inspector and `run_output.json` for a
  human to read, but only the CI-derived Standard Error feeds the Precision Penalty. A p-value
  alone (without the CI it was computed from) doesn't cleanly convert to the same SE-shaped
  penalty, and I didn't want to invent a second, differently-shaped statistical penalty under
  deadline pressure just to use a field that mostly duplicates what the CI already signals.
  **With more time:** a p-value-only fallback penalty (when CI bounds are absent but a p-value
  is present) is a natural, scoped addition.
- **The preregistration bonus trusts the LLM's read of the abstract**, not an independent
  registry lookup — `is_preregistered=True` means "the abstract text mentions something that
  looks like a trial registry ID," not "I verified this NCT number against ClinicalTrials.gov."
  A hallucinated or misread registry mention would incorrectly earn the +0.05 bonus. **With more
  time:** a regex extraction of the actual NCT/ISRCTN identifier plus a live registry lookup
  would convert this from "the LLM says so" to "verified," and is a natural extension of the
  `ci_lower`/`ci_upper`/`p_value` fields already being extracted as plain data, not judgments.
- **`backtest_calibration.py`'s self-optimizing loop has no convergence guarantee or dampening.**
  Each `--iterations` pass re-judges the same static batch and applies feedback again — nothing
  stops `alpha`/`beta` from drifting arbitrarily far from the seed values over many iterations
  against a small, non-independent sample (the same handful of papers, re-critiqued by a model
  that isn't perfectly consistent run to run at `temperature=0.4`). This is disclosed in the
  tool's own docstring/`--help`, not hidden — it's explicitly a calibration/backtesting tool, run
  deliberately, not something `agent.py`'s normal pipeline invokes automatically. **With more
  time:** a decay/cap on cumulative iteration influence, or requiring a larger and more diverse
  backtest corpus before trusting multi-iteration runs, would make repeated runs safer.
- **`audit_status` has only 3 values** (`PASSED`/`FLAGGED`/`OVERRIDDEN`, per the given node
  schema), so a manual prior clamp and an outright quarantine/reject are indistinguishable from
  the node's `audit_status` alone — both land on `OVERRIDDEN`. The finer distinction (which
  specific action happened) is recoverable from `feedback_log.action`, just not from the node
  table itself. A 4th status or a separate `quarantined: bool` column would remove the ambiguity
  — I kept the schema exactly as specified rather than extending it speculatively.
- **The LLM extraction step can be wrong** (e.g. misreading sample size from an abstract that
  reports it awkwardly). I don't hide this: `run_output.json` persists every paper's raw
  telemetry, so a reviewer can spot-check the LLM's extraction against the actual abstract for
  any paper in the ranking. I did not build a second-LLM cross-check or human-in-the-loop
  correction step — that's the next thing I'd add with more time, most likely as a "flag for
  review if two independent extraction calls disagree" step rather than fully trusting a
  single pass.
- **Transient errors (429/503) retry with backoff, but a genuinely malformed LLM response
  (fails Pydantic validation) still just drops the paper** rather than re-prompting. Simple and
  predictable — a single mis-shaped response quietly shrinks the candidate pool by one instead
  of retrying with a corrective prompt. Worth adding, with more time.
- **Single topic, single query.** The agent works for any PubMed query, but I didn't build a
  multi-topic or multi-drug comparison mode. Next step would be parameterizing the whole
  pipeline over a list of queries and adding a cross-topic leaderboard.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
- `ENTREZ_EMAIL` — any real email (NCBI courtesy requirement, not validated as an account).
- `GEMINI_API_KEY` — free key from [Google AI Studio](https://aistudio.google.com/apikey).

## Run

```bash
python agent.py
```

Optional flags:

```bash
python agent.py --query "your own PubMed search" --limit 10 --model gemini-flash-lite-latest --seed 42 --db epistemic_memory.db
python agent.py --batch-size 30   # >10 auto-paces Gemini calls (>=4.2s apart) for the free tier
```

`--limit`/`--batch-size`/`--max-results` are aliases for the same value (5-50, validated). `--seed`
makes the mocked citation numbers reproducible between runs (real telemetry extraction and
arbitration still depend on the live LLM, so wording will vary run to run even with a seed —
only the citation mock is deterministic). `--db` points at a different `epistemic_memory.db` if
you want an isolated store. There is no `--interactive` flag anymore — every flagged paper is
resolved automatically and non-blockingly (see "Fail-safe" above); nothing in this pipeline
waits on a terminal prompt.

Every run writes a full structured record — every paper, every extracted telemetry field,
every score component, the arbiter's justification text, and each paper's audit status — to
`run_output.json`, and persists the knowledge graph + any learned priors to
`epistemic_memory.db`.

Then explore it live:

```bash
streamlit run ui.py
```

The dashboard's own sidebar can also drive a fresh pipeline run directly (no separate
`python agent.py` invocation needed) via its "Live PubMed Search" mode — see "Dashboard" below.

Or run the adversarial calibration engine against a completed run (separate from the normal pipeline):

```bash
python backtest_calibration.py --iterations 5 --input sample_run_output.json
```

Run the offline logic tests any time (no keys, no network, <1s):

```bash
python test_matrix.py
# or, exactly as specified:
python -m pytest test_matrix.py -v
```

## Repo layout

```
agent.py                 run_pipeline(): fetch -> extract -> filter -> score -> non-blocking
                          fail-safe -> arbiter, called by both main() (CLI) and ui.py's "Live
                          PubMed Search" sidebar mode. Configurable batch size (--limit/
                          --batch-size, 5-50) with adaptive rate pacing for batches > 10.
graph_memory.py          persistent knowledge graph (SQLite + networkx) + Bayesian active
                          learning + OOD/Jeffreys priors + signed citation/contradiction
                          topology + ML SurrogateOperator + calibration_history convergence log
                          + prior export/import/reset + node/edge rendering color-and-size math
backtest_calibration.py  autonomous adversarial calibration engine (separate CLI entry point)
ui.py                    two-tab Streamlit dashboard (streamlit run ui.py), dual-mode ingestion
                          (Live PubMed Search / Load Cached Run), no emoji, LaTeX scoring panel
test_matrix.py           79 offline unit tests (posterior formula, anomaly gate, Beta updates,
                          OOD priors, precision penalty, citation sentiment, calibration mapping,
                          batch-size validation, surrogate training/prediction, non-blocking
                          resolution, calibration_history snapshots, signed edge/node styling,
                          prior export/import/reset, run_pipeline wiring)
requirements.txt
.env.example
sample_run_output.json   a real run's full output, committed so the ranking + justifications
                          are visible without anyone needing their own API keys
epistemic_memory.db      that same real run's persisted knowledge graph + learned priors,
                          committed so the dashboard has real data to show without a fresh run
```
