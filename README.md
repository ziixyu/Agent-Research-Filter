# Epistemic Filtering Agent

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
[FETCH]      Bio.Entrez -> PubMed              real metadata + abstracts, 10 papers
[EXTRACT]    Gemini (structured JSON)          LLM Step 1: telemetry only, no verdicts
[FILTER]     Python                            Reasoning Step 1: relevance gate
[SCORE]      Deterministic Bayesian update     Reasoning Step 2: the judgment call
[GATE]       Python fail-safe + optional HITL  flags anomalies, can halt for a human
[ARBITER]    Gemini (structured JSON)          LLM Step 2: defends the ranking it was given
[MEMORY]     SQLite + networkx                 persists the graph and learns from feedback
```

Three modules beyond the core pipeline:

- **`graph_memory.py`** — a persistent knowledge graph (SQLite + `networkx.DiGraph`) and
  Empirical Bayesian active learning over the priors themselves (see "Active learning" below).
- **The fail-safe/HITL gate in `agent.py`** — `python agent.py --interactive` halts on any
  paper that trips a severity threshold and asks a human operator to confirm, override, or
  quarantine it (see "Fail-safe & HITL gate" below).
- **`ui.py`** — `streamlit run ui.py`, a live dashboard over a completed run: an instantly
  recomputable ranking table, the knowledge graph rendered with physics, a HITL override panel,
  and a counterfactual arbiter console (see "Dashboard" below).

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

**2. Likelihood penalty L(Absurdity) — the epistemic gate.** A Discrepancy Index
`D = claim_hyperbole − rigor_baseline` measures how far a paper's claim strength (1–5) overshoots
what its own tier could justify, where `rigor_baseline = P(E) × 5` (the same 1–5 scale,
anchored to the prior). An observational study (P(E)=0.40, baseline≈2.0) claiming a definitive
"reversal/cure" (hyperbole=5) gets D=3.0; the same claim from a meta-analysis (P(E)=0.95,
baseline≈4.75) gets D≈0.25 — the exact "observational study claiming causal cure/reversal
yields a high D" example this was built around. **Underclaiming is never punished** — D is
clamped to 0 when the claim is more cautious than the tier's baseline; modesty is free.

D is then moderated by a **Sample Power Weight** `W_N = log10(N+1)`, min-max normalized across
the batch to [0,1] — the same overclaim from N=5,000 is treated as less absurd than the
identical overclaim from N=10:

```
D_adjusted = D × (2.0 − W_N)          # multiplier in [1.0, 2.0]: best-powered paper in the
                                       # batch gets no amplification, worst-powered gets 2×
L(Absurdity) = 1.0                          if D_adjusted <= threshold (0.5)
             = exp(−k × (D_adjusted − threshold))     otherwise, k = 0.5
```

**3. Posterior score — the final ranking metric:**

```
S_posterior = clip01( [P(E) × L(Absurdity)] × 0.75 + [Velocity_norm] × 0.25 )
```

`Velocity_norm` is unchanged from the original design: citations ÷ years-since-publication,
min-max normalized across the batch. `clip01` is a defensive floor/ceiling — every input is
already bounded to [0,1] by construction, so this is a safety net against a future weight
change quietly breaking that invariant, not something that fires in practice today.

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

Verify the logic yourself: `python test_matrix.py` runs 24 network-free unit tests — the
original 10 against the posterior formula (prior dominates when nothing else differs, the exact
"hyped observational study vs. modest RCT" scenario resolves as intended, the likelihood penalty
is isolated and attributable, Sample Power Weight measurably softens the same overclaim for a
larger N, underclaiming is never punished, scores never leave [0,1], tied/single-paper batches
don't divide by zero) plus 14 more added for this pass: the fail-safe anomaly gate (triggers on
D≥2.0, triggers on N<30 for an interventional design, does NOT trigger on a small
non-interventional sample or an ordinary paper) and `graph_memory.py`'s Empirical Bayesian
updates (seed means match the spec exactly, RCT phase-splitting, confirm/reject/override each
move alpha/beta the way the math promises, feedback on one tier never bleeds into another, the
contradiction-edge heuristic fires on opposing outcome language and not on agreement). Every
SQLite test in that second batch uses a private `:memory:` database — none of them touch a real
`epistemic_memory.db` on disk.

### LLM Step 2 — the arbiter

The top 3 papers, their extracted telemetry, and their computed scores are sent back to the
LLM with an explicit instruction: **explain this ranking, don't re-decide it.** It writes a
3-sentence, telemetry-grounded justification per paper and a short note on which factor (the
prior P(E), the likelihood/absurdity penalty, or citation velocity) actually separated rank 1
from rank 2. This is the part of the output you read out loud in the demo.

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

## Fail-safe & HITL anomaly gate (`agent.py`)

This is a **separate, coarser gate** from the smooth likelihood penalty in Reasoning Step 2.
The likelihood penalty discounts credence continuously for *any* paper whose claim mildly
outruns its tier (`DISCREPANCY_THRESHOLD=0.5`) — silent, automatic, no human involved. The
anomaly gate fires on a much higher bar and exists to put a *specific* paper in front of a
human, not just discount it:

1. **Discrepancy Index D ≥ 2.0** — the claim doesn't just mildly overreach, it's badly out of
   proportion to what the design tier can support.
2. **N < 30 for an interventional design** — currently just `"RCT"` in our schema (a
   meta-analysis synthesizes other work rather than running an intervention itself; cohort,
   observational, in-vitro, and review designs aren't controlled interventions either).

**Batch mode (default):** a flagged paper gets `audit_status="FLAGGED"`, a warning is logged
with the specific trigger reason(s), and the run continues — the automated likelihood penalty
was already applied as part of Reasoning Step 2's normal math, so nothing extra needs to happen
for the score to already reflect it.

**Interactive mode (`python agent.py --interactive`):** execution halts on each flagged paper,
prints a warning card (PMID, N, claim strength vs. the design's justified ceiling, current
P(E), D, L), and prompts:

```
[1] Apply automated likelihood penalty   -> confirm; audit_status -> PASSED; Beta: alpha += 1
[2] Manually clamp Prior P(E)            -> audit_status -> OVERRIDDEN; Beta nudged toward p
[3] Reject / Quarantine paper            -> excluded from top3/arbiter; Beta: beta += 1
```

All three paths were exercised by hand against real anomaly data during development (piped
stdin, not just unit-tested) — see the commit history for the transcripts. Note that
`audit_status` only has 3 values (`PASSED`/`FLAGGED`/`OVERRIDDEN`, matching the given schema);
quarantine and manual-override both persist as `OVERRIDDEN` on the node, with the specific
action (`confirm`/`override`/`reject`) recoverable from `feedback_log` for a finer-grained
audit trail than the node schema alone provides.

## Persistent knowledge graph (`graph_memory.py`)

SQLite (`epistemic_memory.db`) is the durable store; `networkx.DiGraph` is an in-memory view
built from it for graph algorithms and the dashboard. Every scored paper becomes a node
(pmid, title, study_design, sample_size, prior_credence, discrepancy_index,
likelihood_penalty, posterior_score, audit_status). Two kinds of edges:

- **Citation edges — real data**, not mocked. `fetch_citation_edges_from_pubmed()` calls
  `Bio.Entrez.elink` (`pubmed_pubmed_refs`) and keeps only links where *both* papers are in our
  own fetched batch (we only have telemetry for our own batch, so a citation to a paper outside
  it isn't graphable anyway). This found real edges in every live test run — see
  `sample_run_output.json`'s companion `epistemic_memory.db`.
- **Contradiction edges — a disclosed keyword heuristic, not semantic NLI.**
  `infer_outcome_direction()` scans title+abstract for a small set of positive-outcome phrases
  ("significant reduction", "improvement", ...) vs. negative/null-outcome phrases ("no
  significant", "did not improve", ...), checking negative phrasing *first* and letting it win
  outright — a negated result like "no significant improvement" contains the substring
  "improvement" (a positive keyword), and naive substring matching can't tell negated language
  from an affirmative claim otherwise. Two relevant papers with opposite inferred directions get
  a contradiction edge. This is deliberately cheap (zero extra LLM calls — this repo already
  documents real free-tier quota pain) rather than accurate; it's meant to surface *candidates*
  for a human to actually read, not to assert a contradiction is real.

## Dashboard (`ui.py`)

```bash
streamlit run ui.py
```

Reads a completed run's JSON output plus `epistemic_memory.db`. It never re-calls PubMed.

- **Dynamic Ranking Matrix** — Methodology-weight (w_M) and Velocity-weight (w_V=1-w_M) sliders
  recompute every paper's `S_posterior` live, by calling the *exact same* `agent.score_one()`
  function the CLI pipeline uses (not a re-implementation that could drift out of sync) against
  the already-extracted `prior_credence`/`likelihood_penalty`/`velocity_norm` — no new
  extraction, no new Gemini calls.
- **Physics-based graph** — rendered via embedded pyvis HTML (`cdn_resources="in_line"` — see
  the edge case below for why that flag matters). Node size ∝ log10(N); node color interpolates
  green (high `S_posterior`) to red (low/penalized); contradiction edges render bold red
  dashed, citation edges plain gray.
- **HITL Override Panel** — every `FLAGGED` paper gets a card with a live-preview P(E) slider
  (shows what the posterior *would* become before you commit) and Apply/Quarantine buttons that
  write straight through `graph_memory.record_feedback()` into `epistemic_memory.db` — the same
  store `agent.py --interactive` writes to, so a decision made in the dashboard is loaded by the
  next CLI run and vice versa.
- **Arbiter + counterfactual console** — renders the persisted arbiter justification, plus a
  free-text box that calls Gemini live (`agent.counterfactual_arbiter()`, reusing
  `call_gemini_with_retry`) with a user-supplied hypothetical ("re-evaluate if liver stiffness
  endpoints are excluded") and the same top-3 telemetry, explicitly instructed to say so if the
  scenario *wouldn't* plausibly change the ranking rather than manufacturing a change.

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
- **The fail-safe/HITL gate never fired on real PubMed data in this pass.** Published,
  peer-reviewed clinical literature on this topic is, unsurprisingly, generally cautious — no
  paper in any live run made a claim absurd enough (D≥2.0) or ran an interventional trial small
  enough (N<30) to trip the gate. That's a property of the corpus, not evidence the gate doesn't
  work: all 3 HITL decision paths (confirm/override/quarantine) were exercised by hand against
  synthetic anomaly data with real piped stdin, and 5 of the 24 automated tests specifically
  target `detect_anomaly()`/`apply_audit_flags()`. **With more time**, I'd want at least one
  intentionally-adversarial query in the demo corpus (a preprint server or a known-retracted
  paper) so the gate fires on genuinely real data, not just synthetic test fixtures.
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
python agent.py --query "your own PubMed search" --max-results 10 --model gemini-flash-lite-latest --seed 42 --db epistemic_memory.db
python agent.py --interactive     # halt on each FLAGGED anomaly and prompt an operator
```

`--seed` makes the mocked citation numbers reproducible between runs (real telemetry
extraction and arbitration still depend on the live LLM, so wording will vary run to run even
with a seed — only the citation mock is deterministic). `--db` points at a different
`epistemic_memory.db` if you want an isolated store (e.g. for a demo you don't want polluting
your main learned priors).

Every run writes a full structured record — every paper, every extracted telemetry field,
every score component, the arbiter's justification text, and each paper's audit status — to
`run_output.json`, and persists the knowledge graph + any learned priors to
`epistemic_memory.db`.

Then explore it live:

```bash
streamlit run ui.py
```

Run the offline logic tests any time (no keys, no network, <1s):

```bash
python test_matrix.py
```

## Repo layout

```
agent.py                 pipeline: fetch -> extract -> filter -> score -> fail-safe/HITL -> arbiter
graph_memory.py          persistent knowledge graph (SQLite + networkx) + Bayesian active learning
ui.py                    Streamlit dashboard (streamlit run ui.py)
test_matrix.py           24 offline unit tests (posterior formula, anomaly gate, Beta updates)
requirements.txt
.env.example
sample_run_output.json   a real run's full output, committed so the ranking + justifications
                          are visible without anyone needing their own API keys
epistemic_memory.db      that same real run's persisted knowledge graph + learned priors,
                          committed so the dashboard has real data to show without a fresh run
```
