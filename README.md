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
[SCORE]      Deterministic Python matrix       Reasoning Step 2: the judgment call
[ARBITER]    Gemini (structured JSON)          LLM Step 2: defends the ranking it was given
```

### Reasoning Step 1 — the relevance gate

The LLM reads each abstract and returns `is_relevant: bool`. Anything not directly about
semaglutide's effect on liver fibrosis (e.g. a general MASLD guideline that only mentions
semaglutide in passing) is dropped before scoring even starts. This is keyword-independent —
a paper can mention "semaglutide" and "fibrosis" in the same abstract and still get filtered
if it isn't actually reporting on that relationship.

### Reasoning Step 2 — the judgment call (this is the part with no objectively correct answer)

For each surviving paper:

```
S = (0.75 × Methodology) + (0.25 × Velocity_norm) − Penalty
```

- **Methodology** (0.1–1.0): a fixed rubric — Meta-Analysis > RCT > Prospective Cohort >
  Retrospective/Observational > In-Vitro/Animal > Review/Opinion.
- **Velocity_norm** (0–1): citations ÷ years-since-publication, min-max normalized across the
  batch. Rewards papers the field is actually engaging with, independent of raw age.
- **Penalty** (0 or 0.5): applied when a *low-rigour* design (retrospective/observational or
  in-vitro/animal) is paired with *high-hyperbole* language (≥4/5). A cautious in-vitro study
  is not penalized — only overclaiming from weak evidence is.

**Why 0.75/0.25 and not 0.5/0.5, or the reverse?** This is the judgment call the assignment
asks for, and it's defensible rather than "correct":

- A GLP-1/fibrosis result that hasn't been replicated in an RCT is not made more true by going
  viral. Citation velocity measures *attention*, not *validity* — a controversial or clinically
  urgent paper accumulates citations quickly whether or not it holds up. Weighting it at 75%
  would let a hyped-but-flimsy paper outrank a boring, solid one.
- Velocity isn't thrown out entirely (weight 0), because a rigorous paper the field is
  ignoring and a rigorous paper the field is actively building on aren't quite equivalent —
  the second has survived more scrutiny. 0.25 lets velocity break near-ties between
  similar-rigour papers without letting it override a methodology gap.
- **The alternative I rejected:** weighting recency instead of/alongside velocity. I chose
  velocity (citations/age) over raw recency because a 2020 meta-analysis that's still being
  cited heavily represents more *accumulated, tested* evidence than a 2026 single-center
  retrospective study that hasn't had time to be challenged yet — "newest" and "most trusted"
  are not the same axis, and this task is about resolving conflicts in trust, not surfacing
  news.

Verify the logic yourself: `python test_matrix.py` runs 5 network-free unit tests against the
matrix — confirming, among other things, that a Meta-Analysis with zero citations still
outranks an RCT, that a hyped in-vitro study with the highest velocity in the batch still gets
pushed to the bottom by the penalty, and that a batch where every paper ties on velocity
doesn't divide by zero. The formula does what it claims to do, and that claim is checked, not
just asserted in prose.

### LLM Step 2 — the arbiter

The top 3 papers, their extracted telemetry, and their computed scores are sent back to the
LLM with an explicit instruction: **explain this ranking, don't re-decide it.** It writes a
3-sentence, telemetry-grounded justification per paper and a short note on which factor
(methodology, velocity, or the hype penalty) actually separated rank 1 from rank 2. This is
the part of the output you read out loud in the demo.

## An edge case I hit for real, not hypothetically

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
`sample_run_output.json` that 3 of 10 fetched papers were dropped, all for the same reason: the
LLM's relevance gate judged them off-topic (e.g. a general MASLD guideline that only mentions
semaglutide in passing), not extraction failures.

## What I'd flag as a known limitation (and defend anyway)

- **Citation counts are mocked** (`random.randint(0, 200)`), logged loudly at runtime, and
  visible in `run_output.json`. PubMed's base `efetch`/`esummary` endpoints don't return
  citation counts — that requires NCBI's iCite API or Scopus/Crossref, both of which need
  separate registration and rate-limit handling. I chose to mock-and-disclose rather than
  silently fake it or burn the limited build time on a second external integration for a
  secondary (25%-weighted) signal. **With more time**, swapping in the free
  [NIH iCite API](https://icite.od.nih.gov/api) (`GET /api/pubs?pmids=...`) is a ~20-line
  change to `fetch_pubmed()` and nothing downstream needs to change, since the scoring matrix
  only cares about the final `citations` integer.
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
python agent.py --query "your own PubMed search" --max-results 10 --model gemini-flash-lite-latest --seed 42
```

`--seed` makes the mocked citation numbers reproducible between runs (real telemetry
extraction and arbitration still depend on the live LLM, so wording will vary run to run even
with a seed — only the citation mock is deterministic).

Every run writes a full structured record — every paper, every extracted telemetry field,
every score component, and the arbiter's justification text — to `run_output.json`.

Run the offline logic tests any time (no keys, no network, <1s):

```bash
python test_matrix.py
```

## Repo layout

```
agent.py                  single-file pipeline (all 4 modules)
test_matrix.py            5 offline unit tests for the deterministic scoring matrix
requirements.txt
.env.example
sample_run_output.json    a real run's full output, committed so the ranking + justifications
                           are visible without anyone needing their own API keys
```
