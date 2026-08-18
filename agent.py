#!/usr/bin/env python3
"""
Epistemic Filtering Agent
==========================
Ranks conflicted clinical-literature abstracts on Semaglutide's effect on
NASH/MASH liver fibrosis, without trusting any single paper's own framing of
its results.

Pipeline (see README.md for the reasoning behind each design choice):

    [FETCH]      Bio.Entrez -> PubMed          (real metadata, real abstracts)
    [EXTRACT]    Gemini structured JSON         (LLM Step 1: telemetry, not verdicts)
    [FILTER]     Python                         (Reasoning Step 1: relevance gate)
    [SCORE]      Deterministic Bayesian update  (Reasoning Step 2: the judgment call)
    [ARBITER]    Gemini structured JSON         (LLM Step 2: defends the ranking)

Run:
    python agent.py
    python agent.py --query "your own PubMed query" --max-results 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

import graph_memory

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

# Real PubMed titles/abstracts can contain arbitrary Unicode (thin spaces,
# curly quotes, non-Latin characters, ...) that Windows' legacy per-codepage
# console (cp1252 in an ordinary cmd.exe/older terminal) cannot encode —
# this crashed a real run mid-arbiter-output with a UnicodeEncodeError on a
# U+2009 THIN SPACE inside an actual paper title, not a hypothetical. Force
# UTF-8 with a replace-on-error fallback so unusual characters degrade to a
# placeholder glyph instead of taking down the whole run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — never let console setup itself crash the run
            pass

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, markup=True)],
)
log = logging.getLogger("agent")

CURRENT_YEAR = datetime.now().year

DEFAULT_QUERY = 'Semaglutide AND ("NASH" OR "MASH") AND fibrosis'
DEFAULT_MAX_RESULTS = 10

# --------------------------------------------------------------------------
# Module 3 formalization: Bayesian/epistemic state update.
#
# See README.md "The judgment call, formalized" for the full argument. Short
# version: a paper's final ranking score is treated as a POSTERIOR belief in
# its trustworthiness, built from a design-tier PRIOR that gets discounted by
# a LIKELIHOOD penalty when the paper's own claims outrun what its evidence
# tier can support, then blended with citation velocity.
# --------------------------------------------------------------------------

# Prior epistemic credence P(E) is no longer a fixed number per design tier.
# It's now the MEAN of a Beta(alpha, beta) distribution tracked persistently
# in graph_memory.py / epistemic_memory.db, which shifts every time a human
# operator confirms, overrides, or rejects a flagged paper (Empirical
# Bayesian active learning — see graph_memory.py and README.md). What used
# to be a static PRIOR_CREDENCE dict here is now:
#   - graph_memory.seed_prior_means() — pure, I/O-free fallback used when no
#     prior_lookup is supplied (keeps test_matrix.py deterministic), and
#   - graph_memory.get_current_prior_means(conn) — the live, learned values
#     main() actually loads from epistemic_memory.db for a real run.
# Both are keyed by the 6 Beta "tiers", resolved from a paper's
# (study_design, sample_size) via graph_memory.beta_tier_for().

# Likelihood penalty (the "epistemic gate"): a paper's claim_hyperbole
# (1-5) is compared against a rigor baseline derived from its own prior —
# rigor_baseline = P(E) * 5 puts "how strong a claim this evidence tier can
# justify" on the same 1-5 scale claim_hyperbole is scored on. The gap
# (Discrepancy Index D) is only penalized when it's positive (overclaiming);
# a cautious paper is never punished for being modest.
DISCREPANCY_THRESHOLD = 0.5  # tolerate small D as scoring noise, not absurdity
LIKELIHOOD_DECAY_K = 0.5  # exp(-k * D): higher k = harsher penalty per unit of overreach

W_PRIOR = 0.75  # weight on prior*likelihood (rigor + claim-calibration)
W_VELOCITY = 0.25  # weight on citation velocity (attention, not validity)

# These 6 categories are the ones our Beta-tier system knows out of the box
# (see graph_memory.SEED_BETA_PRIORS) and the ones the extraction prompt
# lists as preferred. PaperTelemetry.study_design is NOT restricted to this
# list, though — it's a free-text str, deliberately, so a genuinely novel
# design (e.g. "Mendelian Randomization") can be reported honestly instead
# of forced into the wrong bucket. See graph_memory.get_prior_credence()
# for how an out-of-distribution design gets an uninformative Jeffreys
# prior instead of crashing or borrowing an unrelated tier's credence.
KNOWN_STUDY_DESIGNS = (
    "Meta-Analysis",
    "RCT",
    "Prospective Cohort",
    "Retrospective/Observational",
    "In-Vitro/Animal",
    "Review/Opinion",
)

# Statistical precision penalty (from reported 95% CI bounds, when given):
# SE = (ci_upper - ci_lower) / 3.92 is the standard normal-approximation
# formula (a 95% CI spans ~1.96 SE on each side of the estimate). A wide CI
# for the same design/hyperbole means the underlying estimate is noisier
# than the claim lets on, so it discounts credence the same shape as the
# Discrepancy Index does — smoothly, past a tolerance threshold, never for
# papers that simply didn't report CI bounds at all (SE=None -> no penalty).
SE_PENALTY_THRESHOLD = 0.5
PRECISION_DECAY_K = 0.5

# Preregistration bonus: a trial registry ID (NCT/ISRCTN/...) makes
# after-the-fact outcome-switching and cherry-picking harder to get away
# with, so it's worth a small, capped boost to the design-tier prior — not
# a new evidence tier of its own, just a modifier on top of one.
PREREGISTRATION_BONUS = 0.05


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    """Raw metadata pulled straight from PubMed. No judgment happens here."""

    pmid: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    publication_year: int
    citations: int
    citations_mocked: bool = False
    doi: Optional[str] = None
    url: str = ""  # auto-filled from pmid below if not given explicitly

    @model_validator(mode="after")
    def _fill_url(self):
        if not self.url:
            self.url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return self


class PaperTelemetry(BaseModel):
    """Structured epistemic state extracted from the abstract by the LLM.

    Deliberately narrow: the LLM is only allowed to report *what kind of
    evidence this is*, never *how good the evidence is*. That judgment is
    reserved for the deterministic matrix in Module 3. study_design is
    intentionally a free-text str (not a Literal enum) so a genuinely novel
    design isn't forced into the wrong known bucket — see
    KNOWN_STUDY_DESIGNS and graph_memory.get_prior_credence().
    """

    is_relevant: bool = Field(
        description="True only if the abstract directly discusses Semaglutide "
        "(or a GLP-1 receptor agonist explicitly identified as semaglutide) "
        "and liver fibrosis / NASH / MASH outcomes."
    )
    study_design: str = Field(
        description="Prefer one of: Meta-Analysis, RCT, Prospective Cohort, "
        "Retrospective/Observational, In-Vitro/Animal, Review/Opinion. Only use a "
        "different, more specific label (e.g. 'Mendelian Randomization', "
        "'Organ-on-a-Chip') if the study's actual design genuinely doesn't fit any "
        "of those — do not force a bad fit just to match the list."
    )
    sample_size: int = Field(description="Extracted N. 0 if not explicitly stated.")
    claim_hyperbole: int = Field(
        ge=1,
        le=5,
        description="1 = cautious/grounded language. 5 = definitive causal "
        "claims the study design cannot actually support.",
    )
    ci_lower: Optional[float] = Field(
        default=None,
        description="Reported lower bound of the primary endpoint's 95% CI, if stated "
        "(e.g. '95% CI: 1.2-3.4' -> 1.2). Null if not reported.",
    )
    ci_upper: Optional[float] = Field(
        default=None,
        description="Reported upper bound of the primary endpoint's 95% CI, if stated "
        "(e.g. '95% CI: 1.2-3.4' -> 3.4). Null if not reported.",
    )
    p_value: Optional[float] = Field(
        default=None,
        description="Reported p-value for the primary endpoint, if stated. Null if not reported.",
    )
    is_preregistered: bool = Field(
        description="True if a trial registry identifier (e.g. NCT#########, "
        "ISRCTN########) is mentioned anywhere in the abstract."
    )


class ScoredPaper(BaseModel):
    """A paper after Module 3's Bayesian posterior update has run.

    Naming follows the epistemic-update framing: prior_credence is P(E),
    likelihood_penalty is L(Absurdity), posterior_score is the final
    ranking metric S_posterior. See README.md for the full derivation.
    """

    metadata: PaperMetadata
    telemetry: PaperTelemetry

    base_prior_credence: float  # P(E) before the preregistration bonus, from the Beta tier mean
    preregistration_bonus: float  # +0.05 if telemetry.is_preregistered, else 0.0
    prior_credence: float  # EFFECTIVE P(E) actually used below = base + bonus, clipped [0,1]
    rigor_baseline: float  # prior_credence * 5: the claim strength this tier can justify
    discrepancy_index: float  # D = claim_hyperbole - rigor_baseline (clamped >= 0)
    sample_power_weight: float  # W_N: log10(N+1), batch-normalized to [0, 1]
    discrepancy_adjusted: float  # D moderated by W_N (small-N overclaims hit harder)
    likelihood_penalty: float  # L(Absurdity) = exp(-k * max(0, D_adj - threshold))

    standard_error: Optional[float] = None  # SE = (ci_upper - ci_lower) / 3.92, if CI reported
    precision_penalty: float = 1.0  # exp(-k * max(0, SE - threshold)); 1.0 if no CI reported

    velocity_raw: float
    velocity_norm: float

    posterior_score: float  # S_posterior, clipped [0,1] — see score_one() for the full formula

    is_ood_design: bool = False  # True if telemetry.study_design needed a Jeffreys-prior tier

    # Epistemic fail-safe / HITL gate (see detect_anomaly/apply_audit_flags):
    audit_status: Literal["PASSED", "FLAGGED", "OVERRIDDEN"] = "PASSED"
    audit_reasons: list[str] = Field(default_factory=list)  # why FLAGGED, if it was


class RankJustification(BaseModel):
    pmid: str
    rank: int
    justification: str = Field(description="Exactly 3 sentences, citing specific telemetry.")


class ArbiterVerdict(BaseModel):
    justifications: list[RankJustification]
    overall_defense: str = Field(
        description="2-4 sentences on which criterion (methodology vs velocity vs "
        "hype-penalty) actually decided the top-3 ordering, and why that's defensible."
    )


# --------------------------------------------------------------------------
# Module 1: Ingestion (PubMed)
# --------------------------------------------------------------------------


def fetch_pubmed(query: str, max_results: int) -> list[PaperMetadata]:
    from Bio import Entrez

    email = os.environ.get("ENTREZ_EMAIL")
    if not email:
        log.error(
            "[bold red]ENTREZ_EMAIL is not set.[/] Copy .env.example to .env and fill it in."
        )
        sys.exit(1)
    Entrez.email = email
    api_key = os.environ.get("ENTREZ_API_KEY")
    if api_key:
        Entrez.api_key = api_key

    log.info(f"[FETCHING] Searching PubMed: [cyan]{query}[/]")
    with Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance") as h:
        search_result = Entrez.read(h)
    pmids = search_result.get("IdList", [])
    if not pmids:
        log.warning("No PubMed results for this query.")
        return []
    log.info(f"[FETCHING] Got {len(pmids)} PMIDs, downloading abstracts...")

    with Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml") as h:
        records = Entrez.read(h)

    papers: list[PaperMetadata] = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            art = medline["Article"]
            title = str(art.get("ArticleTitle", "")).strip()

            abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts).strip()
            if not abstract:
                log.warning(f"  PMID {pmid}: no abstract available, dropping.")
                continue

            authors = []
            for a in art.get("AuthorList", []):
                if "LastName" in a and "ForeName" in a:
                    authors.append(f"{a['ForeName']} {a['LastName']}")
                elif "CollectiveName" in a:
                    authors.append(str(a["CollectiveName"]))

            # DOI: prefer ELocationID (EIdType='doi'); some records only carry
            # it in PubmedData.ArticleIdList (IdType='doi') instead — check both.
            doi = None
            for eloc in art.get("ELocationID", []):
                if getattr(eloc, "attributes", {}).get("EIdType") == "doi":
                    doi = str(eloc)
                    break
            if doi is None:
                for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
                    if getattr(aid, "attributes", {}).get("IdType") == "doi":
                        doi = str(aid)
                        break

            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year_str = pub_date.get("Year") or pub_date.get("MedlineDate", "")[:4]
            try:
                publication_year = int(year_str)
            except (ValueError, TypeError):
                publication_year = CURRENT_YEAR
                log.warning(f"  PMID {pmid}: unparseable year '{year_str}', assuming {CURRENT_YEAR}.")

            # PubMed's base esummary/efetch calls do not return citation counts
            # (that needs iCite or Scopus, out of scope for this exercise).
            # We mock it and log it loudly so nobody mistakes it for real data.
            citations = random.randint(0, 200)

            papers.append(
                PaperMetadata(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    publication_year=publication_year,
                    citations=citations,
                    citations_mocked=True,
                    doi=doi,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
            )
        except Exception as e:  # noqa: BLE001 - one bad record shouldn't kill the batch
            log.warning(f"  Skipping a malformed record: {e}")
            continue

    log.info(
        f"[FETCHING] [bold yellow]NOTE: citation counts are MOCKED[/] (random 0-200) "
        f"for all {len(papers)} papers — PubMed's base API does not expose citation "
        f"counts. See README.md for what a real deployment would use instead."
    )
    return papers


# --------------------------------------------------------------------------
# Module 2: Relevance gate & telemetry extraction (LLM Step 1)
# --------------------------------------------------------------------------

TELEMETRY_PROMPT = """You are an elite clinical data architect. Read the provided abstract.

First, determine if it is strictly relevant to the effect of Semaglutide on \
NASH/MASH liver fibrosis. It is NOT relevant if it is about a different drug, \
a different condition, or only tangentially mentions fibrosis without \
reporting outcomes.

If relevant, extract:
- study_design: the single best-fitting category for how the evidence was generated.
- sample_size: the exact N used in the primary analysis. Use 0 if not stated.
- claim_hyperbole: score 1-5, where 5 means the abstract makes definitive, \
absolute causal claims ("proves", "confirms", "eliminates the need for") \
that are not proportionate to the study's actual design and size, and 1 \
means the language is appropriately cautious ("suggests", "associated with", \
"warrants further study").

Do not evaluate whether the paper's findings are TRUE. Only characterize HOW \
the claim was made and what kind of evidence backs it.

Title: {title}

Abstract: {abstract}
"""


def call_gemini_with_retry(fn, *, max_retries: int = 5, base_delay: float = 2.0):
    """Call `fn()` (a zero-arg thunk making one Gemini request), retrying with
    backoff on transient errors (429 rate-limit, 503 overloaded).

    The free tier is aggressively rate-limited (as low as 5 requests/minute
    per model at the time this was written), so a batch of 10 sequential
    telemetry-extraction calls WILL hit 429s in practice — this isn't a
    hypothetical edge case, it reproduced on the very first real run during
    development. We honor the server's suggested `retryDelay` when present,
    otherwise fall back to exponential backoff.
    """
    from google.genai import errors

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except errors.APIError as e:
            last_exc = e
            if e.code not in (429, 503):
                raise  # not transient — don't waste retries on a real error
            delay = base_delay * (2**attempt)
            details = e.details if isinstance(e.details, dict) else {}
            for d in details.get("error", {}).get("details", []):
                if d.get("@type", "").endswith("RetryInfo") and "retryDelay" in d:
                    try:
                        delay = float(str(d["retryDelay"]).rstrip("s")) + 0.5
                    except ValueError:
                        pass
                    break
            log.info(f"    ...rate-limited/overloaded (HTTP {e.code}), retrying in {delay:.1f}s")
            time.sleep(delay)
    raise last_exc  # exhausted retries


def extract_telemetry(client, model: str, paper: PaperMetadata) -> Optional[PaperTelemetry]:
    from google.genai import types

    prompt = TELEMETRY_PROMPT.format(title=paper.title, abstract=paper.abstract)
    try:
        response = call_gemini_with_retry(
            lambda: client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperTelemetry,
                    temperature=0,
                ),
            )
        )
        return PaperTelemetry.model_validate_json(response.text)
    except (ValidationError, Exception) as e:  # noqa: BLE001
        log.warning(f"  PMID {paper.pmid}: telemetry extraction failed ({e}), dropping.")
        return None


# --------------------------------------------------------------------------
# Module 3: Deterministic Bayesian/epistemic state update
#           (Reasoning Step 2 — the judgment call)
# --------------------------------------------------------------------------


def clip01(x: float) -> float:
    """Clamp to [0.0, 1.0]. Explicit safety net even though every input to
    posterior_score is already bounded to [0,1] by construction (P(E) in
    [0,1], L in (0,1], V_norm in [0,1], weights summing to 1.0) — defends
    against a future change to the weights/priors quietly breaking that
    invariant."""
    return max(0.0, min(1.0, x))


def _normalize(values: list[float]) -> tuple[list[float], float, float]:
    """Min-max normalize to [0, 1] with division-by-zero protection: when
    every value in the batch ties (including a batch of size 1), the range
    is 0 and every normalized value is defined as 0.0 rather than raising."""
    lo, hi = min(values, default=0.0), max(values, default=0.0)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in values], lo, hi


def score_one(
    prior: float,
    claim_hyperbole: int,
    w_n: float,
    v_norm: float,
    *,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    is_preregistered: bool = False,
    w_prior: float = W_PRIOR,
    w_velocity: float = W_VELOCITY,
) -> dict:
    """The per-paper posterior-update math, factored out of score_papers()
    so it can be re-run standalone: once when scoring the full batch, again
    (with just a new `prior`) when a human operator manually clamps P(E) in
    the HITL gate, and again from ui.py when the Methodology/Velocity weight
    sliders move. Keeping one implementation avoids drift across call sites.

    `prior` here is the RAW design-tier prior (before preregistration); the
    bonus is applied inside this function so every downstream quantity
    (rigor_baseline, D, ...) is computed against the same effective prior
    the posterior score itself uses — see `base_prior_credence` /
    `preregistration_bonus` / `effective_prior_credence` in the return dict
    for the full breakdown."""
    preregistration_bonus = PREREGISTRATION_BONUS if is_preregistered else 0.0
    effective_prior = clip01(prior + preregistration_bonus)
    rigor_baseline = effective_prior * 5.0

    # Discrepancy Index D: how far the claim overshoots what this evidence
    # tier (now including any preregistration credit) can justify.
    # Underclaiming (D < 0) is never penalized.
    raw_d = claim_hyperbole - rigor_baseline
    d = max(0.0, raw_d)

    # Sample Power Weight moderates the discrepancy: the same overclaim from
    # a large, well-powered study is less absurd than the identical overclaim
    # from a tiny one. Multiplier ranges [1.0, 2.0].
    d_adjusted = d * (2.0 - w_n)

    if d_adjusted > DISCREPANCY_THRESHOLD:
        likelihood = math.exp(-LIKELIHOOD_DECAY_K * (d_adjusted - DISCREPANCY_THRESHOLD))
    else:
        likelihood = 1.0  # no absurdity gate triggered — full credence retained

    # Statistical Precision Penalty: SE from the reported 95% CI (the
    # standard normal-approximation formula: a 95% CI spans ~1.96 SE either
    # side of the point estimate, so width/3.92 = SE). A paper that didn't
    # report CI bounds gets no penalty here — this discounts *reported but
    # noisy* precision, it doesn't punish papers for omitting a CI (that's
    # a separate, unmodeled concern).
    standard_error = None
    precision_penalty = 1.0
    if ci_lower is not None and ci_upper is not None:
        standard_error = abs(ci_upper - ci_lower) / 3.92
        if standard_error > SE_PENALTY_THRESHOLD:
            precision_penalty = math.exp(-PRECISION_DECAY_K * (standard_error - SE_PENALTY_THRESHOLD))

    posterior = clip01((effective_prior * likelihood * precision_penalty) * w_prior + v_norm * w_velocity)

    return {
        "base_prior_credence": prior,
        "preregistration_bonus": preregistration_bonus,
        "effective_prior_credence": effective_prior,
        "rigor_baseline": rigor_baseline,
        "discrepancy_index": d,
        "discrepancy_adjusted": d_adjusted,
        "likelihood_penalty": likelihood,
        "standard_error": standard_error,
        "precision_penalty": precision_penalty,
        "posterior_score": posterior,
    }


def score_papers(
    relevant: list[tuple[PaperMetadata, PaperTelemetry]],
    prior_lookup: Optional[dict[str, float]] = None,
) -> list[ScoredPaper]:
    """prior_lookup maps Beta-tier name -> P(E) (see graph_memory.py). If
    omitted, falls back to graph_memory.seed_prior_means() — a pure,
    I/O-free computation, which is what keeps this function (and
    test_matrix.py, which calls it with no prior_lookup) deterministic and
    network/disk-free. A live run (agent.py main()) instead passes in
    graph_memory.get_current_prior_means(conn), the learned values."""
    if prior_lookup is None:
        prior_lookup = graph_memory.seed_prior_means()

    # --- Citation velocity: citations / years-since-publication, normalized ---
    raw_velocities = []
    for meta, _ in relevant:
        age = CURRENT_YEAR - meta.publication_year + 1
        raw_velocities.append(meta.citations / age if age > 0 else float(meta.citations))
    velocity_norms, _, _ = _normalize(raw_velocities)

    # --- Sample Power Weight W_N: log10(N+1), normalized across the batch ---
    # log-scaled because statistical power grows with N but with steeply
    # diminishing returns (N=100->1000 matters far more than N=10000->10900).
    raw_log_n = [math.log10(tele.sample_size + 1) for _, tele in relevant]
    power_norms, _, _ = _normalize(raw_log_n)

    scored: list[ScoredPaper] = []
    for (meta, tele), v_raw, v_norm, w_n in zip(relevant, raw_velocities, velocity_norms, power_norms):
        tier = graph_memory.beta_tier_for(tele.study_design, tele.sample_size)
        # .get(..., JEFFREYS_MEAN) rather than prior_lookup[tier]: if the
        # caller's prior_lookup doesn't have this tier yet (e.g. the pure
        # seed_prior_means() default, which only knows the 6 seeded tiers,
        # was used for a batch containing a genuinely novel design), that
        # must degrade to the same uninformative Jeffreys mean
        # graph_memory.get_prior_credence() would have registered — never
        # a KeyError, per "do not fail on unknown categories".
        prior = prior_lookup.get(tier, graph_memory.JEFFREYS_MEAN)
        components = score_one(
            prior,
            tele.claim_hyperbole,
            w_n,
            v_norm,
            ci_lower=tele.ci_lower,
            ci_upper=tele.ci_upper,
            is_preregistered=tele.is_preregistered,
        )

        scored.append(
            ScoredPaper(
                metadata=meta,
                telemetry=tele,
                base_prior_credence=round(components["base_prior_credence"], 4),
                preregistration_bonus=round(components["preregistration_bonus"], 4),
                prior_credence=round(components["effective_prior_credence"], 4),
                rigor_baseline=round(components["rigor_baseline"], 4),
                discrepancy_index=round(components["discrepancy_index"], 4),
                sample_power_weight=round(w_n, 4),
                discrepancy_adjusted=round(components["discrepancy_adjusted"], 4),
                likelihood_penalty=round(components["likelihood_penalty"], 4),
                standard_error=(
                    round(components["standard_error"], 4)
                    if components["standard_error"] is not None
                    else None
                ),
                precision_penalty=round(components["precision_penalty"], 4),
                velocity_raw=round(v_raw, 3),
                velocity_norm=round(v_norm, 3),
                posterior_score=round(components["posterior_score"], 4),
                is_ood_design=graph_memory.is_ood_tier(tier),
            )
        )

    scored.sort(key=lambda p: p.posterior_score, reverse=True)
    return scored


# --------------------------------------------------------------------------
# Module 3.5: Epistemic fail-safe & Human-in-the-Loop (HITL) anomaly gate
# --------------------------------------------------------------------------
#
# This is a SEPARATE, coarser gate from the smooth likelihood penalty above.
# The likelihood penalty (DISCREPANCY_THRESHOLD=0.5) silently discounts
# credence for ANY paper whose claim mildly outruns its tier — that's
# continuous and automatic, no human involved. The anomaly gate below fires
# on a much higher bar (D >= 2.0, or N < 30 for an interventional trial) and
# exists to put a SPECIFIC paper in front of a human, not just discount it.

INTERVENTIONAL_DESIGNS = {"RCT"}  # the only tier in our schema that is
# itself a human intervention trial (Meta-Analysis synthesizes other work,
# Cohort/Observational/In-Vitro/Review are not controlled interventions)
ANOMALY_DISCREPANCY_THRESHOLD = 2.0
ANOMALY_MIN_SAMPLE_SIZE = 30


def detect_anomaly(p: ScoredPaper) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if p.discrepancy_index >= ANOMALY_DISCREPANCY_THRESHOLD:
        reasons.append(
            f"Discrepancy Index D={p.discrepancy_index:.2f} >= {ANOMALY_DISCREPANCY_THRESHOLD} "
            f"(claim strength {p.telemetry.claim_hyperbole}/5 far exceeds this design's "
            f"justified ceiling of {p.rigor_baseline:.2f}/5)"
        )
    if (
        p.telemetry.study_design in INTERVENTIONAL_DESIGNS
        and p.telemetry.sample_size < ANOMALY_MIN_SAMPLE_SIZE
    ):
        reasons.append(
            f"N={p.telemetry.sample_size} < {ANOMALY_MIN_SAMPLE_SIZE} for an interventional "
            f"trial design ({p.telemetry.study_design})"
        )
    return bool(reasons), reasons


def apply_audit_flags(scored: list[ScoredPaper]) -> list[ScoredPaper]:
    """Mutates and returns `scored`: any paper tripping detect_anomaly()
    gets audit_status='FLAGGED' and its trigger reason(s) recorded. Papers
    that don't trip it keep the default 'PASSED'."""
    for p in scored:
        is_anomaly, reasons = detect_anomaly(p)
        if is_anomaly:
            p.audit_status = "FLAGGED"
            p.audit_reasons = reasons
    return scored


def show_anomaly_card(p: ScoredPaper) -> None:
    ood_badge = "  [bold magenta][OOD design][/]" if p.is_ood_design else ""
    prereg_badge = "  [bold green][Preregistered][/]" if p.telemetry.is_preregistered else ""
    body = (
        f"[bold]PMID:[/] {p.metadata.pmid}{ood_badge}{prereg_badge}\n"
        f"[bold]Title:[/] {p.metadata.title}\n"
        f"[bold]Link:[/] [link={p.metadata.url}]{p.metadata.url}[/link]\n"
        f"[bold]Study design:[/] {p.telemetry.study_design}    [bold]N:[/] {p.telemetry.sample_size}\n"
        f"[bold]Claim strength:[/] {p.telemetry.claim_hyperbole}/5 vs. "
        f"[bold]design-justified ceiling:[/] {p.rigor_baseline:.2f}/5\n"
        f"[bold]Discrepancy Index D:[/] {p.discrepancy_index:.2f}    "
        f"[bold]Current P(E):[/] {p.prior_credence:.2f}    "
        f"[bold]L(Absurdity):[/] {p.likelihood_penalty:.2f}    "
        f"[bold]Precision penalty:[/] {p.precision_penalty:.2f}    "
        f"[bold]S_posterior:[/] {p.posterior_score:.3f}\n\n"
        f"[bold red]Trigger(s):[/]\n" + "\n".join(f"  • {r}" for r in p.audit_reasons)
    )
    console.print(
        Panel(body, title="[bold red]/!\\ EPISTEMIC ANOMALY — HUMAN REVIEW REQUIRED[/]", border_style="red")
    )


def prompt_operator_choice() -> str:
    while True:
        choice = console.input(
            "\n[bold]Choose an action[/]  "
            "[1] Apply automated likelihood penalty  "
            "[2] Manually clamp Prior P(E)  "
            "[3] Reject / Quarantine paper\n> "
        ).strip()
        if choice in {"1", "2", "3"}:
            return choice
        console.print("[red]Invalid choice — enter 1, 2, or 3.[/]")


def prompt_manual_prior() -> float:
    while True:
        raw = console.input("Enter manual P(E) in [0.0, 1.0]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            console.print("[red]Not a number — try again.[/]")
            continue
        if 0.0 <= value <= 1.0:
            return value
        console.print("[red]Out of range — P(E) must be between 0.0 and 1.0.[/]")


def run_hitl_review(
    flagged: list[ScoredPaper], quarantined: set[str], conn
) -> None:
    """Interactive CLI review loop (agent.py --interactive). Mutates the
    flagged ScoredPaper objects in place and records each decision as
    Empirical Bayesian feedback in epistemic_memory.db via graph_memory."""
    for p in flagged:
        show_anomaly_card(p)
        choice = prompt_operator_choice()
        tier = graph_memory.beta_tier_for(p.telemetry.study_design, p.telemetry.sample_size)

        if choice == "1":
            p.audit_status = "PASSED"
            graph_memory.record_feedback(conn, tier=tier, action="confirm", pmid=p.metadata.pmid)
            console.print(f"[green]Confirmed.[/] Automated penalty retained (L={p.likelihood_penalty}).")

        elif choice == "2":
            new_prior = prompt_manual_prior()
            # is_preregistered=False here deliberately: a manual clamp is the
            # operator's final word on P(E), not a base value for the
            # automatic preregistration bonus to stack on top of — what they
            # type is exactly what S_posterior gets computed from.
            components = score_one(
                new_prior,
                p.telemetry.claim_hyperbole,
                p.sample_power_weight,
                p.velocity_norm,
                ci_lower=p.telemetry.ci_lower,
                ci_upper=p.telemetry.ci_upper,
                is_preregistered=False,
            )
            p.base_prior_credence = round(new_prior, 4)
            p.preregistration_bonus = 0.0
            p.prior_credence = round(components["effective_prior_credence"], 4)
            p.rigor_baseline = round(components["rigor_baseline"], 4)
            p.discrepancy_index = round(components["discrepancy_index"], 4)
            p.discrepancy_adjusted = round(components["discrepancy_adjusted"], 4)
            p.likelihood_penalty = round(components["likelihood_penalty"], 4)
            p.standard_error = (
                round(components["standard_error"], 4) if components["standard_error"] is not None else None
            )
            p.precision_penalty = round(components["precision_penalty"], 4)
            p.posterior_score = round(components["posterior_score"], 4)
            p.audit_status = "OVERRIDDEN"
            graph_memory.record_feedback(conn, tier=tier, action="override", manual_p=new_prior, pmid=p.metadata.pmid)
            console.print(f"[yellow]P(E) manually clamped to {new_prior:.2f}.[/] New S_posterior = {p.posterior_score}")

        else:  # "3"
            quarantined.add(p.metadata.pmid)
            p.audit_status = "OVERRIDDEN"
            graph_memory.record_feedback(conn, tier=tier, action="reject", pmid=p.metadata.pmid)
            console.print(f"[red]Quarantined.[/] PMID {p.metadata.pmid} excluded from arbiter synthesis.")


# --------------------------------------------------------------------------
# Module 4: Defense arbiter (LLM Step 2)
# --------------------------------------------------------------------------

ARBITER_PROMPT = """You are the lead architect of this epistemic filter. You have \
ranked the following papers using a DETERMINISTIC Bayesian posterior update: \
each paper's final score S_posterior = [P(E) * L(Absurdity) * PrecisionPenalty] \
* 0.75 + [citation velocity, normalized] * 0.25, where P(E) is a prior credence \
set by the paper's study-design tier (plus a small +0.05 bonus if the trial was \
preregistered), L(Absurdity) is a likelihood penalty that discounts P(E) when \
the paper's claim_hyperbole outruns what its design and sample size can \
actually justify, and PrecisionPenalty discounts credence further when a \
reported 95% CI is wide relative to the estimate (an observational study with \
a small N claiming a definitive "cure" gets a much larger penalty than the \
same claim from a large, well-powered, preregistered trial with a tight CI).

You did NOT choose these scores or this ranking. Your job now is only to \
explain, in plain language, why each paper landed where it did, using its \
specific telemetry as evidence. Do not second-guess the ranking.

Papers (already sorted by final rank, 1 = highest):

{paper_block}

For each paper, write EXACTLY 3 sentences of justification that name its \
specific study_design, sample_size, claim_hyperbole, prior_credence, and \
likelihood_penalty. Then write a short overall_defense (2-4 sentences) naming \
which factor (the prior, the likelihood/absurdity penalty, or citation \
velocity) was actually decisive in separating rank 1 from rank 2 and rank 2 \
from rank 3.
"""


def build_paper_block(top: list[ScoredPaper]) -> str:
    lines = []
    for i, p in enumerate(top, start=1):
        lines.append(
            f"Rank {i} — PMID {p.metadata.pmid}: \"{p.metadata.title}\"\n"
            f"  study_design={p.telemetry.study_design}"
            f"{' [OOD design]' if p.is_ood_design else ''}, "
            f"sample_size={p.telemetry.sample_size}, "
            f"claim_hyperbole={p.telemetry.claim_hyperbole}/5, "
            f"is_preregistered={p.telemetry.is_preregistered}\n"
            f"  base_prior_credence={p.base_prior_credence}, "
            f"preregistration_bonus={p.preregistration_bonus}, "
            f"prior_credence(P(E))={p.prior_credence}, "
            f"discrepancy_index(D)={p.discrepancy_index}, "
            f"sample_power_weight(W_N)={p.sample_power_weight}, "
            f"likelihood_penalty(L)={p.likelihood_penalty}\n"
            f"  standard_error={p.standard_error}, precision_penalty={p.precision_penalty}, "
            f"velocity_norm={p.velocity_norm}, "
            f"posterior_score(S)={p.posterior_score}"
        )
    return "\n\n".join(lines)


def arbitrate(client, model: str, top: list[ScoredPaper]) -> Optional[ArbiterVerdict]:
    from google.genai import types

    prompt = ARBITER_PROMPT.format(paper_block=build_paper_block(top))
    try:
        response = call_gemini_with_retry(
            lambda: client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ArbiterVerdict,
                    temperature=0.3,
                ),
            )
        )
        return ArbiterVerdict.model_validate_json(response.text)
    except (ValidationError, Exception) as e:  # noqa: BLE001
        log.error(f"Arbiter step failed: {e}")
        return None


COUNTERFACTUAL_PROMPT = """You are the epistemic filter's arbiter. Below are the \
same top-ranked papers and their telemetry you (or a prior run) already \
ranked. A user is asking a COUNTERFACTUAL question — how the ranking or its \
justification would change under a hypothetical scenario. Answer concretely, \
referencing specific telemetry (study_design, sample_size, prior_credence, \
discrepancy_index, likelihood_penalty, posterior_score) for each paper you \
discuss. If the scenario would NOT plausibly change the ranking, say so \
explicitly and explain why — don't manufacture a change that isn't justified \
by the deterministic scoring formula.

Papers:

{paper_block}

Counterfactual question: {scenario}
"""


def counterfactual_arbiter(client, model: str, top: list[dict], scenario: str) -> str:
    """Free-form (not structured-JSON) follow-up used by ui.py's
    counterfactual console. Takes plain dicts (as loaded from run_output
    JSON) rather than ScoredPaper objects, since the dashboard reads
    already-persisted run output rather than holding live Pydantic models."""
    from google.genai import types

    lines = []
    for i, p in enumerate(top, start=1):
        meta, tele = p["metadata"], p["telemetry"]
        lines.append(
            f"Rank {i} — PMID {meta['pmid']}: \"{meta['title']}\"\n"
            f"  study_design={tele['study_design']}, sample_size={tele['sample_size']}, "
            f"claim_hyperbole={tele['claim_hyperbole']}/5\n"
            f"  prior_credence={p['prior_credence']}, discrepancy_index={p['discrepancy_index']}, "
            f"likelihood_penalty={p['likelihood_penalty']}, posterior_score={p['posterior_score']}"
        )
    prompt = COUNTERFACTUAL_PROMPT.format(paper_block="\n\n".join(lines), scenario=scenario)
    response = call_gemini_with_retry(
        lambda: client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(temperature=0.4))
    )
    return response.text


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


_AUDIT_STYLE = {"PASSED": "dim", "FLAGGED": "bold red", "OVERRIDDEN": "bold yellow"}


def print_ranking_table(scored: list[ScoredPaper], quarantined: Optional[set[str]] = None) -> None:
    quarantined = quarantined or set()
    table = Table(title="Bayesian Posterior Ranking (all relevant papers)")
    table.add_column("Rank", justify="right")
    table.add_column("PMID")
    table.add_column("Design")
    table.add_column("N", justify="right")
    table.add_column("Hype", justify="right")
    table.add_column("P(E)", justify="right")
    table.add_column("D", justify="right")
    table.add_column("W_N", justify="right")
    table.add_column("L", justify="right")
    table.add_column("Prec.", justify="right")
    table.add_column("V_norm", justify="right")
    table.add_column("S_post", justify="right", style="bold")
    table.add_column("Flags")
    table.add_column("Audit")

    for i, p in enumerate(scored, start=1):
        style = _AUDIT_STYLE.get(p.audit_status, "")
        audit_label = p.audit_status + (" [quarantined]" if p.metadata.pmid in quarantined else "")
        flags = []
        if p.is_ood_design:
            flags.append("[magenta]OOD[/]")
        if p.telemetry.is_preregistered:
            flags.append("[green]PREREG[/]")
        table.add_row(
            str(i),
            p.metadata.pmid,
            p.telemetry.study_design,
            str(p.telemetry.sample_size),
            f"{p.telemetry.claim_hyperbole}/5",
            f"{p.prior_credence:.2f}",
            f"{p.discrepancy_index:.2f}",
            f"{p.sample_power_weight:.2f}",
            f"{p.likelihood_penalty:.2f}",
            f"{p.precision_penalty:.2f}",
            f"{p.velocity_norm:.2f}",
            f"{p.posterior_score:.3f}",
            " ".join(flags),
            f"[{style}]{audit_label}[/]" if style else audit_label,
        )
    console.print(table)


def print_top3_defense(top: list[ScoredPaper], verdict: Optional[ArbiterVerdict]) -> None:
    console.rule("[bold green]TOP 3 — RANKED WITH JUSTIFICATION")
    just_by_pmid = {j.pmid: j.justification for j in verdict.justifications} if verdict else {}
    for i, p in enumerate(top, start=1):
        console.print(f"\n[bold]#{i}. {p.metadata.title}[/]  (PMID {p.metadata.pmid}, {p.metadata.publication_year})")
        console.print(f"   [link={p.metadata.url}]{p.metadata.url}[/link]")
        console.print(
            f"   S_posterior = {p.posterior_score}  |  design = {p.telemetry.study_design}  |  "
            f"N = {p.telemetry.sample_size}  |  hype = {p.telemetry.claim_hyperbole}/5  |  "
            f"P(E) = {p.prior_credence}  |  L(Absurdity) = {p.likelihood_penalty}  |  "
            f"Precision = {p.precision_penalty}"
        )
        text = just_by_pmid.get(p.metadata.pmid)
        if text:
            console.print(f"   [italic]{text}[/]")
        else:
            console.print("   [dim](no LLM justification available — arbiter step failed or was skipped)[/]")
    if verdict:
        console.print(f"\n[bold]Overall defense:[/] {verdict.overall_defense}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Epistemic Filtering Agent")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument(
        "--model",
        default="gemini-flash-lite-latest",
        help="Gemini model for both LLM steps. Using a '-latest' alias rather than a "
        "pinned version since Google retires dated model names within months (verified "
        "during development: gemini-2.0-flash was already gone as of this writing). "
        "Deliberately the 'lite' tier, not 'gemini-flash-latest': the full flash model's "
        "free tier was measured during development at a 20-requests/DAY cap (unusable for "
        "iterative testing), while flash-lite handled a burst of 8 rapid calls with no "
        "throttling at all. This task needs 10+ calls per run, so lite is the only free "
        "tier that's actually usable — not a quality tradeoff, a quota-survival one.",
    )
    parser.add_argument(
        "--output", default="run_output.json", help="Where to save the full structured run"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the mocked citation RNG (reproducibility)")
    parser.add_argument(
        "--db", default=graph_memory.DB_PATH_DEFAULT,
        help="Path to the persistent epistemic knowledge graph / Beta-prior store (SQLite)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Halt on each FLAGGED_FOR_AUDIT anomaly and prompt an operator for a decision, "
        "instead of just logging a warning and letting the automated penalty stand.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        log.error("[bold red]GEMINI_API_KEY is not set.[/] Copy .env.example to .env and fill it in.")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=gemini_key)

    # ---- Persistent knowledge graph / active-learning store ----
    conn = graph_memory.init_db(args.db)

    # ---- Module 1 ----
    papers = fetch_pubmed(args.query, args.max_results)
    if len(papers) < 3:
        log.error(f"Only {len(papers)} papers with usable abstracts were found — need at least 3. Try a broader --query.")
        sys.exit(1)

    # ---- Module 2 (LLM Step 1) + Reasoning Step 1: relevance filter ----
    log.info(f"[EXTRACTING TELEMETRY] Sending {len(papers)} abstracts to Gemini ({args.model})...")
    relevant: list[tuple[PaperMetadata, PaperTelemetry]] = []
    dropped_irrelevant = 0
    for paper in papers:
        tele = extract_telemetry(client, args.model, paper)
        if tele is None:
            continue
        if not tele.is_relevant:
            dropped_irrelevant += 1
            log.info(f"  PMID {paper.pmid}: [yellow]filtered out[/] (LLM judged not directly relevant)")
            continue
        relevant.append((paper, tele))
        log.info(
            f"  PMID {paper.pmid}: kept — {tele.study_design}, N={tele.sample_size}, hype={tele.claim_hyperbole}/5"
        )

    log.info(
        f"[EXTRACTING TELEMETRY] {len(relevant)}/{len(papers)} papers passed the relevance gate "
        f"({dropped_irrelevant} dropped as off-topic)."
    )
    if len(relevant) < 3:
        log.error(
            f"Only {len(relevant)} relevant papers survived the filter — need at least 3 for a top-3 ranking. "
            f"Try a broader --query or raise --max-results."
        )
        sys.exit(1)

    # Register any out-of-distribution study designs this batch introduced
    # (e.g. an LLM-reported "Mendelian Randomization") with an uninformative
    # Jeffreys prior BEFORE loading current_priors, so this run's own novel
    # designs are included in prior_lookup rather than only ones a previous
    # run already registered.
    ood_designs = set()
    for _, tele in relevant:
        tier = graph_memory.beta_tier_for(tele.study_design, tele.sample_size)
        if graph_memory.is_ood_tier(tier):
            ood_designs.add(tele.study_design)
        graph_memory.get_prior_credence(conn, tele.study_design, tele.sample_size)
    if ood_designs:
        log.info(
            f"[MEMORY] Out-of-distribution study design(s) registered with an uninformative "
            f"Jeffreys prior Beta({graph_memory.JEFFREYS_ALPHA}, {graph_memory.JEFFREYS_BETA}): "
            f"{sorted(ood_designs)}"
        )
    current_priors = graph_memory.get_current_prior_means(conn)
    log.info(f"[MEMORY] Loaded prior credences from [cyan]{args.db}[/]: " + ", ".join(f"{k}={v:.2f}" for k, v in current_priors.items()))

    # ---- Module 3: Reasoning Step 2 — Bayesian posterior scoring (the judgment call) ----
    log.info(
        "[CALCULATING MATRIX] Computing posterior scores: "
        r"\[P(E) x L(Absurdity) x PrecisionPenalty] x 0.75 + \[citation velocity] x 0.25 ..."
    )
    scored = score_papers(relevant, prior_lookup=current_priors)

    # ---- Module 3.5: Epistemic fail-safe & HITL anomaly gate ----
    apply_audit_flags(scored)
    flagged = [p for p in scored if p.audit_status == "FLAGGED"]
    quarantined: set[str] = set()
    if flagged:
        log.warning(f"[FAIL-SAFE] {len(flagged)} paper(s) flagged for epistemic audit.")
        for p in flagged:
            log.warning(f"  PMID {p.metadata.pmid}: {'; '.join(p.audit_reasons)}")
        if args.interactive:
            run_hitl_review(flagged, quarantined, conn)
            scored.sort(key=lambda p: p.posterior_score, reverse=True)  # HITL may have changed scores
        else:
            log.info(
                "[FAIL-SAFE] Non-interactive mode: automated likelihood penalty already applied "
                "to the flagged paper(s) above. Re-run with --interactive to review manually."
            )

    print_ranking_table(scored, quarantined=quarantined)

    candidates = [p for p in scored if p.metadata.pmid not in quarantined]
    top3 = candidates[:3]
    if quarantined:
        log.info(f"[FAIL-SAFE] {len(quarantined)} paper(s) quarantined from arbiter synthesis: {sorted(quarantined)}")

    # ---- Module 4 (LLM Step 2) ----
    log.info(f"[ARBITRATING] Asking Gemini to defend the top {len(top3)} ranking with specific telemetry...")
    verdict = arbitrate(client, args.model, top3)

    print_top3_defense(top3, verdict)

    # ---- Persist to the knowledge graph: nodes for every scored paper, plus
    # citation edges (real PubMed link data) and contradiction edges (keyword
    # heuristic — see graph_memory.py). Both are best-effort: a network hiccup
    # on the citation lookup must not blow up a run that already succeeded. ----
    for p in scored:
        graph_memory.upsert_node(
            conn,
            pmid=p.metadata.pmid,
            title=p.metadata.title,
            study_design=p.telemetry.study_design,
            sample_size=p.telemetry.sample_size,
            prior_credence=p.prior_credence,
            discrepancy_index=p.discrepancy_index,
            likelihood_penalty=p.likelihood_penalty,
            posterior_score=p.posterior_score,
            audit_status=p.audit_status,
            url=p.metadata.url,
            doi=p.metadata.doi,
        )

    # Contradiction edges are tagged sentiment="REFUTING" unconditionally —
    # a detected contradiction IS a refuting relationship by definition.
    contradiction_edges = graph_memory.detect_contradictions(
        [(p.metadata.pmid, p.metadata.title, p.metadata.abstract) for p in scored]
    )
    for src, dst, detail in contradiction_edges:
        graph_memory.add_edge(conn, src, dst, edge_type="contradiction", detail=detail, sentiment="REFUTING")
    if contradiction_edges:
        log.info(f"[MEMORY] {len(contradiction_edges)} candidate contradiction edge(s) detected (keyword heuristic).")

    try:
        texts = {p.metadata.pmid: (p.metadata.title, p.metadata.abstract) for p in scored}
        citation_edges = graph_memory.fetch_citation_edges_from_pubmed(
            [p.metadata.pmid for p in scored],
            email=os.environ["ENTREZ_EMAIL"],
            api_key=os.environ.get("ENTREZ_API_KEY"),
            texts=texts,
        )
        for src, dst, sentiment in citation_edges:
            graph_memory.add_edge(conn, src, dst, edge_type="citation", sentiment=sentiment)
        sentiment_counts = {}
        for _, _, sentiment in citation_edges:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        log.info(
            f"[MEMORY] {len(citation_edges)} citation edge(s) found among this batch's papers "
            f"({dict(sentiment_counts)})."
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never fail the run over this
        log.warning(f"[MEMORY] Citation-edge lookup failed (non-fatal): {e}")

    conn.close()
    log.info(f"[MEMORY] Knowledge graph persisted to [cyan]{args.db}[/].")

    # ---- Persist full run for the demo / grading ----
    out = {
        "generated_at": datetime.now().isoformat(),
        "query": args.query,
        "model": args.model,
        "papers_fetched": len(papers),
        "papers_relevant": len(relevant),
        "papers_dropped_irrelevant": dropped_irrelevant,
        "papers_flagged_for_audit": len(flagged),
        "papers_quarantined": sorted(quarantined),
        "full_ranking": [p.model_dump() for p in scored],
        "top3": [p.model_dump() for p in top3],
        "arbiter_verdict": verdict.model_dump() if verdict else None,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[DONE] Full structured run saved to [cyan]{args.output}[/]")


if __name__ == "__main__":
    main()
