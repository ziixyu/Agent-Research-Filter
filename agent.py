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
    [SCORE]      Deterministic Python matrix    (Reasoning Step 2: the judgment call)
    [ARBITER]    Gemini structured JSON         (LLM Step 2: defends the ranking)

Run:
    python agent.py
    python agent.py --query "your own PubMed query" --max-results 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

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

# Weights for the deterministic judgment matrix. See README.md "Why these
# weights" for the argument that methodology should dominate citation
# velocity roughly 3:1 rather than being ignored or weighted evenly.
W_METHODOLOGY = 0.75
W_VELOCITY = 0.25
HYPE_PENALTY = 0.5
HYPE_PENALTY_THRESHOLD = 4  # claim_hyperbole >= this triggers the penalty
HYPE_PENALTY_DESIGNS = {"Retrospective/Observational", "In-Vitro/Animal"}

METHODOLOGY_BASE_SCORE = {
    "Meta-Analysis": 1.0,
    "RCT": 0.9,
    "Prospective Cohort": 0.7,
    "Retrospective/Observational": 0.5,
    "In-Vitro/Animal": 0.3,
    "Review/Opinion": 0.1,
}

StudyDesign = Literal[
    "Meta-Analysis",
    "RCT",
    "Prospective Cohort",
    "Retrospective/Observational",
    "In-Vitro/Animal",
    "Review/Opinion",
]


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


class PaperTelemetry(BaseModel):
    """Structured epistemic state extracted from the abstract by the LLM.

    Deliberately narrow: the LLM is only allowed to report *what kind of
    evidence this is*, never *how good the evidence is*. That judgment is
    reserved for the deterministic matrix in Module 3.
    """

    is_relevant: bool = Field(
        description="True only if the abstract directly discusses Semaglutide "
        "(or a GLP-1 receptor agonist explicitly identified as semaglutide) "
        "and liver fibrosis / NASH / MASH outcomes."
    )
    study_design: StudyDesign
    sample_size: int = Field(description="Extracted N. 0 if not explicitly stated.")
    claim_hyperbole: int = Field(
        ge=1,
        le=5,
        description="1 = cautious/grounded language. 5 = definitive causal "
        "claims the study design cannot actually support.",
    )


class ScoredPaper(BaseModel):
    """A paper after Module 3's deterministic scoring has run."""

    metadata: PaperMetadata
    telemetry: PaperTelemetry
    velocity_raw: float
    velocity_norm: float
    methodology_score: float
    penalty: float
    epistemic_score: float


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
# Module 3: Deterministic judgment matrix (Reasoning Step 2 — the judgment call)
# --------------------------------------------------------------------------


def score_papers(
    relevant: list[tuple[PaperMetadata, PaperTelemetry]]
) -> list[ScoredPaper]:
    raw_velocities = []
    for meta, _ in relevant:
        age = CURRENT_YEAR - meta.publication_year + 1
        v = meta.citations / age if age > 0 else meta.citations
        raw_velocities.append(v)

    v_min, v_max = min(raw_velocities, default=0), max(raw_velocities, default=0)
    v_range = (v_max - v_min) or 1.0  # avoid div-by-zero when every paper ties

    scored: list[ScoredPaper] = []
    for (meta, tele), v_raw in zip(relevant, raw_velocities):
        v_norm = (v_raw - v_min) / v_range
        m = METHODOLOGY_BASE_SCORE[tele.study_design]

        penalty = 0.0
        if tele.study_design in HYPE_PENALTY_DESIGNS and tele.claim_hyperbole >= HYPE_PENALTY_THRESHOLD:
            penalty = HYPE_PENALTY

        s = (W_METHODOLOGY * m) + (W_VELOCITY * v_norm) - penalty

        scored.append(
            ScoredPaper(
                metadata=meta,
                telemetry=tele,
                velocity_raw=round(v_raw, 3),
                velocity_norm=round(v_norm, 3),
                methodology_score=m,
                penalty=penalty,
                epistemic_score=round(s, 4),
            )
        )

    scored.sort(key=lambda p: p.epistemic_score, reverse=True)
    return scored


# --------------------------------------------------------------------------
# Module 4: Defense arbiter (LLM Step 2)
# --------------------------------------------------------------------------

ARBITER_PROMPT = """You are the lead architect of this epistemic filter. You have \
ranked the following papers using a DETERMINISTIC matrix that prioritizes \
methodological rigour (weight 0.75) over citation velocity (weight 0.25), \
with a penalty applied when a low-rigour design (retrospective/observational \
or in-vitro/animal) is paired with hyperbolic claims (>=4/5).

You did NOT choose these scores or this ranking. Your job now is only to \
explain, in plain language, why each paper landed where it did, using its \
specific telemetry as evidence. Do not second-guess the ranking.

Papers (already sorted by final rank, 1 = highest):

{paper_block}

For each paper, write EXACTLY 3 sentences of justification that name its \
specific study_design, sample_size, claim_hyperbole, and score components. \
Then write a short overall_defense (2-4 sentences) naming which factor \
(methodology, velocity, or the hype penalty) was actually decisive in \
separating rank 1 from rank 2 and rank 2 from rank 3.
"""


def build_paper_block(top: list[ScoredPaper]) -> str:
    lines = []
    for i, p in enumerate(top, start=1):
        lines.append(
            f"Rank {i} — PMID {p.metadata.pmid}: \"{p.metadata.title}\"\n"
            f"  study_design={p.telemetry.study_design}, "
            f"sample_size={p.telemetry.sample_size}, "
            f"claim_hyperbole={p.telemetry.claim_hyperbole}/5\n"
            f"  methodology_score={p.methodology_score}, "
            f"velocity_norm={p.velocity_norm}, "
            f"penalty={p.penalty}, "
            f"epistemic_score={p.epistemic_score}"
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


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def print_ranking_table(scored: list[ScoredPaper]) -> None:
    table = Table(title="Deterministic Epistemic Ranking (all relevant papers)")
    table.add_column("Rank", justify="right")
    table.add_column("PMID")
    table.add_column("Design")
    table.add_column("N", justify="right")
    table.add_column("Hype", justify="right")
    table.add_column("M", justify="right")
    table.add_column("V_norm", justify="right")
    table.add_column("Penalty", justify="right")
    table.add_column("Score S", justify="right", style="bold")

    for i, p in enumerate(scored, start=1):
        table.add_row(
            str(i),
            p.metadata.pmid,
            p.telemetry.study_design,
            str(p.telemetry.sample_size),
            f"{p.telemetry.claim_hyperbole}/5",
            f"{p.methodology_score:.2f}",
            f"{p.velocity_norm:.2f}",
            f"-{p.penalty:.2f}" if p.penalty else "0.00",
            f"{p.epistemic_score:.3f}",
        )
    console.print(table)


def print_top3_defense(top: list[ScoredPaper], verdict: Optional[ArbiterVerdict]) -> None:
    console.rule("[bold green]TOP 3 — RANKED WITH JUSTIFICATION")
    just_by_pmid = {j.pmid: j.justification for j in verdict.justifications} if verdict else {}
    for i, p in enumerate(top, start=1):
        console.print(f"\n[bold]#{i}. {p.metadata.title}[/]  (PMID {p.metadata.pmid}, {p.metadata.publication_year})")
        console.print(f"   score S = {p.epistemic_score}  |  design = {p.telemetry.study_design}  |  N = {p.telemetry.sample_size}  |  hype = {p.telemetry.claim_hyperbole}/5")
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
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        log.error("[bold red]GEMINI_API_KEY is not set.[/] Copy .env.example to .env and fill it in.")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=gemini_key)

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

    # ---- Module 3: Reasoning Step 2 — deterministic scoring (the judgment call) ----
    log.info("[CALCULATING MATRIX] Scoring relevant papers (0.75 x methodology + 0.25 x citation velocity - hype penalty)...")
    scored = score_papers(relevant)
    print_ranking_table(scored)

    top3 = scored[:3]

    # ---- Module 4 (LLM Step 2) ----
    log.info(f"[ARBITRATING] Asking Gemini to defend the top {len(top3)} ranking with specific telemetry...")
    verdict = arbitrate(client, args.model, top3)

    print_top3_defense(top3, verdict)

    # ---- Persist full run for the demo / grading ----
    out = {
        "generated_at": datetime.now().isoformat(),
        "query": args.query,
        "model": args.model,
        "papers_fetched": len(papers),
        "papers_relevant": len(relevant),
        "papers_dropped_irrelevant": dropped_irrelevant,
        "full_ranking": [p.model_dump() for p in scored],
        "top3": [p.model_dump() for p in top3],
        "arbiter_verdict": verdict.model_dump() if verdict else None,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[DONE] Full structured run saved to [cyan]{args.output}[/]")


if __name__ == "__main__":
    main()
