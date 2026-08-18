#!/usr/bin/env python3
"""
backtest_calibration.py
========================
Autonomous agentic calibration engine.

An adversarial LLM-as-a-Judge "red team" critic re-evaluates already-scored
papers against three methodological stress tests (sample power, selective
reporting risk, endpoint validity) and automatically feeds ROBUST/VULNERABLE
verdicts into graph_memory.record_feedback() to move each design tier's
Beta(alpha, beta) — the same mechanism a human operator drives via
`agent.py --interactive` or the dashboard's HITL panel, just automated and
deliberately adversarial instead of manual.

This is a CALIBRATION TOOL, run separately and deliberately from the main
pipeline (`python agent.py`) — it is NOT invoked automatically by a normal
run. Repeated/iterated passes WILL keep shifting the same tiers' priors —
that's the point of `--iterations N`. Point --db at a scratch copy of
epistemic_memory.db if you want to experiment without touching your main
learned priors.

Run:
    python backtest_calibration.py
    python backtest_calibration.py --iterations 5 --input sample_run_output.json

Reuses (does not modify): agent.py's console/log setup (including the
Windows UTF-8 stdout fix, applied automatically on `import agent`),
call_gemini_with_retry, and graph_memory.py's record_feedback/beta_tier_for.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from rich.table import Table

import agent  # importing this applies agent.py's UTF-8 stdout fix as a side effect
import graph_memory

console = agent.console
log = agent.log


# --------------------------------------------------------------------------
# The adversarial judge
# --------------------------------------------------------------------------


class CalibrationVerdict(BaseModel):
    """An adversarial 'red team' judgment against 3 methodological stress
    tests. The LLM is explicitly instructed to be skeptical of the paper's
    own framing, not to summarize it charitably — that skepticism is the
    entire point of using it as a critic rather than a second extractor."""

    sample_power_adequate: bool = Field(
        description="False if N is too small to credibly support the claimed effect size "
        "for this study design."
    )
    selective_reporting_risk: bool = Field(
        description="True if there's a real sign of selective/outcome-switching reporting "
        "risk (e.g. not preregistered, or a suspiciously convenient primary endpoint)."
    )
    endpoint_validity_concern: bool = Field(
        description="True if the primary endpoint is a weak surrogate marker being oversold "
        "as equivalent to a hard clinical outcome."
    )
    verdict: Literal["ROBUST", "VULNERABLE"] = Field(
        description="ROBUST only if the paper clears all three stress tests without serious "
        "concern. VULNERABLE if any single one raises a real red flag."
    )
    rationale: str = Field(description="2-3 sentence adversarial critique justifying the verdict.")


CALIBRATION_PROMPT = """You are an adversarial methodological reviewer — a \
"red team" critic whose job is to find reasons NOT to trust this paper's \
claim, not to summarize it charitably. Stress-test it against three checks:

1. Sample power: is N large enough to credibly detect/support the claimed \
effect, given the study design?
2. Selective reporting risk: is there any sign of outcome-switching, a \
suspiciously convenient endpoint choice, or absence of preregistration that \
would let the authors pick their best-looking result after the fact?
3. Endpoint validity: is the primary endpoint a hard clinical outcome, or a \
surrogate/biomarker being oversold as equivalent to one?

Telemetry already extracted for this paper (context only — you are \
critiquing it, not re-extracting it):
  study_design={study_design}, sample_size={sample_size}, \
claim_hyperbole={claim_hyperbole}/5, is_preregistered={is_preregistered}, \
prior_credence={prior_credence}, posterior_score={posterior_score}

Title: {title}

Abstract: {abstract}

Give a ROBUST verdict only if the paper clears all three stress tests \
without serious concern. Any one real red flag makes it VULNERABLE. Be \
skeptical by design — this is an adversarial review, not a summary.
"""


def judge_paper(client, model: str, paper: dict) -> Optional[CalibrationVerdict]:
    from google.genai import types

    meta, tele = paper["metadata"], paper["telemetry"]
    prompt = CALIBRATION_PROMPT.format(
        study_design=tele["study_design"],
        sample_size=tele["sample_size"],
        claim_hyperbole=tele["claim_hyperbole"],
        is_preregistered=tele.get("is_preregistered", False),
        prior_credence=paper.get("prior_credence"),
        posterior_score=paper.get("posterior_score"),
        title=meta["title"],
        abstract=meta["abstract"],
    )
    try:
        response = agent.call_gemini_with_retry(
            lambda: client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CalibrationVerdict,
                    temperature=0.4,  # an adversarial critic is allowed some variance
                ),
            )
        )
        return CalibrationVerdict.model_validate_json(response.text)
    except (ValidationError, Exception) as e:  # noqa: BLE001
        log.warning(f"  PMID {meta['pmid']}: calibration judging failed ({e}), skipping.")
        return None


# --------------------------------------------------------------------------
# Feeding verdicts back into the Beta priors
# --------------------------------------------------------------------------


def apply_verdict(conn, tier: str, verdict: CalibrationVerdict, pmid: str) -> tuple[float, float]:
    """Maps a calibration verdict onto the SAME confirm/reject vocabulary a
    human HITL decision uses: ROBUST -> 'confirm' (alpha += 1), VULNERABLE
    -> 'reject' (beta += 1) — via graph_memory.record_feedback(), so there
    is exactly one place in the codebase that knows how to move a Beta
    distribution. Kept as its own thin function (rather than inlined) so
    it's independently unit-testable without a live Gemini call."""
    action = "confirm" if verdict.verdict == "ROBUST" else "reject"
    return graph_memory.record_feedback(conn, tier=tier, action=action, pmid=pmid)


def run_calibration_pass(client, model: str, conn, papers: list[dict]) -> dict:
    """One full pass over `papers`: judge each with the adversarial critic,
    apply feedback, return a before/after hyperparameter snapshot."""
    before = graph_memory.get_current_prior_hyperparams(conn)
    results = []
    for paper in papers:
        meta, tele = paper["metadata"], paper["telemetry"]
        verdict = judge_paper(client, model, paper)
        if verdict is None:
            continue
        tier = graph_memory.beta_tier_for(tele["study_design"], tele["sample_size"])
        apply_verdict(conn, tier, verdict, meta["pmid"])
        results.append({"pmid": meta["pmid"], "tier": tier, "verdict": verdict.model_dump()})
        style = "green" if verdict.verdict == "ROBUST" else "red"
        log.info(f"  PMID {meta['pmid']} ({tier}): [bold {style}]{verdict.verdict}[/] — {verdict.rationale}")
    after = graph_memory.get_current_prior_hyperparams(conn)
    return {"results": results, "before": before, "after": after}


def print_delta_table(before: dict, after: dict) -> None:
    table = Table(title="Beta hyperparameter deltas this pass")
    table.add_column("Tier")
    table.add_column("alpha: before -> after", justify="right")
    table.add_column("beta: before -> after", justify="right")
    table.add_column("mean: before -> after", justify="right")
    for tier in sorted(after):
        prev = before.get(tier)
        a1, b1 = after[tier]
        mean1 = f"{a1 / (a1 + b1):.3f}"
        if prev is None:
            table.add_row(tier, f"(new) -> {a1:g}", f"(new) -> {b1:g}", f"(new) -> {mean1}")
        else:
            a0, b0 = prev
            mean0 = f"{a0 / (a0 + b0):.3f}"
            table.add_row(tier, f"{a0:g} -> {a1:g}", f"{b0:g} -> {b1:g}", f"{mean0} -> {mean1}")
    console.print(table)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous agentic calibration engine")
    parser.add_argument(
        "--input", default="sample_run_output.json",
        help="A completed agent.py run's JSON output to backtest against (reads full_ranking)",
    )
    parser.add_argument(
        "--db", default=graph_memory.DB_PATH_DEFAULT,
        help="Epistemic memory store to update. Point at a scratch copy to experiment "
        "without touching your main learned priors.",
    )
    parser.add_argument("--model", default="gemini-flash-lite-latest", help="Gemini model for the adversarial judge")
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of self-critique passes to run. Each pass re-judges the same papers and "
        "applies feedback again — this is a self-optimizing loop, not idempotent.",
    )
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        log.error("[bold red]GEMINI_API_KEY is not set.[/] Copy .env.example to .env and fill it in.")
        raise SystemExit(1)

    from google import genai

    client = genai.Client(api_key=gemini_key)

    with open(args.input, encoding="utf-8") as f:
        run_data = json.load(f)
    papers = run_data.get("full_ranking", [])
    if not papers:
        log.error(f"No papers found in {args.input}'s full_ranking — nothing to calibrate against.")
        raise SystemExit(1)

    conn = graph_memory.init_db(args.db)
    log.info(
        f"[CALIBRATING] Adversarial backtest over {len(papers)} paper(s) from [cyan]{args.input}[/], "
        f"{args.iterations} iteration(s), updating [cyan]{args.db}[/]."
    )

    for i in range(1, args.iterations + 1):
        console.rule(f"[bold]Iteration {i}/{args.iterations}[/]")
        pass_result = run_calibration_pass(client, args.model, conn, papers)
        print_delta_table(pass_result["before"], pass_result["after"])

    conn.close()
    log.info(f"[DONE] Calibration complete. Updated priors persisted to [cyan]{args.db}[/].")


if __name__ == "__main__":
    main()
