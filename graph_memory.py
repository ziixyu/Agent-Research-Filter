#!/usr/bin/env python3
"""
graph_memory.py
================
Persistent epistemic knowledge graph + Empirical Bayesian active learning.

This module is purely additive to the existing pipeline: it does not import
`agent.py` (agent.py imports *this* module, one direction only, so there is
no circular dependency), and it never touches PubMed ingestion, the Gemini
telemetry schemas, or the retry wrapper.

Two independent responsibilities live here:

1. **Knowledge graph persistence** — SQLite (`epistemic_memory.db`) is the
   durable source of truth; `networkx.DiGraph` is an in-memory view built
   from it for graph algorithms and the dashboard. Nodes are scored papers;
   edges are citation links (real, fetched from PubMed's own link graph)
   and detected claim contradictions (a documented keyword heuristic — see
   `detect_contradictions`).

2. **Empirical Bayesian active learning over study-design priors** — instead
   of a fixed number per design tier, each tier's prior credence P(E) is the
   *mean* of a Beta(alpha, beta) distribution that shifts every time a human
   operator confirms, overrides, or rejects a flagged paper (via
   `agent.py --interactive` or the Streamlit HITL panel in `ui.py`).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

DB_PATH_DEFAULT = "epistemic_memory.db"

# --------------------------------------------------------------------------
# Beta-distribution seed hyperparameters (as specified). Read alpha/beta as
# pseudo-counts of (supporting, undermining) evidence for "this design tier
# deserves high credence" — e.g. Beta(19,1) means "19 confirming
# observations baked into the prior, 1 undermining one", mean = 19/20=0.95.
# --------------------------------------------------------------------------
SEED_BETA_PRIORS: dict[str, tuple[float, float]] = {
    "Meta-Analysis": (19.0, 1.0),  # mean 0.95
    "Phase III RCT": (18.0, 2.0),  # mean 0.90
    "Phase II RCT": (14.0, 6.0),  # mean 0.70
    "Prospective Cohort": (11.0, 9.0),  # mean 0.55
    "Retrospective": (8.0, 12.0),  # mean 0.40
    "Review/In Vitro": (3.0, 17.0),  # mean 0.15
}

# agent.py's PaperTelemetry.study_design has 6 categories that don't map
# 1:1 onto the 6 Beta tiers above (the tiers split RCT by phase, which our
# extraction schema doesn't capture, and merge In-Vitro/Animal with
# Review/Opinion into one "Review/In Vitro" tier). Every design category is
# mapped explicitly rather than left to guesswork:
DESIGN_TO_BETA_TIER: dict[str, str] = {
    "Meta-Analysis": "Meta-Analysis",
    "Prospective Cohort": "Prospective Cohort",
    "Retrospective/Observational": "Retrospective",
    "In-Vitro/Animal": "Review/In Vitro",
    "Review/Opinion": "Review/In Vitro",
    # "RCT" is intentionally absent here — see beta_tier_for() below, which
    # splits it into Phase II vs Phase III using a sample-size heuristic
    # instead of a single static mapping.
}

# Our extraction schema doesn't capture trial phase/blinding directly, but
# it DOES capture sample_size, and Phase III confirmatory trials are
# conventionally powered far larger than Phase II ones. This is a disclosed
# heuristic, not a measurement: an RCT with N >= this is assumed Phase III.
RCT_PHASE_III_MIN_N = 300


def beta_tier_for(study_design: str, sample_size: int) -> str:
    """Resolve a (study_design, sample_size) telemetry pair to one of the 6
    seeded Beta tiers. RCT is split by an N-based phase heuristic so that
    all 6 seed tiers are actually reachable; every other design maps
    directly via DESIGN_TO_BETA_TIER."""
    if study_design == "RCT":
        return "Phase III RCT" if sample_size >= RCT_PHASE_III_MIN_N else "Phase II RCT"
    return DESIGN_TO_BETA_TIER.get(study_design, "Retrospective")


def seed_prior_means() -> dict[str, float]:
    """Pure, I/O-free Beta means from the seed hyperparameters. This is the
    default `prior_lookup` agent.py's score_papers() falls back to when no
    persistent store is supplied — keeps unit tests deterministic and
    network/disk-free."""
    return {tier: a / (a + b) for tier, (a, b) in SEED_BETA_PRIORS.items()}


# --------------------------------------------------------------------------
# SQLite schema & connection management
# --------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS design_priors (
    tier  TEXT PRIMARY KEY,
    alpha REAL NOT NULL,
    beta  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    pmid              TEXT PRIMARY KEY,
    title             TEXT,
    study_design      TEXT,
    sample_size       INTEGER,
    prior_credence    REAL,
    discrepancy_index REAL,
    likelihood_penalty REAL,
    posterior_score   REAL,
    audit_status      TEXT DEFAULT 'PASSED',
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    edge_type  TEXT NOT NULL,   -- 'citation' | 'contradiction'
    detail     TEXT,
    created_at TEXT,
    PRIMARY KEY (src, dst, edge_type)
);

CREATE TABLE IF NOT EXISTS feedback_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid       TEXT,
    tier       TEXT NOT NULL,
    action     TEXT NOT NULL,   -- 'confirm' | 'override' | 'reject'
    manual_p   REAL,
    old_alpha  REAL,
    old_beta   REAL,
    new_alpha  REAL,
    new_beta   REAL,
    created_at TEXT
);
"""


def init_db(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """Open (creating if needed) the persistent store, ensure the schema
    exists, and seed design_priors from SEED_BETA_PRIORS the first time.
    Safe to call repeatedly — existing rows are never overwritten."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA_SQL)
    for tier, (a, b) in SEED_BETA_PRIORS.items():
        conn.execute(
            "INSERT OR IGNORE INTO design_priors (tier, alpha, beta) VALUES (?, ?, ?)",
            (tier, a, b),
        )
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Empirical Bayesian active learning
# --------------------------------------------------------------------------


def get_current_prior_means(conn: sqlite3.Connection) -> dict[str, float]:
    """E[P(E)] = alpha / (alpha + beta) for every tier, read fresh from the
    store. This is what agent.py should pass as score_papers()'s
    prior_lookup on any run after the first, so learned feedback actually
    changes future rankings."""
    rows = conn.execute("SELECT tier, alpha, beta FROM design_priors").fetchall()
    return {tier: alpha / (alpha + beta) for tier, alpha, beta in rows}


def get_current_prior_hyperparams(conn: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    rows = conn.execute("SELECT tier, alpha, beta FROM design_priors").fetchall()
    return {tier: (alpha, beta) for tier, alpha, beta in rows}


def record_feedback(
    conn: sqlite3.Connection,
    tier: str,
    action: str,
    pmid: Optional[str] = None,
    manual_p: Optional[float] = None,
    pseudo_weight: float = 2.0,
) -> tuple[float, float]:
    """Update a design tier's Beta(alpha, beta) in response to one human
    decision on one paper, and append an audit row to feedback_log. Returns
    the new (alpha, beta).

    - "confirm"  (operator accepted the automated handling): alpha += 1 —
      one more piece of evidence that this tier's default credence is fine.
    - "reject"   (operator quarantined the paper): beta += 1 — one piece of
      evidence *against* this tier's default credence, at least for this
      instance.
    - "override" (operator manually clamped P(E) to manual_p): the Beta
      distribution is nudged toward the stated belief using `pseudo_weight`
      pseudo-observations — alpha += manual_p*w, beta += (1-manual_p)*w —
      rather than being reset outright, so one human judgment shifts the
      distribution without erasing everything learned before it.
    """
    row = conn.execute("SELECT alpha, beta FROM design_priors WHERE tier = ?", (tier,)).fetchone()
    if row is None:
        old_alpha, old_beta = 1.0, 1.0  # weak uniform fallback if tier was never seeded
    else:
        old_alpha, old_beta = row

    if action == "confirm":
        new_alpha, new_beta = old_alpha + 1.0, old_beta
    elif action == "reject":
        new_alpha, new_beta = old_alpha, old_beta + 1.0
    elif action == "override":
        if manual_p is None:
            raise ValueError("action='override' requires manual_p")
        manual_p = max(0.0, min(1.0, manual_p))
        new_alpha = old_alpha + manual_p * pseudo_weight
        new_beta = old_beta + (1.0 - manual_p) * pseudo_weight
    else:
        raise ValueError(f"unknown feedback action: {action!r}")

    conn.execute(
        "INSERT INTO design_priors (tier, alpha, beta) VALUES (?, ?, ?) "
        "ON CONFLICT(tier) DO UPDATE SET alpha = excluded.alpha, beta = excluded.beta",
        (tier, new_alpha, new_beta),
    )
    conn.execute(
        "INSERT INTO feedback_log "
        "(pmid, tier, action, manual_p, old_alpha, old_beta, new_alpha, new_beta, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pmid, tier, action, manual_p, old_alpha, old_beta, new_alpha, new_beta, _now()),
    )
    conn.commit()
    return new_alpha, new_beta


# --------------------------------------------------------------------------
# Knowledge graph persistence
# --------------------------------------------------------------------------


def upsert_node(
    conn: sqlite3.Connection,
    *,
    pmid: str,
    title: str,
    study_design: str,
    sample_size: int,
    prior_credence: float,
    discrepancy_index: float,
    likelihood_penalty: float,
    posterior_score: float,
    audit_status: str = "PASSED",
) -> None:
    conn.execute(
        "INSERT INTO nodes (pmid, title, study_design, sample_size, prior_credence, "
        "discrepancy_index, likelihood_penalty, posterior_score, audit_status, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(pmid) DO UPDATE SET "
        "title=excluded.title, study_design=excluded.study_design, "
        "sample_size=excluded.sample_size, prior_credence=excluded.prior_credence, "
        "discrepancy_index=excluded.discrepancy_index, "
        "likelihood_penalty=excluded.likelihood_penalty, "
        "posterior_score=excluded.posterior_score, audit_status=excluded.audit_status, "
        "updated_at=excluded.updated_at",
        (
            pmid, title, study_design, sample_size, prior_credence,
            discrepancy_index, likelihood_penalty, posterior_score, audit_status, _now(),
        ),
    )
    conn.commit()


def set_audit_status(conn: sqlite3.Connection, pmid: str, status: str) -> None:
    conn.execute(
        "UPDATE nodes SET audit_status = ?, updated_at = ? WHERE pmid = ?",
        (status, _now(), pmid),
    )
    conn.commit()


def add_edge(
    conn: sqlite3.Connection, src: str, dst: str, edge_type: str, detail: str = ""
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO edges (src, dst, edge_type, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (src, dst, edge_type, detail, _now()),
    )
    conn.commit()


def load_graph(conn: sqlite3.Connection):
    """Build a networkx.DiGraph from the current SQLite state. Imported
    lazily so importing graph_memory doesn't hard-require networkx just to
    do the SQLite/Beta bookkeeping (agent.py uses those without graphing)."""
    import networkx as nx

    g = nx.DiGraph()
    for row in conn.execute(
        "SELECT pmid, title, study_design, sample_size, prior_credence, "
        "discrepancy_index, likelihood_penalty, posterior_score, audit_status FROM nodes"
    ):
        pmid, title, design, n, prior, d, lik, score, status = row
        g.add_node(
            pmid,
            title=title,
            study_design=design,
            sample_size=n,
            prior_credence=prior,
            discrepancy_index=d,
            likelihood_penalty=lik,
            posterior_score=score,
            audit_status=status,
        )
    for row in conn.execute("SELECT src, dst, edge_type, detail FROM edges"):
        src, dst, edge_type, detail = row
        g.add_edge(src, dst, edge_type=edge_type, detail=detail)
    return g


# --------------------------------------------------------------------------
# Citation edges — real data from PubMed's own link graph, restricted to
# pairs where BOTH papers are in our current batch (we only have telemetry
# for our own batch, so a citation to a paper outside it isn't graphable).
# --------------------------------------------------------------------------


def fetch_citation_edges_from_pubmed(
    pmids: list[str], email: str, api_key: Optional[str] = None
) -> list[tuple[str, str]]:
    """Uses Bio.Entrez.elink (pubmed_pubmed_refs) to find which of our
    fetched papers cite which others *within the same batch*. Real PubMed
    link data, not mocked — unlike citation *counts* (see README), the
    link-graph endpoint is free and needs no extra registration."""
    from Bio import Entrez

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    pmid_set = set(pmids)
    edges: list[tuple[str, str]] = []
    with Entrez.elink(
        dbfrom="pubmed", db="pubmed", id=pmids, linkname="pubmed_pubmed_refs"
    ) as handle:
        results = Entrez.read(handle)

    for record in results:
        id_list = record.get("IdList", [])
        if not id_list:
            continue
        src = str(id_list[0])
        for linksetdb in record.get("LinkSetDb", []):
            for link in linksetdb.get("Link", []):
                dst = str(link.get("Id", ""))
                if dst and dst in pmid_set and dst != src:
                    edges.append((src, dst))
    return edges


# --------------------------------------------------------------------------
# Contradiction edges — a disclosed keyword heuristic, not semantic NLI.
# --------------------------------------------------------------------------

_POSITIVE_OUTCOME_KEYWORDS = [
    "improved", "improvement", "reduced fibrosis", "regression of fibrosis",
    "resolution of", "significant reduction", "significantly reduced",
    "significantly improved", "beneficial effect", "reversal of", "efficacious",
]
_NEGATIVE_OUTCOME_KEYWORDS = [
    "no significant", "did not improve", "did not reduce", "failed to",
    "no improvement", "lack of efficacy", "no effect on", "not significantly",
    "no association", "no evidence of benefit",
]


def infer_outcome_direction(title: str, abstract: str) -> int:
    """+1 = abstract's language reports a positive/beneficial outcome,
    -1 = reports a null/negative outcome, 0 = neither phrase family present.

    This is a cheap keyword heuristic, deliberately NOT a semantic
    contradiction detector — see README "known limitations". It exists so
    the graph can flag *candidate* contradictions for a human to actually
    read, without spending a 3rd LLM call per paper (this repo already
    documents real free-tier quota pain from Gemini calls; O(n) string
    matching costs nothing).

    Negative phrasing is checked FIRST and wins outright if present: a
    negated result like "no significant improvement" contains the
    substring "improvement" (a positive keyword), and naive substring
    matching can't tell negated language from an affirmative claim. Since
    negation phrases in this keyword set ("no significant", "did not...")
    are specific enough to rarely false-positive on their own, treating
    them as authoritative avoids that collision without needing real NLP."""
    text = f"{title} {abstract}".lower()
    if any(kw in text for kw in _NEGATIVE_OUTCOME_KEYWORDS):
        return -1
    if any(kw in text for kw in _POSITIVE_OUTCOME_KEYWORDS):
        return 1
    return 0


def detect_contradictions(
    papers: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    """papers: list of (pmid, title, abstract). Returns (pmid_a, pmid_b,
    detail) for every pair whose inferred outcome direction is opposite
    (one positive, one negative) — a candidate "opposing clinical endpoint"
    contradiction for a human (or the dashboard) to inspect."""
    directions = [(pmid, infer_outcome_direction(title, abstract)) for pmid, title, abstract in papers]
    edges: list[tuple[str, str, str]] = []
    for i in range(len(directions)):
        pmid_a, dir_a = directions[i]
        if dir_a == 0:
            continue
        for j in range(i + 1, len(directions)):
            pmid_b, dir_b = directions[j]
            if dir_b == 0 or dir_a == dir_b:
                continue
            label_a = "positive" if dir_a > 0 else "negative"
            label_b = "positive" if dir_b > 0 else "negative"
            edges.append(
                (pmid_a, pmid_b, f"opposing outcome language: {label_a} vs {label_b}")
            )
    return edges
