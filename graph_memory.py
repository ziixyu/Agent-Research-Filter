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

KNOWN_TIERS = frozenset(SEED_BETA_PRIORS.keys())

# Uninformative Jeffreys prior for a design tier we've never seen before —
# e.g. an LLM-reported study_design like "Mendelian Randomization" or
# "Organ-on-a-Chip" that doesn't fit any of the 6 known tiers.
# PaperTelemetry.study_design is a free-text str specifically so the
# extraction step isn't forced to mis-bucket a genuinely novel design; this
# is the other half of that contract — the scoring/prior layer must never
# crash or silently borrow an unrelated tier's credence when it meets one.
# Beta(0.5, 0.5) gives E[P(E)]=0.5 (maximally noncommittal) and is also the
# maximum-variance member of the Beta(a,a) family on [0,1] (Var=0.125) —
# appropriately uncertain rather than falsely confident either way.
JEFFREYS_ALPHA = 0.5
JEFFREYS_BETA = 0.5
JEFFREYS_MEAN = JEFFREYS_ALPHA / (JEFFREYS_ALPHA + JEFFREYS_BETA)  # 0.5


def beta_tier_for(study_design: str, sample_size: int) -> str:
    """Resolve a (study_design, sample_size) telemetry pair to a Beta tier.
    RCT is split by an N-based phase heuristic so both seeded RCT tiers are
    reachable; the 4 other known designs map directly via
    DESIGN_TO_BETA_TIER. Anything else is treated as its OWN out-of-
    distribution tier, named after the design string itself, rather than
    silently folded into an unrelated known tier — see get_prior_credence()
    for how that tier then gets an uninformative Jeffreys prior instead of
    a KeyError or a borrowed number."""
    if study_design == "RCT":
        return "Phase III RCT" if sample_size >= RCT_PHASE_III_MIN_N else "Phase II RCT"
    return DESIGN_TO_BETA_TIER.get(study_design, study_design)


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
    url               TEXT,
    doi               TEXT,
    standard_error    REAL,
    is_preregistered  INTEGER,
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    edge_type  TEXT NOT NULL,   -- 'citation' | 'contradiction'
    sentiment  TEXT,            -- 'SUPPORTING' | 'MENTION' | 'REFUTING'
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

CREATE TABLE IF NOT EXISTS calibration_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT,
    tier               TEXT,
    alpha              REAL,
    beta               REAL,
    expected_credence  REAL,
    iteration          INTEGER,
    trigger_source     TEXT
);

CREATE TABLE IF NOT EXISTS unresolved_audits (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid           TEXT,
    timestamp      TEXT,
    reason         TEXT,
    telemetry_json TEXT
);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """`CREATE TABLE IF NOT EXISTS` only helps on a brand-new database — an
    already-committed epistemic_memory.db from an earlier version of this
    schema still lacks any column added since. SQLite has no
    `ADD COLUMN IF NOT EXISTS`, so this checks PRAGMA table_info() first."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """Open (creating if needed) the persistent store, ensure the schema
    exists (migrating an older on-disk schema forward if needed), and seed
    design_priors from SEED_BETA_PRIORS the first time. Safe to call
    repeatedly — existing rows are never overwritten."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA_SQL)
    _ensure_column(conn, "nodes", "url", "TEXT")
    _ensure_column(conn, "nodes", "doi", "TEXT")
    _ensure_column(conn, "nodes", "standard_error", "REAL")
    _ensure_column(conn, "nodes", "is_preregistered", "INTEGER")
    _ensure_column(conn, "edges", "sentiment", "TEXT")
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


def _snapshot_calibration(conn: sqlite3.Connection, tier: str, alpha: float, beta: float, trigger_source: str) -> None:
    """Append one row to calibration_history: (tier, alpha, beta,
    E[P(E)]) at this moment, tagged with an auto-incrementing per-tier
    `iteration` (just COUNT(*) for that tier so far — simple, and exactly
    what a convergence line chart wants as its x-axis) and a
    `trigger_source` (who/what caused this update: 'human', 'surrogate',
    'calibration_engine', ...). This is pure bookkeeping — callers should
    already have committed the actual design_priors change before calling
    this, since it just records what alpha/beta now say."""
    iteration = conn.execute(
        "SELECT COUNT(*) FROM calibration_history WHERE tier = ?", (tier,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO calibration_history "
        "(timestamp, tier, alpha, beta, expected_credence, iteration, trigger_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_now(), tier, alpha, beta, alpha / (alpha + beta), iteration, trigger_source),
    )
    conn.commit()


def get_calibration_history_df(conn: sqlite3.Connection):
    """The full calibration_history table as a pandas DataFrame, one row
    per (tier, iteration) snapshot, ordered for convergence plotting —
    ui.py's Tab 2 pivots this into wide format (index=iteration,
    columns=tier, values=expected_credence) for st.line_chart."""
    import pandas as pd

    rows = conn.execute(
        "SELECT timestamp, tier, alpha, beta, expected_credence, iteration, trigger_source "
        "FROM calibration_history ORDER BY tier, iteration"
    ).fetchall()
    columns = ["timestamp", "tier", "alpha", "beta", "expected_credence", "iteration", "trigger_source"]
    return pd.DataFrame(rows, columns=columns)


def record_feedback(
    conn: sqlite3.Connection,
    tier: str,
    action: str,
    pmid: Optional[str] = None,
    manual_p: Optional[float] = None,
    pseudo_weight: float = 2.0,
    trigger_source: str = "human",
) -> tuple[float, float]:
    """Update a design tier's Beta(alpha, beta) in response to one
    decision on one paper, append an audit row to feedback_log, and
    snapshot the result into calibration_history. Returns the new
    (alpha, beta).

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

    `trigger_source` identifies WHO made this decision ('human',
    'surrogate', 'calibration_engine', ...) — purely descriptive metadata
    carried into calibration_history/feedback_log's audit trail; it does
    not change the update math.
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
    _snapshot_calibration(conn, tier, new_alpha, new_beta, trigger_source)
    return new_alpha, new_beta


def register_novel_tier(conn: sqlite3.Connection, tier: str, trigger_source: str = "ood_registration") -> tuple[float, float]:
    """Dynamically register a study-design tier the seed table doesn't
    know about, with the uninformative Jeffreys prior Beta(0.5, 0.5).
    Idempotent (INSERT OR IGNORE) — calling it again for a tier that now
    exists (because feedback has since arrived for it) is a no-op and does
    NOT reset that learning back to the Jeffreys prior, and does NOT write
    a redundant calibration_history snapshot (only the actual first
    registration does)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO design_priors (tier, alpha, beta) VALUES (?, ?, ?)",
        (tier, JEFFREYS_ALPHA, JEFFREYS_BETA),
    )
    newly_inserted = cur.rowcount > 0
    conn.commit()
    row = conn.execute("SELECT alpha, beta FROM design_priors WHERE tier = ?", (tier,)).fetchone()
    if newly_inserted:
        _snapshot_calibration(conn, tier, row[0], row[1], trigger_source)
    return tuple(row)


def set_prior_hyperparams(
    conn: sqlite3.Connection, tier: str, alpha: float, beta: float, trigger_source: str = "import"
) -> None:
    """Directly overwrite one tier's Beta(alpha, beta) — used by the prior
    state management controls (ui.py Tab 2: import/reset), NOT by ordinary
    feedback (see record_feedback() for the pseudo-count update path human
    decisions and the surrogate/calibration engine actually go through).
    Snapshots the result into calibration_history like every other
    prior-affecting write, so an import/reset is visible on the convergence
    chart same as any other update."""
    conn.execute(
        "INSERT INTO design_priors (tier, alpha, beta) VALUES (?, ?, ?) "
        "ON CONFLICT(tier) DO UPDATE SET alpha = excluded.alpha, beta = excluded.beta",
        (tier, alpha, beta),
    )
    conn.commit()
    _snapshot_calibration(conn, tier, alpha, beta, trigger_source)


def export_priors(conn: sqlite3.Connection) -> dict:
    """Serializes the full design_priors table — alpha, beta, and the
    derived expected credence for every tier (seeded and dynamically
    registered OOD tiers alike) — to a plain JSON-able dict. ui.py's
    "Export Priors to JSON" button just json.dumps()'s this."""
    rows = conn.execute("SELECT tier, alpha, beta FROM design_priors ORDER BY tier").fetchall()
    return {
        "exported_at": _now(),
        "tiers": {
            tier: {"alpha": alpha, "beta": beta, "expected_credence": alpha / (alpha + beta)}
            for tier, alpha, beta in rows
        },
    }


def import_priors(conn: sqlite3.Connection, payload: dict, trigger_source: str = "import") -> int:
    """Restores design_priors from a dict shaped like export_priors()'s
    output ({"tiers": {tier: {"alpha":.., "beta":..}}}); a bare
    {tier: {"alpha":.., "beta":..}} mapping (no "tiers" wrapper) is also
    accepted so a hand-edited checkpoint still loads. Rows with a
    non-positive alpha/beta are skipped rather than corrupting a Beta
    distribution's support. Returns the number of tiers actually updated.
    Tiers absent from the payload are left untouched (this restores/merges,
    it does not delete)."""
    tiers = payload.get("tiers", payload) if isinstance(payload, dict) else {}
    updated = 0
    for tier, vals in tiers.items():
        try:
            alpha, beta = float(vals["alpha"]), float(vals["beta"])
        except (KeyError, TypeError, ValueError):
            continue
        if alpha <= 0 or beta <= 0:
            continue
        set_prior_hyperparams(conn, tier, alpha, beta, trigger_source=trigger_source)
        updated += 1
    return updated


def reset_priors_to_default(conn: sqlite3.Connection, trigger_source: str = "reset") -> int:
    """Resets every originally-seeded tier's Beta(alpha, beta) back to
    SEED_BETA_PRIORS, snapshotting each into calibration_history so the
    reset is visible as a discontinuity on the convergence chart rather
    than silently rewriting history. Dynamically-registered OOD tiers are
    left as-is — they have no seed value to reset to by definition. Returns
    the number of tiers reset."""
    for tier, (a, b) in SEED_BETA_PRIORS.items():
        set_prior_hyperparams(conn, tier, a, b, trigger_source=trigger_source)
    return len(SEED_BETA_PRIORS)


def get_prior_credence(conn: sqlite3.Connection, study_design: str, sample_size: int) -> float:
    """Live, DB-backed prior lookup for ONE paper: resolve to a Beta tier
    via beta_tier_for(), and if that tier doesn't exist in the store yet —
    a genuinely out-of-distribution design, e.g. an LLM-reported
    'Mendelian Randomization' or 'Organ-on-a-Chip' study that doesn't fit
    any of the 6 known tiers — register it with the Jeffreys prior instead
    of raising or silently borrowing an unrelated tier's number. This is
    the function agent.py's main() calls once per paper (a cheap side
    effect: at most one INSERT) to make sure every design actually present
    in a batch has a row before prior_lookup is loaded for scoring."""
    tier = beta_tier_for(study_design, sample_size)
    row = conn.execute("SELECT alpha, beta FROM design_priors WHERE tier = ?", (tier,)).fetchone()
    if row is None:
        alpha, beta = register_novel_tier(conn, tier)
    else:
        alpha, beta = row
    return alpha / (alpha + beta)


def is_ood_tier(tier: str) -> bool:
    """True if `tier` isn't one of the 6 originally-seeded Beta tiers —
    i.e. it was dynamically registered via register_novel_tier() for an
    out-of-distribution study design."""
    return tier not in KNOWN_TIERS


# --------------------------------------------------------------------------
# Async quarantine queue — papers the non-blocking fail-safe couldn't
# confidently auto-resolve (see agent.py's resolve_anomaly_non_blocking).
# Inserting here never stalls the run; a human reviews these later, via
# ui.py's Async Quarantine Queue accordion, on their own time.
# --------------------------------------------------------------------------


def insert_unresolved_audit(conn: sqlite3.Connection, *, pmid: str, reason: str, telemetry_json: str) -> int:
    cur = conn.execute(
        "INSERT INTO unresolved_audits (pmid, timestamp, reason, telemetry_json) VALUES (?, ?, ?, ?)",
        (pmid, _now(), reason, telemetry_json),
    )
    conn.commit()
    return cur.lastrowid


def get_unresolved_audits(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, pmid, timestamp, reason, telemetry_json FROM unresolved_audits ORDER BY id"
    ).fetchall()
    return [
        {"id": r[0], "pmid": r[1], "timestamp": r[2], "reason": r[3], "telemetry_json": r[4]}
        for r in rows
    ]


def delete_unresolved_audit(conn: sqlite3.Connection, audit_id: int) -> None:
    """Called once a human has manually resolved a queued item (ui.py's
    one-click override) — removes it from the queue. The actual Beta
    update/prior override is a separate record_feedback() call; this only
    clears the queue entry."""
    conn.execute("DELETE FROM unresolved_audits WHERE id = ?", (audit_id,))
    conn.commit()


# --------------------------------------------------------------------------
# ML surrogate operator — learns to predict a HITL decision from telemetry.
# --------------------------------------------------------------------------


class SurrogateOperator:
    """A lightweight ML classifier that learns to predict what a human (or
    the adversarial calibration engine) would decide about a flagged paper
    — PASS / CLAMP / REJECT — from its telemetry, trained on the
    accumulated feedback_log + nodes history in epistemic_memory.db.

    Once trained (>= MIN_TRAINING_ROWS historical decisions, spanning at
    least 2 distinct outcomes), agent.py's non-blocking fail-safe loop can
    auto-resolve a NEW anomaly without a human in the loop, but only when
    the model is confident enough (see agent.py's CONFIDENCE_THRESHOLD).
    Below that confidence — or before there's enough training data at all —
    the caller is expected to fall back to a conservative fixed penalty
    instead of trusting an under-trained or uncertain prediction.

    Feature vector (5 dims), matching the task spec exactly:
        x = [log10(sample_size + 1), discrepancy_D, standard_error_SE,
             is_preregistered, study_design_prior]
    Target label: 0=PASS ('confirm'), 1=CLAMP ('override'), 2=REJECT ('reject')
    — the exact 3 actions record_feedback() already accepts, so training
    labels come directly from feedback_log.action with no extra mapping
    table to keep in sync.
    """

    MIN_TRAINING_ROWS = 10
    ACTION_TO_LABEL = {"confirm": 0, "override": 1, "reject": 2}
    LABEL_TO_ACTION = {0: "confirm", 1: "override", 2: "reject"}
    LABEL_NAMES = {0: "PASS", 1: "CLAMP", 2: "REJECT"}

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.model = None
        self.is_trained = False
        self.n_training_rows = 0
        self._fit()

    def _training_rows(self) -> list[tuple]:
        """Join feedback_log (the label) against nodes (the features that
        existed for that pmid at last upsert) — a paper must have both a
        recorded decision AND telemetry on file to be usable as a training
        example."""
        return self.conn.execute(
            "SELECT f.action, n.sample_size, n.discrepancy_index, n.standard_error, "
            "n.is_preregistered, n.prior_credence "
            "FROM feedback_log f JOIN nodes n ON f.pmid = n.pmid "
            "WHERE f.action IN ('confirm', 'override', 'reject') AND n.sample_size IS NOT NULL"
        ).fetchall()

    @staticmethod
    def feature_vector(
        sample_size: float, discrepancy_d: float, standard_error: Optional[float],
        is_preregistered: bool, study_design_prior: float,
    ) -> list[float]:
        import math

        return [
            math.log10((sample_size or 0) + 1),
            discrepancy_d or 0.0,
            standard_error if standard_error is not None else 0.0,
            1.0 if is_preregistered else 0.0,
            study_design_prior if study_design_prior is not None else JEFFREYS_MEAN,
        ]

    def _fit(self) -> None:
        rows = self._training_rows()
        self.n_training_rows = len(rows)
        if self.n_training_rows < self.MIN_TRAINING_ROWS:
            return  # not enough history yet — stays untrained, predict() returns None

        X, y = [], []
        for action, n, d, se, prereg, prior in rows:
            X.append(self.feature_vector(n, d, se, bool(prereg), prior))
            y.append(self.ACTION_TO_LABEL[action])

        if len(set(y)) < 2:
            return  # LogisticRegression needs at least 2 distinct classes to fit

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self.model = model
        self.is_trained = True

    def predict(
        self, sample_size: float, discrepancy_d: float, standard_error: Optional[float],
        is_preregistered: bool, study_design_prior: float,
    ) -> Optional[tuple[int, str, float]]:
        """Returns (label, label_name, confidence) or None if the model
        isn't trained. `confidence` is the predicted class's probability
        (softmax-normalized across PASS/CLAMP/REJECT) — the caller decides
        what confidence bar is high enough to act on automatically."""
        if not self.is_trained:
            return None
        x = [self.feature_vector(sample_size, discrepancy_d, standard_error, is_preregistered, study_design_prior)]
        proba = self.model.predict_proba(x)[0]
        # predict_proba's columns follow model.classes_ (sorted, but not
        # necessarily [0,1,2] — e.g. if training data so far only contains
        # 'confirm'/'reject' decisions, classes_ is [0,2] and proba has only
        # 2 columns), so the argmax INDEX must be mapped back through
        # classes_ rather than assumed to equal the label itself.
        best_idx = int(proba.argmax())
        label = int(self.model.classes_[best_idx])
        return label, self.LABEL_NAMES[label], float(proba[best_idx])


# --------------------------------------------------------------------------
# Graph rendering calibration — pure, network/UI-free color and size
# mappings shared by ui.py's pyvis rendering AND test_matrix.py (a single
# source of truth instead of a value dict mirrored/hand-copied into the
# dashboard, which is how a color-spec typo would previously go untested).
# --------------------------------------------------------------------------

# Node diameter is strictly proportional to S_posterior: Radius = 12 + 28*S.
# S_posterior is already clip01()-bounded to [0,1] by construction, so this
# maps to a [12, 40] pixel radius with no additional clamping needed.
NODE_RADIUS_BASE = 12.0
NODE_RADIUS_SCALE = 28.0


def node_radius(posterior_score: Optional[float]) -> float:
    s = max(0.0, min(1.0, posterior_score or 0.0))
    return NODE_RADIUS_BASE + NODE_RADIUS_SCALE * s


# Node fill is a continuous luminance gradient over log10(N+1): light
# silver/slate at N≈0 up to deep electric blue at N>=2000 (clamped beyond
# that ceiling so one N=100000 outlier can't wash out the rest of a batch).
NODE_COLOR_LOW = (0xCB, 0xD5, 0xE1)  # #CBD5E1 — low sample power
NODE_COLOR_HIGH = (0x02, 0x84, 0xC7)  # #0284C7 — high sample power
NODE_COLOR_N_CEILING = 2000


def node_fill_color(sample_size: Optional[float]) -> str:
    import math

    n = max(0.0, sample_size or 0.0)
    t = math.log10(n + 1) / math.log10(NODE_COLOR_N_CEILING + 1)
    t = max(0.0, min(1.0, t))
    rgb = tuple(round(lo + t * (hi - lo)) for lo, hi in zip(NODE_COLOR_LOW, NODE_COLOR_HIGH))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


# Node border encodes a categorical flag, not a continuous value, so it's a
# small lookup rather than a gradient. Quarantine is checked first when a
# node qualifies for more than one flag: an active anomaly is the single
# most actionable signal a reviewer can get from border color alone, ahead
# of OOD design or preregistration.
NODE_BORDER_STYLE = {
    "preregistered": {"color": "#10B981", "width": 3, "dashes": False},
    "ood": {"color": "#8B5CF6", "width": 3, "dashes": False},
    "quarantined": {"color": "#EF4444", "width": 3, "dashes": True},
    "default": {"color": "#334155", "width": 1, "dashes": False},
}


def node_border_style(
    *, is_preregistered: bool = False, is_ood: bool = False, is_quarantined: bool = False
) -> dict:
    if is_quarantined:
        return NODE_BORDER_STYLE["quarantined"]
    if is_ood:
        return NODE_BORDER_STYLE["ood"]
    if is_preregistered:
        return NODE_BORDER_STYLE["preregistered"]
    return NODE_BORDER_STYLE["default"]


# Signed directed citation/contradiction edges — exact spec: SUPPORTING
# solid emerald green width=2, MENTION slate gray width=1, REFUTING bold
# red dashed width=3 (an unconditional contradiction edge is REFUTING too).
EDGE_STYLE = {
    "SUPPORTING": {"color": "#10B981", "width": 2, "dashes": False},
    "MENTION": {"color": "#64748B", "width": 1, "dashes": False},
    "REFUTING": {"color": "#EF4444", "width": 3, "dashes": True},
}


def edge_style_for(edge_type: str, sentiment: Optional[str]) -> dict:
    """Resolve one edge's (edge_type, sentiment) pair to its EDGE_STYLE
    entry — a contradiction edge is unconditionally REFUTING regardless of
    its own sentiment column; a citation edge with no sentiment on file
    defaults to MENTION rather than crashing on a dict lookup."""
    resolved = "REFUTING" if edge_type == "contradiction" else (sentiment or "MENTION")
    return EDGE_STYLE.get(resolved, EDGE_STYLE["MENTION"])


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
    url: Optional[str] = None,
    doi: Optional[str] = None,
    standard_error: Optional[float] = None,
    is_preregistered: Optional[bool] = None,
) -> None:
    conn.execute(
        "INSERT INTO nodes (pmid, title, study_design, sample_size, prior_credence, "
        "discrepancy_index, likelihood_penalty, posterior_score, audit_status, url, doi, "
        "standard_error, is_preregistered, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(pmid) DO UPDATE SET "
        "title=excluded.title, study_design=excluded.study_design, "
        "sample_size=excluded.sample_size, prior_credence=excluded.prior_credence, "
        "discrepancy_index=excluded.discrepancy_index, "
        "likelihood_penalty=excluded.likelihood_penalty, "
        "posterior_score=excluded.posterior_score, audit_status=excluded.audit_status, "
        "url=excluded.url, doi=excluded.doi, standard_error=excluded.standard_error, "
        "is_preregistered=excluded.is_preregistered, updated_at=excluded.updated_at",
        (
            pmid, title, study_design, sample_size, prior_credence,
            discrepancy_index, likelihood_penalty, posterior_score, audit_status,
            url, doi, standard_error, (None if is_preregistered is None else int(is_preregistered)), _now(),
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
    conn: sqlite3.Connection,
    src: str,
    dst: str,
    edge_type: str,
    detail: str = "",
    sentiment: Optional[str] = None,
) -> None:
    """sentiment is one of 'SUPPORTING' / 'MENTION' / 'REFUTING' for
    citation edges (see fetch_citation_edges_from_pubmed); contradiction
    edges are conventionally tagged 'REFUTING' by their caller, since a
    detected contradiction IS a refuting relationship by definition."""
    conn.execute(
        "INSERT OR REPLACE INTO edges (src, dst, edge_type, sentiment, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (src, dst, edge_type, sentiment, detail, _now()),
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
        "discrepancy_index, likelihood_penalty, posterior_score, audit_status, url, doi, "
        "standard_error, is_preregistered FROM nodes"
    ):
        pmid, title, design, n, prior, d, lik, score, status, url, doi, se, prereg = row
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
            url=url,
            doi=doi,
            standard_error=se,
            is_preregistered=bool(prereg) if prereg is not None else False,
            is_ood_design=is_ood_tier(beta_tier_for(design, n or 0)),
        )
    for row in conn.execute("SELECT src, dst, edge_type, sentiment, detail FROM edges"):
        src, dst, edge_type, sentiment, detail = row
        g.add_edge(src, dst, edge_type=edge_type, sentiment=sentiment, detail=detail)
    return g


# --------------------------------------------------------------------------
# Citation edges — real data from PubMed's own link graph, restricted to
# pairs where BOTH papers are in our current batch (we only have telemetry
# for our own batch, so a citation to a paper outside it isn't graphable).
# --------------------------------------------------------------------------


def citation_sentiment(
    src: str, dst: str, texts: Optional[dict[str, tuple[str, str]]]
) -> str:
    """Heuristic sentiment for a citation edge src->dst, reusing the SAME
    outcome-direction heuristic as detect_contradictions(): if both papers'
    abstracts report the same non-neutral outcome direction, the citation
    is tagged SUPPORTING; opposite directions, REFUTING; anything ambiguous
    (either side's direction is neutral/undetermined, or we don't have that
    paper's text) defaults to MENTION. Pure and network-free — this is the
    same disclosed keyword heuristic as contradiction detection, not a real
    citation-context classifier (that would require reading how paper A
    actually characterizes paper B in its own text, which we don't have)."""
    if not texts:
        return "MENTION"
    src_text = texts.get(src)
    dst_text = texts.get(dst)
    if not src_text or not dst_text:
        return "MENTION"
    dir_src = infer_outcome_direction(*src_text)
    dir_dst = infer_outcome_direction(*dst_text)
    if dir_src == 0 or dir_dst == 0:
        return "MENTION"
    return "SUPPORTING" if dir_src == dir_dst else "REFUTING"


def fetch_citation_edges_from_pubmed(
    pmids: list[str],
    email: str,
    api_key: Optional[str] = None,
    texts: Optional[dict[str, tuple[str, str]]] = None,
) -> list[tuple[str, str, str]]:
    """Uses Bio.Entrez.elink (pubmed_pubmed_refs) to find which of our
    fetched papers cite which others *within the same batch*. Real PubMed
    link data, not mocked — unlike citation *counts* (see README), the
    link-graph endpoint is free and needs no extra registration.

    Returns (src, dst, sentiment) triples. `texts` — {pmid: (title,
    abstract)} — is optional; without it every edge is tagged 'MENTION'
    (no sentiment signal available), which is what happens if the caller
    doesn't have title/abstract handy rather than crashing."""
    from Bio import Entrez

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    pmid_set = set(pmids)
    edges: list[tuple[str, str, str]] = []
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
                    edges.append((src, dst, citation_sentiment(src, dst, texts)))
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
