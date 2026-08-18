"""
Sanity checks for:
  - Module 3's Bayesian/epistemic state update in agent.py: prior credence
    P(E) -> likelihood penalty L(Absurdity) -> posterior S.
  - Module 3.5's fail-safe/HITL anomaly gate (detect_anomaly/apply_audit_flags).
  - graph_memory.py's Empirical Bayesian active learning (Beta hyperparameter
    updates from simulated human feedback) and contradiction-edge heuristic.

No network, no API keys — pure logic, and every SQLite test uses a private
in-memory database (":memory:"), so nothing here touches a real
epistemic_memory.db on disk. Run with:
    python test_matrix.py
"""

import graph_memory
from agent import (
    ScoredPaper,
    PaperMetadata,
    PaperTelemetry,
    apply_audit_flags,
    clip01,
    detect_anomaly,
    score_one,
    score_papers,
)


def make_case(
    pmid, design, year, n, citations, hype,
    *, ci_lower=None, ci_upper=None, is_preregistered=False, doi=None,
):
    meta = PaperMetadata(
        pmid=pmid,
        title=f"Paper {pmid}",
        abstract="placeholder",
        publication_year=year,
        citations=citations,
        citations_mocked=True,
        doi=doi,
    )
    tele = PaperTelemetry(
        is_relevant=True,
        study_design=design,
        sample_size=n,
        claim_hyperbole=hype,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_preregistered=is_preregistered,
    )
    return meta, tele


def by_pmid(scored):
    return {p.metadata.pmid: p for p in scored}


# --------------------------------------------------------------------------
# Prior credence: methodology tier should dominate when nothing else
# distinguishes two papers.
# --------------------------------------------------------------------------


def test_prior_dominates_when_velocity_and_hype_are_equal():
    """A Meta-Analysis (P(E)=0.95) with zero citations should still outrank
    an RCT (P(E)=0.90) with zero citations — the prior must matter even with
    no velocity signal and no absurdity penalty on either side."""
    cases = [
        make_case("meta", "Meta-Analysis", 2020, 100, 0, 1),
        make_case("rct", "RCT", 2020, 100, 0, 1),
    ]
    scored = score_papers(cases)
    assert scored[0].metadata.pmid == "meta", scored
    assert scored[0].posterior_score > scored[1].posterior_score
    assert scored[0].likelihood_penalty == 1.0 and scored[1].likelihood_penalty == 1.0


# --------------------------------------------------------------------------
# The requested scenario: observational study, high citation velocity, high
# absurdity — demoted below a modest, cautious RCT.
# --------------------------------------------------------------------------


def test_hyped_observational_study_demoted_below_modest_rct():
    cases = [
        # Big, heavily-cited, but overclaiming relative to its design tier:
        # a retrospective/observational study with a "cure" level claim.
        make_case("hyped_observational", "Retrospective/Observational", 2025, 500, 190, 5),
        # Small, quiet, cautious RCT.
        make_case("modest_rct", "RCT", 2023, 100, 5, 2),
    ]
    scored = by_pmid(score_papers(cases))

    obs, rct = scored["hyped_observational"], scored["modest_rct"]
    assert rct.posterior_score > obs.posterior_score, (rct, obs)
    # The RCT's claim (hype=2) is well within what a Phase-III-assumed RCT
    # prior (4.5/5) can justify, so it should pay no absurdity penalty at all.
    assert rct.likelihood_penalty == 1.0
    # The observational study's claim (hype=5) blows past its tier's
    # justified ceiling (rigor_baseline = 0.40*5 = 2.0) -> real penalty.
    assert obs.likelihood_penalty < 1.0
    assert obs.discrepancy_index > 0


# --------------------------------------------------------------------------
# Isolate the likelihood penalty: same design, same N, same velocity —
# only claim_hyperbole differs.
# --------------------------------------------------------------------------


def test_likelihood_penalty_isolated_from_prior_and_velocity():
    cases = [
        make_case("cautious", "Retrospective/Observational", 2024, 100, 50, 1),
        make_case("hyped", "Retrospective/Observational", 2024, 100, 50, 5),
    ]
    scored = by_pmid(score_papers(cases))
    cautious, hyped = scored["cautious"], scored["hyped"]

    # Same prior (same design), same N, same citations/year -> the only
    # thing that can explain a score difference is the likelihood penalty.
    assert cautious.prior_credence == hyped.prior_credence
    assert cautious.velocity_norm == hyped.velocity_norm
    assert cautious.likelihood_penalty == 1.0
    assert hyped.likelihood_penalty < 1.0
    assert cautious.posterior_score > hyped.posterior_score


def test_underclaiming_is_never_penalized():
    """A cautious claim from a low-rigour design (hype comfortably below
    what the tier could justify) must not trigger any penalty — only
    overreach is punished, never modesty. Uses Retrospective/Observational
    (baseline = 0.40*5 = 2.0) rather than In-Vitro/Animal here specifically
    because its baseline is comfortably above hype=1, giving an unambiguous
    D=0.0 — see test_small_discrepancy_within_threshold_not_penalized below
    for the tighter, tier-merge-affected case."""
    case = make_case("cautious_observational", "Retrospective/Observational", 2024, 100, 5, 1)
    scored = score_papers([case])[0]
    assert scored.discrepancy_index == 0.0
    assert scored.likelihood_penalty == 1.0


def test_small_discrepancy_within_threshold_not_penalized():
    """In-Vitro/Animal and Review/Opinion share one Beta tier ("Review/In
    Vitro", mean 0.15 -> rigor_baseline=0.75), so even the most cautious
    possible claim (hype=1, the Pydantic field's minimum) technically
    overshoots that very low baseline by D=0.25. This is real, not a bug —
    it reflects how little even a modest claim from this evidence tier can
    be taken at face value. The DISCREPANCY_THRESHOLD (0.5) exists exactly
    to absorb small gaps like this as noise rather than absurdity: for a
    single-paper batch (sample_power_weight=0.0), D_adjusted = 0.25*2 =
    0.5, which is NOT > threshold, so no penalty fires."""
    case = make_case("cautious_in_vitro", "In-Vitro/Animal", 2024, 10, 5, 1)
    scored = score_papers([case])[0]
    assert scored.discrepancy_index > 0.0  # a real, if small, gap
    assert scored.likelihood_penalty == 1.0  # but still fully forgiven


# --------------------------------------------------------------------------
# Sample Power Weight: identical overclaim, different N -> different penalty.
# --------------------------------------------------------------------------


def test_sample_power_weight_moderates_the_same_overclaim():
    """Two papers make the identical overclaim (same design, same
    claim_hyperbole -> identical raw Discrepancy Index) but one has a much
    larger sample. The larger study's overclaim should be penalized less
    harshly — big N doesn't excuse overclaiming, but it does make an
    identical overclaim less absurd than the same words from N=10."""
    cases = [
        make_case("small_n", "Retrospective/Observational", 2024, 10, 50, 5),
        make_case("large_n", "Retrospective/Observational", 2024, 5000, 50, 5),
    ]
    scored = by_pmid(score_papers(cases))
    small, large = scored["small_n"], scored["large_n"]

    assert small.discrepancy_index == large.discrepancy_index  # same raw overreach
    assert large.sample_power_weight > small.sample_power_weight
    assert large.discrepancy_adjusted < small.discrepancy_adjusted
    assert large.likelihood_penalty > small.likelihood_penalty  # less harsh
    assert large.posterior_score > small.posterior_score


# --------------------------------------------------------------------------
# Boundary clipping and division-by-zero protection.
# --------------------------------------------------------------------------


def test_clip01_boundaries():
    assert clip01(-0.5) == 0.0
    assert clip01(1.5) == 1.0
    assert clip01(0.3) == 0.3
    assert clip01(0.0) == 0.0
    assert clip01(1.0) == 1.0


def test_posterior_score_always_in_unit_interval():
    cases = [
        make_case("extreme_high", "Meta-Analysis", 2026, 100000, 500, 1),
        make_case("extreme_low", "Review/Opinion", 1990, 0, 0, 5),
        make_case("mid", "Prospective Cohort", 2015, 40, 30, 3),
    ]
    for p in score_papers(cases):
        assert 0.0 <= p.posterior_score <= 1.0, p


def test_tied_batch_does_not_divide_by_zero():
    """When every paper in the batch has identical velocity AND identical
    sample size, both normalizations (velocity and sample power weight) must
    default to 0.0 rather than raising ZeroDivisionError."""
    cases = [
        make_case("a", "RCT", 2024, 10, 20, 1),
        make_case("b", "RCT", 2024, 10, 20, 1),
    ]
    scored = score_papers(cases)  # should not raise
    for p in scored:
        assert p.velocity_norm == 0.0
        assert p.sample_power_weight == 0.0


def test_single_paper_batch_does_not_divide_by_zero():
    case = make_case("solo", "RCT", 2024, 50, 10, 2)
    scored = score_papers([case])  # should not raise
    assert scored[0].velocity_norm == 0.0
    assert scored[0].sample_power_weight == 0.0


# --------------------------------------------------------------------------
# Module 3.5: fail-safe / HITL anomaly gate.
# --------------------------------------------------------------------------


def test_anomaly_triggers_on_high_discrepancy():
    """D >= 2.0 alone must flag a paper, regardless of sample size."""
    cases = [make_case("big_overclaim", "Retrospective/Observational", 2024, 500, 10, 5)]
    scored = score_papers(cases)
    is_anomaly, reasons = detect_anomaly(scored[0])
    assert scored[0].discrepancy_index >= 2.0, scored[0]
    assert is_anomaly
    assert any("Discrepancy Index" in r for r in reasons)


def test_anomaly_triggers_on_small_interventional_sample():
    """N < 30 on an interventional (RCT) design alone must flag a paper,
    even with a perfectly cautious claim (D=0)."""
    cases = [make_case("tiny_rct", "RCT", 2024, 15, 5, 1)]
    scored = score_papers(cases)
    is_anomaly, reasons = detect_anomaly(scored[0])
    assert scored[0].discrepancy_index == 0.0  # not flagged for overclaiming
    assert is_anomaly  # but still flagged, for the sample size
    assert any("N=" in r and "interventional" in r for r in reasons)


def test_anomaly_does_not_trigger_on_small_non_interventional_sample():
    """N < 30 should NOT flag a non-interventional design (e.g. a small
    in-vitro study) — the sample-size criterion is scoped to interventional
    trials only, per spec."""
    cases = [make_case("tiny_in_vitro", "In-Vitro/Animal", 2024, 12, 5, 1)]
    scored = score_papers(cases)
    is_anomaly, _ = detect_anomaly(scored[0])
    assert not is_anomaly


def test_anomaly_does_not_trigger_on_ordinary_paper():
    """A well-powered RCT with a cautious claim should pass clean — the
    gate must not cry wolf on unremarkable papers."""
    cases = [make_case("solid_rct", "RCT", 2024, 400, 20, 1)]
    scored = score_papers(cases)
    is_anomaly, reasons = detect_anomaly(scored[0])
    assert not is_anomaly
    assert reasons == []


def test_apply_audit_flags_sets_status_and_reasons():
    cases = [
        make_case("flagged_one", "Retrospective/Observational", 2024, 500, 10, 5),  # D>=2.0
        make_case("clean_one", "Meta-Analysis", 2024, 5000, 10, 1),
    ]
    scored = score_papers(cases)
    apply_audit_flags(scored)
    by = {p.metadata.pmid: p for p in scored}
    assert by["flagged_one"].audit_status == "FLAGGED"
    assert len(by["flagged_one"].audit_reasons) >= 1
    assert by["clean_one"].audit_status == "PASSED"
    assert by["clean_one"].audit_reasons == []


# --------------------------------------------------------------------------
# graph_memory.py: Empirical Bayesian active learning over design priors.
# --------------------------------------------------------------------------


def _fresh_db():
    """A private in-memory SQLite store — isolated per test, never touches
    a real epistemic_memory.db file on disk."""
    return graph_memory.init_db(":memory:")


def test_beta_seed_means_match_spec():
    means = graph_memory.seed_prior_means()
    assert means["Meta-Analysis"] == 19 / 20
    assert means["Phase III RCT"] == 18 / 20
    assert means["Phase II RCT"] == 14 / 20
    assert means["Prospective Cohort"] == 11 / 20
    assert means["Retrospective"] == 8 / 20
    assert means["Review/In Vitro"] == 3 / 20


def test_beta_tier_for_splits_rct_by_sample_size():
    assert graph_memory.beta_tier_for("RCT", 500) == "Phase III RCT"
    assert graph_memory.beta_tier_for("RCT", 50) == "Phase II RCT"
    assert graph_memory.beta_tier_for("RCT", graph_memory.RCT_PHASE_III_MIN_N) == "Phase III RCT"  # boundary is inclusive
    assert graph_memory.beta_tier_for("Meta-Analysis", 9999) == "Meta-Analysis"
    assert graph_memory.beta_tier_for("In-Vitro/Animal", 5) == "Review/In Vitro"
    assert graph_memory.beta_tier_for("Review/Opinion", 5) == "Review/In Vitro"


def test_confirm_feedback_increases_alpha_only():
    conn = _fresh_db()
    old_a, old_b = graph_memory.get_current_prior_hyperparams(conn)["Retrospective"]
    new_a, new_b = graph_memory.record_feedback(conn, tier="Retrospective", action="confirm", pmid="1")
    assert new_a == old_a + 1.0
    assert new_b == old_b
    conn.close()


def test_reject_feedback_increases_beta_only():
    conn = _fresh_db()
    old_a, old_b = graph_memory.get_current_prior_hyperparams(conn)["Retrospective"]
    new_a, new_b = graph_memory.record_feedback(conn, tier="Retrospective", action="reject", pmid="1")
    assert new_a == old_a
    assert new_b == old_b + 1.0
    conn.close()


def test_override_feedback_shifts_mean_toward_manual_value():
    """Repeatedly overriding a low-credence tier toward P(E)=1.0 should
    monotonically increase its Beta mean, without ever exceeding 1.0."""
    conn = _fresh_db()
    tier = "Review/In Vitro"  # starts at mean 0.15, the lowest seed tier
    means = [graph_memory.get_current_prior_means(conn)[tier]]
    for i in range(5):
        graph_memory.record_feedback(conn, tier=tier, action="override", manual_p=1.0, pmid=str(i))
        means.append(graph_memory.get_current_prior_means(conn)[tier])
    for earlier, later in zip(means, means[1:]):
        assert later > earlier, means
    assert means[-1] < 1.0  # nudged, never reset/clamped to the target outright
    conn.close()


def test_feedback_updates_are_persisted_and_isolated_per_tier():
    """Feedback on one tier must not affect any other tier's hyperparameters."""
    conn = _fresh_db()
    before = graph_memory.get_current_prior_hyperparams(conn)
    graph_memory.record_feedback(conn, tier="Meta-Analysis", action="confirm", pmid="1")
    after = graph_memory.get_current_prior_hyperparams(conn)
    assert after["Meta-Analysis"] != before["Meta-Analysis"]
    for tier in before:
        if tier != "Meta-Analysis":
            assert after[tier] == before[tier]
    conn.close()


def test_feedback_log_records_audit_trail():
    conn = _fresh_db()
    graph_memory.record_feedback(conn, tier="Retrospective", action="confirm", pmid="42")
    rows = conn.execute("SELECT pmid, tier, action FROM feedback_log").fetchall()
    assert rows == [("42", "Retrospective", "confirm")]
    conn.close()


# --------------------------------------------------------------------------
# graph_memory.py: contradiction-edge heuristic.
# --------------------------------------------------------------------------


def test_detect_contradictions_flags_opposing_language():
    papers = [
        ("1", "Semaglutide and fibrosis", "significant reduction in fibrosis observed"),
        ("2", "Semaglutide and fibrosis outcomes", "no significant improvement was observed"),
        ("3", "Unrelated topic", "neutral text with nothing decisive either way"),
    ]
    edges = graph_memory.detect_contradictions(papers)
    assert len(edges) == 1
    assert edges[0][:2] == ("1", "2")


def test_detect_contradictions_no_edge_when_directions_agree():
    papers = [
        ("1", "Study A", "significant improvement was observed"),
        ("2", "Study B", "significantly improved outcomes were reported"),
    ]
    assert graph_memory.detect_contradictions(papers) == []


# --------------------------------------------------------------------------
# graph_memory.py: signed citation sentiment heuristic.
# --------------------------------------------------------------------------


def test_citation_sentiment_supporting_when_directions_agree():
    texts = {
        "1": ("Study A", "significant reduction in fibrosis observed"),
        "2": ("Study B", "significantly reduced fibrosis was also seen"),
    }
    assert graph_memory.citation_sentiment("1", "2", texts) == "SUPPORTING"


def test_citation_sentiment_refuting_when_directions_oppose():
    texts = {
        "1": ("Study A", "significant reduction in fibrosis observed"),
        "2": ("Study B", "no significant improvement was observed"),
    }
    assert graph_memory.citation_sentiment("1", "2", texts) == "REFUTING"


def test_citation_sentiment_mention_when_ambiguous_or_missing():
    texts = {"1": ("Study A", "neutral text, nothing decisive")}
    assert graph_memory.citation_sentiment("1", "2", texts) == "MENTION"  # dst missing
    assert graph_memory.citation_sentiment("1", "1", texts) == "MENTION"  # neutral direction
    assert graph_memory.citation_sentiment("1", "2", None) == "MENTION"  # no texts at all


# --------------------------------------------------------------------------
# Statistical Precision Penalty (CI -> SE) and preregistration bonus.
# --------------------------------------------------------------------------


def test_precision_penalty_absent_without_ci():
    comp = score_one(0.90, claim_hyperbole=1, w_n=0.5, v_norm=0.5)
    assert comp["standard_error"] is None
    assert comp["precision_penalty"] == 1.0


def test_precision_penalty_absent_for_tight_ci():
    """SE = (ci_upper - ci_lower) / 3.92. A tight CI (SE <= 0.5, the
    threshold) must not be penalized."""
    comp = score_one(0.90, claim_hyperbole=1, w_n=0.5, v_norm=0.5, ci_lower=1.0, ci_upper=2.96)
    expected_se = (2.96 - 1.0) / 3.92
    assert abs(comp["standard_error"] - expected_se) < 1e-9
    assert expected_se <= 0.5 + 1e-9
    assert comp["precision_penalty"] == 1.0


def test_precision_penalty_applies_for_wide_ci():
    """A wide CI (SE > 0.5) must trigger a real (<1.0) precision penalty,
    and posterior_score must be strictly lower than the same paper with a
    tight CI, all else equal."""
    tight = score_one(0.90, claim_hyperbole=1, w_n=0.5, v_norm=0.5, ci_lower=1.0, ci_upper=2.0)
    wide = score_one(0.90, claim_hyperbole=1, w_n=0.5, v_norm=0.5, ci_lower=1.0, ci_upper=10.0)
    assert tight["precision_penalty"] == 1.0
    assert wide["standard_error"] > 0.5
    assert wide["precision_penalty"] < 1.0
    assert wide["posterior_score"] < tight["posterior_score"]


def test_preregistration_bonus_boosts_effective_prior():
    not_prereg = score_one(0.90, claim_hyperbole=1, w_n=0.5, v_norm=0.5, is_preregistered=False)
    prereg = score_one(0.90, claim_hyperbole=1, w_n=0.5, v_norm=0.5, is_preregistered=True)
    assert not_prereg["preregistration_bonus"] == 0.0
    assert prereg["preregistration_bonus"] == 0.05
    assert abs(prereg["effective_prior_credence"] - (0.90 + 0.05)) < 1e-9
    assert prereg["posterior_score"] > not_prereg["posterior_score"]


def test_preregistration_bonus_is_capped_at_one():
    comp = score_one(0.99, claim_hyperbole=1, w_n=0.5, v_norm=0.5, is_preregistered=True)
    assert comp["effective_prior_credence"] == 1.0  # 0.99 + 0.05 clipped, not 1.04


def test_score_papers_end_to_end_with_ci_and_preregistration():
    """Integration check: score_papers() actually threads ci_lower/ci_upper/
    is_preregistered from telemetry through to the ScoredPaper output."""
    cases = [
        make_case(
            "prereg_tight_ci", "RCT", 2024, 500, 20, 1,
            ci_lower=1.0, ci_upper=1.8, is_preregistered=True,
        ),
    ]
    scored = score_papers(cases)
    p = scored[0]
    assert p.preregistration_bonus == 0.05
    assert p.standard_error is not None and p.standard_error < 0.5
    assert p.precision_penalty == 1.0
    assert p.prior_credence == round(p.base_prior_credence + 0.05, 4)


# --------------------------------------------------------------------------
# Out-of-distribution study designs & Jeffreys priors.
# --------------------------------------------------------------------------


def test_ood_design_registers_jeffreys_prior():
    conn = _fresh_db()
    assert conn.execute(
        "SELECT 1 FROM design_priors WHERE tier = ?", ("Mendelian Randomization",)
    ).fetchone() is None

    p = graph_memory.get_prior_credence(conn, "Mendelian Randomization", 5000)
    assert p == graph_memory.JEFFREYS_MEAN == 0.5

    row = conn.execute(
        "SELECT alpha, beta FROM design_priors WHERE tier = ?", ("Mendelian Randomization",)
    ).fetchone()
    assert row == (graph_memory.JEFFREYS_ALPHA, graph_memory.JEFFREYS_BETA)
    conn.close()


def test_ood_design_registration_is_idempotent_after_feedback():
    """Calling get_prior_credence() again for a design that has since
    received human feedback must NOT reset it back to the Jeffreys prior."""
    conn = _fresh_db()
    graph_memory.get_prior_credence(conn, "Organ-on-a-Chip", 10)
    graph_memory.record_feedback(conn, tier="Organ-on-a-Chip", action="confirm", pmid="1")
    a1, b1 = graph_memory.get_current_prior_hyperparams(conn)["Organ-on-a-Chip"]
    assert (a1, b1) == (graph_memory.JEFFREYS_ALPHA + 1.0, graph_memory.JEFFREYS_BETA)

    graph_memory.get_prior_credence(conn, "Organ-on-a-Chip", 10)  # should not reset
    a2, b2 = graph_memory.get_current_prior_hyperparams(conn)["Organ-on-a-Chip"]
    assert (a2, b2) == (a1, b1)
    conn.close()


def test_beta_tier_for_novel_design_is_its_own_tier():
    """An unrecognized design must resolve to a tier named after itself,
    not silently fold into an unrelated known tier (e.g. Retrospective)."""
    tier = graph_memory.beta_tier_for("Organ-on-a-Chip", 10)
    assert tier == "Organ-on-a-Chip"
    assert graph_memory.is_ood_tier(tier)
    assert not graph_memory.is_ood_tier("Retrospective")


def test_score_papers_handles_ood_design_without_crashing():
    """score_papers() must never raise on a study_design outside the 6
    known tiers ('do not fail on unknown categories') — it should fall back
    to the Jeffreys mean and flag is_ood_design=True."""
    cases = [make_case("novel", "Mendelian Randomization", 2024, 5000, 20, 1)]
    scored = score_papers(cases)  # default prior_lookup = seed_prior_means(), no DB
    assert scored[0].is_ood_design is True
    assert scored[0].base_prior_credence == graph_memory.JEFFREYS_MEAN


# --------------------------------------------------------------------------
# backtest_calibration.py: automated calibration updates.
# --------------------------------------------------------------------------


def test_apply_verdict_robust_maps_to_confirm():
    import backtest_calibration as bc

    conn = _fresh_db()
    before = graph_memory.get_current_prior_hyperparams(conn)["Retrospective"]
    verdict = bc.CalibrationVerdict(
        sample_power_adequate=True, selective_reporting_risk=False,
        endpoint_validity_concern=False, verdict="ROBUST", rationale="Solid.",
    )
    new_a, new_b = bc.apply_verdict(conn, "Retrospective", verdict, pmid="1")
    assert (new_a, new_b) == (before[0] + 1.0, before[1])
    conn.close()


def test_apply_verdict_vulnerable_maps_to_reject():
    import backtest_calibration as bc

    conn = _fresh_db()
    before = graph_memory.get_current_prior_hyperparams(conn)["Retrospective"]
    verdict = bc.CalibrationVerdict(
        sample_power_adequate=False, selective_reporting_risk=True,
        endpoint_validity_concern=True, verdict="VULNERABLE", rationale="Weak.",
    )
    new_a, new_b = bc.apply_verdict(conn, "Retrospective", verdict, pmid="1")
    assert (new_a, new_b) == (before[0], before[1] + 1.0)
    conn.close()


def test_run_calibration_pass_logs_a_result_per_judged_paper():
    """run_calibration_pass() drives judge_paper() per paper — verified here
    with a stub judge (no live Gemini call) standing in for judge_paper."""
    import backtest_calibration as bc

    conn = _fresh_db()
    papers = [
        {
            "metadata": {"pmid": "1", "title": "T1", "abstract": "A1"},
            "telemetry": {"study_design": "RCT", "sample_size": 400, "claim_hyperbole": 1, "is_preregistered": True},
            "prior_credence": 0.9, "posterior_score": 0.8,
        },
        {
            "metadata": {"pmid": "2", "title": "T2", "abstract": "A2"},
            "telemetry": {"study_design": "Retrospective/Observational", "sample_size": 50, "claim_hyperbole": 5, "is_preregistered": False},
            "prior_credence": 0.4, "posterior_score": 0.1,
        },
    ]
    stub_verdicts = {
        "1": bc.CalibrationVerdict(sample_power_adequate=True, selective_reporting_risk=False, endpoint_validity_concern=False, verdict="ROBUST", rationale="ok"),
        "2": bc.CalibrationVerdict(sample_power_adequate=False, selective_reporting_risk=True, endpoint_validity_concern=True, verdict="VULNERABLE", rationale="weak"),
    }
    original_judge = bc.judge_paper
    bc.judge_paper = lambda client, model, paper: stub_verdicts[paper["metadata"]["pmid"]]
    try:
        result = bc.run_calibration_pass(client=None, model="stub", conn=conn, papers=papers)
    finally:
        bc.judge_paper = original_judge

    assert len(result["results"]) == 2
    assert result["results"][0]["verdict"]["verdict"] == "ROBUST"
    assert result["results"][1]["verdict"]["verdict"] == "VULNERABLE"
    conn.close()


# --------------------------------------------------------------------------
# URL formatting and metadata preservation.
# --------------------------------------------------------------------------


def test_paper_metadata_url_auto_fills_from_pmid():
    meta = PaperMetadata(
        pmid="12345678", title="T", abstract="A", publication_year=2024, citations=0,
    )
    assert meta.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert meta.doi is None


def test_paper_metadata_url_explicit_value_is_preserved():
    meta = PaperMetadata(
        pmid="1", title="T", abstract="A", publication_year=2024, citations=0,
        url="https://example.org/custom",
    )
    assert meta.url == "https://example.org/custom"


def test_paper_metadata_doi_round_trips_through_scoring():
    case = make_case("1", "RCT", 2024, 100, 10, 1, doi="10.1000/xyz123")
    scored = score_papers([case])
    dumped = scored[0].model_dump()
    assert dumped["metadata"]["doi"] == "10.1000/xyz123"
    assert dumped["metadata"]["url"] == "https://pubmed.ncbi.nlm.nih.gov/1/"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
