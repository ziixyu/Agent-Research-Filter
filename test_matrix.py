"""
Sanity checks for Module 3's Bayesian/epistemic state update in agent.py:
prior credence P(E) -> likelihood penalty L(Absurdity) -> posterior S.

No network, no API keys — pure logic. Run with:
    python test_matrix.py
"""

from agent import PaperMetadata, PaperTelemetry, clip01, score_papers


def make_case(pmid, design, year, n, citations, hype):
    meta = PaperMetadata(
        pmid=pmid,
        title=f"Paper {pmid}",
        abstract="placeholder",
        publication_year=year,
        citations=citations,
        citations_mocked=True,
    )
    tele = PaperTelemetry(is_relevant=True, study_design=design, sample_size=n, claim_hyperbole=hype)
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
    """A cautious claim from a low-rigour design (hype below what the tier
    could justify) must not trigger any penalty — only overreach is
    punished, never modesty."""
    case = make_case("cautious_animal_study", "In-Vitro/Animal", 2024, 10, 5, 1)
    scored = score_papers([case])[0]
    assert scored.discrepancy_index == 0.0
    assert scored.likelihood_penalty == 1.0


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
