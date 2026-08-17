"""
Sanity checks for the deterministic scoring matrix in agent.py Module 3.

No network, no API keys — pure logic. Run with:
    python test_matrix.py
"""

from agent import PaperMetadata, PaperTelemetry, score_papers


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


def test_methodology_dominates_when_velocity_is_zero():
    """A Meta-Analysis with zero citations should still outrank an RCT with
    zero citations — methodology weight (0.75) must matter even with no
    velocity signal at all."""
    cases = [
        make_case("meta", "Meta-Analysis", 2020, 100, 0, 1),
        make_case("rct", "RCT", 2020, 100, 0, 1),
    ]
    scored = score_papers(cases)
    assert scored[0].metadata.pmid == "meta", scored
    assert scored[0].epistemic_score > scored[1].epistemic_score


def test_hype_penalty_can_sink_a_high_velocity_paper():
    """A low-rigour design (in-vitro/animal) with maximum citation velocity
    AND high hyperbole should still rank BELOW a boring, cautious RCT with
    zero velocity. This is the whole point of the penalty term."""
    cases = [
        make_case("hyped_animal_study", "In-Vitro/Animal", 2026, 20, 500, 5),  # huge velocity
        make_case("boring_rct", "RCT", 2020, 50, 0, 1),  # zero velocity
    ]
    scored = score_papers(cases)
    assert scored[0].metadata.pmid == "boring_rct", scored
    assert scored[-1].metadata.pmid == "hyped_animal_study"
    assert scored[-1].penalty == 0.5


def test_no_penalty_for_cautious_low_rigour_study():
    """Low rigour alone is not penalized — only low rigour PAIRED with
    overclaiming (hyperbole >= 4) triggers the penalty."""
    case = make_case("cautious_animal_study", "In-Vitro/Animal", 2024, 10, 5, 2)
    scored = score_papers([case])
    assert scored[0].penalty == 0.0


def test_velocity_normalizes_to_0_1_range_across_batch():
    cases = [
        make_case("low", "RCT", 2024, 10, 0, 1),
        make_case("mid", "RCT", 2024, 10, 50, 1),
        make_case("high", "RCT", 2024, 10, 100, 1),
    ]
    scored = {p.metadata.pmid: p for p in score_papers(cases)}
    assert scored["low"].velocity_norm == 0.0
    assert scored["high"].velocity_norm == 1.0
    assert 0.0 < scored["mid"].velocity_norm < 1.0


def test_single_tie_batch_does_not_divide_by_zero():
    """When every paper in the batch has identical velocity, min==max and the
    normalization must not raise ZeroDivisionError."""
    cases = [
        make_case("a", "RCT", 2024, 10, 20, 1),
        make_case("b", "RCT", 2024, 10, 20, 1),
    ]
    scored = score_papers(cases)  # should not raise
    assert scored[0].velocity_norm == 0.0
    assert scored[1].velocity_norm == 0.0


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
