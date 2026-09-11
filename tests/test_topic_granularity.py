"""Topic granularity: the knob a researcher needs before a real corpus.

A 13-video trial showed small batches under-splitting badly — 90-minute
homogeneous videos collapsing into a single topic. Granularity scales
HDBSCAN's minimum cluster size relative to the corpus-size baseline, so it
composes with batch size rather than fighting it.

These tests pin the mapping and the invariants; they do not claim anything
about whether "fine" produces better topics on real transcripts, which is a
judgement only a researcher looking at output can make.
"""

from __future__ import annotations

import pytest

from panekmodel2.topic_model_params import (
    DEFAULT_GRANULARITY,
    GRANULARITY_DESCRIPTIONS,
    GRANULARITY_LEVELS,
    min_cluster_size_for,
    validate_granularity,
)

LEVELS = ["coarse", "standard", "fine"]


def test_standard_is_the_default():
    assert DEFAULT_GRANULARITY == "standard"


def test_every_level_has_a_plain_language_description():
    """The UI shows these; a missing one would render blank."""
    assert set(GRANULARITY_DESCRIPTIONS) == set(GRANULARITY_LEVELS) == set(LEVELS)
    for level in LEVELS:
        assert len(GRANULARITY_DESCRIPTIONS[level]) > 20


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_complete_curve(level):
    curve = GRANULARITY_LEVELS[level]
    assert set(curve) == {"tiny", "small", "ceiling", "divisor"}
    assert all(isinstance(v, int) and v > 0 for v in curve.values())


@pytest.mark.parametrize("bad", ["COARSE", "medium", "", "fine ", None, 1])
def test_unknown_granularity_raises(bad):
    """A typo in a .env must fail loudly, not silently pick a default."""
    with pytest.raises(ValueError, match="Unknown topic granularity"):
        validate_granularity(bad)


def test_standard_preserves_the_previous_baseline():
    """Existing runs must not shift behaviour just because the knob exists."""
    assert min_cluster_size_for(8, "standard") == 2
    assert min_cluster_size_for(30, "standard") == 3
    assert min_cluster_size_for(200, "standard") == 5
    assert min_cluster_size_for(5000, "standard") == 5


@pytest.mark.parametrize("n", [12, 30, 60, 200, 2000])
def test_coarse_splits_less_than_standard_and_fine_splits_more(n):
    """The ordering is the whole contract: bigger minimum → broader topics."""
    coarse = min_cluster_size_for(n, "coarse")
    standard = min_cluster_size_for(n, "standard")
    fine = min_cluster_size_for(n, "fine")
    assert coarse > standard > fine


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 50, 10_000])
@pytest.mark.parametrize("level", LEVELS)
def test_never_below_hdbscans_floor(n, level):
    """min_cluster_size < 2 is meaningless to HDBSCAN."""
    assert min_cluster_size_for(n, level) >= 2


@pytest.mark.parametrize("level", LEVELS)
def test_never_exceeds_what_the_corpus_supports(level):
    """A minimum above n_samples makes HDBSCAN raise.

    That is how a Short used to kill a whole batch, so coarse granularity on a
    tiny batch must not reintroduce it.
    """
    for n in (2, 3, 4, 6, 12):
        assert min_cluster_size_for(n, level) <= max(2, n)


def test_coarse_on_a_tiny_batch_is_still_clusterable():
    """The specific reintroduction risk: coarse doubles the minimum."""
    assert min_cluster_size_for(4, "coarse") <= 4


@pytest.mark.parametrize("level", LEVELS)
def test_result_is_an_int(level):
    """HDBSCAN rejects a float min_cluster_size."""
    value = min_cluster_size_for(200, level)
    assert isinstance(value, int) and not isinstance(value, bool)


def test_modeler_validates_granularity_at_construction():
    """Fail when the runner is built, not minutes into a batch."""
    from panekmodel2.topic_model import TopicModeler

    with pytest.raises(ValueError, match="Unknown topic granularity"):
        TopicModeler(embedding_model="irrelevant", topic_granularity="medium")


def test_modeler_defaults_to_standard():
    from panekmodel2.topic_model import TopicModeler

    assert TopicModeler(embedding_model="irrelevant").topic_granularity == "standard"


def test_settings_expose_granularity_and_validate_it(monkeypatch):
    from panekmodel2.config import Settings

    assert Settings().topic_granularity == "standard"
    assert Settings(topic_granularity="fine").topic_granularity == "fine"


def test_get_settings_rejects_a_bad_env_value(monkeypatch):
    """A bad .env should stop startup rather than surprise a run."""
    import panekmodel2.config as config_mod

    monkeypatch.setenv("TOPIC_GRANULARITY", "medium")
    config_mod.get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Unknown topic granularity"):
            config_mod.get_settings()
    finally:
        monkeypatch.delenv("TOPIC_GRANULARITY", raising=False)
        config_mod.get_settings.cache_clear()


def test_the_spa_offers_exactly_the_supported_levels():
    """The picker and the backend must not drift apart."""
    from pathlib import Path

    app_js = (Path(__file__).resolve().parents[1]
              / "src/panekmodel2/server/static/app.js").read_text()
    block = app_js.split("const GRANULARITY_OPTS = [", 1)[1].split("];", 1)[0]
    for level in LEVELS:
        assert f"'{level}'" in block
    assert "'medium'" not in block


def test_params_module_does_not_import_bertopic():
    """config.py validates granularity; it must not pay for torch to do it."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "src/panekmodel2/topic_model_params.py").read_text()
    assert "bertopic" not in source
    assert "import numpy" not in source


# ── N-7: the multiplier scheme was broken in a way the arithmetic hid ──
def _pre_knob_standard(n: int) -> int:
    """The function that predates the granularity knob, transcribed verbatim.

    Standard staying byte-identical to this is load-bearing: adding an option
    must not silently move anyone's existing results.
    """
    if n <= 10:
        return 2
    if n <= 50:
        return 3
    return max(3, min(5, n // 40))


ALL_SIZES = range(1, 5001)


def test_standard_is_byte_identical_to_the_pre_knob_function():
    drift = [
        n for n in ALL_SIZES
        if min_cluster_size_for(n, "standard") != max(2, min(_pre_knob_standard(n), max(2, n)))
    ]
    assert drift == [], f"standard drifted at {drift[:10]}"


def test_fine_actually_varies_with_corpus_size():
    """N-7: 'fine' was the constant 2 at every size from 1 to 5000.

    The baseline capped at 5, 5 * 0.5 banker-rounds to 2, and 2 is the floor —
    so the documented multiplier design collapsed silently.
    """
    values = {min_cluster_size_for(n, "fine") for n in ALL_SIZES}
    assert len(values) > 1, f"fine is constant at {values}, so the knob does nothing"


def test_coarse_actually_varies_with_corpus_size():
    assert len({min_cluster_size_for(n, "coarse") for n in ALL_SIZES}) > 1


def test_fine_is_strictly_finer_than_standard_wherever_the_floor_allows():
    """Below 11 chunks standard is already 2, so nothing can be finer."""
    offenders = [
        n for n in ALL_SIZES
        if min_cluster_size_for(n, "standard") > 2
        and min_cluster_size_for(n, "fine") >= min_cluster_size_for(n, "standard")
    ]
    assert offenders == [], f"fine not strictly finer at {offenders[:10]}"


def test_coarse_is_strictly_coarser_than_standard_wherever_the_corpus_allows():
    """At 2 chunks the corpus cap and the floor meet, so nothing can be coarser."""
    offenders = [
        n for n in ALL_SIZES
        if n > 2
        and min_cluster_size_for(n, "coarse") <= min_cluster_size_for(n, "standard")
    ]
    assert offenders == [], f"coarse not strictly coarser at {offenders[:10]}"


def test_fine_and_standard_coincide_only_where_the_floor_forces_it():
    """Documented honestly rather than papered over: the knob is inert here."""
    coincide = [n for n in ALL_SIZES
                if min_cluster_size_for(n, "fine") == min_cluster_size_for(n, "standard")]
    assert coincide == list(range(1, 11))


def test_ordering_holds_at_every_size():
    for n in ALL_SIZES:
        c = min_cluster_size_for(n, "coarse")
        s = min_cluster_size_for(n, "standard")
        f = min_cluster_size_for(n, "fine")
        assert c >= s >= f, f"ordering broken at n={n}: {c}/{s}/{f}"


def test_bounds_hold_at_every_size_and_level():
    """2 <= m <= n, so HDBSCAN can never be handed an impossible minimum."""
    for n in ALL_SIZES:
        for level in LEVELS:
            m = min_cluster_size_for(n, level)
            assert 2 <= m <= max(2, n), f"{level} out of bounds at n={n}: {m}"
