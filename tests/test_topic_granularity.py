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
    GRANULARITY_FACTORS,
    granularity_factor,
    min_cluster_size_for,
)

LEVELS = ["coarse", "standard", "fine"]


def test_standard_is_the_default():
    assert DEFAULT_GRANULARITY == "standard"


def test_every_level_has_a_plain_language_description():
    """The UI shows these; a missing one would render blank."""
    assert set(GRANULARITY_DESCRIPTIONS) == set(GRANULARITY_FACTORS) == set(LEVELS)
    for level in LEVELS:
        assert len(GRANULARITY_DESCRIPTIONS[level]) > 20


@pytest.mark.parametrize("level", LEVELS)
def test_factor_is_positive(level):
    assert granularity_factor(level) > 0


@pytest.mark.parametrize("bad", ["COARSE", "medium", "", "fine ", None, 1])
def test_unknown_granularity_raises(bad):
    """A typo in a .env must fail loudly, not silently pick a default."""
    with pytest.raises(ValueError, match="Unknown topic granularity"):
        granularity_factor(bad)


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
    assert coarse > standard >= fine


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
