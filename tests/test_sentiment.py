"""The single sentiment-normalization function must be the only conversion."""

from __future__ import annotations

import pytest

from panekmodel2.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    UnmappableSentimentLabel,
    normalize_all,
    normalize_sentiment,
    produces_neutral,
    validate_labels,
)


@pytest.mark.parametrize(
    "label,score,expected",
    [
        ("positive", 0.9, 0.9),
        ("POSITIVE", 0.9, 0.9),
        ("pos", 1.0, 1.0),
        ("negative", 0.75, -0.75),
        ("NEG", 0.5, -0.5),
        ("neutral", 0.95, 0.0),
        ("neu", 1.0, 0.0),
    ],
)
def test_normalize_sentiment(label, score, expected):
    assert normalize_sentiment(label, score) == pytest.approx(expected)


def test_neutral_never_becomes_negative():
    """The bug this function exists to prevent: sign-flipping a neutral label."""
    assert normalize_sentiment("neutral", 0.95) == 0.0


@pytest.mark.parametrize("label", ["LABEL_0", "LABEL_1", "5 stars", "", "mixed"])
def test_unmappable_labels_raise(label):
    with pytest.raises(UnmappableSentimentLabel):
        normalize_sentiment(label, 0.9)


def test_validate_labels_names_the_offenders():
    with pytest.raises(UnmappableSentimentLabel) as excinfo:
        validate_labels(["positive", "LABEL_2", "negative"])
    assert "label_2" in str(excinfo.value)


def test_validate_labels_accepts_known_set():
    validate_labels(["positive", "NEGATIVE", "neutral"])


def test_normalize_all():
    results = [SentimentResult("positive", 0.5), SentimentResult("negative", 0.25)]
    assert normalize_all(results) == pytest.approx([0.5, -0.25])


def test_produces_neutral_is_honest_about_binary_models():
    assert produces_neutral(["positive", "negative"]) is False
    assert produces_neutral(["positive", "neutral"]) is True


def test_aggregate_uses_the_shared_mapping():
    analyzer = SentimentAnalyzer("test-model")
    agg = analyzer.aggregate(
        [SentimentResult("positive", 1.0), SentimentResult("negative", 1.0), SentimentResult("neutral", 1.0)]
    )
    assert agg["mean"] == pytest.approx(0.0)
    assert agg["counts"] == {"positive": 1, "negative": 1, "neutral": 1}
    assert agg["sd"] == pytest.approx(0.816, abs=1e-3)


def test_aggregate_empty():
    analyzer = SentimentAnalyzer("test-model")
    agg = analyzer.aggregate([])
    assert agg["mean"] == 0.0 and agg["counts"] == {}


def test_analyzer_does_not_load_weights_on_construction():
    """`panekmodel2 fetch` must not pull a 1.4 GB model."""
    analyzer = SentimentAnalyzer("definitely-not-a-real-model-name")
    assert analyzer._pipe is None
    assert analyzer.analyze([]) == []
    assert analyzer._pipe is None


def test_analyze_rejects_unmappable_model_output(monkeypatch):
    """An unknown label set fails loudly instead of rendering as all-neutral."""
    analyzer = SentimentAnalyzer("test-model")
    analyzer._pipe = lambda texts, **kwargs: [{"label": "LABEL_0", "score": 0.99} for _ in texts]

    class FakeChunk:
        text = "hello"

    with pytest.raises(UnmappableSentimentLabel):
        analyzer.analyze([FakeChunk()])


@pytest.mark.slow
def test_default_model_labels_are_all_mappable():
    """The configured default must emit labels normalize_sentiment understands.

    Resolves the model's config from the Hub (no inference weights) so a
    default swap cannot silently ship a model whose labels we would refuse.
    """
    from transformers import AutoConfig

    from panekmodel2.config import Settings

    config = AutoConfig.from_pretrained(Settings().sentiment_model)
    labels = list(config.id2label.values())

    validate_labels(labels)
    assert produces_neutral(labels), "the default is documented as having a neutral class"
    assert {l.lower() for l in labels} == {"negative", "neutral", "positive"}
