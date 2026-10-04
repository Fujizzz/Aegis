from mafnd.detector import MultiAgentDetector
from mafnd.models import NewsItem
from mafnd.providers import ASPECTS, HeuristicProvider


def test_offline_detector_runs_both_routes():
    detector = MultiAgentDetector(HeuristicProvider())
    real = detector.detect(
        NewsItem(content="Researchers published a verified study according to the official report.")
    )
    fake = detector.detect(
        NewsItem(title="SHOCKING SECRET!", content="Unbelievable miracle exposed! Everyone is brainwashed.")
    )
    assert real.label == "real"
    assert fake.label == "fake"
    assert set(real.aspect_scores) == set(ASPECTS)
    assert 0.0 <= fake.fake_probability <= 1.0


def test_weight_keys_are_validated():
    try:
        MultiAgentDetector(HeuristicProvider(), weights={"emotion": 1.0})
    except ValueError as error:
        assert "weights must contain exactly" in str(error)
    else:
        raise AssertionError("Invalid weights should fail")
