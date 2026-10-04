"""Two-stage orchestration for specialist fake-news detection agents."""

from __future__ import annotations

import statistics
from typing import Mapping

from .models import DetectionResult, NewsItem
from .providers import ASPECTS, AgentProvider
from .probability import probability, normalized_weights, thresholds


DEFAULT_WEIGHTS = {
    "emotion": 0.30,
    "exaggeration": 0.25,
    "bias": 0.30,
    "commonsense": 0.15,
}


class MultiAgentDetector:
    """Four specialist agents followed by a conditional verification layer."""

    def __init__(
        self,
        provider: AgentProvider,
        *,
        weights: Mapping[str, float] | None = None,
        real_threshold: float = 0.30,
        fake_threshold: float = 0.70,
        second_layer_fake_cap: float = 0.90,
        self_consistency: int = 1,
    ) -> None:
        real_threshold, fake_threshold, second_layer_fake_cap = thresholds(
            real_threshold, fake_threshold, second_layer_fake_cap)
        if type(self_consistency) is not int or self_consistency < 1:
            raise ValueError("self_consistency must be at least 1")
        selected = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if set(selected) != set(ASPECTS):
            raise ValueError(f"weights must contain exactly: {', '.join(ASPECTS)}")
        self.provider = provider
        self.weights = normalized_weights(selected, ASPECTS)
        self.real_threshold = real_threshold
        self.fake_threshold = fake_threshold
        self.second_layer_fake_cap = second_layer_fake_cap
        self.self_consistency = self_consistency

    def detect(self, item: NewsItem) -> DetectionResult:
        scores = {}
        try:
            scores = self._screen(item)
        except Exception as error:
            return DetectionResult('abstain', None, 'screening', scores, '', 'error', type(error).__name__)
        try:
            return self._decide(item, scores)
        except Exception as error:
            return DetectionResult('abstain', None, 'verification', scores, '', 'error', type(error).__name__)

    def _screen(self, item):
        scores = {
            aspect: statistics.median(
                probability(self.provider.score(aspect, item)) for _ in range(self.self_consistency)
            )
            for aspect in ASPECTS
        }
        return scores

    def _decide(self, item, scores):
        stage_one_probability = sum(scores[name] * self.weights[name] for name in ASPECTS)
        if stage_one_probability >= self.fake_threshold:
            initial_label = "fake"
        elif stage_one_probability <= self.real_threshold:
            initial_label = "real"
        else:
            initial_label = "uncertain"

        needs_verification = initial_label == "uncertain" or (
            initial_label == "fake" and stage_one_probability < self.second_layer_fake_cap
        )
        if needs_verification:
            verification = self.provider.verify(item, scores)
            p = probability(verification.fake_probability)
            return DetectionResult(
                label="fake" if p >= 0.5 else "real",
                fake_probability=p,
                stage="verification",
                aspect_scores=scores,
                explanation=verification.rationale,
            )

        explanation = (
            "The first-stage specialist consensus was outside the uncertainty region; "
            "the verification layer was not invoked."
        )
        return DetectionResult(
            label=initial_label,
            fake_probability=stage_one_probability,
            stage="screening",
            aspect_scores=scores,
            explanation=explanation,
        )
