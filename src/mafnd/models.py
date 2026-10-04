from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NewsItem:
    content: str
    title: str = ""
    subject: str = ""
    date: str = ""
    identifier: str = ""

    def render(self) -> str:
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.date:
            parts.append(f"Date: {self.date}")
        parts.append(f"Content: {self.content}")
        return "\n".join(parts)


@dataclass(frozen=True)
class VerificationResult:
    fake_probability: float
    rationale: str


@dataclass(frozen=True)
class DetectionResult:
    label: str
    fake_probability: float | None
    stage: str
    aspect_scores: dict[str, float]
    explanation: str
    status: str = "success"
    error_type: str | None = None

    @property
    def p_fake(self):
        return self.fake_probability

    @property
    def class_confidence(self):
        if self.fake_probability is None:
            return None
        return self.fake_probability if self.label == "fake" else 1 - self.fake_probability

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self), p_fake=self.p_fake, class_confidence=self.class_confidence)
