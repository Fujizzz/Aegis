"""Pluggable offline and OpenAI-compatible providers for the detector agents."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from typing import Mapping, Protocol

from .models import NewsItem, VerificationResult
from .probability import probability, parse_probability


ASPECTS = ("emotion", "exaggeration", "bias", "commonsense")


class AgentProvider(Protocol):
    def score(self, aspect: str, item: NewsItem) -> float: ...

    def verify(self, item: NewsItem, aspect_scores: Mapping[str, float]) -> VerificationResult: ...


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class HeuristicProvider:
    """Deterministic local provider for demos, tests, and pipeline integration."""

    lexicons = {
        "emotion": {
            "outrage", "furious", "terrifying", "hate", "panic", "disaster",
            "evil", "betrayal", "shameful", "destroy", "fear", "horrifying",
        },
        "exaggeration": {
            "shocking", "unbelievable", "miracle", "secret", "forbidden", "always",
            "never", "everyone", "everything", "ultimate", "guaranteed", "exposed",
        },
        "bias": {
            "traitor", "enemy", "corrupt", "evil", "propaganda", "brainwashed",
            "liar", "rigged", "agenda", "disgrace", "puppet", "criminals",
        },
        "commonsense": {
            "impossible", "immortal", "flat-earth", "time-travel", "alien cure",
            "instant cure", "never ages", "perpetual motion", "mind control",
        },
    }
    credibility_terms = {
        "according", "audited", "dataset", "evidence", "official", "published",
        "report", "researchers", "study", "trial", "verified", "minutes",
    }

    @staticmethod
    def _words(item: NewsItem) -> tuple[str, list[str]]:
        text = f"{item.title} {item.content}".lower()
        return text, re.findall(r"[a-z0-9-]+", text)

    def score(self, aspect: str, item: NewsItem) -> float:
        if aspect not in ASPECTS:
            raise ValueError(f"Unknown aspect: {aspect}")
        text, words = self._words(item)
        vocabulary = self.lexicons[aspect]
        hits = sum(1 for term in vocabulary if term in text)
        score = 0.06 + min(0.72, hits * 0.18)
        if aspect in {"emotion", "exaggeration"}:
            score += min(0.15, text.count("!") * 0.04)
            uppercase = sum(1 for token in item.title.split() if len(token) > 2 and token.isupper())
            score += min(0.12, uppercase * 0.04)
        if aspect == "bias" and words:
            score += min(0.12, sum(word in {"they", "them", "those"} for word in words) / len(words))
        if aspect == "commonsense" and re.search(r"\b(100%|zero evidence|proves? everyone)\b", text):
            score += 0.20
        return _clip(score)

    def verify(self, item: NewsItem, aspect_scores: Mapping[str, float]) -> VerificationResult:
        text, _ = self._words(item)
        base = sum(float(aspect_scores[name]) for name in ASPECTS) / len(ASPECTS)
        credibility_hits = sum(term in text for term in self.credibility_terms)
        sensational_hits = sum(term in text for term in self.lexicons["exaggeration"])
        adjusted = base - min(0.25, credibility_hits * 0.05) + min(0.25, sensational_hits * 0.05)
        probability = _clip(adjusted)
        rationale = (
            "Offline verifier combined four specialist scores with transparent lexical "
            f"evidence ({credibility_hits} credibility cues; {sensational_hits} sensational cues)."
        )
        return VerificationResult(probability, rationale)


class DeepSeekProvider:
    """OpenAI-compatible DeepSeek provider with no SDK dependency."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 45.0,
    ) -> None:
        if not api_key or api_key == "replace_me":
            raise ValueError("A non-empty API key is required for the DeepSeek provider")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail[:300]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error
        return str(payload["choices"][0]["message"]["content"]).strip()

    @staticmethod
    def _number(text: str) -> float:
        return parse_probability(text)

    @staticmethod
    def _json_object(text: str) -> dict[str, object]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in model response: {text[:120]}")
            return json.loads(match.group(0))

    def score(self, aspect: str, item: NewsItem) -> float:
        criteria = {
            "emotion": "emotional incitement, polarizing language, and fear or outrage cues",
            "exaggeration": "sensationalism, absolutist claims, clickbait, and unsupported certainty",
            "bias": "one-sided framing, selective evidence, and partisan or derogatory language",
            "commonsense": "conflict with common knowledge, internal logic, or basic plausibility",
        }
        if aspect not in criteria:
            raise ValueError(f"Unknown aspect: {aspect}")
        response = self._chat(
            f"You are the {aspect} specialist in a fake-news detection team.",
            "Score the following item only for " + criteria[aspect] + ". "
            "Return exactly one number from 0 (not suspicious) to 1 (highly suspicious).\n\n"
            + item.render(),
            temperature=0.1,
            max_tokens=32,
        )
        return self._number(response)

    def verify(self, item: NewsItem, aspect_scores: Mapping[str, float]) -> VerificationResult:
        response = self._chat(
            "You are the verification layer of a multi-agent fake-news detection system. "
            "Do not claim external fact-checking unless evidence is present in the supplied text.",
            "Review the article and the specialist suspicion scores. Return JSON only as "
            '{"fake_probability": 0.0, "rationale": "brief evidence-based explanation"}.\n\n'
            f"Specialist scores: {json.dumps(dict(aspect_scores), sort_keys=True)}\n\n{item.render()}",
            temperature=0.1,
            max_tokens=220,
        )
        data = self._json_object(response)
        return VerificationResult(
            probability(data["fake_probability"]),
            str(data.get("rationale", "The verifier returned no rationale.")),
        )
