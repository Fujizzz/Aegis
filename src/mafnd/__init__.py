"""Two-stage multi-agent fake-news detection."""

from .detector import MultiAgentDetector
from .models import DetectionResult, NewsItem
from .providers import DeepSeekProvider, HeuristicProvider

__all__ = [
    "DeepSeekProvider",
    "DetectionResult",
    "HeuristicProvider",
    "MultiAgentDetector",
    "NewsItem",
]
