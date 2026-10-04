"""Strict probability contract for the standalone detector (no SDK required)."""
import math
import re


def probability(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a probability')
    p = float(value)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Probability must be finite and in [0, 1]')
    return p


def parse_probability(text):
    # Accept a complete numeric token, optionally a percent; never extract a
    # substring from a negative, out-of-range, or ambiguous model answer.
    token = text.strip()
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?', token):
        raise ValueError('Expected one probability number or percentage')
    return probability(float(token[:-1]) / 100 if token.endswith('%') else token)


def normalized_weights(weights, names):
    if set(weights) != set(names):
        raise ValueError('Weights must contain exactly the configured aspects')
    values = {name: float(weights[name]) for name in names}
    if any(isinstance(weights[k], bool) or not math.isfinite(v) or v < 0 for k, v in values.items()):
        raise ValueError('Weights must be finite and nonnegative')
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Weights must have a finite positive sum')
    return {name: value / total for name, value in values.items()}


def thresholds(real, fake, cap):
    real, fake, cap = map(probability, (real, fake, cap))
    if not real < .5 <= fake <= cap:
        raise ValueError('Require 0 <= real < 0.5 <= fake <= second-layer cap <= 1')
    return real, fake, cap


def normalize_comparison(data):
    if not isinstance(data, dict) or data.get('closer') not in ('true', 'fake', 'tie'):
        raise ValueError('Invalid comparison object or closer label')
    pt, pf = probability(data['proximity_true']), probability(data['proximity_fake'])
    if pt + pf <= 0:
        raise ValueError('Comparison probabilities cannot both be zero')
    pt, pf = pt / (pt + pf), pf / (pt + pf)
    # Numeric evidence determines closer; retain explicit, documented normalization.
    return {'closer': 'fake' if pf > pt else 'true' if pt > pf else 'tie',
            'proximity_true': pt, 'proximity_fake': pf}
