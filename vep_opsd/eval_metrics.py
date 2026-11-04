"""Text metrics shared by evaluators, without importing training code."""
from __future__ import annotations

import re
import string


def normalise(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text)


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalise(prediction).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for token in p:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in g:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)
