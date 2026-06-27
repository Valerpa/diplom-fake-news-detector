import numpy as np

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _label(prob: float, threshold: float = 0.5) -> str:
    return "ПРАВДИВАЯ" if prob >= threshold else "ЛОЖНАЯ"
