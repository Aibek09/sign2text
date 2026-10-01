"""Local ASL fingerspelling classifier over MediaPipe's 21 hand landmarks.

Architecture and checkpoint by Tharun Chatteti (MIT); see
https://github.com/TharunCh-tch/asl-detection . The checkpoint was trained on
third-party images whose dataset card does not specify a redistribution license.
Keep the checkpoint in the local work/ cache; do not bundle it in outputs.
"""
from functools import lru_cache
from pathlib import Path
from collections import deque
from itertools import combinations
import hashlib

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "work" / "models" / "asl_landmark_mlp.pt"
SHA256 = "5be517be8996f18636679c84497eecb240d8691645167f091361bd721ca2be7d"
# J and Z trace a path in the air; a still frame cannot establish either letter.
LETTERS = frozenset("ABCDEFGHIKLMNOPQRSTUVWXY")


class MotionLetterTracker:
    """Conservative fingertip-path detector for the moving ASL letters J and Z."""

    def __init__(self):
        self.points = deque(maxlen=18)
        self.shape = None
        self.latched = None
        self.latch_until = 0.0

    def reset(self):
        self.points.clear()
        self.shape = None
        self.latched = None
        self.latch_until = 0.0

    def observe(self, hand, state, timestamp):
        if self.latched and timestamp < self.latch_until:
            return self.latched
        self.latched = None
        shape = "J" if state == (False, False, False, False, True) else (
            "Z" if state == (False, True, False, False, False) else None)
        if shape is None or len(hand) != 21:
            self.reset()
            return None
        if self.shape != shape or (self.points and timestamp - self.points[-1][0] > .7):
            self.points.clear()
        self.shape = shape
        tip = hand[20 if shape == "J" else 8]
        palm = max(((hand[0].x - hand[9].x) ** 2 + (hand[0].y - hand[9].y) ** 2) ** .5, 1e-5)
        self.points.append((timestamp, tip.x, tip.y, palm))
        while self.points and timestamp - self.points[0][0] > 2.5:
            self.points.popleft()
        if len(self.points) < 4:
            return None
        points = list(self.points)
        scale = points[0][3]
        found = self._j_path(points, scale) if shape == "J" else self._z_path(points, scale)
        if found:
            self.latched = shape
            self.latch_until = timestamp + 1.0
            self.points.clear()
            return shape
        return None

    @staticmethod
    def _j_path(points, palm):
        start = points[0]
        end = points[-1]
        for low in points[2:-1]:
            if low[2] - start[2] > .65 * palm and low[2] - end[2] > .22 * palm \
                    and abs(end[1] - low[1]) > .38 * palm:
                return True
        return False

    @staticmethod
    def _z_path(points, palm):
        start, end = points[0], points[-1]
        for first, second in combinations(points[1:-1], 2):
            direction = 1 if first[1] > start[1] else -1
            if (direction * (first[1] - start[1]) > .55 * palm
                    and abs(first[2] - start[2]) < .4 * palm
                    and direction * (second[1] - first[1]) < -.45 * palm
                    and second[2] - first[2] > .25 * palm
                    and direction * (end[1] - second[1]) > .55 * palm
                    and abs(end[2] - second[2]) < .4 * palm):
                return True
        return False


def installed():
    try:
        stat = CHECKPOINT.stat()
        return _verified(stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False


@lru_cache(maxsize=2)
def _verified(mtime, size):
    return size == 83001 and hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA256


@lru_cache(maxsize=1)
def _model():
    import torch
    if not installed():
        return None
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint.get("input_dim") != 63 or checkpoint.get("classes") != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError("ASL letter model metadata is invalid")
    model = torch.nn.Sequential(
        torch.nn.Linear(63, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(.3),
        torch.nn.Linear(128, 64), torch.nn.BatchNorm1d(64), torch.nn.ReLU(), torch.nn.Dropout(.3),
        torch.nn.Linear(64, 26),
    )
    model.load_state_dict({key.removeprefix("net."): value for key, value in checkpoint["state_dict"].items()}, strict=True)
    model.eval()
    return model, checkpoint["scaler"]


def predict(hand, allowed=LETTERS):
    """Return a supported letter and probability, or (None, None)."""
    import numpy as np
    import torch
    loaded = _model()
    if loaded is None or len(hand) != 21:
        return None, None
    model, scaler = loaded
    raw = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
    centered = raw - raw[0]
    scale = max(float(np.linalg.norm(centered[9])), 1e-6)
    features = (centered / scale).reshape(63)
    features = (features - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.from_numpy(features.reshape(1, 63))), dim=1)[0]
    first, second = torch.topk(probabilities, 2)
    letter = chr(65 + int(second[0]))
    confidence = float(first[0])
    # Gate unsupported classes and ambiguous shapes rather than forcing a letter.
    if (allowed is not None and letter not in allowed) or confidence < .82 or confidence - float(first[1]) < .42:
        return None, None
    return letter, confidence
