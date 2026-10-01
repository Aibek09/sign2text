"""Offline, closed-vocabulary hand gestures. No generative text fallback."""
import os
import hashlib
import math
import threading
from functools import lru_cache
from pathlib import Path
from collections import Counter
from time import perf_counter
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))
ASSET = ROOT / "work" / "models" / "gesture_recognizer.task"
MODEL_ID = "simple-gestures-v3"
ASSET_SHA = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
WARNING = "Recognizes only the gestures shown in the app, not sign-language sentences."
WORDS = {"Thumb_Up": "Yes", "Thumb_Down": "No", "Open_Palm": "Stop",
         "Victory": "Peace", "ILoveYou": "I love you", "Pointing_Up": "One",
         "Closed_Fist": "Ready", "ASL_Three": "Three", "Four": "Four",
         "Okay": "Okay", "Call_Me": "Call", "Rock_On": "Rock",
         "Letter_L": "L", "Pinky": "Little", "Pinch": "Pinch",
         "Three_Fingers": "Triple",
         "Thumb_Middle": "Two fingers: thumb + middle",
         "Thumb_Ring": "Two fingers: thumb + ring",
         "Thumb_Middle_Ring": "Three fingers: thumb + middle + ring",
         "Thumb_Middle_Pinky": "Three fingers: thumb + middle + pinky",
         "Thumb_Ring_Pinky": "Three fingers: thumb + ring + pinky",
         "Index_Ring": "Two fingers: index + ring",
         "Index_Middle_Pinky": "Three fingers: index + middle + pinky",
         "Index_Ring_Pinky": "Three fingers: index + ring + pinky",
         "Middle_Ring_Pinky": "Three fingers: middle + ring + pinky",
         "Middle_Pinky": "Two fingers: middle + pinky"}
ASL_WORDS = {f"ASL_{number}": name for number, name in enumerate(
    ("", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")) if number}
ASL_LETTERS = {f"LETTER_{letter}": letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


class LiveRecognizer:
    """Reuse one local image recognizer for webcam frames, without saving them."""
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        from ml.asl_letters import MotionLetterTracker
        self.letter_motion = MotionLetterTracker()

    def recognize_jpeg(self, payload, mode="gestures"):
        import cv2
        import mediapipe as mp
        import numpy as np
        if not installed():
            raise ValueError("Gesture recognition is not installed yet.")
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Camera frame could not be read.")
        height, width = frame.shape[:2]
        if width > 1280 or height > 1280:
            raise ValueError("Camera frame is too large.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.lock:
            if self.model is None:
                options = mp.tasks.vision.GestureRecognizerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(ASSET)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2)
                self.model = mp.tasks.vision.GestureRecognizer.create_from_options(options)
            result = self.model.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            original_hand_count = len(result.hand_landmarks)
            hand_present = bool(original_hand_count)
            if mode != "asl_letters" or original_hand_count != 1:
                self.letter_motion.reset()
            prediction = live_prediction(result, .65, mode,
                                         self.letter_motion if mode == "asl_letters" else None,
                                         monotonic())
            if prediction["gesture"] is None:
                # A front-facing camera may show the opposite hand/orientation.
                mirrored = cv2.flip(rgb, 1)
                result = self.model.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored))
                hand_present = hand_present or bool(result.hand_landmarks)
                prediction = live_prediction(result, .75, mode)
                if original_hand_count == 2 and len(result.hand_landmarks) < 2:
                    prediction = {"gesture": None, "text": None, "confidence": None}
        prediction["handPresent"] = hand_present
        return prediction


def distance(hand, a, b):
    return math.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y)


def hand_state(hand):
    """Five extended-finger flags, independent of image scale and handedness."""
    if len(hand) != 21:
        return None
    palm = max(distance(hand, 0, 9), 1e-5)
    thumb = distance(hand, 4, 17) > distance(hand, 2, 17) + .20 * palm
    fingers = []
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                               (13, 14, 15, 16), (17, 18, 19, 20)):
        bend = distance(hand, mcp, tip) / max(distance(hand, mcp, pip), 1e-5)
        fingers.append(bend > 1.48 and distance(hand, 0, tip) > distance(hand, 0, pip) * 1.08)
    return (thumb, *fingers)


def geometric_gesture(hand):
    state = hand_state(hand)
    if state is None:
        return None
    thumb, index, middle, ring, pinky = state
    palm = max(distance(hand, 0, 9), 1e-5)
    pinch = distance(hand, 4, 8) < .30 * palm
    if pinch and middle and ring and pinky: return "Okay"
    if pinch and not middle and not ring and not pinky: return "Pinch"
    four_tucked = distance(hand, 4, 5) < .45 * palm
    return {
        (False, False, False, False, False): "Closed_Fist",
        (True, True, True, False, False): "ASL_Three",
        (False, True, True, True, True): "Four" if four_tucked else None,
        (True, False, False, False, True): "Call_Me",
        (False, True, False, False, True): "Rock_On",
        (True, True, False, False, False): "Letter_L",
        (False, False, False, False, True): "Pinky",
        (False, True, True, True, False): "Three_Fingers",
        (True, False, True, False, False): "Thumb_Middle",
        (True, False, False, True, False): "Thumb_Ring",
        (True, False, True, True, False): "Thumb_Middle_Ring",
        (True, False, True, False, True): "Thumb_Middle_Pinky",
        (True, False, False, True, True): "Thumb_Ring_Pinky",
        (False, True, False, True, False): "Index_Ring",
        (False, True, True, False, True): "Index_Middle_Pinky",
        (False, True, False, True, True): "Index_Ring_Pinky",
        (False, False, True, True, True): "Middle_Ring_Pinky",
        (False, False, True, False, True): "Middle_Pinky",
    }.get(state)


def asl_number(hand):
    state = hand_state(hand)
    if state is None:
        return None
    thumb, index, middle, ring, pinky = state
    palm = max(distance(hand, 0, 9), 1e-5)
    for number, tip, other, extended in (
            (6, 20, (8, 12, 16), (index, middle, ring)),
            (7, 16, (8, 12, 20), (index, middle, pinky)),
            (8, 12, (8, 16, 20), (index, ring, pinky)),
            (9, 8, (12, 16, 20), (middle, ring, pinky))):
        if all(extended) and distance(hand, 4, tip) < .34 * palm and all(distance(hand, 4, finger) > .25 * palm for finger in other):
            return f"ASL_{number}"
    if middle and ring and pinky and index and thumb:
        return "ASL_5"
    mapping = {
        (False, True, False, False, False): 1,
        (False, True, True, False, False): 2,
        (True, True, True, False, False): 3,
        (False, True, True, True, True): 4 if distance(hand, 4, 5) < .45 * palm else None,
    }
    number = mapping.get(state)
    return f"ASL_{number}" if number else None


def live_prediction(result, threshold, mode="gestures", motion_tracker=None, timestamp=None):
    hands = result.hand_landmarks
    if not 1 <= len(hands) <= 2:
        if motion_tracker is not None:
            motion_tracker.reset()
        return {"gesture": None, "text": None, "confidence": None}
    if mode == "asl_letters" and len(hands) != 1:
        if motion_tracker is not None:
            motion_tracker.reset()
        return {"gesture": None, "text": None, "confidence": None}
    recognized = []
    for index, hand in enumerate(hands):
        if mode == "asl_letters":
            from ml.asl_letters import predict
            moving = motion_tracker.observe(hand, hand_state(hand), timestamp if timestamp is not None else monotonic()) if motion_tracker else None
            if moving:
                recognized.append((f"LETTER_{moving}", None))
                continue
            letter, score = predict(hand)
            if letter is None:
                return {"gesture": None, "text": None, "confidence": None}
            recognized.append((f"LETTER_{letter}", score))
            continue
        if mode == "asl":
            label = asl_number(hand)
            if not label:
                return {"gesture": None, "text": None, "confidence": None}
            recognized.append((label, None))
            continue
        custom = geometric_gesture(hand)
        if custom:
            recognized.append((custom, None))
            continue
        if open_palm_landmarks(hand):
            recognized.append(("Open_Palm", None))
            continue
        categories = result.gestures[index] if index < len(result.gestures) else []
        if not categories or categories[0].category_name not in WORDS or categories[0].score < threshold:
            # Do not pretend that one recognized hand is the entire two-hand gesture.
            return {"gesture": None, "text": None, "confidence": None}
        recognized.append((categories[0].category_name, categories[0].score))
    # Canonical order prevents a pair from flickering when hand tracking swaps indices.
    recognized.sort(key=lambda item: item[0])
    names = [name for name, _ in recognized]
    scores = [score for _, score in recognized]
    vocabulary = ASL_WORDS if mode == "asl" else ASL_LETTERS if mode == "asl_letters" else WORDS
    return {"gesture": "|".join(names), "text": " + ".join(vocabulary[name] for name in names),
            "confidence": round(sum(scores) / len(scores), 3) if all(score is not None for score in scores) else None}


def open_palm_landmarks(hand):
    """Conservative four-extended-finger check for an upright/open palm."""
    if len(hand) != 21:
        return False

    def distance(a, b):
        return math.hypot(hand[a].x - hand[b].x, hand[a].y - hand[b].y)

    def straight(mcp, pip, tip):
        a = (hand[mcp].x - hand[pip].x, hand[mcp].y - hand[pip].y)
        b = (hand[tip].x - hand[pip].x, hand[tip].y - hand[pip].y)
        length = math.hypot(*a) * math.hypot(*b)
        if length < 1e-5:
            return False
        cosine = (a[0] * b[0] + a[1] * b[1]) / length
        return (cosine < -.8 and
                distance(mcp, tip) > 1.65 * distance(mcp, pip) and
                distance(0, tip) > 1.12 * distance(0, pip))

    return all(straight(mcp, pip, tip) for mcp, pip, tip in
               ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)))


def installed():
    try:
        stat = ASSET.stat()
        return _verified_asset(stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False


@lru_cache(maxsize=2)
def _verified_asset(mtime, size):
    return size > 1_000_000 and hashlib.sha256(ASSET.read_bytes()).hexdigest() == ASSET_SHA


def download_status():
    return {"percent": 100 if installed() else 0}


def decide(samples, mode="gestures"):
    """Require a held one- or two-hand gesture and agreement across the clip."""
    votes = Counter()
    previous, run = None, 0
    for label, confidence in samples:
        label = label if valid_gesture(label, mode) and confidence >= .65 else None
        run = run + 1 if label and label == previous else (1 if label else 0)
        previous = label
        if label and run >= 4:
            votes[label] += 1
    if not votes:
        raise ValueError("Gesture not recognized. Hold one or two supported gestures clearly for a second.")
    winner, count = votes.most_common(1)[0]
    if count < max(2, len(samples) * .12) or count / sum(votes.values()) < .8:
        raise ValueError("Hold your gesture or pair of gestures still for a second.")
    return winner


def valid_gesture(label, mode="gestures"):
    if not isinstance(label, str):
        return False
    parts = label.split("|")
    vocabulary = ASL_WORDS if mode == "asl" else ASL_LETTERS if mode == "asl_letters" else WORDS
    return 1 <= len(parts) <= 2 and all(part in vocabulary for part in parts)


def words_for_gesture(label, mode="gestures"):
    vocabulary = ASL_WORDS if mode == "asl" else ASL_LETTERS if mode == "asl_letters" else WORDS
    return " + ".join(vocabulary[part] for part in label.split("|"))


class Translator:
    model = None

    def translate(self, path, progress=lambda _: None, mode="gestures"):
        import cv2
        import mediapipe as mp
        started = perf_counter()
        if not installed():
            raise ValueError("Gesture recognition is not installed yet.")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not 1 <= fps <= 240:
            cap.release()
            raise ValueError("This video cannot be opened. Choose an MP4 or record again.")
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ASSET)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
        samples, index = [], 0
        from ml.asl_letters import MotionLetterTracker
        motion_tracker = MotionLetterTracker() if mode == "asl_letters" else None
        progress("Reading your gesture…")
        try:
            with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    timestamp = (index / fps)
                    index += 1
                    if index / fps > 10.5:
                        raise ValueError("Choose a recording of 1–10 seconds.")
                    h, w = frame.shape[:2]
                    if h * w > 3840 * 2160:
                        raise ValueError("Choose a video no larger than 4K.")
                    if timestamp + 1e-6 < len(samples) / 12:
                        continue
                    scale = min(1, 960 / max(h, w))
                    frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = recognizer.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(timestamp * 1000))
                    prediction = live_prediction(result, .65, mode, motion_tracker, timestamp)
                    samples.append((prediction["gesture"], prediction["confidence"] or .7))
        finally:
            cap.release()
        if index / fps < 1:
            raise ValueError("Record for at least one second.")
        label = decide(samples, mode)
        return {"text": words_for_gesture(label, mode), "gesture": label, "confidence": None,
                "latencyMs": round((perf_counter() - started) * 1000), "model": MODEL_ID if mode == "gestures" else "asl-letters-v1" if mode == "asl_letters" else "asl-numbers-v1",
                "experimental": True, "warning": WARNING, "frames": len(samples), "device": "cpu"}
