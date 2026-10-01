"""Local, experimental ASL inference. All weights are loaded offline.

Model and training reference: https://huggingface.co/mamounyosef/sign-language-bridge
Live preprocessing follows its crop/CLAHE/landmark-overlay recipe; frame sampling
and greedy decoding are reduced for a 16 GB Mac and differ from published eval.
"""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from ml.download_weights import WEIGHTS, CHUNK

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "work" / "models"
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_ID = "sign-language-bridge / Qwen3-VL-2B"
WARNING = "Experimental ASL output. This model often mistranslates signing; verify the meaning with a fluent signer."


def installed():
    files_ready = all((MODEL_DIR / p).is_file() for p in [
        "base/config.json", "base/model.safetensors", "base/tokenizer.json",
        "base/chat_template.json", "base/tokenizer_config.json", "base/preprocessor_config.json",
        "base/video_preprocessor_config.json",
        "asl/adapter/adapter_config.json", "asl/adapter/adapter_model.safetensors",
    ])
    if not files_ready:
        return False
    try:
        return all((MODEL_DIR / local).stat().st_size == size and
                   (MODEL_DIR / local).with_suffix(".verified").read_text().strip() == sha
                   for _, _, _, local, size, sha in WEIGHTS)
    except (OSError, UnicodeError):
        return False


def download_status():
    """Count persisted ranges, excluding incomplete and legacy cache files."""
    complete = 0
    total = sum(spec[4] for spec in WEIGHTS)
    for _, _, _, local, size, sha in WEIGHTS:
        path = MODEL_DIR / local
        try:
            if path.stat().st_size == size:
                complete += size
                continue
        except OSError:
            pass
        for part in (MODEL_DIR / "range-parts" / sha).glob("*"):
            if part.name.isdigit():
                index = int(part.name)
                expected = min(CHUNK, size - index * CHUNK)
                try:
                    if expected > 0 and part.stat().st_size == expected:
                        complete += expected
                except OSError:
                    pass  # The installer may be assembling/deleting ranges.
    return {"bytes": complete, "totalBytes": total, "percent": round(complete / total * 100)}


def prepare_video(path):
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not 1 <= fps <= 240:
        cap.release()
        raise ValueError("This video cannot be decoded. Try an MP4 or a new camera recording.")
    frames = []
    # Decode sequentially: browser WebM often has no reliable frame count.
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index / fps > 10.5:
                raise ValueError("Use a short clip of 1–10 seconds. Longer videos are not truncated silently.")
            if frame.shape[0] * frame.shape[1] > 3840 * 2160:
                raise ValueError("Use a video no larger than 4K.")
            if not frames or (index - 1) / fps >= len(frames) / 8:
                h, w = frame.shape[:2]
                scale = min(1, 640 / max(h, w))
                frames.append(cv2.resize(frame, (round(w * scale), round(h * scale))))
    finally:
        cap.release()
    if index / fps < 1 or len(frames) < 2:
        raise ValueError("Record at least one second with your face and hands visible.")
    if len(frames) > 64:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 64).astype(int)]
    detected = []
    crop_points = []
    hands_seen = 0
    with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=.5,
                                      min_tracking_confidence=.5) as holistic:
        for frame in frames:
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            groups = []
            for landmarks, indices in [
                (result.pose_landmarks, [11, 12, 13, 14, 15, 16]),
                (result.left_hand_landmarks, list(range(21))),
                (result.right_hand_landmarks, list(range(21))),
            ]:
                groups.append([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in indices] if landmarks else None)
            if result.pose_landmarks:
                crop_points.extend((p.x, p.y) for p in result.pose_landmarks.landmark[:25] if p.visibility > .5)
            for group in groups[1:]:
                if group:
                    crop_points.extend(group)
            hands_seen += bool(groups[1] or groups[2])
            detected.append(groups)
    if hands_seen < max(2, len(frames) // 8) or not crop_points:
        raise ValueError("No clear signing hands detected. Keep hands, face, and upper body visible and try again.")
    points = np.array(crop_points)
    lower, upper = points.min(0), points.max(0)
    margin = np.maximum((upper - lower) * .15, .08)
    lower, upper = np.clip(lower - margin, 0, 1), np.clip(upper + margin, 0, 1)
    h, w = frames[0].shape[:2]
    x1, y1 = (lower * [w, h]).astype(int)
    x2, y2 = (upper * [w, h]).astype(int)
    if x2 - x1 < 32 or y2 - y1 < 32:
        raise ValueError("Move into the frame so your hands and upper body are visible.")
    pose_edges = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]
    hand_edges = [(0, s) for s in [1, 5, 9, 13, 17]] + [(i, i + 1) for s in [1, 5, 9, 13, 17] for i in range(s, s + 3)] + [(5, 9), (9, 13), (13, 17)]
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    output = []
    for frame, groups in zip(frames, detected):
        frame = frame[y1:y2, x1:x2].copy()
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        for group, edges, color in zip(groups, [pose_edges, hand_edges, hand_edges], [(0,255,255), (0,200,0), (255,80,0)]):
            if group:
                coords = [(round(x*w-x1), round(y*h-y1)) for x,y in group]
                for a,b in edges:
                    cv2.line(frame, coords[a], coords[b], color, 1, cv2.LINE_AA)
                for point in coords:
                    cv2.circle(frame, point, 1, color, -1, cv2.LINE_AA)
        output.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(output), index / fps


class Translator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        if not installed():
            raise RuntimeError("Model download is incomplete. Run ml/install_model.py with the model environment.")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        processor = AutoProcessor.from_pretrained(str(MODEL_DIR / "base"), local_files_only=True)
        base = AutoModelForImageTextToText.from_pretrained(
            str(MODEL_DIR / "base"), dtype=dtype, local_files_only=True,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, str(MODEL_DIR / "asl" / "adapter"),
                                              local_files_only=True, autocast_adapter_dtype=False)
        model.to(self.device).eval()
        self.processor, self.model = processor, model

    def translate(self, path, progress=lambda _: None):
        import torch
        started = perf_counter()
        progress("Checking video and tracking hands…")
        frames, duration = prepare_video(path)
        if self.model is None:
            progress("Loading the local ASL model for the first time…")
            self.load()
        progress("Translating ASL on your device…")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a sign language translator."}]},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Translate this American Sign Language video into English."}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], videos=[frames], return_tensors="pt",
                                do_sample_frames=False,
                                video_metadata=[{"fps":len(frames)/duration, "total_num_frames":len(frames),
                                                 "frames_indices":list(range(len(frames)))}],
                                size={"shortest_edge":len(frames)*4*32*32, "longest_edge":len(frames)*64*32*32}).to(self.device)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                         num_beams=1, repetition_penalty=1.1, no_repeat_ngram_size=4)
        text = self.processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError("The model returned no text. Try a clearer, shorter ASL phrase.")
        return {"text": text, "confidence": None, "latencyMs": round((perf_counter()-started)*1000),
                "model": MODEL_ID, "experimental": True, "warning": WARNING,
                "frames": len(frames), "device": self.device}
