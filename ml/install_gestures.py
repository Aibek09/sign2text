"""Install the versioned official gesture bundle, then verify its checksum."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "work" / "models" / "gesture_recognizer.task"
SHA = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == SHA:
        print("Gesture recognition is installed.")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(20 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA:
        raise RuntimeError("Model checksum mismatch; the existing file was not changed")
    temporary = TARGET.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(TARGET)
    print("Gesture recognition installed. Start with python3 start.py")

if __name__ == "__main__":
    main()
