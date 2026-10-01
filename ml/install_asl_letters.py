"""Install the small ASL alphabet checkpoint in the local, ignored model cache."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "work" / "models" / "asl_landmark_mlp.pt"
SHA256 = "5be517be8996f18636679c84497eecb240d8691645167f091361bd721ca2be7d"
URL = "https://raw.githubusercontent.com/TharunCh-tch/asl-detection/main/models/asl_landmark_mlp.pt"


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == SHA256:
        print("ASL letter model is installed.")
        return
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read(1_000_000)
    if len(data) != 83001 or hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("ASL letter checkpoint checksum mismatch")
    temporary = TARGET.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(TARGET)
    print("ASL letter model installed locally.")


if __name__ == "__main__":
    main()
