"""Download pinned, safetensors-only ASL inference assets into the workspace."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
os.environ.setdefault("HF_XET_CACHE", str(ROOT / "work" / "hf-cache" / "xet"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

def main():
    from huggingface_hub import snapshot_download
    packages = [
        ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "base", ["*.json", "*.txt", "*.safetensors", "README.md"]),
        ("mamounyosef/sign-language-bridge", "8315997afd45c703d2fb6dcb4dfc96821464fcea", "asl", ["adapter/*.json", "adapter/*.safetensors", "README.md"]),
    ]
    for repo, revision, folder, patterns in packages:
        print(f"Downloading {repo} at {revision}", flush=True)
        snapshot_download(repo, revision=revision, local_dir=ROOT / "work" / "models" / folder,
                          allow_patterns=[p for p in patterns if "safetensors" not in p], max_workers=2)
    from download_weights import download, WEIGHTS
    for spec in WEIGHTS:
        download(ROOT / "work" / "models", spec)
    print("Download complete. Start with: python3 start.py", flush=True)

if __name__ == "__main__":
    main()
