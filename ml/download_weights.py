"""Bounded, resumable range transfers with final SHA-256 verification."""
from __future__ import annotations
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WEIGHTS = [
    ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "model.safetensors", "base/model.safetensors", 4255140312, "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"),
    ("mamounyosef/sign-language-bridge", "8315997afd45c703d2fb6dcb4dfc96821464fcea", "adapter/adapter_model.safetensors", "asl/adapter/adapter_model.safetensors", 755500712, "aea9664e2bde31bdeb2934dd0a863616145f687153035a7cff14c279570b1405"),
]
CHUNK = 4 * 1024 * 1024

def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()

def download(root, spec):
    # Advisory locks are released even after an interrupted/killed installer.
    import fcntl
    locks = root / ".download-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / spec[5]).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another installer is downloading {spec[3]}; wait for it to finish") from None
        _download(root, spec)


def _download(root, spec):
    import requests
    repo, revision, remote, local, size, sha = spec
    destination = root / local
    marker = destination.with_suffix(".verified")
    if destination.exists() and destination.stat().st_size == size and digest(destination) == sha:
        marker.write_text(sha)
        return
    marker.unlink(missing_ok=True)  # Never retain proof for a now-corrupted file.
    parts = root / "range-parts" / sha
    parts.mkdir(parents=True, exist_ok=True)
    count = (size + CHUNK - 1) // CHUNK
    def fetch(index):
        start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(6):
            try:
                # A unique range query prevents intermediaries caching another byte range.
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote}?download=true&part={index}"
                with requests.get(url, headers={"Range":f"bytes={start}-{end}"}, timeout=(10,60), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server returned an unexpected byte range")
                    with path.with_suffix(".tmp").open("wb") as out:
                        for block in response.iter_content(256 * 1024):
                            out.write(block)
                if path.with_suffix(".tmp").stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                path.with_suffix(".tmp").replace(path)
                return
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise RuntimeError(f"Range {index} failed after six attempts; rerun to resume") from None
                time.sleep(attempt + 1)
    print(f"Downloading {local}: {size / 1e9:.2f} GB in recoverable ranges", flush=True)
    workers = max(1, min(16, int(os.environ.get("SIGN2TEXT_DOWNLOAD_WORKERS", "8"))))
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(fetch, i) for i in range(count)]
    failures = []
    try:
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except RuntimeError as error:
                failures.append(str(error))
            if completed % 20 == 0 or completed == count:
                print(f"{local}: {completed}/{count} parts", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if failures:
        raise RuntimeError(f"{len(failures)} ranges remain incomplete. Rerun to resume only missing parts.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    assembling = destination.with_suffix(".assembling")
    with assembling.open("wb") as out:
        for index in range(count):
            with (parts / str(index)).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(block)
    if digest(assembling) != sha:
        raise RuntimeError("Checksum mismatch. Weights were not activated; remove range-parts for this hash and retry.")
    assembling.replace(destination)
    marker.write_text(sha)
    # Verified chunks are redundant, generated solely by this downloader.
    for index in range(count):
        (parts / str(index)).unlink()
    parts.rmdir()
    print(f"Verified {local}", flush=True)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "work" / "models"
    for spec in WEIGHTS:
        download(root, spec)
