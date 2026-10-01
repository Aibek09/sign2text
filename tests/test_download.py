import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml import download_weights as transfer
from ml import runtime


class DownloadTests(unittest.TestCase):
    def test_verified_ranges_assemble_and_resume_without_network(self):
        payload = b"abcdefghij"
        sha = hashlib.sha256(payload).hexdigest()
        spec = ("repo", "revision", "remote", "base/model.safetensors", len(payload), sha)

        class Response:
            status_code = 206
            def __init__(self, start, end):
                self.start, self.end = start, end
                self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, _): yield payload[self.start:self.end+1]

        def get(_url, headers, **_):
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return Response(start, end)

        with tempfile.TemporaryDirectory() as directory, patch.object(transfer, "CHUNK", 4):
            root = Path(directory)
            # Simulate a previous successful range; only the other two are fetched.
            parts = root / "range-parts" / sha
            parts.mkdir(parents=True)
            (parts / "0").write_bytes(payload[:4])
            with patch("requests.get", side_effect=get) as fetch:
                transfer.download(root, spec)
                self.assertEqual(fetch.call_count, 2)
            self.assertEqual((root / spec[3]).read_bytes(), payload)
            self.assertEqual((root / spec[3]).with_suffix(".verified").read_text(), sha)
            self.assertFalse(parts.exists())
            with patch("requests.get", side_effect=AssertionError("must remain offline")):
                transfer.download(root, spec)

    def test_checksum_mismatch_never_activates_weights(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(transfer, "CHUNK", 4):
            root = Path(directory)
            sha = hashlib.sha256(b"good").hexdigest()
            parts = root / "range-parts" / sha
            parts.mkdir(parents=True)
            (parts / "0").write_bytes(b"evil")
            spec = ("repo", "revision", "remote", "base/model.safetensors", 4, sha)
            destination = root / spec[3]
            destination.parent.mkdir(parents=True)
            destination.with_suffix(".verified").write_text(sha)
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                transfer.download(root, spec)
            self.assertFalse((root / spec[3]).exists())
            self.assertFalse((root / spec[3]).with_suffix(".verified").exists())

    def test_concurrent_installer_cannot_overwrite_active_transfer(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_dir = root / ".download-locks"
            lock_dir.mkdir()
            with (lock_dir / "hash").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Another installer"):
                    transfer.download(root, ("repo", "rev", "remote", "base/model", 4, "hash"))

    def test_progress_ignores_partial_and_out_of_range_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = root / "range-parts" / "hash"
            parts.mkdir(parents=True)
            (parts / "0").write_bytes(b"1234")
            (parts / "1.tmp").write_bytes(b"1234")
            (parts / "99").write_bytes(b"1234")
            with patch.object(runtime, "MODEL_DIR", root), patch.object(runtime, "CHUNK", 4), \
                 patch.object(runtime, "WEIGHTS", [("repo", "rev", "remote", "base/model", 8, "hash")]):
                self.assertEqual(runtime.download_status(), {"bytes": 4, "totalBytes": 8, "percent": 50})
