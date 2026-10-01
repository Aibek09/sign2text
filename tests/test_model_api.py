import json
import threading
import unittest
import urllib.request
import urllib.error
import time
import tempfile
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from app import server


class ModelAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_missing_model_is_explicit(self):
        with patch.object(server, "installed", return_value=False):
            result = json.load(urllib.request.urlopen(self.url + "/api/model"))
        self.assertFalse(result["installed"])
        self.assertTrue(result["experimental"])

    def test_cross_origin_inference_is_rejected(self):
        request = urllib.request.Request(self.url + "/api/translate", data=b"video",
            headers={"Origin":"https://example.com", "Content-Type":"application/octet-stream"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)

    def test_live_frame_returns_only_recognized_result(self):
        request = urllib.request.Request(self.url + "/api/live", data=b"jpeg frame",
            headers={"Content-Type": "image/jpeg"})
        expected = {"gesture": "Thumb_Up", "text": "Yes", "confidence": .91}
        with patch.object(server, "installed", return_value=True), \
             patch.object(server, "runtime_ready", return_value=True), \
             patch.object(server.live_recognizer, "recognize_jpeg", return_value=expected) as recognize:
            result = json.load(urllib.request.urlopen(request))
        self.assertEqual(result, expected)
        recognize.assert_called_once_with(b"jpeg frame")

    def test_asl_live_frame_uses_separate_mode(self):
        request = urllib.request.Request(self.url + "/api/live", data=b"jpeg frame",
            headers={"Content-Type": "image/jpeg", "X-Sign-Mode": "asl"})
        expected = {"gesture": "ASL_2", "text": "Two", "confidence": None}
        with patch.object(server, "installed", return_value=True), \
             patch.object(server, "runtime_ready", return_value=True), \
             patch.object(server.live_recognizer, "recognize_jpeg", return_value=expected) as recognize:
            result = json.load(urllib.request.urlopen(request))
        self.assertEqual(result, expected)
        recognize.assert_called_once_with(b"jpeg frame", mode="asl")

    def test_asl_letter_frame_uses_letter_model_only(self):
        request = urllib.request.Request(self.url + "/api/live", data=b"jpeg frame",
            headers={"Content-Type": "image/jpeg", "X-Sign-Mode": "asl_letters"})
        expected = {"gesture": "LETTER_L", "text": "L", "confidence": .99}
        with patch.object(server, "installed", return_value=True), \
             patch.object(server, "runtime_ready", return_value=True), \
             patch.object(server, "asl_letters_installed", return_value=True), \
             patch.object(server, "letters_runtime_ready", return_value=True), \
             patch.object(server.live_recognizer, "recognize_jpeg", return_value=expected) as recognize:
            result = json.load(urllib.request.urlopen(request))
        self.assertEqual(result, expected)
        recognize.assert_called_once_with(b"jpeg frame", mode="asl_letters")

    def test_asl_letter_mode_refuses_missing_checkpoint(self):
        request = urllib.request.Request(self.url + "/api/live", data=b"jpeg frame",
            headers={"Content-Type": "image/jpeg", "X-Sign-Mode": "asl_letters"})
        with patch.object(server, "installed", return_value=True), \
             patch.object(server, "runtime_ready", return_value=True), \
             patch.object(server, "asl_letters_installed", return_value=False):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 503)

    def test_live_frame_rejects_wrong_type_and_cross_origin(self):
        wrong_type = urllib.request.Request(self.url + "/api/live", data=b"frame",
            headers={"Content-Type": "application/octet-stream"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(wrong_type)
        self.assertEqual(caught.exception.code, 415)
        cross_origin = urllib.request.Request(self.url + "/api/live", data=b"frame",
            headers={"Content-Type": "image/jpeg", "Origin": "https://example.com"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(cross_origin)
        self.assertEqual(caught.exception.code, 403)

    def test_no_successful_translation_without_weights(self):
        request = urllib.request.Request(self.url + "/api/translate", data=b"video",
            headers={"Content-Type":"application/octet-stream"})
        with patch.object(server, "installed", return_value=False):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 503)

    def test_invalid_length_rejected(self):
        request = urllib.request.Request(self.url + "/api/translate", data=b"",
            headers={"Content-Type":"application/octet-stream"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 413)

    def test_missing_runtime_rejected_before_job(self):
        request = urllib.request.Request(self.url + "/api/translate", data=b"video",
            headers={"Content-Type":"application/octet-stream"})
        with patch.object(server, "installed", return_value=True), patch.object(server, "runtime_ready", return_value=False):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 503)

    def test_result_expiration_preserves_running_jobs(self):
        now = time.time()
        entries = {"old": {"status": "complete", "created": now-2000, "finished": now-1900},
                   "new": {"status": "complete", "created": now-2000, "finished": now},
                   "running": {"status": "running", "created": now-2000}}
        with patch.object(server, "jobs", entries):
            server.purge_jobs()
            self.assertEqual(set(entries), {"new", "running"})

    def test_failed_inference_deletes_upload_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.video"
            path.write_bytes(b"video")
            entries = {"job": {"status": "running", "created": time.time()}}
            server.job_lock.acquire()
            with patch.object(server, "jobs", entries), patch.object(server.translator, "translate", side_effect=ValueError("bad clip")):
                server.perform_job("job", path)
            self.assertFalse(path.exists())
            self.assertFalse(server.job_lock.locked())
            self.assertEqual(entries["job"]["status"], "error")

    def test_job_api_delivers_translator_result(self):
        request = urllib.request.Request(self.url + "/api/translate", data=b"synthetic test video",
            headers={"Content-Type":"application/octet-stream"})
        seen_paths = []
        expected = {"text": "mocked output, not an ASL prediction", "confidence": None}
        def translate(path, progress):
            self.assertEqual(path.read_bytes(), b"synthetic test video")
            seen_paths.append(path)
            progress("Test inference")
            return expected
        with patch.object(server, "installed", return_value=True), \
             patch.object(server, "runtime_ready", return_value=True), \
             patch.object(server.translator, "translate", side_effect=translate):
            response = urllib.request.urlopen(request)
            self.assertEqual(response.status, 202)
            job_id = json.load(response)["jobId"]
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                job = json.load(urllib.request.urlopen(self.url + "/api/jobs/" + job_id))
                if job["status"] != "running" and not server.job_lock.locked():
                    break
                time.sleep(.01)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["result"], expected)
        self.assertTrue(seen_paths)
        self.assertFalse(seen_paths[0].exists())
