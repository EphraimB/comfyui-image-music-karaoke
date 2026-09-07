from __future__ import annotations

import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

from ace_step_base import (AceStepBaseClient, BASE_MODEL, MissingBaseModelError)


def wav_bytes(seconds=0.25, sample_rate=8000):
    frames = b"\x00\x00" * int(seconds * sample_rate)
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return output.getvalue()


class FakeResponse:
    def __init__(self, payload=None, content=b"", status=200):
        self.payload = payload
        self.content = content
        self.status_code = status
        self.text = json.dumps(payload) if payload is not None else ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content


class FakeSession:
    def __init__(self, models=None, loaded_model=None):
        self.headers = {}
        self.models = list(models or [])
        self.loaded_model = loaded_model
        self.calls = []
        self.output = wav_bytes()

    @staticmethod
    def wrapped(data):
        return FakeResponse({"data": data, "code": 200, "error": None})

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/health"):
            return self.wrapped({"status": "ok", "loaded_model": self.loaded_model})
        if url.endswith("/v1/model_inventory"):
            return self.wrapped({"models": [{"name": name} for name in self.models]})
        if "/v1/audio?" in url:
            return FakeResponse(content=self.output)
        raise AssertionError(url)

    def post(self, url, **kwargs):
        recorded = dict(kwargs)
        if "files" in recorded:
            recorded["files"] = {"src_audio": recorded["files"]["src_audio"][0]}
        self.calls.append(("POST", url, recorded))
        if url.endswith("/v1/init"):
            return self.wrapped({"loaded_model": BASE_MODEL})
        if url.endswith("/release_task"):
            return self.wrapped({"task_id": "lego-test"})
        if url.endswith("/query_result"):
            result = json.dumps([{"file": "/v1/audio?path=lead.wav", "dit_model": BASE_MODEL}])
            return self.wrapped([{"task_id": "lego-test", "status": 1, "result": result}])
        raise AssertionError(url)


class AceStepBaseClientTests(unittest.TestCase):
    def test_lego_vocals_uploads_exact_track_request_and_saves_wav(self):
        session = FakeSession([BASE_MODEL], loaded_model=BASE_MODEL)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "instrumental.wav"
            output = Path(directory) / "lead_vocal.wav"
            source.write_bytes(wav_bytes())
            client = AceStepBaseClient(session=session, poll_seconds=0.01)
            result = client.generate_lego_vocals(
                source, output, caption="clear pop lead vocal", lyrics="Sing the light",
                seed=42, inference_steps=50,
            )
            self.assertEqual(result.model, BASE_MODEL)
            self.assertTrue(output.is_file())
            self.assertGreater(result.duration, 0)

        release = next(call for call in session.calls if call[1].endswith("/release_task"))
        fields = release[2]["data"]
        self.assertEqual(fields["model"], "acestep-v15-base")
        self.assertEqual(fields["task_type"], "lego")
        self.assertEqual(fields["track_name"], "vocals")
        self.assertEqual(fields["audio_format"], "wav")
        self.assertEqual(release[2]["files"]["src_audio"], "instrumental.wav")
        self.assertFalse(any(call[1].endswith("/v1/init") for call in session.calls))

    def test_missing_base_is_actionable_and_never_initializes_or_submits(self):
        session = FakeSession(["acestep-v15-xl-sft"])
        client = AceStepBaseClient(session=session)
        with self.assertRaisesRegex(MissingBaseModelError, "will not download"):
            client.preflight_and_load_base()
        called_urls = [call[1] for call in session.calls]
        self.assertFalse(any(url.endswith("/v1/init") for url in called_urls))
        self.assertFalse(any(url.endswith("/release_task") for url in called_urls))

    def test_installed_base_is_initialized_when_another_model_is_loaded(self):
        session = FakeSession([BASE_MODEL, "acestep-v15-xl-sft"],
                              loaded_model="acestep-v15-xl-sft")
        client = AceStepBaseClient(session=session)
        result = client.preflight_and_load_base()
        self.assertEqual(result["loaded_model"], BASE_MODEL)
        init = next(call for call in session.calls if call[1].endswith("/v1/init"))
        self.assertEqual(init[2]["json"], {
            "model": BASE_MODEL, "slot": 1, "init_llm": False,
        })


if __name__ == "__main__":
    unittest.main()
