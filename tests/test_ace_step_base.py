from __future__ import annotations

import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from ace_step_base import (AceStepBaseClient, AceStepBaseError,
                           AceStepBaseRuntimeManager, BASE_MODEL,
                           create_managed_base_client, MissingBaseModelError)


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
            recorded["files"] = {
                name: upload[0] for name, upload in recorded["files"].items()
            }
        self.calls.append(("POST", url, recorded))
        if url.endswith("/v1/init"):
            return self.wrapped({"loaded_model": BASE_MODEL})
        if url.endswith("/release_task"):
            return self.wrapped({"task_id": "lego-test"})
        if url.endswith("/query_result"):
            result = json.dumps([{"file": "/v1/audio?path=lead.wav", "dit_model": BASE_MODEL}])
            return self.wrapped([{"task_id": "lego-test", "status": 1, "result": result}])
        raise AssertionError(url)


class HealthSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse({"data": {
            "status": "ok", "service": "ACE-Step API", "models_initialized": True,
        }, "code": 200})


class FakeProcess:
    def __init__(self, return_code=None):
        self.return_code = return_code
        self.pid = 4321
        self.terminated = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = 0

    def wait(self, timeout=None):
        del timeout
        return self.return_code

    def kill(self):
        self.terminated = True
        self.return_code = -9


def make_runtime(root: Path):
    required = [
        root / ".venv" / "Scripts" / "python.exe",
        root / "acestep" / "api_server.py",
        root / "checkpoints" / BASE_MODEL / "model.safetensors",
        root / "checkpoints" / "vae" / "diffusion_pytorch_model.safetensors",
        root / "checkpoints" / "Qwen3-Embedding-0.6B" / "model.safetensors",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"installed")


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
        self.assertNotIn("ref_audio", release[2]["files"])
        self.assertFalse(any(call[1].endswith("/v1/init") for call in session.calls))

    def test_reference_voice_is_uploaded_separately_from_instrumental(self):
        session = FakeSession([BASE_MODEL], loaded_model=BASE_MODEL)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "instrumental.wav"
            reference = Path(directory) / "voice_reference.wav"
            output = Path(directory) / "lead_vocal.wav"
            source.write_bytes(wav_bytes())
            reference.write_bytes(wav_bytes())
            client = AceStepBaseClient(session=session, poll_seconds=0.01)
            client.generate_lego_vocals(
                source, output, caption="clear pop lead vocal", lyrics="Sing the light",
                seed=42, inference_steps=50, reference_wav=reference,
            )

        release = next(call for call in session.calls if call[1].endswith("/release_task"))
        self.assertEqual(release[2]["files"], {
            "src_audio": "instrumental.wav",
            "ref_audio": "voice_reference.wav",
        })

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


class AceStepBaseRuntimeManagerTests(unittest.TestCase):
    def test_server_already_running_is_reused_without_starting_process(self):
        session = HealthSession([True])
        starts = []
        manager = AceStepBaseRuntimeManager(
            runtime_root=Path("missing-on-purpose"),
            popen_factory=lambda *args, **kwargs: starts.append((args, kwargs)),
        )

        result = manager.ensure_running(session)

        self.assertFalse(result["started"])
        self.assertEqual(starts, [])

    def test_absent_server_is_started_once_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_runtime(root)
            session = HealthSession([ConnectionError("refused"), True])
            process = FakeProcess()
            starts = []

            def start(*args, **kwargs):
                starts.append((args, kwargs))
                return process

            manager = AceStepBaseRuntimeManager(
                runtime_root=root, popen_factory=start, poll_seconds=0.01,
            )
            first = manager.ensure_running(session)
            second = manager.ensure_running(session)

        self.assertTrue(first["started"])
        self.assertFalse(second["started"])
        self.assertEqual(len(starts), 1)
        command = starts[0][0][0]
        environment = starts[0][1]["env"]
        self.assertEqual(command[-4:], ["--host", "127.0.0.1", "--port", "8001"])
        self.assertEqual(environment["ACESTEP_CONFIG_PATH"], BASE_MODEL)
        self.assertEqual(environment["ACESTEP_INIT_LLM"], "false")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertTrue(environment["HF_MODULES_CACHE"].endswith("huggingface\\modules"))

    def test_startup_failure_reports_runtime_logs_and_no_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_runtime(root)
            session = HealthSession([ConnectionError("refused")])
            manager = AceStepBaseRuntimeManager(
                runtime_root=root, popen_factory=lambda *args, **kwargs: FakeProcess(17),
                poll_seconds=0.01,
            )
            with self.assertRaises(AceStepBaseError) as raised:
                manager.ensure_running(session)

        message = str(raised.exception)
        self.assertIn("exited during startup with code 17", message)
        self.assertIn("base_api.stderr.log", message)
        self.assertIn("No models were downloaded", message)

    def test_legacy_mode_never_creates_or_starts_base_runtime(self):
        with patch("ace_step_base.get_base_runtime_manager") as get_manager:
            client = create_managed_base_client(False)
        self.assertIsNone(client)
        get_manager.assert_not_called()


if __name__ == "__main__":
    unittest.main()
