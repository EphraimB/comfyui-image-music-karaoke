from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse


BASE_MODEL = "acestep-v15-base"
PREFERRED_LM_MODEL = "acestep-5Hz-lm-1.7B"
LEGO_VOCALS_INSTRUCTION = "Generate the VOCALS track based on the audio context:"


class AceStepBaseError(RuntimeError):
    """An actionable failure returned by the local ACE-Step Base service."""


class MissingBaseModelError(AceStepBaseError):
    """The local official ACE-Step server does not have the 2B Base model."""


@dataclass(frozen=True)
class LegoVocalResult:
    output_path: Path
    task_id: str
    model: str
    duration: float
    server_item: dict


class AceStepBaseClient:
    """Small client for the official ACE-Step 1.5 asynchronous HTTP API.

    The client deliberately refuses to initialize a model that the server does not
    already report. That keeps this project from causing ACE-Step's model downloader
    to run implicitly.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8001", *,
                 timeout_seconds: float = 3600.0, poll_seconds: float = 2.0,
                 session=None):
        self.server_url = self._validate_local_url(server_url)
        self.timeout_seconds = max(10.0, float(timeout_seconds))
        self.poll_seconds = max(0.05, float(poll_seconds))
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise AceStepBaseError(
                    "The ACE-Step Base integration requires the 'requests' package. "
                    "Install this custom node's requirements.txt in the ComfyUI environment."
                ) from exc
            session = requests.Session()
        self.session = session
        token = os.environ.get("ACESTEP_API_KEY", "").strip()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    @staticmethod
    def _validate_local_url(value: str) -> str:
        text = str(value or "").strip().rstrip("/")
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
                "127.0.0.1", "localhost", "::1"}:
            raise ValueError("ACE-Step server_url must be a local HTTP URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("ACE-Step server_url must not contain credentials, a query, or a fragment.")
        return text

    @staticmethod
    def _unwrap(response, operation: str):
        try:
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            body = getattr(response, "text", "")
            detail = str(body).strip()[:600]
            raise AceStepBaseError(
                f"Official ACE-Step API {operation} failed: {detail or exc}"
            ) from exc
        if isinstance(payload, dict) and payload.get("code") not in (None, 200):
            raise AceStepBaseError(
                f"Official ACE-Step API {operation} failed: "
                f"{payload.get('error') or payload.get('detail') or payload}"
            )
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    def _url(self, path: str) -> str:
        return urljoin(self.server_url + "/", path.lstrip("/"))

    def preflight_and_load_base(self) -> dict:
        try:
            health = self.session.get(self._url("/health"), timeout=10)
            health_data = self._unwrap(health, "health check")
            response = self.session.get(self._url("/v1/model_inventory"), timeout=30)
        except AceStepBaseError:
            raise
        except Exception as exc:
            raise AceStepBaseError(
                f"Cannot reach the official ACE-Step API at {self.server_url}. "
                "Start its local API server with the 2B Base checkpoint already installed."
            ) from exc
        data = self._unwrap(response, "model listing")
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("models", data.get("data", []))
        else:
            entries = []
        available = set()
        for item in entries:
            if isinstance(item, dict):
                for key in ("name", "id"):
                    value = str(item.get(key, "")).strip()
                    if value:
                        available.add(value)
                        available.add(value.rsplit("/", 1)[-1])
            else:
                available.add(str(item).strip())
        if BASE_MODEL not in available:
            visible = ", ".join(sorted(name for name in available if name)) or "none"
            raise MissingBaseModelError(
                f"ACE-Step 2B Base model '{BASE_MODEL}' is not available on the local server "
                f"(reported models: {visible}). Install/configure it in the official ACE-Step "
                "service first; this custom node will not download it automatically."
            )

        if isinstance(health_data, dict) and health_data.get("loaded_model") == BASE_MODEL:
            return health_data

        response = self.session.post(
            self._url("/v1/init"),
            json={"model": BASE_MODEL, "slot": 1, "init_llm": False},
            timeout=self.timeout_seconds,
        )
        loaded = self._unwrap(response, "2B Base model initialization")
        loaded_name = str(loaded.get("loaded_model", "")) if isinstance(loaded, dict) else ""
        if loaded_name and loaded_name != BASE_MODEL:
            raise AceStepBaseError(
                f"ACE-Step initialized '{loaded_name}' instead of required '{BASE_MODEL}'."
            )
        return loaded if isinstance(loaded, dict) else {"loaded_model": BASE_MODEL}

    def generate_lego_vocals(self, source_wav: Path, output_wav: Path, *,
                             caption: str, lyrics: str, vocal_language: str = "en",
                             seed: int = 0, inference_steps: int = 50,
                             guidance_scale: float = 7.0, reference_wav: Path | None = None,
                             on_progress=None) -> LegoVocalResult:
        source_wav = Path(source_wav).resolve()
        output_wav = Path(output_wav).resolve()
        if not source_wav.is_file():
            raise ValueError(f"Instrumental source WAV does not exist: {source_wav}")
        if source_wav.suffix.lower() != ".wav":
            raise ValueError("ACE-Step LEGO vocals milestone requires a WAV source file.")
        if not str(caption).strip():
            raise ValueError("Describe the desired lead-vocal performance.")
        if not str(lyrics).strip():
            raise ValueError("Provide the lyrics for the lead-vocal track.")
        if reference_wav is not None:
            reference_wav = Path(reference_wav).resolve()
            if not reference_wav.is_file():
                raise ValueError(f"Voice-reference WAV does not exist: {reference_wav}")
            if reference_wav.suffix.lower() != ".wav":
                raise ValueError("ACE-Step voice-reference conditioning requires a WAV file.")
            if reference_wav == source_wav:
                raise ValueError("Voice-reference audio must be separate from the instrumental source audio.")

        loaded = self.preflight_and_load_base()
        if on_progress:
            on_progress(f"Loaded {loaded.get('loaded_model', BASE_MODEL)} for LEGO vocals")

        fields = {
            "model": BASE_MODEL,
            "task_type": "lego",
            "track_name": "vocals",
            "instruction": LEGO_VOCALS_INSTRUCTION,
            "prompt": str(caption).strip(),
            "lyrics": str(lyrics).strip(),
            "vocal_language": str(vocal_language or "en").strip(),
            "thinking": "false",
            "use_format": "false",
            "instrumental": "false",
            "audio_format": "wav",
            "inference_steps": str(int(inference_steps)),
            "guidance_scale": str(float(guidance_scale)),
            "use_random_seed": "false",
            "seed": str(int(seed)),
            "batch_size": "1",
            "repainting_start": "0",
            "repainting_end": "-1",
        }
        try:
            with ExitStack() as stack:
                source_handle = stack.enter_context(source_wav.open("rb"))
                files = {"src_audio": (source_wav.name, source_handle, "audio/wav")}
                if reference_wav is not None:
                    reference_handle = stack.enter_context(reference_wav.open("rb"))
                    files["ref_audio"] = (reference_wav.name, reference_handle, "audio/wav")
                response = self.session.post(
                    self._url("/release_task"), data=fields,
                    files=files, timeout=120,
                )
        except Exception as exc:
            raise AceStepBaseError(f"Could not submit ACE-Step LEGO vocals task: {exc}") from exc
        released = self._unwrap(response, "LEGO vocals submission")
        task_id = str(released.get("task_id", "")).strip() if isinstance(released, dict) else ""
        if not task_id:
            raise AceStepBaseError("ACE-Step accepted the request but returned no task_id.")
        if on_progress:
            on_progress(f"ACE-Step LEGO vocals task {task_id} queued")

        deadline = time.monotonic() + self.timeout_seconds
        item = None
        while time.monotonic() < deadline:
            try:
                response = self.session.post(
                    self._url("/query_result"), json={"task_id_list": [task_id]}, timeout=30,
                )
            except Exception as exc:
                raise AceStepBaseError(f"Could not query ACE-Step task {task_id}: {exc}") from exc
            queried = self._unwrap(response, "LEGO vocals status query")
            tasks = queried if isinstance(queried, list) else []
            task = tasks[0] if tasks else {}
            status = task.get("status")
            if status in (1, "1", "succeeded", "success"):
                raw_result = task.get("result", [])
                try:
                    results = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                except json.JSONDecodeError as exc:
                    raise AceStepBaseError("ACE-Step returned malformed generation result JSON.") from exc
                if not isinstance(results, list) or not results:
                    raise AceStepBaseError("ACE-Step completed without an audio result.")
                item = results[0]
                break
            if status in (2, "2", "failed", "error"):
                detail = task.get("error") or task.get("message") or task.get("result")
                raise AceStepBaseError(f"ACE-Step LEGO vocals task failed: {detail}")
            if on_progress:
                on_progress(f"ACE-Step LEGO vocals task {task_id} is running")
            time.sleep(self.poll_seconds)
        if item is None:
            raise AceStepBaseError(
                f"ACE-Step LEGO vocals task {task_id} exceeded {self.timeout_seconds:g} seconds."
            )

        audio_ref = str(item.get("file") or item.get("url") or "").strip()
        if not audio_ref:
            raise AceStepBaseError("ACE-Step completed without a downloadable audio URL.")
        audio_url = audio_ref if audio_ref.startswith(("http://", "https://")) else self._url(audio_ref)
        if urlparse(audio_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise AceStepBaseError("ACE-Step returned a non-local audio URL; refusing to download it.")
        try:
            response = self.session.get(audio_url, stream=True, timeout=120)
            response.raise_for_status()
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_wav.with_suffix(".download")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            shutil.move(str(temporary), str(output_wav))
        except Exception as exc:
            raise AceStepBaseError(f"Could not save ACE-Step vocal WAV: {exc}") from exc

        try:
            import soundfile as sf
            info = sf.info(output_wav)
        except Exception as exc:
            output_wav.unlink(missing_ok=True)
            raise AceStepBaseError(f"ACE-Step result is not a readable WAV file: {exc}") from exc
        if info.frames <= 0 or info.samplerate <= 0:
            output_wav.unlink(missing_ok=True)
            raise AceStepBaseError("ACE-Step returned an empty vocal WAV.")
        return LegoVocalResult(
            output_path=output_wav, task_id=task_id, model=BASE_MODEL,
            duration=float(info.frames) / float(info.samplerate), server_item=dict(item),
        )
