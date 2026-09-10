from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import atexit
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse


BASE_MODEL = "acestep-v15-base"
PREFERRED_LM_MODEL = "acestep-5Hz-lm-1.7B"
LEGO_VOCALS_INSTRUCTION = "Generate the VOCALS track based on the audio context:"
BASE_RUNTIME_ENV = "IMAGE_MUSIC_KARAOKE_ACESTEP_BASE_ROOT"


class AceStepBaseError(RuntimeError):
    """An actionable failure returned by the local ACE-Step Base service."""


class MissingBaseModelError(AceStepBaseError):
    """The local official ACE-Step server does not have the 2B Base model."""


class AceStepBaseRuntimeManager:
    """Start and reuse the project's isolated official ACE-Step Base API."""

    def __init__(self, server_url: str = "http://127.0.0.1:8001", *,
                 runtime_root: Path | None = None, startup_seconds: float = 360.0,
                 poll_seconds: float = 1.0, popen_factory=None, sleep=None,
                 monotonic=None):
        self.server_url = AceStepBaseClient._validate_local_url(server_url)
        self.runtime_root = Path(runtime_root).resolve() if runtime_root else None
        self.startup_seconds = max(5.0, float(startup_seconds))
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.popen_factory = popen_factory or subprocess.Popen
        self.sleep = sleep or time.sleep
        self.monotonic = monotonic or time.monotonic
        self._lock = threading.Lock()
        self._process = None
        self._shutdown_registered = False

    @property
    def health_url(self) -> str:
        return urljoin(self.server_url + "/", "health")

    def _health_ready(self, session) -> bool:
        try:
            response = session.get(self.health_url, timeout=3)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("code") not in (None, 200):
                return False
            data = payload.get("data", payload) if isinstance(payload, dict) else {}
            return (isinstance(data, dict)
                    and data.get("status") == "ok"
                    and data.get("service") in (None, "ACE-Step API"))
        except Exception:
            return False

    @staticmethod
    def _candidate_roots() -> list[Path]:
        candidates = []
        configured = os.environ.get(BASE_RUNTIME_ENV, "").strip()
        if configured:
            candidates.append(Path(configured).expanduser())
        module_root = Path(__file__).resolve().parent
        candidates.append(module_root.parent / "work" / "ACE-Step-1.5-main")
        for documents in (Path.home() / "Documents", Path.home() / "OneDrive" / "Documents"):
            codex = documents / "Codex"
            if codex.is_dir():
                candidates.extend(sorted(codex.glob("*/*/work/ACE-Step-1.5-main"), reverse=True))
            candidates.append(documents / "ACE-Step-1.5-main")
        unique = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        return unique

    @staticmethod
    def _required_paths(root: Path) -> dict[str, Path]:
        return {
            "isolated Python": root / ".venv" / "Scripts" / "python.exe",
            "official API module": root / "acestep" / "api_server.py",
            "2B Base checkpoint": root / "checkpoints" / BASE_MODEL / "model.safetensors",
            "ACE VAE": root / "checkpoints" / "vae" / "diffusion_pytorch_model.safetensors",
            "ACE text encoder": root / "checkpoints" / "Qwen3-Embedding-0.6B" / "model.safetensors",
        }

    def _resolve_runtime(self) -> tuple[Path, dict[str, Path]]:
        candidates = [self.runtime_root] if self.runtime_root else self._candidate_roots()
        checked = []
        for root in candidates:
            if root is None:
                continue
            required = self._required_paths(root)
            missing = [label for label, path in required.items()
                       if not path.is_file() or path.stat().st_size <= 0]
            checked.append(f"{root} ({', '.join(missing) if missing else 'complete'})")
            if not missing:
                return root, required
        searched = "; ".join(checked) or "no candidate paths"
        raise AceStepBaseError(
            "ACE LEGO → RVC could not find the installed official ACE-Step 1.5 Base runtime "
            f"and required local checkpoint files. Set {BASE_RUNTIME_ENV} to the ACE-Step "
            "repository containing .venv, checkpoints/acestep-v15-base, checkpoints/vae, "
            f"and checkpoints/Qwen3-Embedding-0.6B. Checked: {searched}. "
            "No models were downloaded."
        )

    @staticmethod
    def _tail(path: Path, limit: int = 6000) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
        except OSError:
            return ""

    def _stop_managed_process(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def ensure_running(self, session, on_progress=None) -> dict:
        with self._lock:
            if self._health_ready(session):
                if on_progress:
                    on_progress(f"Reusing healthy official ACE-Step Base API at {self.server_url}")
                return {"started": False, "server_url": self.server_url}

            root, required = self._resolve_runtime()
            if self._process is None or self._process.poll() is not None:
                log_dir = root / ".cache" / "image_music_karaoke"
                log_dir.mkdir(parents=True, exist_ok=True)
                huggingface_cache = root / ".cache" / "huggingface"
                (huggingface_cache / "modules").mkdir(parents=True, exist_ok=True)
                (huggingface_cache / "hub").mkdir(parents=True, exist_ok=True)
                stdout_path = log_dir / "base_api.stdout.log"
                stderr_path = log_dir / "base_api.stderr.log"
                environment = os.environ.copy()
                environment.update({
                    "ACESTEP_CONFIG_PATH": BASE_MODEL,
                    "ACESTEP_INIT_LLM": "false",
                    "ACESTEP_NO_INIT": "false",
                    "ACESTEP_API_HOST": "127.0.0.1",
                    "ACESTEP_API_PORT": str(urlparse(self.server_url).port or 8001),
                    "HF_HUB_OFFLINE": "1",
                    "HF_HOME": str(huggingface_cache),
                    "HF_MODULES_CACHE": str(huggingface_cache / "modules"),
                    "HUGGINGFACE_HUB_CACHE": str(huggingface_cache / "hub"),
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONUNBUFFERED": "1",
                })
                bootstrap = Path(__file__).resolve().with_name("ace_step_base_server.py")
                if not bootstrap.is_file():
                    raise AceStepBaseError(
                        f"ACE LEGO → RVC local-only API bootstrap is missing: {bootstrap}"
                    )
                command = [str(required["isolated Python"]), "-u", str(bootstrap),
                           "--host", "127.0.0.1", "--port",
                           str(urlparse(self.server_url).port or 8001)]
                if on_progress:
                    on_progress(f"Starting official ACE-Step 1.5 2B Base API from {root}")
                try:
                    with stdout_path.open("ab", buffering=0) as stdout_handle, \
                            stderr_path.open("ab", buffering=0) as stderr_handle:
                        self._process = self.popen_factory(
                            command, cwd=str(root), env=environment, stdin=subprocess.DEVNULL,
                            stdout=stdout_handle, stderr=stderr_handle,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                except Exception as exc:
                    raise AceStepBaseError(
                        f"ACE LEGO → RVC could not start the official Base API with "
                        f"{required['isolated Python']}: {exc}. Runtime: {root}. "
                        f"No models were downloaded."
                    ) from exc
                if not self._shutdown_registered:
                    atexit.register(self._stop_managed_process)
                    self._shutdown_registered = True
            else:
                stdout_path = root / ".cache" / "image_music_karaoke" / "base_api.stdout.log"
                stderr_path = root / ".cache" / "image_music_karaoke" / "base_api.stderr.log"

            deadline = self.monotonic() + self.startup_seconds
            while self.monotonic() < deadline:
                if self._health_ready(session):
                    if on_progress:
                        on_progress(f"Official ACE-Step Base API is ready at {self.server_url}")
                    return {"started": True, "server_url": self.server_url,
                            "runtime_root": str(root), "pid": getattr(self._process, "pid", None)}
                return_code = self._process.poll()
                if return_code is not None:
                    detail = self._tail(stderr_path) or self._tail(stdout_path)
                    raise AceStepBaseError(
                        f"ACE LEGO → RVC Base API exited during startup with code {return_code}. "
                        f"Runtime: {root}. Check {stderr_path} and {stdout_path}. "
                        f"Last log output: {detail or 'none'}. No models were downloaded."
                    )
                self.sleep(self.poll_seconds)

            self._stop_managed_process()
            detail = self._tail(stderr_path) or self._tail(stdout_path)
            raise AceStepBaseError(
                f"ACE LEGO → RVC Base API did not become healthy at {self.server_url} within "
                f"{self.startup_seconds:g} seconds. Runtime: {root}. Check {stderr_path} and "
                f"{stdout_path}. Last log output: {detail or 'none'}. No models were downloaded."
            )


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
                 session=None, service_manager: AceStepBaseRuntimeManager | None = None):
        self.server_url = self._validate_local_url(server_url)
        self.timeout_seconds = max(10.0, float(timeout_seconds))
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.service_manager = service_manager
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

    def preflight_and_load_base(self, on_progress=None) -> dict:
        if self.service_manager is not None:
            self.service_manager.ensure_running(self.session, on_progress=on_progress)
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

        loaded = self.preflight_and_load_base(on_progress=on_progress)
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


_BASE_RUNTIME_MANAGER = AceStepBaseRuntimeManager()


def get_base_runtime_manager() -> AceStepBaseRuntimeManager:
    return _BASE_RUNTIME_MANAGER


def create_managed_base_client(enabled: bool) -> AceStepBaseClient | None:
    """Create the production client only when ACE LEGO mode is selected."""
    if not enabled:
        return None
    return AceStepBaseClient(service_manager=get_base_runtime_manager())
