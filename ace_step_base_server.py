"""Launch the official ACE-Step API with local-model-only startup guards."""

from __future__ import annotations

import sys
from pathlib import Path


def _local_model(model_name: str, checkpoint_dir) -> str:
    path = Path(checkpoint_dir) / model_name
    if path.is_dir() and any(path.iterdir()):
        return str(path)
    raise RuntimeError(
        f"Required local ACE-Step model '{model_name}' is missing at {path}. "
        "Automatic model downloads are disabled for Image Music Karaoke."
    )


def main() -> None:
    runtime_root = Path.cwd().resolve()
    sys.path.insert(0, str(runtime_root))

    import acestep.model_downloader as downloader
    import acestep.api.model_download as api_download

    # Base LEGO does not use the Turbo DiT or LM, but the official startup checker
    # treats those optional components as part of the unified "main" download.
    # The parent process has already verified Base, VAE, and text-encoder weights.
    downloader.check_main_model_exists = lambda _checkpoint_dir=None: True

    def local_only(model_name, checkpoint_dir, *args, **kwargs):
        del args, kwargs
        return _local_model(model_name, checkpoint_dir)

    api_download.ensure_model_downloaded = local_only

    from acestep.api_server import main as official_main
    official_main()


if __name__ == "__main__":
    main()
