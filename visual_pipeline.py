"""Generate a scene sequence with actual pixel-conditioned Flux edits."""

from __future__ import annotations

import math
from pathlib import Path


def _working_size(width: int, height: int) -> tuple[int, int]:
    if height > width:
        return 576, 1024
    return (1024, 576) if width >= 1600 else (896, 512)


def _fit_reference(source, width: int, height: int):
    from PIL import Image, ImageFilter
    source = source.convert("RGB")
    foreground = source.copy()
    foreground.thumbnail((width, height), Image.Resampling.LANCZOS)
    scale = max(width / source.width, height / source.height)
    cover = source.resize((max(1, round(source.width * scale)),
                           max(1, round(source.height * scale))), Image.Resampling.LANCZOS)
    left, top = (cover.width - width) // 2, (cover.height - height) // 2
    canvas = cover.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(18))
    canvas.paste(foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2))
    return canvas


def _reference_canvas(paths: list[str], width: int, height: int):
    """Compose every assigned source into one pixel-conditioning canvas."""
    from PIL import Image, ImageFilter
    sources = [Image.open(path).convert("RGB") for path in paths]
    if len(sources) == 1:
        return _fit_reference(sources[0], width, height)
    background = _fit_reference(sources[0], width, height).filter(ImageFilter.GaussianBlur(10))
    columns = math.ceil(math.sqrt(len(sources)))
    rows = math.ceil(len(sources) / columns)
    margin = max(6, width // 160)
    cell_w, cell_h = width // columns, height // rows
    for index, source in enumerate(sources):
        source.thumbnail((cell_w - margin * 2, cell_h - margin * 2), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x = col * cell_w + (cell_w - source.width) // 2
        y = row * cell_h + (cell_h - source.height) // 2
        background.paste(source, (x, y))
    return background


def _tensor(image):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


def create_scene_images(call_node, common_ksampler, plan: dict, output_dir,
                        *, width: int, height: int, seed: int,
                        image_model: str, image_clip: str, image_t5: str,
                        image_vae: str, steps: int = 4, edit_denoise: float = 0.30,
                        use_original: bool = False, identity_blend: float = 0.58,
                        on_progress=None) -> list[dict]:
    """Render every planned scene, reusing models and its assigned source pixels."""
    import numpy as np
    import torch
    import comfy.model_management as mm
    from PIL import Image

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    assets = list(plan.get("reference_assets") or [])
    if not assets and (plan.get("source_image_path") or plan.get("image_path")):
        assets = [{"path": plan.get("source_image_path") or plan.get("image_path"),
                   "instruction": "Preserve the important subject and identity."}]
    scenes = list(plan.get("scenes") or [{
        "scene_id": "scene_001", "section": "Song", "start": 0.0,
        "duration": float(plan.get("target_duration", 1.0)),
        "reference_indices": [0] if assets else [],
        "generation_prompt": plan.get("background_prompt"),
        "edit_prompt": plan.get("visual_edit_prompt"), "continuity_notes": "",
    }])
    render_width, render_height = _working_size(width, height)
    model = clip = vae = None
    results = []
    try:
        model = call_node("UNETLoader", unet_name=image_model, weight_dtype="default")[0]
        clip = call_node("DualCLIPLoader", clip_name1=image_clip, clip_name2=image_t5,
                         type="flux", device="default")[0]
        vae = call_node("VAELoader", vae_name=image_vae)[0]
        for index, scene in enumerate(scenes):
            references = [assets[value] for value in scene.get("reference_indices", [])
                          if isinstance(value, int) and 0 <= value < len(assets)
                          and Path(str(assets[value].get("path", ""))).is_file()]
            paths = [str(item["path"]) for item in references]
            has_reference = bool(paths)
            if on_progress:
                on_progress(f"Rendering visual scene {index + 1}/{len(scenes)}: {scene.get('section', 'scene')}")
            source_canvas = _reference_canvas(paths, render_width, render_height) if paths else None
            output_path = destination / f"scene_{index + 1:04d}.png"
            if use_original and source_canvas is not None:
                source_canvas.save(output_path)
            else:
                prompt = str((scene.get("edit_prompt") if has_reference else scene.get("generation_prompt")) or
                             (plan.get("visual_edit_prompt") if has_reference else plan.get("background_prompt")) or
                             plan.get("request") or "cinematic music scene")
                if has_reference:
                    instructions = "; ".join(str(item.get("instruction") or "") for item in references)
                    prompt += (f". Source-asset instructions: {instructions}. Preserve the exact same important "
                               "people, facial identity, body proportions, pose cues, and defining objects from "
                               "the supplied pixels; edit their environment, lighting and atmosphere.")
                prompt += (f". Continuity: {scene.get('continuity_notes', '')}. Clean composition, no writing, "
                           "no letters, no logo, no watermark, leave lower-center space readable for lyrics.")
                positive = call_node("CLIPTextEncode", clip=clip, text=prompt)[0]
                negative = call_node("ConditioningZeroOut", conditioning=positive)[0]
                positive = call_node("FluxGuidance", conditioning=positive, guidance=3.5)[0]
                if source_canvas is not None:
                    latent = call_node("VAEEncode", pixels=_tensor(source_canvas), vae=vae)[0]
                    denoise = max(0.08, min(0.45, float(edit_denoise)))
                else:
                    latent = call_node("EmptySD3LatentImage", width=render_width,
                                       height=render_height, batch_size=1)[0]
                    denoise = 1.0
                with torch.inference_mode():
                    sampled = common_ksampler(model, (int(seed) + index * 1009) % (2 ** 64),
                                              int(steps), 1.0, "euler", "simple", positive,
                                              negative, latent, denoise=denoise)[0]
                    decoded = call_node("VAEDecode", samples=sampled, vae=vae)[0]
                pixels = (decoded[0].detach().float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                generated = Image.fromarray(pixels).convert("RGB")
                if source_canvas is not None:
                    # This pixel-space safety pass stops an aggressive diffusion
                    # edit from silently replacing a reference person's face.
                    preservation = max(0.35, min(0.80, float(identity_blend)))
                    generated = Image.blend(generated, source_canvas, preservation)
                generated.save(output_path)
            results.append({**scene, "image_path": str(output_path),
                            "source_paths": paths,
                            "render_mode": "reference-conditioned identity-preserving edit"
                                           if paths and not use_original else
                                           "supplied reference" if paths else "generated from director plan"})
        return results
    finally:
        model = clip = vae = None
        mm.unload_all_models()
        mm.soft_empty_cache()


def create_karaoke_background(call_node, common_ksampler, plan: dict, output_path,
                              *, width: int, height: int, seed: int,
                              image_model: str, image_clip: str, image_t5: str,
                              image_vae: str, steps: int = 4, edit_denoise: float = 0.30,
                              use_original: bool = False, on_progress=None) -> str:
    """Compatibility entry point retained for older callers."""
    from PIL import Image
    result = create_scene_images(
        call_node, common_ksampler, plan, Path(output_path).parent / "visual_scenes",
        width=width, height=height, seed=seed, image_model=image_model,
        image_clip=image_clip, image_t5=image_t5, image_vae=image_vae,
        steps=steps, edit_denoise=edit_denoise, use_original=use_original,
        on_progress=on_progress)
    source = Path(result[0]["image_path"])
    target = Path(output_path)
    if source != target:
        Image.open(source).convert("RGB").save(target)
    return str(target)
