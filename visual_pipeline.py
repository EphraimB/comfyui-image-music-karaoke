"""Generate a scene sequence with actual pixel-conditioned Flux edits."""

from __future__ import annotations

import math
import re
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


def _clothing_change_requested(prompt: str) -> bool:
    """Only expose clothing to the edit when the direction explicitly asks for it."""
    text = str(prompt or "").casefold()
    return bool(re.search(
        r"\b(?:wear(?:ing|s)?|dress(?:ed|ing)?\s+(?:him|her|them|the person)|"
        r"(?:change|replace|swap|transform|redesign)\b.{0,40}\b"
        r"(?:clothes|clothing|outfit|wardrobe|costume|jacket|shirt|dress|suit))\b",
        text,
    ))


def _requests_unchanged_source(prompt: str) -> bool:
    text = str(prompt or "").casefold()
    return bool(re.search(
        r"\b(?:no (?:visual )?(?:change|transformation|edit)|use (?:the )?(?:source|original) "
        r"(?:unchanged|as[- ]is)|keep (?:the )?(?:source|original) unchanged)\b", text))


def _identity_protection_mask(image, *, keep_clothing: bool):
    """Make a feathered mask for exact face/hair/glasses and optional wardrobe pixels."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    width, height = image.size
    faces = []
    try:
        import cv2
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        faces = list(cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4,
            minSize=(max(32, width // 12), max(32, height // 12))))
    except Exception:
        faces = []

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if faces:
        x, y, face_w, face_h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        head = (max(0, x - int(face_w * 0.55)), max(0, y - int(face_h * 0.80)),
                min(width, x + int(face_w * 1.55)), min(height, y + int(face_h * 1.45)))
    else:
        # Portrait-safe fallback: retain the central head and defining accessories,
        # while leaving most of the canvas available for a new environment.
        head = (int(width * 0.28), int(height * 0.02),
                int(width * 0.72), int(height * 0.76))
    draw.ellipse(head, fill=255)
    if keep_clothing:
        center = (head[0] + head[2]) // 2
        shoulder_y = max(head[1], int(head[3] - (head[3] - head[1]) * 0.18))
        half_shoulder = int((head[2] - head[0]) * 0.88)
        draw.polygon([
            (max(0, center - half_shoulder), shoulder_y),
            (min(width, center + half_shoulder), shoulder_y),
            (min(width, center + int(half_shoulder * 1.18)), height),
            (max(0, center - int(half_shoulder * 1.18)), height),
        ], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(5, min(width, height) // 80)))


def _reference_canvas(paths: list[str], width: int, height: int, *, keep_clothing: bool):
    """Compose assigned sources and return pixels plus identity-protection mask."""
    from PIL import Image, ImageChops, ImageFilter
    sources = [Image.open(path).convert("RGB") for path in paths]
    if len(sources) == 1:
        canvas = _fit_reference(sources[0], width, height)
        return canvas, _identity_protection_mask(canvas, keep_clothing=keep_clothing)
    background = _fit_reference(sources[0], width, height).filter(ImageFilter.GaussianBlur(10))
    protection = Image.new("L", (width, height), 0)
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
        local = _identity_protection_mask(source, keep_clothing=keep_clothing)
        layer = Image.new("L", (width, height), 0)
        layer.paste(local, (x, y))
        protection = ImageChops.lighter(protection, layer)
    return background, protection


def _tensor(image):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


def _mask_tensor(mask):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)


def _background_difference(source, generated, protection_mask) -> float:
    """Mean normalized pixel change outside the protected identity region."""
    import numpy as np
    source_pixels = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    generated_pixels = np.asarray(generated.convert("RGB"), dtype=np.float32) / 255.0
    editable = 1.0 - np.asarray(protection_mask, dtype=np.float32) / 255.0
    weight = float(editable.sum())
    if weight < 1.0:
        return 0.0
    difference = np.abs(source_pixels - generated_pixels).mean(axis=2)
    return float((difference * editable).sum() / weight)


def create_scene_images(call_node, common_ksampler, plan: dict, output_dir,
                        *, width: int, height: int, seed: int,
                        image_model: str, image_clip: str, image_t5: str,
                        image_vae: str, steps: int = 4, edit_denoise: float = 0.30,
                        use_original: bool = False, identity_blend: float = 0.58,
                        on_progress=None) -> list[dict]:
    """Render each scene with text, masked identity edit, or multi-reference composition."""
    import numpy as np
    import torch
    import comfy.model_management as mm
    import folder_paths
    from PIL import Image, ImageOps

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
    clip = vae = None
    models = {}
    results = []

    def load_model(name: str):
        if name not in models:
            models[name] = call_node("UNETLoader", unet_name=name, weight_dtype="default")[0]
        return models[name]

    def decode_sample(model_name: str, positive, negative, latent, *, scene_seed: int,
                      sample_steps: int, denoise: float):
        with torch.inference_mode():
            sampled = common_ksampler(
                load_model(model_name), scene_seed % (2 ** 64), sample_steps, 1.0,
                "euler", "simple", positive, negative, latent, denoise=denoise)[0]
            decoded = call_node("VAEDecode", samples=sampled, vae=vae)[0]
        pixels = (decoded[0].detach().float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        return Image.fromarray(pixels).convert("RGB")

    try:
        clip = call_node("DualCLIPLoader", clip_name1=image_clip, clip_name2=image_t5,
                         type="flux", device="default")[0]
        vae = call_node("VAELoader", vae_name=image_vae)[0]
        available_models = set(folder_paths.get_filename_list("diffusion_models"))
        fill_model = "flux1-fill-dev.safetensors" if "flux1-fill-dev.safetensors" in available_models else None
        for index, scene in enumerate(scenes):
            assigned_indices = [value for value in scene.get("reference_indices", [])
                                if isinstance(value, int) and 0 <= value < len(assets)
                                and Path(str(assets[value].get("path", ""))).is_file()]
            references = [assets[value] for value in assigned_indices]
            paths = [str(item["path"]) for item in references]
            has_reference = bool(paths)
            edit_mode = ("text-to-image" if not paths else
                         "identity-preserving image-to-image" if len(paths) == 1 else
                         "multi-reference composition")
            if on_progress:
                on_progress(f"Rendering visual scene {index + 1}/{len(scenes)} with {edit_mode}: "
                            f"{scene.get('section', 'scene')}")
            output_path = destination / f"scene_{index + 1:04d}.png"
            prompt = str((scene.get("edit_prompt") if has_reference else scene.get("generation_prompt")) or
                         (plan.get("visual_edit_prompt") if has_reference else plan.get("background_prompt")) or
                         plan.get("request") or "cinematic music scene")
            source_canvas = protection_mask = None
            clothing_change = _clothing_change_requested(prompt)
            if has_reference:
                source_canvas, protection_mask = _reference_canvas(
                    paths, render_width, render_height, keep_clothing=not clothing_change)
                instructions = "; ".join(
                    str(item.get("instruction") or "Preserve the important subject and exact identity.")
                    for item in references)
                prompt += (
                    f". Source-asset instructions: {instructions}. Keep the exact same referenced person or "
                    "subject. Preserve recognizable facial geometry, eyes, glasses, hair, skin details and "
                    "identity. Materially redesign the environment, lighting, atmosphere, framing, props, "
                    "background and composition for this scene. Do not create a lookalike or replacement "
                    "person. " +
                    ("The direction explicitly requests a wardrobe change; change clothing while retaining "
                     "the exact head, face, hair and glasses."
                     if clothing_change else
                     "Keep the source clothing unchanged; the direction did not explicitly request a wardrobe change."))
            prompt += (f". Continuity: {scene.get('continuity_notes', '')}. Cinematic still, clean composition, "
                       "no writing, no letters, no logo, no watermark, leave lower-center space readable for lyrics.")

            requested_edit_strength = max(0.08, min(0.95, float(edit_denoise)))
            identity_setting = max(0.0, min(1.0, float(identity_blend)))
            render_status = "newly-rendered"
            fallback_reason = ""
            actual_edit_strength = 1.0 if has_reference else 1.0
            model_used = image_model

            if (use_original and source_canvas is not None
                    and _requests_unchanged_source(prompt)):
                source_canvas.save(output_path)
                generated = source_canvas
                render_status = "reused"
                actual_edit_strength = 0.0
            else:
                positive = call_node("CLIPTextEncode", clip=clip, text=prompt)[0]
                negative = call_node("ConditioningZeroOut", conditioning=positive)[0]
                positive = call_node("FluxGuidance", conditioning=positive, guidance=3.5)[0]
                scene_seed = (int(seed) + index * 1009) % (2 ** 64)
                if source_canvas is None:
                    latent = call_node("EmptySD3LatentImage", width=render_width,
                                       height=render_height, batch_size=1)[0]
                    generated = decode_sample(
                        image_model, positive, negative, latent, scene_seed=scene_seed,
                        sample_steps=max(1, int(steps)), denoise=1.0)
                else:
                    generated = None
                    if fill_model:
                        try:
                            edit_mask = ImageOps.invert(protection_mask)
                            edit_positive, edit_negative, latent = call_node(
                                "InpaintModelConditioning", positive=positive, negative=negative,
                                vae=vae, pixels=_tensor(source_canvas), mask=_mask_tensor(edit_mask),
                                noise_mask=True)
                            generated = decode_sample(
                                fill_model, edit_positive, edit_negative, latent,
                                scene_seed=scene_seed, sample_steps=max(20, int(steps)), denoise=1.0)
                            model_used = fill_model
                        except Exception as error:
                            fallback_reason = f"FLUX Fill edit failed: {type(error).__name__}: {error}"
                            if on_progress:
                                on_progress(f"Scene {index + 1} FLUX Fill edit failed; rendering a new "
                                            "background and compositing protected identity pixels.")
                    if generated is None:
                        latent = call_node("EmptySD3LatentImage", width=render_width,
                                           height=render_height, batch_size=1)[0]
                        generated = decode_sample(
                            image_model, positive, negative, latent, scene_seed=scene_seed,
                            sample_steps=max(4, int(steps)), denoise=1.0)
                        render_status = "fallback-rendered"
                        model_used = image_model

                    # Restore only identity-critical pixels. Unlike the previous full-frame
                    # blend, this keeps the face/hair/glasses exact while leaving the scene new.
                    identity_opacity = 0.88 + 0.12 * identity_setting
                    identity_alpha = protection_mask.point(
                        lambda value: round(value * identity_opacity))
                    generated = Image.composite(source_canvas, generated, identity_alpha)
                    difference = _background_difference(source_canvas, generated, protection_mask)
                    if difference < 0.075:
                        latent = call_node("EmptySD3LatentImage", width=render_width,
                                           height=render_height, batch_size=1)[0]
                        replacement = decode_sample(
                            image_model, positive, negative, latent,
                            scene_seed=(scene_seed + 7919) % (2 ** 64),
                            sample_steps=max(4, int(steps)), denoise=1.0)
                        generated = Image.composite(source_canvas, replacement, identity_alpha)
                        difference = _background_difference(source_canvas, generated, protection_mask)
                        render_status = "fallback-rendered"
                        model_used = image_model
                        fallback_reason = (fallback_reason + "; " if fallback_reason else "") + (
                            "first render did not materially change editable pixels")
                    scene["background_difference"] = round(difference, 4)
                generated.save(output_path)
            reference_labels = [f"{asset_index + 1}:{Path(str(assets[asset_index].get('path'))).name}"
                                for asset_index in assigned_indices]
            results.append({
                **scene,
                "image_path": str(output_path),
                "source_paths": paths,
                "assigned_references": reference_labels,
                "visual_prompt": prompt,
                "edit_mode": edit_mode,
                "render_mode": edit_mode,
                "edit_strength": actual_edit_strength,
                "requested_edit_strength": requested_edit_strength,
                "identity_preservation": identity_setting,
                "render_status": render_status,
                "fallback_reason": fallback_reason,
                "image_model_used": model_used,
                "clothing_change_requested": clothing_change,
            })
        return results
    finally:
        models.clear()
        clip = vae = None
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
