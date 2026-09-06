from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime
from pathlib import Path

import folder_paths
import nodes

from .planner import plan_song
from .exporter import export_directed_package
from .audio_pipeline import (_channels, as_audio, master_audio, mix_sfx,
                             build_sfx_schedule, strip_sfx_markers,
                             generate_described_sfx, VocalMixBackend)
from .visual_pipeline import create_scene_images


def parse_duration(value):
    """Accept seconds, MM:SS, or HH:MM:SS without a fixed song-length ceiling."""
    text = str(value).strip()
    parts = text.split(":")
    if not 1 <= len(parts) <= 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
        raise ValueError("Duration must be positive seconds, MM:SS, or HH:MM:SS; e.g. 210, 3:30, 1:00:00.")
    if len(parts) > 1 and any(float(p) >= 60 for p in parts[1:]):
        raise ValueError("Seconds and minutes after a colon must be less than 60.")
    seconds = sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    if not math.isfinite(seconds) or seconds < 0.1:
        raise ValueError("Duration must be finite and at least 0.1 second.")
    return seconds


def check_cancel():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    return False


def report(stage, fraction=0):
    check_cancel()
    print(f"[Image Song Karaoke] {stage}", flush=True)


def invoke(node_name, **inputs):
    """Use the node implementations registered by this installed ComfyUI."""
    cls = nodes.NODE_CLASS_MAPPINGS.get(node_name)
    if cls is None:
        raise RuntimeError(f"Required installed node is unavailable: {node_name}")
    instance = cls()
    result = getattr(instance, getattr(cls, "FUNCTION", "execute"))(**inputs)
    if isinstance(result, dict):
        return result["result"]
    if hasattr(result, "result"):
        return result.result
    return result


def new_job(parent=None):
    root = Path(parent or folder_paths.get_output_directory())
    if parent is None:
        root = root / "image_music_karaoke"
    destination = root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _save_image(image, path):
    import numpy as np
    from PIL import Image
    if image is None or image.shape[0] != 1:
        raise ValueError("Each reference asset must contain exactly one image.")
    pixels = (image[0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).convert("RGB").save(path)


def _save_audio(audio, path):
    import soundfile as sf
    rate = int(audio["sample_rate"])
    sound = _channels(audio, rate)
    sf.write(path, sound.T.numpy(), rate, format="FLAC", subtype="PCM_24")


MEDIA_INPUTS_DEFAULT = '{"references":[],"sound_effects":[]}'
REFERENCE_INPUTS_DEFAULT = '{"version":1,"references":[]}'
SFX_INPUTS_DEFAULT = '{"version":1,"sound_effects":[]}'


def _parse_media_inputs(value):
    """Validate the frontend editor state without changing planner semantics."""
    if value in (None, ""):
        value = MEDIA_INPUTS_DEFAULT
    try:
        document = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("The reference image / sound effect editor contains invalid JSON.") from exc
    references = document.get("references", [])
    effects = document.get("sound_effects", [])
    if not isinstance(references, list) or not isinstance(effects, list):
        raise ValueError("Media editor references and sound_effects must be lists.")
    if len(references) > 100 or len(effects) > 100:
        raise ValueError("The media editor supports at most 100 images and 100 sound effects per run.")
    if any(not isinstance(item, dict) for item in references + effects):
        raise ValueError("Every media editor entry must be an object.")
    return references, effects


def _resolve_input_asset(item, label, extensions):
    filename = str(item.get("filename") or "").strip()
    subfolder = str(item.get("subfolder") or "").strip().replace("\\", "/")
    asset_type = str(item.get("type") or "input").strip().lower()
    if not filename:
        return None
    if asset_type != "input" or Path(filename).name != filename:
        raise ValueError(f"{label} must be an uploaded ComfyUI input file.")
    if Path(filename).suffix.lower() not in extensions:
        raise ValueError(f"{label} has an unsupported file type: {Path(filename).suffix or '(none)'}")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / subfolder / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the ComfyUI input directory.") from exc
    if not path.is_file():
        raise ValueError(f"{label} was not found in the ComfyUI input directory: {filename}")
    return path


def _materialize_media_inputs(value, reference_dir, sfx_dir):
    from PIL import Image
    import soundfile as sf

    reference_items, effect_items = _parse_media_inputs(value)
    references = []
    for index, item in enumerate(reference_items):
        source = _resolve_input_asset(
            item, f"Reference image {index + 1}",
            {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
        instruction = str(item.get("instruction") or "").strip()
        if source is None:
            raise ValueError(f"Choose a file for reference image {index + 1}, or remove that entry.")
        if not instruction:
            raise ValueError(f"Describe reference image {index + 1} and how it should be used.")
        destination = reference_dir / f"reference_{index + 1:03d}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(destination)
        references.append({"path": str(destination), "instruction": instruction})

    effects = []
    for index, item in enumerate(effect_items):
        source = _resolve_input_asset(
            item, f"Sound effect {index + 1}",
            {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"})
        description = str(item.get("description") or "").strip()
        if source is None and not description:
            raise ValueError(f"Upload audio or describe sound effect {index + 1}, or remove that entry.")
        metadata = {
            "path": None,
            "description": description or "uploaded effect",
            "placement": str(item.get("placement") or "automatic").strip(),
            "occurrences": str(item.get("occurrences") or "automatic").strip(),
            "duration": max(0.5, min(60.0, float(item.get("duration", 4.0)))),
            "gain_db": max(-36.0, min(3.0, float(item.get("gain_db", -18.0)))),
        }
        if source is not None:
            destination = sfx_dir / f"effect_{index + 1:03d}.flac"
            samples, rate = sf.read(source, dtype="float32", always_2d=True)
            sf.write(destination, samples, int(rate), format="FLAC", subtype="PCM_24")
            metadata["path"] = str(destination)
        effects.append(metadata)
    return references, effects


def _merge_dynamic_media_inputs(value, reference_media=None, sfx_media=None):
    """Overlay connected UI-node lists onto the legacy combined document."""
    references, effects = _parse_media_inputs(value)
    if reference_media is not None:
        references, _unused = _parse_media_inputs(reference_media)
    if sfx_media is not None:
        _unused, effects = _parse_media_inputs(sfx_media)
    return json.dumps({"version": 1, "references": references,
                       "sound_effects": effects}, ensure_ascii=False)


class KaraokeReferenceImagesInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references_json": ("STRING", {"multiline": True, "default": REFERENCE_INPUTS_DEFAULT,
                "tooltip": "Managed by the + Add Image editor."}),
        }}

    RETURN_TYPES = ("KARAOKE_REFERENCE_INPUTS",)
    RETURN_NAMES = ("reference_media",)
    FUNCTION = "collect"
    CATEGORY = "audio/Image Music Karaoke/Director inputs"
    DESCRIPTION = "Variable-length reference image picker for the Karaoke Director."

    def collect(self, references_json):
        references, _effects = _parse_media_inputs(references_json)
        return (json.dumps({"version": 1, "references": references}, ensure_ascii=False),)


class KaraokeSoundEffectsInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sound_effects_json": ("STRING", {"multiline": True, "default": SFX_INPUTS_DEFAULT,
                "tooltip": "Managed by the + Add Sound Effect editor."}),
        }}

    RETURN_TYPES = ("KARAOKE_SFX_INPUTS",)
    RETURN_NAMES = ("sfx_media",)
    FUNCTION = "collect"
    CATEGORY = "audio/Image Music Karaoke/Director inputs"
    DESCRIPTION = "Variable-length uploaded or described sound-effect picker for the Karaoke Director."

    def collect(self, sound_effects_json):
        _references, effects = _parse_media_inputs(sound_effects_json)
        return (json.dumps({"version": 1, "sound_effects": effects}, ensure_ascii=False),)


class KaraokeReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "instruction": ("STRING", {"multiline": True, "default":
                "This is the main character. Preserve their exact face and identity; adapt the setting and lighting to the scene."}),
        }, "optional": {"previous_references": ("KARAOKE_REFERENCES",)}}

    RETURN_TYPES = ("KARAOKE_REFERENCES", "STRING")
    RETURN_NAMES = ("references", "summary")
    FUNCTION = "append"
    CATEGORY = "audio/Image Music Karaoke/Director inputs"
    DESCRIPTION = "Add one source image and its editable use/identity instruction. Chain nodes for multiple references."

    def append(self, image, instruction, previous_references=None):
        if image is None or image.shape[0] != 1:
            raise ValueError("Reference Image accepts one image per node.")
        if not str(instruction).strip():
            raise ValueError("Describe what this reference represents and how it may be used.")
        result = list(previous_references or [])
        result.append({"image": image, "instruction": str(instruction).strip()})
        return result, f"{len(result)} reference image(s) collected"


class KaraokeSoundEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "description": ("STRING", {"multiline": True, "default": "distant train horn"}),
            "placement": ("STRING", {"multiline": True, "default": "automatic",
                "tooltip": "Examples: intro, after the first chorus, 12s and 48s, or automatic."}),
            "occurrences": ("STRING", {"default": "automatic", "tooltip": "Examples: once, twice, 3, every chorus."}),
            "generated_duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 60.0, "step": 0.5}),
            "gain_db": ("FLOAT", {"default": -18.0, "min": -36.0, "max": 3.0, "step": 1.0}),
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "Optional uploaded WAV/MP3. Leave disconnected to generate from the description."}),
            "previous_effects": ("KARAOKE_SFX_LIST",),
        }}

    RETURN_TYPES = ("KARAOKE_SFX_LIST", "STRING")
    RETURN_NAMES = ("sound_effects", "summary")
    FUNCTION = "append"
    CATEGORY = "audio/Image Music Karaoke/Director inputs"
    DESCRIPTION = "Add an uploaded or described sound effect. Chain nodes for an arbitrary list."

    def append(self, description, placement, occurrences, generated_duration, gain_db,
               audio=None, previous_effects=None):
        if not str(description).strip() and audio is None:
            raise ValueError("Supply an audio file or describe a sound to generate.")
        result = list(previous_effects or [])
        result.append({"audio": audio, "description": str(description or "uploaded effect").strip(),
                       "placement": str(placement or "automatic").strip(),
                       "occurrences": str(occurrences or "automatic").strip(),
                       "duration": float(generated_duration), "gain_db": float(gain_db)})
        return result, f"{len(result)} sound effect(s) collected"


class ImageSongPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "song_request": ("STRING", {"multiline": True, "default": "Write an uplifting indie pop song with clear expressive singing."}),
            "duration": ("STRING", {"default": "3:30", "tooltip": "Seconds, MM:SS, or HH:MM:SS."}),
            "section_seconds": ("INT", {"default": 60, "min": 10, "max": 180, "step": 1}),
            "lyrics_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional exact lyrics; split section blocks with a line containing ---."}),
            "writer_model": (["gemma4:12b"],),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "media_inputs_json": ("STRING", {"multiline": True, "default": MEDIA_INPUTS_DEFAULT,
                "tooltip": "Managed by the + Add Image / + Add Sound Effect editor."}),
        }, "optional": {
            "reference_media": ("KARAOKE_REFERENCE_INPUTS",),
            "sfx_media": ("KARAOKE_SFX_INPUTS",),
            "reference_images": ("KARAOKE_REFERENCES",),
            "sound_effects": ("KARAOKE_SFX_LIST",),
            "image": ("IMAGE", {"tooltip": "Legacy optional single reference image."}),
            "sfx_audio": ("AUDIO", {"tooltip": "Legacy optional single uploaded sound."}),
        }}

    RETURN_TYPES = ("IMAGE_SONG_PLAN", "STRING")
    RETURN_NAMES = ("song_plan", "lyrics_and_plan")
    FUNCTION = "plan"
    CATEGORY = "audio/Image Music Karaoke"
    DESCRIPTION = "Director/planner for song structure, reference-conditioned scenes, voice and arbitrary SFX."

    def plan(self, song_request, duration, section_seconds, lyrics_override, writer_model, seed,
             media_inputs_json=MEDIA_INPUTS_DEFAULT, sfx_description="", reference_images=None,
             sound_effects=None, image=None, sfx_audio=None, reference_media=None, sfx_media=None):
        import comfy.model_management as mm
        seconds = parse_duration(duration)
        if not song_request.strip():
            raise ValueError("Describe the song you want in song_request.")
        job = new_job()
        reference_dir, sfx_dir = job / "references", job / "sound_effects"
        reference_dir.mkdir()
        sfx_dir.mkdir()
        media_inputs_json = _merge_dynamic_media_inputs(
            media_inputs_json, reference_media=reference_media, sfx_media=sfx_media)
        references, effects = _materialize_media_inputs(media_inputs_json, reference_dir, sfx_dir)
        for index, item in enumerate(reference_images or []):
            path = reference_dir / f"reference_{len(references) + 1:03d}.png"
            _save_image(item["image"], path)
            references.append({"path": str(path), "instruction": str(item["instruction"])})
        if image is not None:
            path = reference_dir / f"reference_{len(references) + 1:03d}.png"
            _save_image(image, path)
            references.append({"path": str(path),
                               "instruction": "Legacy reference: preserve its important subject and exact identity."})
        for index, item in enumerate(sound_effects or []):
            metadata = {key: item.get(key) for key in
                        ("description", "placement", "occurrences", "duration", "gain_db")}
            if item.get("audio") is not None:
                path = sfx_dir / f"effect_{len(effects) + 1:03d}.flac"
                _save_audio(item["audio"], path)
                metadata["path"] = str(path)
            else:
                metadata["path"] = None
            effects.append(metadata)
        if sfx_audio is not None:
            path = sfx_dir / f"effect_{len(effects) + 1:03d}.flac"
            _save_audio(sfx_audio, path)
            effects.append({"path": str(path), "description": str(sfx_description or "uploaded effect"),
                            "placement": "automatic", "occurrences": "automatic",
                            "duration": 4.0, "gain_db": -18.0})
        mm.unload_all_models()
        mm.soft_empty_cache()
        plan = plan_song(references[0]["path"] if references else None, song_request, seconds,
                         section_seconds, lyrics_override=lyrics_override,
                         ollama_model=writer_model, seed=seed,
                         sfx_present=bool(effects), sfx_description=sfx_description,
                         references=references, sound_effects=effects,
                         on_progress=report, cancelled=check_cancel)
        plan.update(source_image_path=references[0]["path"] if references else None,
                    image_path=None, sfx_path=effects[0].get("path") if len(effects) == 1 else None,
                    job_dir=str(job), seed=seed, target_duration=seconds, request=song_request)
        (job / "song_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [plan.get("title", "Image song"),
                 f"Duration: {seconds:g} seconds; {len(plan['segments'])} ACE section(s); {len(plan.get('scenes', []))} visual scene(s)",
                 plan["tags"], f"References: {len(references)}; sound effects: {len(effects)}"]
        for scene in plan.get("scenes", []):
            lines.append(f"SCENE {scene['scene_id']} {scene['section']} @ {scene['start']:.2f}s; references={scene['reference_indices']}")
        for index, segment in enumerate(plan["segments"], 1):
            lines.extend([f"\nSECTION {index} ({segment['duration']:g} seconds)", segment["lyrics"]])
        preview = "\n".join(lines)
        (job / "lyrics.txt").write_text(preview, encoding="utf-8")
        return plan, preview


class ImageSongRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "song_plan": ("IMAGE_SONG_PLAN",),
            "music_model": (folder_paths.get_filename_list("diffusion_models"), {"default": "acestep_v1.5_xl_sft_bf16.safetensors"}),
            "text_encoder": (folder_paths.get_filename_list("text_encoders"), {"default": "qwen_0.6b_ace15.safetensors"}),
            "audio_code_model": (folder_paths.get_filename_list("text_encoders"), {"default": "qwen_4b_ace15.safetensors"}),
            "audio_vae": (folder_paths.get_filename_list("vae"), {"default": "ace_1.5_vae.safetensors"}),
            "image_model": (folder_paths.get_filename_list("diffusion_models"), {"default": "flux1-schnell-fp8.safetensors"}),
            "image_clip": (folder_paths.get_filename_list("text_encoders"), {"default": "clip_l.safetensors"}),
            "image_t5": (folder_paths.get_filename_list("text_encoders"), {"default": "t5xxl_fp16.safetensors"}),
            "image_vae": (folder_paths.get_filename_list("vae"), {"default": "ae.safetensors"}),
            "visual_treatment": (["identity-preserving reference edits", "use supplied references unchanged"],),
            "image_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
            "image_edit_strength": ("FLOAT", {"default": 0.30, "min": 0.08, "max": 0.45, "step": 0.02}),
            "identity_preservation": ("FLOAT", {"default": 0.58, "min": 0.35, "max": 0.80, "step": 0.05,
                "tooltip": "Pixel-space reference preservation after editing; higher values retain identity more strongly."}),
            "steps": ("INT", {"default": 65, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 3.0, "min": 0, "max": 20, "step": 0.1}),
            "generate_audio_codes": ("BOOLEAN", {"default": True}),
            "lyric_timing": (["auto", "required", "estimated"], {"default": "auto"}),
            "resolution": (["1280x720", "1920x1080", "720x1280"],),
            "filename": ("STRING", {"default": "song"}),
        }, "optional": {
            "trained_voice_model": ("RVC_MODEL",),
            "sfx_engine": ("TTS_ENGINE", {"tooltip": "Optional compatible local sound-effect engine, such as MOSS v2."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("song_FLAC", "song_MP3", "music_video_MP4", "karaoke_video_MP4", "export_report")
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "audio/Image Music Karaoke"
    DESCRIPTION = "ACE-Step song generation, directed identity-safe scenes, lead-only RVC, SFX, dual mixes and four final files."

    def render(self, song_plan, music_model, text_encoder, audio_code_model, audio_vae,
               image_model, image_clip, image_t5, image_vae, visual_treatment,
               image_steps, image_edit_strength, identity_preservation, steps, cfg,
               generate_audio_codes, lyric_timing, resolution, filename,
               trained_voice_model=None, sfx_engine=None):
        import torch
        import soundfile as sf
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        plan = song_plan
        if not plan.get("segments"):
            raise ValueError("The song plan has no sections. Run the planner again.")
        check_cancel()
        job = new_job(plan["job_dir"])
        segment_dir = job / "sections"
        segment_dir.mkdir()
        name = re.sub(r"[^\w.-]+", "_", filename, flags=re.UNICODE).strip("._")[:80] or "song"
        manifest = {"status": "generating", "plan": plan,
                    "models": [music_model, text_encoder, audio_code_model, audio_vae,
                               image_model, image_clip, image_t5, image_vae],
                    "settings": {"steps": steps, "cfg": cfg, "sampler": "euler", "scheduler": "simple",
                                 "shift": 6.0, "visual_treatment": visual_treatment,
                                 "image_steps": image_steps, "image_edit_strength": image_edit_strength,
                                 "identity_preservation": identity_preservation,
                                 "voice_conversion": bool(trained_voice_model),
                                 "sound_effects": len(plan.get("sfx_entries") or [])},
                    "segments": [], "scenes": []}
        manifest_path = job / "generation.json"

        def save_manifest():
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        save_manifest()
        model = clip = vae = None
        try:
            width, height = map(int, resolution.split("x"))
            rendered_scenes = create_scene_images(
                invoke, nodes.common_ksampler, plan, job / "visual_scenes",
                width=width, height=height, seed=int(plan.get("seed", 0)),
                image_model=image_model, image_clip=image_clip, image_t5=image_t5,
                image_vae=image_vae, steps=image_steps, edit_denoise=image_edit_strength,
                identity_blend=identity_preservation,
                use_original=visual_treatment == "use supplied references unchanged",
                on_progress=report)
            manifest["scenes"] = rendered_scenes
            save_manifest()

            report("Loading the existing ACE-Step Text to music models")
            model = invoke("UNETLoader", unet_name=music_model, weight_dtype="default")[0]
            model = invoke("ModelSamplingAuraFlow", model=model, shift=6.0)[0]
            clip = invoke("DualCLIPLoader", clip_name1=text_encoder, clip_name2=audio_code_model, type="ace", device="default")[0]
            for module in clip.cond_stage_model.modules():
                if hasattr(module, "fixed_kv") and hasattr(module, "init_kv_cache"):
                    module.fixed_kv = False
            manifest["settings"]["audio_code_attention"] = "pytorch_kv_cache"
            vae = invoke("VAELoader", vae_name=audio_vae)[0]
            progress = ProgressBar(len(plan["segments"]) * 2 + len(rendered_scenes) + 2)
            for index, item in enumerate(plan["segments"]):
                check_cancel()
                duration = float(item["duration"])
                generate_seconds = max(10.0, duration)
                section_seed = (int(plan.get("seed", 0)) + index) % (2 ** 64)
                report(f"Generating ACE-Step section {index + 1}/{len(plan['segments'])}: {duration:g} seconds")
                with torch.inference_mode():
                    positive = invoke("TextEncodeAceStepAudio1.5", clip=clip, tags=plan["tags"],
                                      lyrics=strip_sfx_markers(item["lyrics"]), seed=section_seed,
                                      bpm=int(plan.get("bpm", 120)), duration=generate_seconds,
                                      timesignature="4", language=plan.get("language", "en"),
                                      keyscale=plan.get("keyscale", "E minor"),
                                      generate_audio_codes=generate_audio_codes, cfg_scale=2.0,
                                      temperature=0.85, top_p=1.0, top_k=0, min_p=0.0)[0]
                    negative = invoke("ConditioningZeroOut", conditioning=positive)[0]
                    latent = invoke("EmptyAceStep1.5LatentAudio", seconds=generate_seconds, batch_size=1)[0]
                    sampled = nodes.common_ksampler(model, section_seed, steps, cfg, "euler", "simple",
                                                    positive, negative, latent, denoise=1.0)[0]
                    audio = invoke("VAEDecodeAudioTiled", vae=vae, samples=sampled,
                                   tile_size=512, overlap=64)[0]
                rate = int(audio["sample_rate"])
                waveform = audio["waveform"][0].detach().float().cpu()
                target_samples = round(duration * rate)
                if waveform.shape[-1] < target_samples - round(rate * 0.025):
                    raise RuntimeError(f"Section {index + 1} decoded shorter than requested.")
                path = segment_dir / f"section_{index + 1:05d}_ace.flac"
                sf.write(path, waveform[:, :target_samples].T.numpy(), rate, format="FLAC", subtype="PCM_24")
                manifest["segments"].append({"audio_path": str(path), "raw_audio_path": str(path),
                                             "lyrics": item["lyrics"], "duration": duration,
                                             "seed": section_seed, "sample_rate": rate})
                save_manifest()
                del waveform, audio, sampled, latent, positive, negative
                progress.update_absolute(index + 1)
            model = clip = vae = None
            mm.unload_all_models()
            mm.soft_empty_cache()

            generated_effects = []
            for index, entry in enumerate(plan.get("sfx_entries") or []):
                if entry.get("path"):
                    samples, effect_rate = sf.read(entry["path"], dtype="float32", always_2d=True)
                    effect_audio = as_audio(torch.from_numpy(samples.T.copy()), int(effect_rate))
                    source = "uploaded audio"
                elif sfx_engine is not None:
                    report(f"Generating sound effect {index + 1}: {entry.get('description', 'effect')}")
                    effect_audio = invoke("UnifiedSoundEffectsNode", TTS_engine=sfx_engine,
                                          description=str(entry.get("description") or "ambient sound"),
                                          duration_seconds=float(entry.get("duration", 4.0)),
                                          seed=(int(plan.get("seed", 0)) + 50000 + index),
                                          crossfade_seconds=1.0, enable_audio_cache=True)[0]
                    source = "connected neural SFX engine"
                else:
                    effect_audio = generate_described_sfx(str(entry.get("description") or "ambient sound"),
                                                          float(entry.get("duration", 4.0)),
                                                          int(plan.get("seed", 0)) + 50000 + index)
                    source = "offline procedural fallback"
                generated_effects.append(effect_audio)
                entry["render_source"] = source
            if sfx_engine is not None:
                mm.unload_all_models()
                mm.soft_empty_cache()
            schedule = build_sfx_schedule(plan)
            voice_backend = VocalMixBackend(invoke, trained_voice_model)
            if trained_voice_model is not None and (not isinstance(trained_voice_model, dict)
                                                     or not trained_voice_model.get("model_path")):
                raise ValueError("The connected trained voice model is invalid or missing its .pth file.")

            for index, item in enumerate(manifest["segments"]):
                check_cancel()
                samples, rate = sf.read(item["raw_audio_path"], dtype="float32", always_2d=True)
                original = as_audio(torch.from_numpy(samples.T.copy()), int(rate))
                target_samples = round(float(item["duration"]) * int(rate))
                report(f"Separating lead/backing vocals for section {index + 1}/{len(manifest['segments'])}")
                full, karaoke, voice_info = voice_backend.process(original, sample_rate=int(rate),
                                                                  target_samples=target_samples)
                item["voice_processing"] = voice_info
                item["lead_voice_converted"] = bool(trained_voice_model)
                cues = schedule.get(index, [])
                for cue in cues:
                    effect = generated_effects[cue["effect_index"]]
                    full = mix_sfx(full, effect, [cue["offset"]], sample_rate=int(rate),
                                   target_samples=target_samples, level_db=cue["gain_db"])
                    karaoke = mix_sfx(karaoke, effect, [cue["offset"]], sample_rate=int(rate),
                                      target_samples=target_samples, level_db=cue["gain_db"])
                full = master_audio(full, sample_rate=int(rate), target_samples=target_samples)
                karaoke = master_audio(karaoke, sample_rate=int(rate), target_samples=target_samples)
                full_path = segment_dir / f"section_{index + 1:05d}_full.flac"
                karaoke_path = segment_dir / f"section_{index + 1:05d}_karaoke.flac"
                sf.write(full_path, _channels(full, int(rate), target_samples).T.numpy(), int(rate),
                         format="FLAC", subtype="PCM_24")
                sf.write(karaoke_path, _channels(karaoke, int(rate), target_samples).T.numpy(), int(rate),
                         format="FLAC", subtype="PCM_24")
                item.update(audio_path=str(full_path), karaoke_audio_path=str(karaoke_path),
                            sfx_cues=cues,
                            processing_order=["ACE-Step", "Demucs lead/backing separation",
                                              "lead-only RVC" if trained_voice_model else "original lead",
                                              "SFX in both mixes" if cues else "no SFX in this section",
                                              "separate mastering", "alignment against final full mix"])
                save_manifest()
                progress.update_absolute(len(plan["segments"]) + index + 1)

            report("Aligning final lead performance and exporting four deliverables")
            outputs = export_directed_package(manifest["segments"], rendered_scenes, job,
                                              basename=name, width=width, height=height,
                                              alignment=lyric_timing, language=plan.get("language", "en"),
                                              on_progress=report, cancelled=check_cancel)
            manifest.update(status="complete", outputs=outputs)
            save_manifest()
            visual_report = []
            for scene_index, scene in enumerate(rendered_scenes, 1):
                assigned = ", ".join(scene.get("assigned_references") or []) or "none"
                visual_report.extend([
                    f"VISUAL SCENE {scene_index}",
                    f"  References: {assigned}",
                    f"  Prompt: {scene.get('visual_prompt', '')}",
                    f"  Edit mode: {scene.get('edit_mode', scene.get('render_mode', 'unknown'))}",
                    f"  Edit strength: {scene.get('edit_strength', image_edit_strength)}",
                    f"  Identity preservation: {scene.get('identity_preservation', identity_preservation)}",
                    f"  Result: {scene.get('render_status', 'newly-rendered')}",
                ])
                if scene.get("fallback_reason"):
                    visual_report.append(f"  Fallback reason: {scene['fallback_reason']}")
            report_text = (f"Created {len(plan['segments'])} ACE sections and {len(rendered_scenes)} visual scenes.\n"
                           f"FLAC: {outputs['flac_path']}\nMP3: {outputs['mp3_path']}\n"
                           f"Music video: {outputs['music_video_path']}\n"
                           f"Karaoke video: {outputs['karaoke_video_path']}\n"
                           f"Lyric timing: {outputs.get('timing_mode', lyric_timing)}; source is the final full mix.\n"
                           f"Voice: {'trained lead voice only' if trained_voice_model else 'original lead'}; "
                           "karaoke uses instrumental plus feasible stereo backing vocals.\n"
                           f"References: {len(plan.get('reference_assets') or [])}; SFX: {len(generated_effects)}"
                           "\n\nVisual scene report:\n" + "\n".join(visual_report))
            if outputs.get("warnings"):
                report_text += "\n" + "\n".join(outputs["warnings"])
            output_root = Path(folder_paths.get_output_directory()).resolve()

            def ui_file(path):
                relative = Path(path).resolve().relative_to(output_root)
                return {"filename": relative.name, "subfolder": relative.parent.as_posix(), "type": "output"}

            return {"ui": {"text": [report_text],
                           "song_flac": [ui_file(outputs["flac_path"])],
                           "song_mp3": [ui_file(outputs["mp3_path"])],
                           "music_video": [ui_file(outputs["music_video_path"])],
                           "karaoke_video": [ui_file(outputs["karaoke_video_path"])]},
                    "result": (str(outputs["flac_path"]), str(outputs["mp3_path"]),
                               str(outputs["music_video_path"]), str(outputs["karaoke_video_path"]),
                               report_text)}
        except BaseException as error:
            manifest.update(status="interrupted" if "interrupt" in type(error).__name__.lower() else "failed",
                            error=str(error))
            save_manifest()
            raise
        finally:
            model = clip = vae = None


NODE_CLASS_MAPPINGS = {
    "KaraokeReferenceImagesInput": KaraokeReferenceImagesInput,
    "KaraokeSoundEffectsInput": KaraokeSoundEffectsInput,
    "KaraokeReferenceImage": KaraokeReferenceImage,
    "KaraokeSoundEffect": KaraokeSoundEffect,
    "ImageSongPlan": ImageSongPlan,
    "ImageSongRender": ImageSongRender,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KaraokeReferenceImagesInput": "Reference Images",
    "KaraokeSoundEffectsInput": "Sound Effects",
    "KaraokeReferenceImage": "Karaoke Director — Reference Image",
    "KaraokeSoundEffect": "Karaoke Director — Sound Effect",
    "ImageSongPlan": "Karaoke Director — Plan Song + Scenes",
    "ImageSongRender": "Generate Song + Music Video + Karaoke",
}
