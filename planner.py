"""Local, cancellable image-to-song planning for the karaoke workflow.

Only Ollama on 127.0.0.1:11434 is contacted. Music segments are balanced so
there is no short final tail. A total below ten seconds stays below ten here;
the generation node pads its model request and trims the rendered audio.
"""

from __future__ import annotations

import base64
import http.client
import json
import math
from pathlib import Path
import queue
import re
import socket
import threading
import time
from typing import Any, Callable


OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
REQUEST_TIMEOUT_SECONDS = 900.0
CONNECT_TIMEOUT_SECONDS = 10.0

LANGUAGES = set("ar az bg bn ca cs da de el en es fa fi fr he hi hr ht hu id is it ja ko la lt ms ne nl no pa pl pt ro ru sa sk sr sv sw ta te th tl tr uk ur vi yue zh unknown".split())
KEYS = {f"{root} {quality}" for quality in ("major", "minor")
        for root in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")}


class PlanningError(RuntimeError):
    """An actionable local model or planning failure."""


class PlanningCancelled(PlanningError):
    """The caller cancelled planning."""


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise PlanningCancelled("Song planning was cancelled.")


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _local_post(payload: dict[str, Any], *, timeout: float = REQUEST_TIMEOUT_SECONDS,
                cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """POST locally with a wall-clock deadline and active socket cancellation.

Ollama may take minutes to load a model before returning its first token.
A daemon transport thread lets the caller check ComfyUI cancellation while
that happens. Closing/shutting down its socket also ends Ollama's request.
"""
    _check_cancel(cancelled)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    result: queue.Queue = queue.Queue(maxsize=1)
    connection = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT,
                                             timeout=CONNECT_TIMEOUT_SECONDS)
    aborted = threading.Event()
    transport_socket: list[socket.socket] = []

    def stop_transport() -> None:
        aborted.set()
        sock = transport_socket[0] if transport_socket else connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    def request() -> None:
        try:
            connection.connect()
            if aborted.is_set():
                return
            if connection.sock is not None:
                transport_socket.append(connection.sock)
                connection.sock.settimeout(timeout)
            connection.request("POST", "/api/generate", body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise PlanningError(f"Local Ollama returned HTTP {response.status}: "
                                    f"{raw.decode('utf-8', errors='replace')[:600]}")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise PlanningError("Local Ollama returned a non-object API response.")
            if parsed.get("error"):
                raise PlanningError(f"Local Ollama: {parsed['error']}")
            result.put((True, parsed))
        except Exception as error:
            if not aborted.is_set():
                result.put((False, error))
        finally:
            connection.close()

    threading.Thread(target=request, name="karaoke-local-ollama", daemon=True).start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            _check_cancel(cancelled)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlanningError(f"Local Ollama did not finish within {timeout:g} seconds. "
                                    "Check Ollama and available memory, then retry.")
            try:
                succeeded, value = result.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            if succeeded:
                return value
            if isinstance(value, PlanningError):
                raise value
            raise PlanningError(f"Could not complete a request to local Ollama at "
                                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}: {value}") from value
    finally:
        stop_transport()


def _ollama_json(model: str, system: str, prompt: str, *, seed: int,
                 images: list[str] | None = None, cancelled=None) -> Any:
    payload: dict[str, Any] = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "5m",
        "options": {"seed": int(seed) % 2147483647, "temperature": 0.75,
                    "num_ctx": 8192, "num_predict": 4096},
    }
    if images:
        payload["images"] = images
    response = _local_post(payload, cancelled=cancelled)
    content = response.get("response")
    if not isinstance(content, str):
        raise PlanningError("Local Ollama returned no text response.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"The model response was not valid JSON ({error.msg}).") from error


def _validated_request(model: str, system: str, prompt: str, validator,
                       *, seed: int, images=None, cancelled=None, on_progress=None):
    """Retry malformed content exactly once; transport failures are explicit."""
    correction = ""
    for attempt in range(2):
        _check_cancel(cancelled)
        try:
            content = _ollama_json(model, system, prompt + correction, seed=seed + attempt,
                                   images=images, cancelled=cancelled)
            _check_cancel(cancelled)
            return validator(content)
        except (ValueError, TypeError, KeyError) as error:
            if attempt == 1:
                raise PlanningError(f"Ollama produced invalid song data twice: {error}. "
                                    "Try a simpler prompt or another installed vision model.") from error
            _progress(on_progress, "Ollama returned invalid song data; asking it to correct the JSON once.")
            correction = ("\nYour previous answer failed validation: " + str(error)[:500]
                          + "\nReturn a complete corrected JSON object matching the requested schema.")
    raise AssertionError("unreachable")


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _singable_lines(text: str) -> list[str]:
    # ACE uses bracketed section markers; those are not sung lyric lines.
    return [line.strip() for line in text.splitlines()
            if line.strip() and not re.fullmatch(r"\s*\[[^\]]+\]\s*", line)]


def _validate_style(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("The song style must be a JSON object")
    checked = {key: _nonempty_text(value.get(key), key)
               for key in ("tags", "title", "image_description", "language", "keyscale", "refrain",
                           "background_prompt", "visual_edit_prompt")}
    bpm = value.get("bpm")
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(bpm):
        raise ValueError("bpm must be a number between 10 and 300")
    if int(bpm) != bpm or not 10 <= bpm <= 300:
        raise ValueError("bpm must be a whole number between 10 and 300")
    checked["bpm"] = int(bpm)
    if checked["language"] not in LANGUAGES:
        raise ValueError("language must be a supported two/three-letter ACE language code")
    if checked["keyscale"] not in KEYS:
        raise ValueError("keyscale must use a key such as 'C major' or 'E minor'")
    if not _singable_lines(checked["refrain"]):
        raise ValueError("refrain must contain actual sung words")
    return checked


def _validate_lyrics(value: Any) -> str:
    if not isinstance(value, dict):
        raise ValueError("A lyric segment must be a JSON object")
    lyrics = _nonempty_text(value.get("lyrics"), "lyrics")
    if not _singable_lines(lyrics):
        raise ValueError("Every segment needs sung lyrics, not only section headings")
    if len(lyrics) > 16000:
        raise ValueError("One lyric segment exceeds 16,000 characters; shorten its lyrics")
    return lyrics


def _validate_visual(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("The karaoke visual plan must be a JSON object")
    return {key: _nonempty_text(value.get(key), key)
            for key in ("background_prompt", "visual_edit_prompt")}


def _scene_skeleton(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn ACE lyric sections into a duration-exact music-video timeline."""
    scenes: list[dict[str, Any]] = []
    song_cursor = 0.0
    for segment_index, segment in enumerate(segments):
        duration = float(segment["duration"])
        blocks: list[dict[str, Any]] = []
        current = {"section": f"Section {segment_index + 1}", "lines": []}
        for raw in str(segment.get("lyrics", "")).splitlines():
            line = raw.strip()
            marker = re.fullmatch(r"\[([^\]]+)\]", line)
            if marker and not marker.group(1).strip().casefold().startswith("sfx"):
                if current["lines"]:
                    blocks.append(current)
                current = {"section": marker.group(1).strip(), "lines": []}
            elif line and not (marker and marker.group(1).strip().casefold().startswith("sfx")):
                current["lines"].append(line)
        if current["lines"] or not blocks:
            blocks.append(current)
        weights = [max(1, sum(len(line.split()) for line in block["lines"])) for block in blocks]
        total_weight = sum(weights)
        local_cursor = 0.0
        for block_index, (block, weight) in enumerate(zip(blocks, weights)):
            scene_duration = (duration - local_cursor if block_index == len(blocks) - 1
                              else duration * weight / total_weight)
            scene_id = f"scene_{len(scenes) + 1:03d}"
            scenes.append({
                "scene_id": scene_id,
                "segment_index": segment_index,
                "section": block["section"],
                "start": song_cursor + local_cursor,
                "duration": scene_duration,
                "lyrics_excerpt": "\n".join(block["lines"])[:1800],
            })
            local_cursor += scene_duration
        song_cursor += duration
    return scenes


def _validate_direction(value: Any, skeleton: list[dict[str, Any]],
                        reference_count: int, sfx_count: int,
                        total_duration: float) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("The director plan must be a JSON object")
    supplied = value.get("scenes")
    if not isinstance(supplied, list):
        raise ValueError("director scenes must be a list")
    by_id = {str(item.get("scene_id")): item for item in supplied if isinstance(item, dict)}
    scenes = []
    for index, base in enumerate(skeleton):
        item = by_id.get(base["scene_id"], {})
        generation = str(item.get("generation_prompt") or item.get("background_prompt") or "").strip()
        editing = str(item.get("edit_prompt") or item.get("visual_edit_prompt") or "").strip()
        if not generation:
            raise ValueError(f"{base['scene_id']} needs generation_prompt")
        if not editing:
            editing = generation
        refs = item.get("reference_indices", [])
        if not isinstance(refs, list):
            raise ValueError(f"{base['scene_id']} reference_indices must be a list")
        refs = list(dict.fromkeys(int(value) for value in refs
                                  if isinstance(value, int) and 0 <= value < reference_count))
        edit_mode = ("text-to-image" if not refs else
                     "identity-preserving image-to-image" if len(refs) == 1 else
                     "multi-reference composition")
        scenes.append({**base,
                       "reference_indices": refs,
                       "edit_mode": edit_mode,
                       "generation_prompt": generation,
                       "edit_prompt": editing,
                       "continuity_notes": str(item.get("continuity_notes") or "").strip()})
    events = []
    for item in value.get("sfx_events", []):
        if not isinstance(item, dict):
            continue
        effect_index = item.get("effect_index")
        at = item.get("time_seconds")
        if (isinstance(effect_index, int) and 0 <= effect_index < sfx_count
                and isinstance(at, (int, float)) and math.isfinite(at)):
            events.append({"effect_index": effect_index,
                           "time_seconds": max(0.0, min(float(total_duration), float(at))),
                           "gain_db": max(-36.0, min(3.0, float(item.get("gain_db", -18.0))))})
    return {"scenes": scenes, "sfx_events": events}


def _segment_count(total: float, preferred: float) -> int:
    # If the requested segment size is ten seconds and total=11, two 5.5s
    # segments would violate the minimum. Use one 11s segment in that case.
    return max(1, min(math.ceil(total / preferred), max(1, math.floor(total / 10.0))))


def _override_blocks(text: str, count: int) -> tuple[list[str], str]:
    if re.search(r"(?m)^\s*---\s*$", text):
        blocks = re.split(r"(?m)^\s*---\s*$", text)
        if len(blocks) != count:
            raise PlanningError(f"The lyrics override contains {len(blocks)} blocks separated by ---; "
                                f"this duration needs {count} segments. Supply exactly one block per "
                                "segment or remove the --- separators for automatic splitting.")
        for block in blocks:
            _validate_lyrics({"lyrics": block})
        return [block.strip() for block in blocks], "Lyrics used exactly as the supplied per-segment blocks."
    # Keep marker lines with their following lyric line. Blank lines are
    # layout only. Never create an instrumental-only segment.
    units: list[str] = []
    pending: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if re.fullmatch(r"\s*\[[^\]]+\]\s*", line):
            pending.append(line.strip())
        else:
            units.append("\n".join(pending + [line.strip()]))
            pending = []
    if not units:
        raise PlanningError("The lyrics override contains no sung words.")
    if count > len(units):
        return [units[index % len(units)] for index in range(count)], (
            "Lyrics override has fewer sung lines than segments; its lines repeat cyclically "
            "so every segment has lyrics. Use --- blocks to control each segment explicitly.")
    blocks = []
    for index in range(count):
        start, end = index * len(units) // count, (index + 1) * len(units) // count
        blocks.append(_validate_lyrics({"lyrics": "\n".join(units[start:end])}))
    return blocks, "Lyrics override split in order across segments at sung-line boundaries."


def plan_song(image_path, prompt, total_seconds, segment_seconds,
              lyrics_override="", ollama_model="gemma4:12b", seed=0,
              sfx_present=False, sfx_description="", references=None,
              sound_effects=None, on_progress=None, cancelled=None) -> dict[str, Any]:
    """Plan a consistent style and a complete set of lyric-bearing segments.

``duration`` is each segment's actual requested output duration, including
totals below ten seconds. The caller handles model padding and final trim.
The optional ``notes`` field explicitly reports splitting/repeating overrides.
"""
    total, preferred = float(total_seconds), float(segment_seconds)
    if not math.isfinite(total) or total <= 0:
        raise PlanningError("Song duration must be a finite positive number of seconds.")
    if not math.isfinite(preferred) or not 10 <= preferred <= 180:
        raise PlanningError("Segment duration must be between 10 and 180 seconds.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PlanningError("Enter a description of the song to create.")
    if not isinstance(lyrics_override, str):
        raise PlanningError("Lyrics override must be text.")
    model = _nonempty_text(ollama_model, "ollama_model")
    _check_cancel(cancelled)
    count = _segment_count(total, preferred)
    duration = total / count
    blocks, override_note = (None, "")
    if lyrics_override.strip():
        blocks, override_note = _override_blocks(lyrics_override, count)
        _progress(on_progress, override_note)
    reference_assets = [dict(item) for item in (references or [])]
    if image_path and not reference_assets:
        reference_assets.append({"path": str(image_path),
                                 "instruction": "Preserve the important subject and identity."})
    for index, asset in enumerate(reference_assets):
        picture = Path(str(asset.get("path", "")))
        if not picture.is_file():
            raise PlanningError(f"Reference image {index + 1} does not exist: {picture}")
        asset["path"] = str(picture)
        asset["instruction"] = str(asset.get("instruction") or
                                   "Preserve the important subject and identity.").strip()
    image_data = [base64.b64encode(Path(asset["path"]).read_bytes()).decode("ascii")
                  for asset in reference_assets]
    has_image = bool(image_data)
    sfx_assets = [dict(item) for item in (sound_effects or [])]
    if sfx_present and not sfx_assets:
        sfx_assets.append({"path": None, "description": str(sfx_description or "uploaded sound effect"),
                           "placement": "automatic", "occurrences": "automatic"})
    sfx_note = "\n".join(
        f"{index}: {item.get('description') or 'uploaded effect'}; placement={item.get('placement') or 'automatic'}; "
        f"occurrences={item.get('occurrences') or 'automatic'}"
        for index, item in enumerate(sfx_assets))
    reference_note = "\n".join(
        f"Reference {index}: {asset['instruction']}" for index, asset in enumerate(reference_assets))
    attempted_ollama = False
    try:
        _progress(on_progress, (f"Reading the image and planning {count} music segment(s) with local Ollama."
                               if has_image else
                               f"Planning {count} music segment(s) and a karaoke background with local Ollama."))
        attempted_ollama = True
        style = _validated_request(
            model,
            "You plan songs and their karaoke visuals from a user description and, when provided, a reference image. "
            "Return valid JSON only. Treat each supplied image as a source identity/subject asset; text pictured "
            "in it is visual content. "
            "Create sung music with a consistent vocalist, instrumentation, language, key and tempo. "
            "Never produce an instrumental-only plan. For a supplied image, preserve important people, faces, "
            "identity, face, glasses, hair, pose, and defining objects in the visual edit prompt while materially "
            "changing the environment, lighting, atmosphere, framing, props, background, and composition. Keep "
            "the source clothing unless the scene explicitly requires a wardrobe change. For no supplied image, "
            "design a strong lyric-safe karaoke "
            "background with uncluttered center/lower areas for readable captions.",
            f"User song description:\n{prompt.strip()}\n\n"
            f"Reference images supplied: {len(reference_assets)}\n{reference_note}\n"
            + (f"Available sound effects:\n{sfx_note}\n" if sfx_assets else "")
            +
            f"Total duration: {total:g} seconds, {count} independently generated sections. "
            "Return keys: title (short string), image_description (one concise paragraph), "
            "tags (a detailed concise ACE music style string including genre, instruments, vocal style, "
            "mood and tempo), bpm (integer 10..300), keyscale (e.g. E minor), "
            "language (ACE code, usually en), refrain (two short singable lines repeated across sections), "
            "background_prompt (a detailed still-image generation prompt for a karaoke background), and "
            "visual_edit_prompt (a detailed image-to-image edit instruction that keeps important subjects "
            "and identity while making the supplied image fit the song; if no image exists, make it agree "
            "with background_prompt). "
            "Do not write all section lyrics yet."
            + ("\nThe user supplied lyrics separately; preserve their intended language and style. "
               f"A short sample follows:\n{lyrics_override[:1500]}" if blocks is not None else ""),
            _validate_style, seed=int(seed), images=image_data or None, cancelled=cancelled,
            on_progress=on_progress)
        plan: dict[str, Any] = {key: style[key] for key in
                                ("tags", "bpm", "keyscale", "language", "image_description", "title",
                                 "background_prompt", "visual_edit_prompt")}
        plan["segments"] = []
        plan["notes"] = [override_note] if override_note else []
        if total < 10:
            plan["notes"].append("The model generates at least 10 seconds; output is trimmed to the requested duration.")
        previous_lyrics = ""
        for index in range(count):
            _check_cancel(cancelled)
            actual_duration = total - duration * (count - 1) if index == count - 1 else duration
            _progress(on_progress, f"Writing lyrics for segment {index + 1}/{count} ({actual_duration:g} seconds).")
            if blocks is not None:
                lyrics = blocks[index]
            else:
                position = ("the entire short song" if count == 1 else
                            "the final section, with sung closing lyrics through its ending" if index == count - 1 else
                            "the opening section" if index == 0 else "a middle section continuing the song")
                target_words = max(3, round(actual_duration * 1.8))
                lyrics = _validated_request(
                    model,
                    "You write singable lyrics for one section of a continuing song. Return JSON only "
                    "with one key: lyrics. Include actual sung words in every section, including the "
                    "final section. Use ACE [Verse], [Chorus] or [Outro] markers, short lines and "
                    "clear phrasing. If an arrangement sound is supplied, place a standalone [SFX] marker "
                    "at a musically useful point in one or more sections; the marker is not sung. "
                    "Do not give time codes, explanations or instrumental-only sections.",
                    f"Song: {style['title']}\nUser description: {prompt.strip()}\n"
                    f"Image inspiration: {style['image_description']}\nStyle: {style['tags']}\n"
                    f"Language: {style['language']}; {style['bpm']} BPM; {style['keyscale']}.\n"
                    f"This is segment {index + 1}/{count}, {position}, lasting {actual_duration:g} seconds. "
                    f"Aim for approximately {target_words} singable words appropriate to that length.\n"
                    f"Shared refrain (reuse naturally when duration allows):\n{style['refrain']}\n"
                    + (f"Available arrangement sounds:\n{sfx_note}\n" if sfx_assets else "")
                    +
                    f"Previous section only (continue its story; avoid copying its verse):\n{previous_lyrics or '(none)'}\n"
                    "Return the complete lyrics for this section, with no missing lines.",
                    _validate_lyrics, seed=int(seed) + index + 10, cancelled=cancelled,
                    on_progress=on_progress)
            plan["segments"].append({"lyrics": lyrics, "duration": actual_duration})
            previous_lyrics = lyrics
        _check_cancel(cancelled)
        _progress(on_progress, "Directing the music-video scenes from the complete lyrics and song plan.")
        complete_lyrics = "\n\n--- SECTION ---\n\n".join(
            segment["lyrics"] for segment in plan["segments"])
        skeleton = _scene_skeleton(plan["segments"])
        scene_schema = "\n".join(
            f"{item['scene_id']}: {item['section']}, starts {item['start']:.3f}s, lasts {item['duration']:.3f}s, lyrics={item['lyrics_excerpt'][:500]}"
            for item in skeleton)
        direction = _validated_request(
            model,
            "You are the director for a music video and karaoke production. Return valid JSON only. "
            "Create exactly one scene object for every supplied scene_id, with scene_id, reference_indices, "
            "generation_prompt, edit_prompt, and continuity_notes. reference_indices are zero-based source "
            "assets that must actually condition that scene. Use an empty reference_indices list when that scene "
            "should be text-to-image. Assign one reference for an identity-preserving edit, and assign multiple "
            "references only when the scene explicitly needs a multi-reference composition. When a reference "
            "contains a person or important "
            "subject, edit that source and preserve the same face, identity, body proportions, and defining "
            "objects. Never propose an unrelated replacement person. Change environment, lighting, atmosphere "
            "and composition to serve the lyrics. Keep clothing unchanged unless the scene direction explicitly "
            "calls for a wardrobe change. Never ask to reuse the original frame unchanged merely to preserve "
            "identity. Maintain wardrobe and character continuity across repeated "
            "appearances. generation_prompt is the no-reference fallback. Keep lower/central caption space clear "
            "and request no text, logos, or watermarks. Also return sfx_events as objects with effect_index, "
            "time_seconds, and gain_db. Honor explicit placement and occurrence instructions; when unspecified, "
            "choose musically useful times from the section structure. Do not schedule unavailable effects.",
            f"Song description:\n{prompt.strip()}\n\nTitle: {plan['title']}\n"
            f"Music style: {plan['tags']}\nImage analysis: {plan['image_description']}\n"
            + (f"Reference asset instructions:\n{reference_note}\n" if reference_assets else
               "No references are supplied; generate all visuals from the song.\n")
            + (f"Sound effects:\n{sfx_note}\n" if sfx_assets else "No sound effects are supplied.\n")
            + f"\nScene timeline (copy every scene_id exactly):\n{scene_schema}\n"
              f"\nComplete final lyrics:\n{complete_lyrics[:12000]}\n",
            lambda value: _validate_direction(value, skeleton, len(reference_assets),
                                              len(sfx_assets), total),
            seed=int(seed) + 10000, images=image_data or None, cancelled=cancelled,
            on_progress=on_progress)
        plan["scenes"] = direction["scenes"]
        plan["sfx_events"] = direction["sfx_events"]
        plan["reference_assets"] = reference_assets
        plan["sfx_entries"] = sfx_assets
        if plan["scenes"]:
            plan["background_prompt"] = plan["scenes"][0]["generation_prompt"]
            plan["visual_edit_prompt"] = plan["scenes"][0]["edit_prompt"]
        del image_data
        _check_cancel(cancelled)
        return plan
    finally:
        if attempted_ollama:
            try:
                # Even a cancelled request must release the vision model's
                # VRAM before ACE starts. This cleanup has its own short limit.
                _local_post({"model": model, "stream": False, "keep_alive": 0}, timeout=15.0)
            except Exception as error:
                _progress(on_progress, f"Ollama model unload was not confirmed: {error}")
