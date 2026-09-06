"""Disk-backed FLAC and karaoke MP4 export for segmented ComfyUI music.

Audio/video are streamed by FFmpeg, never materialized as full-track tensors.
Timing is an estimate unless the supplied words can be anchored to local
faster-whisper recognition; all interpolation is recorded in the timeline.
No models are downloaded and existing output files are never overwritten.
"""
from __future__ import annotations

import difflib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata

SAMPLE_RATE = 48000


class ExportCancelled(RuntimeError):
    pass


def find_ffmpeg(explicit=None):
    candidates = [explicit, os.environ.get("IMAGEIO_FFMPEG_EXE"), shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass
    candidates.extend([
        Path.home() / "Documents/ComfyUI/.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe",
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
    ])
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("FFmpeg with libass and libx264 is required. Install imageio-ffmpeg or set IMAGEIO_FFMPEG_EXE.")


def _check_cancel(cancelled):
    if cancelled and cancelled():
        raise ExportCancelled("Music export cancelled.")


def _run(args, cwd=None, cancelled=None):
    """Keep subprocess logs on disk and read only the last 12 KB on failure."""
    _check_cancel(cancelled)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [str(arg) for arg in args], cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            while process.poll() is None:
                _check_cancel(cancelled)
                time.sleep(0.15)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if process.returncode:
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - 12000))
            raise RuntimeError("FFmpeg failed: " + errors.read().decode("utf-8", errors="replace"))


def flac_info(path):
    """Read lossless sample count and precision without decoding the track."""
    with open(path, "rb") as stream:
        header = stream.read(42)
    if len(header) != 42 or header[:4] != b"fLaC" or header[4] & 127 != 0:
        raise RuntimeError(f"Invalid native FLAC file: {path}")
    packed = int.from_bytes(header[18:26], "big")
    rate = packed >> 44
    return {"sample_rate": rate, "channels": ((packed >> 41) & 7) + 1,
            "bits_per_sample": ((packed >> 36) & 31) + 1,
            "samples": packed & ((1 << 36) - 1),
            "duration": (packed & ((1 << 36) - 1)) / rate}


def _tokens(lyrics):
    if not isinstance(lyrics, str):
        raise ValueError("Each segment must include its actual lyrics as text.")
    result = []
    for line_index, line in enumerate(lyrics.splitlines()):
        if re.fullmatch(r"\s*\[[^\]]+\]\s*", line):
            continue
        for word in line.split():
            result.append({"text": word, "line": line_index})
    if not result:
        raise ValueError("Every segment needs non-empty lyrics; section labels alone are not lyrics.")
    return result


def _normal(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


def _estimated(tokens, duration):
    weights = [max(1, len(_normal(token["text"]))) for token in tokens]
    total = sum(weights)
    cursor = 0
    words = []
    for token, weight in zip(tokens, weights):
        start = duration * cursor / total
        cursor += weight
        words.append({**token, "start": start, "end": duration * cursor / total, "timing": "estimated"})
    return words


def _align(tokens, recognized, duration):
    """Anchor exact words, then interpolate every unmatched supplied word."""
    matcher = difflib.SequenceMatcher(
        None, [_normal(token["text"]) for token in tokens],
        [_normal(word["text"]) for word in recognized], autojunk=False,
    )
    anchors = {}
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            token_index, speech_index = block.a + i, block.b + i
            if not _normal(tokens[token_index]["text"]):
                continue
            word = recognized[speech_index]
            start = max(0.0, min(duration, word["start"]))
            end = max(start, min(duration, word["end"]))
            if end > start:
                anchors[token_index] = (start, end)
    ratio = len(anchors) / len(tokens)
    if ratio < 0.35:
        return _estimated(tokens, duration), {"mode": "estimated", "matched_words": len(anchors), "match_ratio": ratio}
    words = [{**token, "start": None, "end": None, "timing": "interpolated"} for token in tokens]
    for index, (start, end) in anchors.items():
        words[index].update(start=start, end=end, timing="recognized")
    boundaries = [-1] + sorted(anchors) + [len(tokens)]
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 2:
            continue
        begin = words[left]["end"] if left >= 0 else 0.0
        finish = words[right]["start"] if right < len(tokens) else duration
        if finish <= begin + 0.005 * (right - left - 1):
            # There is no honest space for missed words between these anchors.
            return _estimated(tokens, duration), {"mode": "estimated", "matched_words": len(anchors), "match_ratio": ratio, "reason": "Recognized anchors leave no time for all supplied lyrics."}
        interpolated = _estimated(tokens[left + 1:right], finish - begin)
        for index, item in enumerate(interpolated, left + 1):
            words[index].update(start=begin + item["start"], end=begin + item["end"])
    last_end = 0.0
    for word in words:
        if word["start"] < last_end - 0.05:
            return _estimated(tokens, duration), {"mode": "estimated", "matched_words": len(anchors), "match_ratio": ratio, "reason": "Recognition timings overlap or run backwards."}
        word["start"] = max(last_end, word["start"])
        word["end"] = max(word["start"], word["end"])
        last_end = word["end"]
    return words, {"mode": "whisper" if len(anchors) == len(tokens) else "whisper+interpolation",
                   "matched_words": len(anchors), "match_ratio": ratio}


def _local_model(whisper_model=None):
    failures = []
    try:
        from faster_whisper import WhisperModel
    except Exception as error:
        WhisperModel = None
        failures.append(str(error))
    candidates = [str(whisper_model)] if whisper_model else []
    cache = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    if not whisper_model:
        for name in ("base", "small.en", "small", "large-v2"):
            candidates.extend(str(path) for path in sorted((cache / f"models--Systran--faster-whisper-{name}" / "snapshots").glob("*")) if (path / "model.bin").exists())
    for candidate in candidates if WhisperModel else []:
        try:
            return WhisperModel(candidate, device="cpu", compute_type="int8", local_files_only=True)
        except Exception as error:
            failures.append(str(error))
    # Comfy's installed speech suite may keep OpenAI-format weights instead.
    pt_candidates = []
    if whisper_model and Path(whisper_model).suffix == ".pt":
        pt_candidates.append(Path(whisper_model))
    try:
        import folder_paths
        pt_candidates.append(Path(folder_paths.models_dir) / "stt/whisper/base.en.pt")
    except ImportError:
        pass
    pt_candidates.extend([Path.home() / "Documents/ComfyUI/models/stt/whisper/base.en.pt",
                          Path.home() / ".cache/whisper/base.en.pt"])
    for candidate in pt_candidates:
        if candidate.is_file():
            try:
                return _OpenAIWhisperAdapter(candidate)
            except Exception as error:
                failures.append(str(error))
    raise RuntimeError("Cannot load a downloaded local Whisper model: " + "; ".join(failures))


class _OpenAIWhisperAdapter:
    """Normalize the already-installed OpenAI Whisper interface, CPU only."""
    def __init__(self, model_path):
        import whisper
        self.model = whisper.load_model(str(model_path), device="cpu")

    def transcribe(self, filename, language=None, **options):
        import numpy as np
        from types import SimpleNamespace
        process = subprocess.run([find_ffmpeg(), "-v", "error", "-nostdin", "-i", filename,
                                  "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace")[-1500:])
        audio = np.frombuffer(process.stdout, dtype=np.float32).copy()
        result = self.model.transcribe(audio, language=language, word_timestamps=True,
                                       fp16=False, verbose=False, condition_on_previous_text=False)
        phrases = [SimpleNamespace(words=[SimpleNamespace(word=w["word"], start=w["start"], end=w["end"])
                                           for w in phrase.get("words", [])])
                   for phrase in result.get("segments", [])]
        return phrases, result.get("language")


def _ass_time(seconds):
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{fraction:02d}"


def _safe_ass(text):
    # Prevent supplied lyrics from becoming ASS override tags or line breaks.
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")


def _ass_header(width, height):
    font = max(18, round(height * 0.056))
    margin = max(16, round(height * 0.085))
    return f"""[Script Info]
Title: Image + Lyrics Music Karaoke
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,{font},&H0000DCFF,&H00FFFFFF,&H00181818,&H90000000,-1,0,0,0,100,100,0,0,1,3,1,2,{margin},{margin},{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_events(words, offset, duration):
    groups, group = [], []
    for word in words:
        length = sum(len(item["text"]) + 1 for item in group)
        if group and (len(group) >= 7 or length + len(word["text"]) > 42 or word["line"] != group[-1]["line"]):
            groups.append(group)
            group = []
        group.append(word)
    if group:
        groups.append(group)
    lines = []
    for index, group in enumerate(groups):
        start = 0.0 if index == 0 else group[0]["start"]
        end = groups[index + 1][0]["start"] if index + 1 < len(groups) else duration
        cursor = int(round(start * 100))
        parts = []
        for word in group:
            word_start = max(cursor, int(round(word["start"] * 100)))
            word_end = max(word_start, int(round(word["end"] * 100)))
            if word_start > cursor:
                parts.append("{\\k" + str(word_start - cursor) + "}\\h")
            parts.append("{\\kf" + str(word_end - word_start) + "}" + _safe_ass(word["text"]) + " ")
            cursor = word_end
        lines.append(f"Dialogue: 0,{_ass_time(offset + start)},{_ass_time(offset + end)},Karaoke,,0,0,0,," + "".join(parts).rstrip())
    return lines


def export_karaoke(segments, image_path, output_dir, basename="song", *,
                   width=1280, height=720, fps=24, alignment="auto",
                   whisper_model=None, language=None, ffmpeg_path=None,
                   on_progress=None, cancelled=None, fade_seconds=0.01):
    """Export dict segments with audio_path, lyrics, duration (seconds).

    Returns absolute output paths, duration, timing_mode and warnings.
    alignment='auto' uses available local recognition, recording fallback;
    'estimated' distributes supplied words by character count; 'required'
    raises if recognition cannot anchor at least 35% of any segment's words.
    on_progress(stage, fraction) is optional; cancelled() may stop FFmpeg.
    Each audio source must cover its requested duration (25 ms tolerance).
    Repeated sources and lyrics are normalized/aligned only once.
    Duration has no application cap; storage and compute remain finite.
    """
    if alignment not in {"auto", "estimated", "required"}:
        raise ValueError("alignment must be auto, estimated or required")
    if not re.fullmatch(r"[\w.-]+", basename) or basename in {".", ".."}:
        raise ValueError("basename must be a simple filename without directories")
    width, height, fps = int(width), int(height), int(fps)
    if min(width, height) < 64 or width % 2 or height % 2 or not 1 <= fps <= 60:
        raise ValueError("Use even video dimensions >=64 and a frame rate from 1 to 60")
    if not math.isfinite(fade_seconds) or not 0 <= fade_seconds <= 0.5:
        raise ValueError("fade_seconds must be between 0 and 0.5")
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    specs, accumulated, sample_cursor = [], 0.0, 0
    for index, segment in enumerate(segments):
        duration = float(segment["duration"])
        if not math.isfinite(duration) or duration < 0.05:
            raise ValueError(f"Segment {index + 1} duration must be a finite number >=0.05 seconds")
        source = Path(segment["audio_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        tokens = _tokens(segment.get("lyrics"))
        accumulated += duration
        end_sample = round(accumulated * SAMPLE_RATE)
        specs.append({"index": index + 1, "audio_path": str(source), "lyrics": segment["lyrics"],
                      "source_id": segment.get("source_id", str(source)), "tokens": tokens,
                      "offset_samples": sample_cursor, "samples": end_sample - sample_cursor,
                      "duration": (end_sample - sample_cursor) / SAMPLE_RATE})
        sample_cursor = end_sample
    if not specs:
        raise ValueError("At least one music segment is required")
    total_duration = sample_cursor / SAMPLE_RATE
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {kind + "_path": output_dir / f"{basename}.{extension}"
                    for kind, extension in (("flac", "flac"), ("mp4", "mp4"), ("ass", "ass"), ("timeline", "timeline.json"))}
    for destination in destinations.values():
        if destination.exists():
            raise FileExistsError(f"Choose a new output name; this file already exists: {destination}")
    ffmpeg = find_ffmpeg(ffmpeg_path)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    warnings, model, timeline = [], None, []

    def report(stage, fraction):
        _check_cancel(cancelled)
        if on_progress:
            on_progress(stage, fraction)

    if alignment != "estimated":
        report("Loading local lyric timing model", 0.01)
        try:
            model = _local_model(whisper_model)
        except Exception as error:
            if alignment == "required":
                raise RuntimeError(f"Lyric alignment is required: {error}") from error
            warnings.append(f"Word timing is estimated: {error}")
    else:
        warnings.append("Word timing is estimated from supplied lyrics and segment durations; it is not vocal alignment.")

    with tempfile.TemporaryDirectory(prefix=".karaoke-", dir=output_dir) as temporary:
        scratch = Path(temporary)
        normalized_cache, alignment_cache, concat = {}, {}, []
        subtitle = [_ass_header(width, height)]
        for spec in specs:
            report(f"Preparing segment {spec['index']}/{len(specs)}", 0.05 + 0.45 * (spec["index"] - 1) / len(specs))
            key = (spec["audio_path"], spec["samples"])
            if key not in normalized_cache:
                normalized = scratch / f"segment-{len(normalized_cache):06d}.flac"
                filters = [f"aresample={SAMPLE_RATE}", f"atrim=end_sample={spec['samples']}", "asetpts=N/SR/TB"]
                if fade_seconds:
                    fade = min(fade_seconds, spec["duration"] / 4)
                    filters += [f"afade=t=in:st=0:d={fade:.8f}", f"afade=t=out:st={spec['duration'] - fade:.8f}:d={fade:.8f}"]
                _run(base + ["-i", spec["audio_path"], "-map", "0:a:0", "-vn", "-af", ",".join(filters),
                             "-ac", "2", "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", str(normalized)], cancelled=cancelled)
                actual = flac_info(normalized)["samples"]
                shortfall = spec["samples"] - actual
                if shortfall > round(0.025 * SAMPLE_RATE):
                    raise ValueError(f"Segment {spec['index']} has only {actual / SAMPLE_RATE:.3f}s of audio for a {spec['duration']:.3f}s request. Generate more audio; missing music will not be silently padded.")
                if shortfall > 0:
                    padded = scratch / f"segment-{len(normalized_cache):06d}-exact.flac"
                    _run(base + ["-i", str(normalized), "-af", f"apad,atrim=end_sample={spec['samples']}",
                                 "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", str(padded)], cancelled=cancelled)
                    normalized = padded
                normalized_cache[key] = normalized
            normalized = normalized_cache[key]
            concat.append(f"file '{normalized.name}'")
            timing_key = (key, spec["lyrics"])
            if timing_key not in alignment_cache:
                words = _estimated(spec["tokens"], spec["duration"])
                stats = {"mode": "estimated", "matched_words": 0, "match_ratio": 0.0}
                if model:
                    report(f"Timing lyrics for segment {spec['index']}/{len(specs)}", 0.08 + 0.45 * (spec["index"] - 1) / len(specs))
                    try:
                        recognized = []
                        speech, _info = model.transcribe(str(normalized), language=language or None,
                                                        word_timestamps=True, vad_filter=False, beam_size=3,
                                                        condition_on_previous_text=False)
                        for phrase in speech:
                            _check_cancel(cancelled)
                            for word in phrase.words or []:
                                if word.word.strip():
                                    recognized.append({"text": word.word.strip(), "start": float(word.start), "end": float(word.end)})
                        words, stats = _align(spec["tokens"], recognized, spec["duration"])
                        if stats["mode"] == "estimated":
                            stats.setdefault("reason", "Too few supplied words matched the vocal recognition.")
                    except ExportCancelled:
                        raise
                    except Exception as error:
                        _check_cancel(cancelled)
                        stats["reason"] = str(error)
                if alignment == "required" and stats["mode"] == "estimated":
                    raise RuntimeError(f"Segment {spec['index']} could not be aligned: {stats.get('reason', 'No reliable vocal matches')}")
                alignment_cache[timing_key] = (words, stats)
            words, stats = alignment_cache[timing_key]
            if model and stats["mode"] != "whisper":
                warnings.append(f"Segment {spec['index']}: {stats['mode']} lyric timing ({stats['matched_words']}/{len(words)} recognized words). " + stats.get("reason", "Unrecognized words use interpolated times."))
            offset = spec["offset_samples"] / SAMPLE_RATE
            timeline.append({key: value for key, value in spec.items() if key != "tokens"} |
                            {"offset": offset, "timing": stats,
                             "words": [{**word, "start": word["start"] + offset, "end": word["end"] + offset} for word in words]})
            subtitle.extend(_ass_events(words, offset, spec["duration"]))
        # Do not retain the recognition model during video encoding.
        del model
        (scratch / "segments.txt").write_text("\n".join(concat) + "\n", encoding="utf-8")
        (scratch / "lyrics.ass").write_text("\n".join(subtitle) + "\n", encoding="utf-8-sig")
        report("Encoding 24-bit FLAC", 0.55)
        _run(base + ["-f", "concat", "-safe", "0", "-i", "segments.txt", "-map", "0:a:0",
                     "-af", f"atrim=end_sample={sample_cursor},asetpts=N/SR/TB", "-c:a", "flac",
                     "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "audio.flac"], cwd=scratch, cancelled=cancelled)
        audio_info = flac_info(scratch / "audio.flac")
        if audio_info["samples"] != sample_cursor or audio_info["bits_per_sample"] != 24:
            raise RuntimeError(f"FLAC validation failed: expected {sample_cursor} samples at 24 bits; got {audio_info}")
        report("Encoding karaoke MP4", 0.65)
        video_filter = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                        f"drawbox=x=0:y=ih*0.70:w=iw:h=ih*0.30:color=black@0.28:t=fill,ass=lyrics.ass")
        _run(base + ["-loop", "1", "-framerate", str(fps), "-i", str(image_path), "-i", "audio.flac",
                     "-map", "0:v:0", "-map", "1:a:0", "-vf", video_filter,
                     "-c:v", "libx264", "-preset", "fast", "-tune", "stillimage", "-crf", "20",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE),
                     "-t", f"{total_duration:.8f}", "-movflags", "+faststart", "video.mp4"], cwd=scratch, cancelled=cancelled)
        modes = sorted({item["timing"]["mode"] for item in timeline})
        result = {key: str(value) for key, value in destinations.items()} | {
            "duration": total_duration, "timing_mode": ", ".join(modes), "warnings": warnings,
            "sample_rate": SAMPLE_RATE, "channels": 2, "bits_per_sample": 24, "samples": sample_cursor,
        }
        document = {"schema_version": 1, **result, "video": {"width": width, "height": height, "fps": fps},
                    "timing_note": "Recognition timestamps and interpolated highlights are approximate; inspect playback for sung-word synchronization.",
                    "segments": timeline}
        (scratch / "timeline.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        report("Saving music and karaoke files", 0.98)
        for source, key in (("audio.flac", "flac_path"), ("video.mp4", "mp4_path"), ("lyrics.ass", "ass_path"), ("timeline.json", "timeline_path")):
            # Re-check after the long render; never replace somebody else's result.
            if destinations[key].exists():
                raise FileExistsError(destinations[key])
            os.rename(scratch / source, destinations[key])
    report("Export complete", 1.0)
    return result


def export_directed_package(segments, scenes, output_dir, basename="song", *,
                            width=1280, height=720, fps=24, alignment="auto",
                            whisper_model=None, language=None, ffmpeg_path=None,
                            on_progress=None, cancelled=None, fade_seconds=0.01):
    """Export full/karaoke mixes and two scene-sequenced H.264 videos.

    Word recognition always listens to ``audio_path``: the final full mix that
    contains the final lead vocal after optional conversion and SFX insertion.
    ``karaoke_audio_path`` is muxed only after those timestamps are established.
    """
    if alignment not in {"auto", "estimated", "required"}:
        raise ValueError("alignment must be auto, estimated or required")
    if not re.fullmatch(r"[\w.-]+", basename) or basename in {".", ".."}:
        raise ValueError("basename must be a simple filename without directories")
    width, height, fps = int(width), int(height), int(fps)
    if min(width, height) < 64 or width % 2 or height % 2 or not 1 <= fps <= 60:
        raise ValueError("Use even video dimensions >=64 and a frame rate from 1 to 60")
    specs, accumulated, sample_cursor = [], 0.0, 0
    for index, segment in enumerate(segments):
        duration = float(segment["duration"])
        full_source = Path(segment["audio_path"]).resolve()
        karaoke_source = Path(segment.get("karaoke_audio_path", "")).resolve()
        if not full_source.is_file() or not karaoke_source.is_file():
            raise FileNotFoundError(f"Final full/karaoke section files are missing for segment {index + 1}")
        accumulated += duration
        end_sample = round(accumulated * SAMPLE_RATE)
        specs.append({"index": index + 1, "audio_path": str(full_source),
                      "karaoke_audio_path": str(karaoke_source),
                      "lyrics": segment["lyrics"], "tokens": _tokens(segment["lyrics"]),
                      "offset_samples": sample_cursor, "samples": end_sample - sample_cursor,
                      "duration": (end_sample - sample_cursor) / SAMPLE_RATE})
        sample_cursor = end_sample
    if not specs:
        raise ValueError("At least one music segment is required")
    total_duration = sample_cursor / SAMPLE_RATE
    scene_specs = []
    scene_cursor = 0.0
    for index, scene in enumerate(scenes):
        path = Path(str(scene.get("image_path", ""))).resolve()
        duration = float(scene.get("duration", 0))
        if not path.is_file() or duration <= 0:
            raise ValueError(f"Visual scene {index + 1} has no usable image or duration")
        scene_specs.append({**scene, "image_path": str(path), "start": scene_cursor,
                            "duration": min(duration, max(0.001, total_duration - scene_cursor))})
        scene_cursor += duration
        if scene_cursor >= total_duration - 0.001:
            break
    if not scene_specs:
        raise ValueError("At least one visual scene is required")
    if scene_cursor < total_duration - 0.025:
        scene_specs[-1]["duration"] += total_duration - scene_cursor

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    music_video_name = "music_video.mp4" if basename == "song" else f"{basename}_music_video.mp4"
    karaoke_video_name = "karaoke_video.mp4" if basename == "song" else f"{basename}_karaoke_video.mp4"
    destinations = {
        "flac_path": output_dir / f"{basename}.flac",
        "mp3_path": output_dir / f"{basename}.mp3",
        "music_video_path": output_dir / music_video_name,
        "karaoke_video_path": output_dir / karaoke_video_name,
        "ass_path": output_dir / f"{basename}.ass",
        "timeline_path": output_dir / f"{basename}_timeline.json",
    }
    for destination in destinations.values():
        if destination.exists():
            raise FileExistsError(f"Choose a new output name; this file already exists: {destination}")
    ffmpeg = find_ffmpeg(ffmpeg_path)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    warnings, model, timeline = [], None, []

    def report(stage, fraction):
        _check_cancel(cancelled)
        if on_progress:
            on_progress(stage, fraction)

    if alignment != "estimated":
        report("Loading local lyric timing model", 0.01)
        try:
            model = _local_model(whisper_model)
        except Exception as error:
            if alignment == "required":
                raise RuntimeError(f"Lyric alignment is required: {error}") from error
            warnings.append(f"Word timing is estimated: {error}")
    else:
        warnings.append("Word timing is estimated from supplied lyrics and segment durations.")

    with tempfile.TemporaryDirectory(prefix=".directed-karaoke-", dir=output_dir) as temporary:
        scratch = Path(temporary)
        full_concat, karaoke_concat = [], []
        subtitle = [_ass_header(width, height)]
        for spec in specs:
            report(f"Preparing final mixes {spec['index']}/{len(specs)}", 0.05 + 0.32 * (spec["index"] - 1) / len(specs))
            normalized = {}
            for kind, source_key in (("full", "audio_path"), ("karaoke", "karaoke_audio_path")):
                target = scratch / f"{kind}-{spec['index']:05d}.flac"
                filters = [f"aresample={SAMPLE_RATE}", f"atrim=end_sample={spec['samples']}", "asetpts=N/SR/TB"]
                if fade_seconds:
                    fade = min(fade_seconds, spec["duration"] / 4)
                    filters += [f"afade=t=in:st=0:d={fade:.8f}",
                                f"afade=t=out:st={spec['duration'] - fade:.8f}:d={fade:.8f}"]
                _run(base + ["-i", spec[source_key], "-map", "0:a:0", "-vn", "-af", ",".join(filters),
                             "-ac", "2", "-c:a", "flac", "-sample_fmt", "s32",
                             "-bits_per_raw_sample", "24", str(target)], cancelled=cancelled)
                actual = flac_info(target)["samples"]
                if spec["samples"] - actual > round(0.025 * SAMPLE_RATE):
                    raise ValueError(f"Segment {spec['index']} {kind} mix is shorter than requested")
                if actual < spec["samples"]:
                    exact = scratch / f"{kind}-{spec['index']:05d}-exact.flac"
                    _run(base + ["-i", str(target), "-af", f"apad,atrim=end_sample={spec['samples']}",
                                 "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24",
                                 str(exact)], cancelled=cancelled)
                    target = exact
                normalized[kind] = target
            full_concat.append(f"file '{normalized['full'].name}'")
            karaoke_concat.append(f"file '{normalized['karaoke'].name}'")
            words = _estimated(spec["tokens"], spec["duration"])
            stats = {"mode": "estimated", "matched_words": 0, "match_ratio": 0.0}
            if model:
                report(f"Aligning final lead vocal {spec['index']}/{len(specs)}", 0.08 + 0.32 * (spec["index"] - 1) / len(specs))
                try:
                    recognized = []
                    speech, _info = model.transcribe(str(normalized["full"]), language=language or None,
                                                     word_timestamps=True, vad_filter=False, beam_size=3,
                                                     condition_on_previous_text=False)
                    for phrase in speech:
                        for word in phrase.words or []:
                            if word.word.strip():
                                recognized.append({"text": word.word.strip(), "start": float(word.start),
                                                   "end": float(word.end)})
                    words, stats = _align(spec["tokens"], recognized, spec["duration"])
                except ExportCancelled:
                    raise
                except Exception as error:
                    stats["reason"] = str(error)
            if alignment == "required" and stats["mode"] == "estimated":
                raise RuntimeError(f"Segment {spec['index']} could not be aligned against the final lead vocal")
            if model and stats["mode"] != "whisper":
                warnings.append(f"Segment {spec['index']}: {stats['mode']} lyric timing "
                                f"({stats['matched_words']}/{len(words)} recognized words).")
            offset = spec["offset_samples"] / SAMPLE_RATE
            timeline.append({key: value for key, value in spec.items() if key != "tokens"} |
                            {"offset": offset, "timing": stats,
                             "words": [{**word, "start": word["start"] + offset,
                                        "end": word["end"] + offset} for word in words]})
            subtitle.extend(_ass_events(words, offset, spec["duration"]))
        del model
        (scratch / "full.txt").write_text("\n".join(full_concat) + "\n", encoding="utf-8")
        (scratch / "karaoke.txt").write_text("\n".join(karaoke_concat) + "\n", encoding="utf-8")
        (scratch / "lyrics.ass").write_text("\n".join(subtitle) + "\n", encoding="utf-8-sig")
        for listing, target in (("full.txt", "song.flac"), ("karaoke.txt", "karaoke.flac")):
            _run(base + ["-f", "concat", "-safe", "0", "-i", listing, "-map", "0:a:0",
                         "-af", f"atrim=end_sample={sample_cursor},asetpts=N/SR/TB", "-c:a", "flac",
                         "-sample_fmt", "s32", "-bits_per_raw_sample", "24", target],
                 cwd=scratch, cancelled=cancelled)
        audio_info = flac_info(scratch / "song.flac")
        karaoke_info = flac_info(scratch / "karaoke.flac")
        if audio_info["samples"] != sample_cursor or karaoke_info["samples"] != sample_cursor:
            raise RuntimeError("Final full or karaoke mix has an incorrect duration")
        report("Encoding lossless FLAC and 320 kbps MP3", 0.48)
        _run(base + ["-i", "song.flac", "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "320k",
                     "song.mp3"], cwd=scratch, cancelled=cancelled)

        report("Encoding the directed visual sequence", 0.58)
        clip_files = []
        scale_filter = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps}")
        for index, scene in enumerate(scene_specs):
            clip = f"scene-{index:04d}.mp4"
            _run(base + ["-loop", "1", "-framerate", str(fps), "-i", scene["image_path"],
                         "-vf", scale_filter, "-t", f"{scene['duration']:.8f}", "-an", "-c:v", "libx264",
                         "-preset", "fast", "-tune", "stillimage", "-crf", "20", "-pix_fmt", "yuv420p", clip],
                 cwd=scratch, cancelled=cancelled)
            clip_files.append(f"file '{clip}'")
        (scratch / "visuals.txt").write_text("\n".join(clip_files) + "\n", encoding="utf-8")
        _run(base + ["-f", "concat", "-safe", "0", "-i", "visuals.txt", "-c", "copy", "visuals.mp4"],
             cwd=scratch, cancelled=cancelled)
        report("Encoding full-song music video", 0.72)
        _run(base + ["-i", "visuals.mp4", "-i", "song.flac", "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE),
                     "-t", f"{total_duration:.8f}", "-movflags", "+faststart", "music_video.mp4"],
             cwd=scratch, cancelled=cancelled)
        report("Encoding lyric-timed karaoke video", 0.84)
        karaoke_filter = "drawbox=x=0:y=ih*0.70:w=iw:h=ih*0.30:color=black@0.28:t=fill,ass=lyrics.ass"
        _run(base + ["-i", "visuals.mp4", "-i", "karaoke.flac", "-map", "0:v:0", "-map", "1:a:0",
                     "-vf", karaoke_filter, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE),
                     "-t", f"{total_duration:.8f}", "-movflags", "+faststart", "karaoke_video.mp4"],
             cwd=scratch, cancelled=cancelled)
        modes = sorted({item["timing"]["mode"] for item in timeline})
        result = {key: str(value) for key, value in destinations.items()} | {
            "duration": total_duration, "timing_mode": ", ".join(modes), "warnings": warnings,
            "sample_rate": SAMPLE_RATE, "channels": 2, "bits_per_sample": 24,
            "samples": sample_cursor,
        }
        document = {"schema_version": 2, **result,
                    "video": {"width": width, "height": height, "fps": fps},
                    "alignment_source": "final full mix after voice conversion and SFX insertion",
                    "segments": timeline, "scenes": scene_specs}
        (scratch / "timeline.json").write_text(json.dumps(document, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        report("Saving the four requested outputs", 0.97)
        for source, key in (("song.flac", "flac_path"), ("song.mp3", "mp3_path"),
                            ("music_video.mp4", "music_video_path"),
                            ("karaoke_video.mp4", "karaoke_video_path"),
                            ("lyrics.ass", "ass_path"), ("timeline.json", "timeline_path")):
            if destinations[key].exists():
                raise FileExistsError(destinations[key])
            os.rename(scratch / source, destinations[key])
    report("Export complete", 1.0)
    return result
