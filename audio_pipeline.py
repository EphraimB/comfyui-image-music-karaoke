"""Modular post-processing for vocals, arrangement sounds, and mastering.

All processing happens before karaoke alignment.  The renderer passes the
resulting section files to exporter.py, so Whisper always listens to the final
voice-converted and SFX-mixed performance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


SFX_MARKER = re.compile(r"^\s*\[SFX(?:\s*(?:@|:)\s*([^\]]+))?\]\s*$", re.IGNORECASE)


def strip_sfx_markers(lyrics: str) -> str:
    """Remove arrangement directions before lyrics are sent to ACE-Step."""
    return "\n".join(line for line in str(lyrics).splitlines() if not SFX_MARKER.match(line))


def _marker_offset(detail: str, fallback: float, duration: float) -> float:
    detail = str(detail or "")
    percent = re.search(r"@?\s*(\d+(?:\.\d+)?)\s*%", detail)
    seconds = re.search(r"(?:^|@)\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b", detail, re.IGNORECASE)
    if percent:
        return duration * float(percent.group(1)) / 100.0
    if seconds:
        return float(seconds.group(1))
    return fallback


def explicit_sfx_offsets(lyrics: str, duration: float) -> list[float]:
    lines = [line.strip() for line in str(lyrics).splitlines() if line.strip()]
    offsets = []
    for index, line in enumerate(lines):
        match = SFX_MARKER.match(line)
        if match:
            fallback = duration * (index + 0.5) / max(1, len(lines))
            offsets.append(max(0.0, min(duration, _marker_offset(match.group(1), fallback, duration))))
    return offsets


def plan_sfx_events(segments: list[dict], enabled: bool) -> dict[int, list[float]]:
    """Map explicit markers, or add conservative whole-song automatic cues."""
    if not enabled:
        return {}
    events = {index: explicit_sfx_offsets(item.get("lyrics", ""), float(item["duration"]))
              for index, item in enumerate(segments)}
    if any(events.values()):
        return {index: values for index, values in events.items() if values}

    durations = [float(item["duration"]) for item in segments]
    total = sum(durations)
    fractions = [0.62] if total <= 90 else ([0.32, 0.76] if total <= 240 else [0.2, 0.5, 0.82])
    automatic: dict[int, list[float]] = {}
    for fraction in fractions:
        absolute = total * fraction
        elapsed = 0.0
        for index, duration in enumerate(durations):
            if absolute <= elapsed + duration or index == len(durations) - 1:
                automatic.setdefault(index, []).append(max(0.0, min(duration, absolute - elapsed)))
                break
            elapsed += duration
    return automatic


def build_sfx_schedule(plan: dict) -> dict[int, list[dict]]:
    """Resolve director events and explicit lyric markers into segment-local cues."""
    entries = list(plan.get("sfx_entries") or [])
    if not entries:
        return {}
    segments = list(plan.get("segments") or [])
    starts, cursor = [], 0.0
    for segment in segments:
        starts.append(cursor)
        cursor += float(segment["duration"])
    events = []
    explicitly_used = set()
    sequential = 0
    for segment_index, segment in enumerate(segments):
        lines = [line.strip() for line in str(segment.get("lyrics", "")).splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            match = SFX_MARKER.match(line)
            if not match:
                continue
            detail = str(match.group(1) or "").strip()
            effect_index = sequential % len(entries)
            descriptor = re.sub(r"@?\s*\d+(?:\.\d+)?\s*(?:s|sec|second|%)\w*", "", detail,
                                flags=re.IGNORECASE)
            for candidate, entry in enumerate(entries):
                description = str(entry.get("description") or "").casefold()
                if descriptor:
                    words = [word for word in re.findall(r"[a-z0-9]+", descriptor.casefold()) if len(word) > 2]
                    if words and any(word in description for word in words):
                        effect_index = candidate
                        break
            fallback = float(segment["duration"]) * (line_index + 0.5) / max(1, len(lines))
            offset = _marker_offset(detail, fallback, float(segment["duration"]))
            events.append({"effect_index": effect_index,
                           "time_seconds": starts[segment_index] + max(0.0, min(float(segment["duration"]), offset)),
                           "gain_db": float(entries[effect_index].get("gain_db", -18.0))})
            explicitly_used.add(effect_index)
            sequential += 1
    for event in plan.get("sfx_events") or []:
        index = int(event.get("effect_index", -1)) if isinstance(event, dict) else -1
        if 0 <= index < len(entries) and index not in explicitly_used:
            events.append({"effect_index": index,
                           "time_seconds": float(event.get("time_seconds", 0.0)),
                           "gain_db": float(event.get("gain_db", entries[index].get("gain_db", -18.0)))})
    scheduled_indices = {event["effect_index"] for event in events}
    for index, entry in enumerate(entries):
        if index in scheduled_indices:
            continue
        placement = str(entry.get("placement") or "automatic")
        occurrences = str(entry.get("occurrences") or "automatic").casefold()
        number = re.search(r"\d+", occurrences)
        count = max(1, min(12, int(number.group()) if number else
                           2 if "twice" in occurrences else 3 if "three" in occurrences else 1))
        seconds = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second)", placement, re.I)]
        absolute_times = seconds[:count]
        if not absolute_times:
            matching = [scene for scene in plan.get("scenes") or []
                        if str(scene.get("section", "")).casefold() in placement.casefold()]
            if matching:
                absolute_times = [float(scene["start"]) + float(scene["duration"]) * 0.5
                                  for scene in matching[:count]]
        if not absolute_times:
            absolute_times = [cursor * (position + 1) / (count + 1) for position in range(count)]
        events.extend({"effect_index": index, "time_seconds": max(0.0, min(cursor, at)),
                       "gain_db": float(entry.get("gain_db", -18.0))}
                      for at in absolute_times)
    result: dict[int, list[dict]] = {}
    for event in sorted(events, key=lambda item: item["time_seconds"]):
        absolute = event["time_seconds"]
        for segment_index, (start, segment) in enumerate(zip(starts, segments)):
            duration = float(segment["duration"])
            if absolute <= start + duration or segment_index == len(segments) - 1:
                local = max(0.0, min(duration, absolute - start))
                cue = {**event, "offset": local}
                if not any(existing["effect_index"] == cue["effect_index"] and
                           abs(existing["offset"] - cue["offset"]) < 0.15
                           for existing in result.get(segment_index, [])):
                    result.setdefault(segment_index, []).append(cue)
                break
    return result


def _channels(audio, target_rate: int, target_samples: int | None = None):
    import torch
    import torch.nn.functional as functional

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Expected a ComfyUI AUDIO value with waveform and sample_rate.")
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError("Audio waveform is empty or has an unsupported shape.")
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    source_rate = int(audio["sample_rate"])
    if source_rate <= 0:
        raise ValueError("Audio sample rate must be positive.")
    if source_rate != target_rate:
        output_samples = max(1, round(waveform.shape[-1] * target_rate / source_rate))
        waveform = functional.interpolate(waveform.unsqueeze(0), size=output_samples,
                                          mode="linear", align_corners=False)[0]
    if target_samples is not None:
        if waveform.shape[-1] < target_samples:
            waveform = functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
        waveform = waveform[:, :target_samples]
    return waveform


def as_audio(waveform, sample_rate: int) -> dict:
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _rms(waveform) -> float:
    import torch
    return float(torch.sqrt(torch.mean(waveform.square()) + 1e-12))


def recombine_voice(instrumental, converted_vocal, original_vocal, *, sample_rate: int,
                    target_samples: int) -> dict:
    inst = _channels(instrumental, sample_rate, target_samples)
    converted = _channels(converted_vocal, sample_rate, target_samples)
    original = _channels(original_vocal, sample_rate, target_samples)
    gain = max(0.4, min(2.5, _rms(original) / max(_rms(converted), 1e-5)))
    return as_audio(inst + converted * gain, sample_rate)


def split_lead_and_backing(vocal_audio, *, sample_rate: int,
                           target_samples: int) -> tuple[dict, dict]:
    """Approximate centered lead versus stereo backing from the Demucs vocal stem."""
    vocal = _channels(vocal_audio, sample_rate, target_samples)
    center = vocal.mean(dim=0, keepdim=True).repeat(2, 1)
    side = vocal - center
    # Stereo-side material is usually harmony/double-track ambience. Suppress
    # tiny separation residue so a centered lead does not leak into karaoke.
    if _rms(side) < _rms(vocal) * 0.08:
        side.zero_()
    return as_audio(center, sample_rate), as_audio(side * 0.85, sample_rate)


def mix_sfx(song_audio, effect_audio, offsets: list[float], *, sample_rate: int,
            target_samples: int, level_db: float = -18.0, duck_db: float = -3.0) -> dict:
    import torch

    song = _channels(song_audio, sample_rate, target_samples).clone()
    effect = _channels(effect_audio, sample_rate)
    effect_rms = _rms(effect)
    if effect_rms <= 1e-6:
        return as_audio(song, sample_rate)
    desired_rms = 10.0 ** (float(level_db) / 20.0)
    effect = effect * min(8.0, desired_rms / effect_rms)
    fade = min(effect.shape[-1] // 3, max(1, round(sample_rate * 0.015)))
    if fade > 1:
        effect[:, :fade] *= torch.linspace(0, 1, fade)
        effect[:, -fade:] *= torch.linspace(1, 0, fade)
    duck = 10.0 ** (float(duck_db) / 20.0)
    for offset in offsets:
        start = max(0, min(target_samples - 1, round(float(offset) * sample_rate)))
        end = min(target_samples, start + effect.shape[-1])
        if end <= start:
            continue
        chunk = effect[:, :end - start]
        song[:, start:end] = song[:, start:end] * duck + chunk
    return as_audio(song, sample_rate)


def generate_described_sfx(description: str, duration: float, seed: int,
                           sample_rate: int = 48000) -> dict:
    """Offline procedural fallback when no compatible neural SFX engine is connected."""
    import torch
    import torch.nn.functional as functional

    duration = max(0.5, min(60.0, float(duration)))
    samples = max(1, round(duration * sample_rate))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 63 - 1))
    time_axis = torch.arange(samples, dtype=torch.float32) / sample_rate
    text = str(description or "ambient sound").casefold()
    noise = torch.randn(samples, generator=generator) * 0.12
    if any(word in text for word in ("ocean", "wave", "wind", "rain", "ambient")):
        kernel = torch.ones(1, 1, max(8, sample_rate // 1200))
        signal = functional.conv1d(noise[None, None], kernel / kernel.numel(), padding="same")[0, 0]
        signal *= 0.55 + 0.45 * torch.sin(2 * math.pi * 0.17 * time_axis).square()
    elif any(word in text for word in ("train", "horn", "siren")):
        base = 185.0 if "train" in text else 420.0
        signal = sum(torch.sin(2 * math.pi * base * multiple * time_axis) / multiple
                     for multiple in (1.0, 1.5, 2.0, 3.0)) * 0.09
        signal *= torch.sin(math.pi * torch.clamp(time_axis / duration, 0, 1)).sqrt()
    elif any(word in text for word in ("laugh", "laughter", "giggle")):
        signal = torch.zeros_like(time_axis)
        bursts = max(2, round(duration * 2.5))
        for index in range(bursts):
            center = duration * (index + 0.6) / (bursts + 0.2)
            envelope = torch.exp(-((time_axis - center) / 0.075).square())
            pitch = 170 + 45 * (index % 3)
            signal += envelope * (torch.sin(2 * math.pi * pitch * time_axis) + noise * 2.5) * 0.13
    elif any(word in text for word in ("crowd", "cheer", "applause", "clap")):
        kernel = torch.ones(1, 1, max(3, sample_rate // 5000))
        signal = functional.conv1d(noise[None, None], kernel / kernel.numel(), padding="same")[0, 0]
        for center in torch.linspace(0.1, max(0.1, duration - 0.1), max(3, round(duration * 4))):
            signal += torch.exp(-((time_axis - center) / 0.018).square()) * 0.12
    else:
        kernel = torch.ones(1, 1, max(4, sample_rate // 2400))
        signal = functional.conv1d(noise[None, None], kernel / kernel.numel(), padding="same")[0, 0]
    fade = min(samples // 3, max(1, round(sample_rate * 0.04)))
    signal[:fade] *= torch.linspace(0, 1, fade)
    signal[-fade:] *= torch.linspace(1, 0, fade)
    peak = max(float(signal.abs().max()), 1e-6)
    stereo = (signal / peak * 0.75).repeat(2, 1)
    return as_audio(stereo, sample_rate)


def master_audio(audio, *, sample_rate: int, target_samples: int,
                 target_rms_db: float = -15.0, peak_db: float = -1.0) -> dict:
    """Apply DC removal, conservative RMS normalization, and peak limiting."""
    import torch

    waveform = _channels(audio, sample_rate, target_samples)
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    desired_rms = 10.0 ** (float(target_rms_db) / 20.0)
    gain = max(0.25, min(4.0, desired_rms / max(_rms(waveform), 1e-6)))
    waveform = waveform * gain
    ceiling = 10.0 ** (float(peak_db) / 20.0)
    peak = float(waveform.abs().max())
    if peak > ceiling:
        waveform = torch.tanh(waveform / ceiling) * ceiling / math.tanh(1.0)
        peak = float(waveform.abs().max())
        if peak > ceiling:
            waveform *= ceiling / peak
    return as_audio(waveform, sample_rate)


@dataclass
class VocalMixBackend:
    """Replaceable Demucs lead/backing split with optional RVC lead conversion."""

    call_node: object
    rvc_model: dict | None = None
    pitch: int = 0
    _separator: object = None
    _engine: object = None
    _converter: object = None

    def load(self) -> None:
        self._separator = self.call_node("Demucs_Loader", d_model="htdemucs",
                                        overlap=0.25, shifts=1, split=True)[0]
        if self.rvc_model is not None:
            self._engine = self.call_node("RVCEngineNode", pitch=int(self.pitch), index_ratio=0.75,
                                          consonant_protection=0.25, volume_envelope=0.25,
                                          hubert_model="content-vec-best: Content Vec 768 (Recommended)",
                                          output_sample_rate=0, enable_custom_chunking=False,
                                          device="auto")[0]

    def process(self, music_audio, *, sample_rate: int,
                target_samples: int) -> tuple[dict, dict, str]:
        if self._separator is None:
            self.load()
        separated = self.call_node("Demucs_Sampler", model=self._separator, audio=music_audio,
                                   ext="flac", bits_per_sample=24, as_float="float32",
                                   clip_mode="rescale", mp3_bitrate=320,
                                   audio_save=False, preset=2)
        instrumental, original_vocal = separated[0], separated[4]
        inst = _channels(instrumental, sample_rate, target_samples)
        lead, backing = split_lead_and_backing(original_vocal, sample_rate=sample_rate,
                                               target_samples=target_samples)
        lead_wave = _channels(lead, sample_rate, target_samples)
        backing_wave = _channels(backing, sample_rate, target_samples)
        info = "Lead vocal separated; original lead retained."
        if self.rvc_model is not None:
            converted, conversion_info = self.call_node(
                "UnifiedVoiceChangerNode", TTS_engine=self._engine,
                source_audio=lead, narrator_target=self.rvc_model,
                refinement_passes=1, max_chunk_duration=30, chunk_method="smart")[:2]
            converted_wave = _channels(converted, sample_rate, target_samples)
            gain = max(0.4, min(2.5, _rms(lead_wave) / max(_rms(converted_wave), 1e-5)))
            lead_wave = converted_wave * gain
            info = str(conversion_info)
        full = as_audio(inst + lead_wave + backing_wave, sample_rate)
        karaoke = as_audio(inst + backing_wave, sample_rate)
        return full, karaoke, info


@dataclass
class RVCVocalBackend(VocalMixBackend):
    """Compatibility wrapper for earlier workflows and API callers."""

    def convert(self, music_audio, *, sample_rate: int, target_samples: int) -> tuple[dict, str]:
        full, _karaoke, info = self.process(music_audio, sample_rate=sample_rate,
                                           target_samples=target_samples)
        return full, info
