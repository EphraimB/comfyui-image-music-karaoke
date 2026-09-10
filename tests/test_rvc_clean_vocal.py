from __future__ import annotations

import unittest

import torch

from audio_pipeline import VocalMixBackend


class RecordingInvoker:
    def __init__(self):
        self.calls = []

    def __call__(self, node_name, **kwargs):
        self.calls.append((node_name, kwargs))
        if node_name == "RVCEngineNode":
            return ("configured-rvc-engine",)
        if node_name == "UnifiedVoiceChangerNode":
            converted = {
                "waveform": kwargs["source_audio"]["waveform"] * 0.5,
                "sample_rate": kwargs["source_audio"]["sample_rate"],
            }
            return converted, "conversion complete"
        raise AssertionError(f"Unexpected node call: {node_name}")


class LegacyRecordingInvoker(RecordingInvoker):
    def __call__(self, node_name, **kwargs):
        if node_name == "Demucs_Loader":
            self.calls.append((node_name, kwargs))
            return (kwargs["d_model"],)
        if node_name == "Demucs_Sampler":
            self.calls.append((node_name, kwargs))
            source = kwargs["audio"]
            instrumental = {
                "waveform": source["waveform"] * 0.6,
                "sample_rate": source["sample_rate"],
            }
            vocal = {
                "waveform": source["waveform"] * 0.4,
                "sample_rate": source["sample_rate"],
            }
            return instrumental, None, None, None, vocal
        return super().__call__(node_name, **kwargs)


class CleanVocalRVCTests(unittest.TestCase):
    def test_clean_vocal_uses_the_established_rvc_settings_without_demucs(self):
        invoke = RecordingInvoker()
        model = {"type": "rvc_model", "model_path": "trained.pth", "index_path": "trained.index"}
        source = {"waveform": torch.full((1, 2, 4800), 0.2), "sample_rate": 48000}
        backend = VocalMixBackend(invoke, model)

        converted, info = backend.convert_lead_vocal(
            source, sample_rate=48000, target_samples=4800)

        self.assertEqual(info, "conversion complete")
        self.assertEqual([name for name, _kwargs in invoke.calls],
                         ["RVCEngineNode", "UnifiedVoiceChangerNode"])
        engine = invoke.calls[0][1]
        self.assertEqual(engine, {
            "pitch": 0,
            "index_ratio": 0.75,
            "consonant_protection": 0.25,
            "volume_envelope": 0.25,
            "hubert_model": "content-vec-best: Content Vec 768 (Recommended)",
            "output_sample_rate": 0,
            "enable_custom_chunking": False,
            "device": "auto",
        })
        changer = invoke.calls[1][1]
        self.assertEqual(changer["TTS_engine"], "configured-rvc-engine")
        self.assertIs(changer["source_audio"], source)
        self.assertIs(changer["narrator_target"], model)
        self.assertEqual(changer["refinement_passes"], 1)
        self.assertEqual(changer["max_chunk_duration"], 30)
        self.assertEqual(changer["chunk_method"], "smart")
        self.assertTrue(torch.allclose(converted["waveform"], source["waveform"]))

    def test_disconnected_rvc_model_keeps_the_clean_vocal(self):
        invoke = RecordingInvoker()
        source = {"waveform": torch.randn(1, 2, 800), "sample_rate": 8000}
        backend = VocalMixBackend(invoke, None)

        converted, info = backend.convert_lead_vocal(
            source, sample_rate=8000, target_samples=800)

        self.assertEqual(invoke.calls, [])
        self.assertIn("skipped", info.lower())
        self.assertTrue(torch.equal(converted["waveform"], source["waveform"]))

    def test_direct_clean_vocal_remix_never_invokes_demucs(self):
        invoke = RecordingInvoker()
        model = {"type": "rvc_model", "model_path": "trained.pth", "index_path": "trained.index"}
        instrumental = {"waveform": torch.full((1, 2, 4800), 0.1), "sample_rate": 48000}
        vocal = {"waveform": torch.full((1, 2, 4800), 0.2), "sample_rate": 48000}
        backend = VocalMixBackend(invoke, model)

        full, karaoke, converted, info = backend.process_clean_vocal(
            instrumental, vocal, sample_rate=48000, target_samples=4800)

        self.assertEqual(info, "conversion complete")
        self.assertEqual([name for name, _kwargs in invoke.calls],
                         ["RVCEngineNode", "UnifiedVoiceChangerNode"])
        self.assertFalse(any("Demucs" in name for name, _kwargs in invoke.calls))
        self.assertTrue(torch.allclose(converted["waveform"], vocal["waveform"]))
        self.assertTrue(torch.allclose(karaoke["waveform"], instrumental["waveform"]))
        self.assertTrue(torch.allclose(full["waveform"], torch.full((1, 2, 4800), 0.3)))

    def test_legacy_processing_still_uses_demucs_and_rvc(self):
        invoke = LegacyRecordingInvoker()
        model = {"type": "rvc_model", "model_path": "trained.pth", "index_path": "trained.index"}
        full_mix = {"waveform": torch.full((1, 2, 4800), 0.3), "sample_rate": 48000}
        backend = VocalMixBackend(invoke, model)

        full, karaoke, info = backend.process(
            full_mix, sample_rate=48000, target_samples=4800)

        names = [name for name, _kwargs in invoke.calls]
        self.assertGreaterEqual(names.count("Demucs_Loader"), 2)
        self.assertGreaterEqual(names.count("Demucs_Sampler"), 2)
        self.assertIn("RVCEngineNode", names)
        self.assertIn("UnifiedVoiceChangerNode", names)
        self.assertEqual(full["waveform"].shape[-1], 4800)
        self.assertEqual(karaoke["waveform"].shape[-1], 4800)
        self.assertIn("Karaoke separator: Demucs", info)


if __name__ == "__main__":
    unittest.main()
