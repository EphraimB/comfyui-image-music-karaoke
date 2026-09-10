from __future__ import annotations

import json
import unittest
from pathlib import Path


class WorkflowVocalModeTests(unittest.TestCase):
    def test_bundled_workflow_exposes_mode_and_connects_singing_voice(self):
        root = Path(__file__).resolve().parents[1]
        workflow = json.loads(
            (root / "workflows" / "image-music-karaoke.json").read_text(encoding="utf-8")
        )
        nodes = {node["id"]: node for node in workflow["nodes"]}
        render = nodes[3]
        singing_voice = nodes[9]
        inputs = {item["name"]: item for item in render["inputs"]}

        self.assertIn("vocal_mode", inputs)
        self.assertIn("ace_voice_reference", inputs)
        self.assertEqual(inputs["vocal_mode"]["localized_name"], "Vocal Mode")
        self.assertIn("Legacy / separated vocal", render["widgets_values"])
        self.assertEqual(
            render["widgets_values_named"]["vocal_mode"], "Legacy / separated vocal"
        )
        self.assertIsNone(inputs["ace_voice_reference"]["link"])

        link_id = inputs["trained_voice_model"]["link"]
        self.assertIsNotNone(link_id)
        self.assertIn(link_id, singing_voice["outputs"][0]["links"])
        self.assertIn(
            [link_id, singing_voice["id"], 0, render["id"], 1, "RVC_MODEL"],
            workflow["links"],
        )

    def test_renderer_widget_values_match_serializable_input_order(self):
        root = Path(__file__).resolve().parents[1]
        workflow = json.loads(
            (root / "workflows" / "image-music-karaoke.json").read_text(encoding="utf-8")
        )
        render = next(node for node in workflow["nodes"] if node["type"] == "ImageSongRender")
        expected = [
            "acestep_v1.5_xl_sft_bf16.safetensors",
            "qwen_0.6b_ace15.safetensors",
            "qwen_4b_ace15.safetensors",
            "ace_1.5_vae.safetensors",
            "flux1-schnell-fp8.safetensors",
            "clip_l.safetensors",
            "t5xxl_fp16.safetensors",
            "ae.safetensors",
            "identity-preserving reference edits",
            "Legacy / separated vocal",
            4,
            0.3,
            0.58,
            65,
            3,
            True,
            "auto",
            "1280x720",
            "song",
        ]
        self.assertEqual(render["widgets_values"], expected)
        self.assertEqual(list(render["widgets_values_named"]), [
            "music_model", "text_encoder", "audio_code_model", "audio_vae",
            "image_model", "image_clip", "image_t5", "image_vae",
            "visual_treatment", "vocal_mode", "image_steps", "image_edit_strength",
            "identity_preservation", "steps", "cfg", "generate_audio_codes",
            "lyric_timing", "resolution", "filename",
        ])
        self.assertEqual(list(render["widgets_values_named"].values()), expected)


if __name__ == "__main__":
    unittest.main()
