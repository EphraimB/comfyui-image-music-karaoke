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
        self.assertIn("Legacy / separated vocal", render["widgets_values"])
        self.assertIsNone(inputs["ace_voice_reference"]["link"])

        link_id = inputs["trained_voice_model"]["link"]
        self.assertIsNotNone(link_id)
        self.assertIn(link_id, singing_voice["outputs"][0]["links"])
        self.assertIn(
            [link_id, singing_voice["id"], 0, render["id"], 1, "RVC_MODEL"],
            workflow["links"],
        )


if __name__ == "__main__":
    unittest.main()
