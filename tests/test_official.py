from __future__ import annotations
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import models
import official


class OfficialModelTests(unittest.TestCase):
    def setUp(self):
        self.info = models.get_manifest(ROOT / "model-manifest.json")

    def test_models_are_exact_official_bf16(self):
        self.assertEqual(self.info["officialPrecision"], "bfloat16")
        self.assertEqual(self.info["officialImageSize"], [832, 480])
        self.assertEqual(self.info["officialFrames"], 81)
        self.assertEqual(self.info["officialFps"], 30)
        self.assertEqual(len(list(models.expected_files(self.info, Path("/tmp/nowhere")))), 9)
        self.assertEqual(sum(f["size"] for m in self.info["hfRepositories"] for f in m["files"]),
                         97_600_801_337)
        self.assertEqual(self.info["officialGitRevision"],
                         "aa503085c3b985e1c44ae4f7786c2a9402df53c1")

    def test_missing_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(models.verify(self.info, Path(d), quiet=True))

    def test_only_path_placeholders_change(self):
        example = '\n'.join([
            *['"/path/to/diffusion_pytorch_model-00001-of-00006.safetensors"'] * 6,
            '"/path/to/Wan2.1-T2V-14B/Wan2.1_VAE.pth"',
            '"/path/to/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"',
            'num_inference_steps=50, num_frames=81, fps=30, dtype=torch.bfloat16'
        ])
        with tempfile.TemporaryDirectory() as d:
            patched = official.path_only_patch(example, self.info, Path(d))
            self.assertEqual(patched.count(str(Path(d) / "weights/wan/diffusion_pytorch_model-")), 6)
            self.assertNotIn("/path/to/", patched)
            self.assertIn("num_inference_steps=50, num_frames=81, fps=30, dtype=torch.bfloat16", patched)
            with self.assertRaises(ValueError):
                official.path_only_patch(example.replace("num_frames=81", "/path/to/new"), self.info, Path(d))
            with self.assertRaises(ValueError):
                official.path_only_patch(example.replace("diffusion_pytorch_model-", "oops"), self.info, Path(d))

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            for name in ("video.mp4", "mask.png", "character.png"):
                (directory / name).write_bytes(b"synthetic test")
            table = directory / "job.csv"
            with table.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=official.REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerow({"source_video": str(directory / "video.mp4"),
                                 "source_mask": str(directory / "mask.png"),
                                 "reference_1": str(directory / "character.png"),
                                 "reference_2": "None"})
            row = official.validate_csv(table)
            self.assertEqual(row["reference_2"], "None")
            (directory / "mask.png").unlink()
            with self.assertRaises(SystemExit):
                official.validate_csv(table)


if __name__ == "__main__":
    unittest.main()
