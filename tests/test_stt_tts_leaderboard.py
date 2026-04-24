"""
Tests for STT and TTS leaderboard generation.

Covers:
- Dynamic metric discovery from metrics.json (no hardcoded metric list)
- Excel workbook is produced with summary sheet + per-provider sheets
- Handles single-criterion (`llm_judge_score`) and multi-criterion metrics.json
- Skips the `leaderboard` subdir inside output_dir

Run with:
    python -m pytest tests/test_stt_tts_leaderboard.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import openpyxl  # noqa: F401 — ensures xlsx reading works

from calibrate.stt.leaderboard import generate_leaderboard as generate_stt_leaderboard
from calibrate.tts.leaderboard import generate_leaderboard as generate_tts_leaderboard


def _write_provider(
    base: Path,
    provider: str,
    metrics: dict,
    results_rows: list[dict] | None = None,
) -> None:
    provider_dir = base / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "metrics.json").write_text(json.dumps(metrics))
    if results_rows is not None:
        pd.DataFrame(results_rows).to_csv(
            provider_dir / "results.csv", index=False
        )


# ---------------------------------------------------------------------------
# STT leaderboard
# ---------------------------------------------------------------------------


class TestSTTLeaderboard(unittest.TestCase):

    def test_default_single_criterion_produces_llm_judge_score_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "deepgram", {
                "wer": 0.1,
                "string_similarity": 0.9,
                "llm_judge_score": 0.85,
            }, results_rows=[
                {"id": 1, "gt": "hello", "pred": "hello", "llm_judge_score": True},
            ])
            _write_provider(base, "google", {
                "wer": 0.2,
                "string_similarity": 0.8,
                "llm_judge_score": 0.75,
            }, results_rows=[
                {"id": 1, "gt": "hello", "pred": "hallo", "llm_judge_score": False},
            ])

            save_dir = base / "leaderboard"
            generate_stt_leaderboard(str(base), str(save_dir))

            # Charts: one per metric
            self.assertTrue((save_dir / "wer.png").exists())
            self.assertTrue((save_dir / "string_similarity.png").exists())
            self.assertTrue((save_dir / "llm_judge_score.png").exists())

            # Excel workbook exists with summary sheet
            xlsx = save_dir / "stt_leaderboard.xlsx"
            self.assertTrue(xlsx.exists())
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("wer", summary.columns)
            self.assertIn("llm_judge_score", summary.columns)
            self.assertIn("string_similarity", summary.columns)
            self.assertEqual(set(summary["run"]), {"deepgram", "google"})

    def test_custom_criterion_metrics_surface_dynamically(self):
        """A provider with custom criterion `semantic_match` should produce a chart
        named semantic_match.png and a column in the summary."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "provider-a", {
                "wer": 0.05,
                "string_similarity": 0.95,
                "semantic_match_score": 0.9,
                "completeness_score": 0.7,
            })

            save_dir = base / "leaderboard"
            generate_stt_leaderboard(str(base), str(save_dir))

            self.assertTrue((save_dir / "semantic_match_score.png").exists())
            self.assertTrue((save_dir / "completeness_score.png").exists())

            xlsx = save_dir / "stt_leaderboard.xlsx"
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("semantic_match_score", summary.columns)
            self.assertIn("completeness_score", summary.columns)

    def test_skips_existing_leaderboard_folder(self):
        """A pre-existing `leaderboard` subdir under output_dir must not be
        treated as a provider (hardcoded skip in the leaderboard code)."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "provider-x", {
                "wer": 0.1, "string_similarity": 0.9, "llm_judge_score": 1.0,
            }, results_rows=[
                {"id": 1, "gt": "hi", "pred": "hi", "llm_judge_score": True},
            ])
            # Pre-existing leaderboard directory — must be skipped
            (base / "leaderboard").mkdir()
            (base / "leaderboard" / "metrics.json").write_text(
                json.dumps({"wer": 999.0})
            )

            # Save inside the default location (base/leaderboard)
            generate_stt_leaderboard(str(base))

            xlsx = base / "leaderboard" / "stt_leaderboard.xlsx"
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertEqual(list(summary["run"]), ["provider-x"])


# ---------------------------------------------------------------------------
# TTS leaderboard
# ---------------------------------------------------------------------------


class TestTTSLeaderboard(unittest.TestCase):

    def test_default_single_criterion_and_ttfb(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "openai", {
                "llm_judge_score": 0.95,
                "ttfb": {"mean": 0.4, "std": 0.05, "values": [0.35, 0.45]},
            }, results_rows=[
                {"id": 1, "text": "hi", "llm_judge_score": True, "ttfb": 0.4},
            ])
            _write_provider(base, "elevenlabs", {
                "llm_judge_score": 0.8,
                "ttfb": {"mean": 0.3, "std": 0.02, "values": [0.29, 0.31]},
            }, results_rows=[
                {"id": 1, "text": "hi", "llm_judge_score": True, "ttfb": 0.3},
            ])

            save_dir = base / "leaderboard"
            generate_tts_leaderboard(str(base), str(save_dir))

            self.assertTrue((save_dir / "llm_judge_score.png").exists())
            self.assertTrue((save_dir / "ttfb.png").exists())

            xlsx = save_dir / "tts_leaderboard.xlsx"
            self.assertTrue(xlsx.exists())
            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("llm_judge_score", summary.columns)
            self.assertIn("ttfb", summary.columns)

    def test_multi_criterion_metrics_surface_dynamically(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_provider(base, "provider-a", {
                "intelligibility_score": 0.9,
                "pronunciation_score": 0.85,
                "ttfb": {"mean": 0.4},
            })

            save_dir = base / "leaderboard"
            generate_tts_leaderboard(str(base), str(save_dir))

            self.assertTrue((save_dir / "intelligibility_score.png").exists())
            self.assertTrue((save_dir / "pronunciation_score.png").exists())
            self.assertTrue((save_dir / "ttfb.png").exists())


if __name__ == "__main__":
    unittest.main()
