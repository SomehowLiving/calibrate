"""
Regression tests for SDK-path judge behavior.

Covers two bugs flagged on PR #47:

1. Simulation SDK (`calibrate.llm.simulations.run`) must write a `type`
   field into each criterion in `metrics.json`. Without it, the simulation
   leaderboard's default-binary fallback multiplies rating means by 100
   (corrupting e.g. `4.2` → `420`).

2. Voice-sim SDK (`calibrate.agent.simulation.run_single`) must fall back
   to `judge.model` for the STT judge when `judge.stt_model` is omitted,
   matching the config-driven flow in
   `calibrate/agent/run_simulation.py::_run_single_simulation_inner`.
   Otherwise `simulation.run(...)` and `simulation.run_single(...)` can
   produce different STT scores for the same judge config.

Run with:
    python -m pytest tests/test_sdk_judge_regressions.py -v
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Bug 1: SDK simulation metrics.json must carry `type`
# ---------------------------------------------------------------------------


class TestSDKSimulationMetricsCarryType(unittest.IsolatedAsyncioTestCase):

    async def test_rating_criterion_aggregate_gets_type_field(self):
        """After `calibrate.llm.simulations.run(...)` completes, the saved
        metrics.json should include `type: "rating"` for each rating
        criterion so the leaderboard doesn't multiply its mean by 100."""
        from calibrate.llm import simulations

        # Fake `run_single_simulation_task` that returns the shape the SDK expects:
        # (simulation_metrics_row, evaluation_results)
        # Each evaluation_result carries {name, type, value, reasoning}.
        async def fake_task(
            semaphore, config, persona_index, user_persona, scenario_index,
            scenario, output_dir, args, agent=None,
        ):
            async with semaphore:
                sim_name = f"sim_{persona_index}_{scenario_index}"
                eval_results = [
                    {"name": "tool_usage", "type": "binary", "value": 1.0, "reasoning": "ok"},
                    {"name": "fluency", "type": "rating", "value": 4.0, "reasoning": "good"},
                ]
                sim_metrics = {"name": sim_name, "tool_usage": 1.0, "fluency": 4.0}
                return sim_metrics, eval_results

        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "calibrate.llm.run_simulation.run_single_simulation_task",
                 side_effect=fake_task,
             ):
            await simulations.run(
                system_prompt="...",
                tools=[],
                personas=[{"characteristics": "friendly", "language": "english"}],
                scenarios=[{"description": "basic inquiry"}],
                evaluation_criteria=[
                    {"name": "tool_usage", "description": "calls tool"},
                    {
                        "name": "fluency",
                        "type": "rating",
                        "scale_min": 1,
                        "scale_max": 5,
                        "description": "rate fluency",
                    },
                ],
                output_dir=tmp,
                model="gpt-4.1",
                provider="openai",
            )

            # Find the model subfolder that was created (single model path uses _flat_output=True)
            with open(os.path.join(tmp, "metrics.json")) as f:
                metrics = json.load(f)

        self.assertIn("tool_usage", metrics)
        self.assertIn("fluency", metrics)
        self.assertEqual(metrics["tool_usage"]["type"], "binary")
        self.assertEqual(metrics["fluency"]["type"], "rating")
        # Raw means are preserved (no 100x scaling at this stage)
        self.assertAlmostEqual(metrics["fluency"]["mean"], 4.0)
        self.assertAlmostEqual(metrics["tool_usage"]["mean"], 1.0)


# ---------------------------------------------------------------------------
# Bug 2: judge.model falls back to STT judge in agent.simulation.run_single
# ---------------------------------------------------------------------------


class TestAgentSimulationRunSingleSTTFallback(unittest.IsolatedAsyncioTestCase):

    async def _call_run_single(self, judge):
        """Invoke agent.simulation.run_single with the given judge dict,
        capturing the kwargs passed to the underlying _run_simulation."""
        from calibrate.agent import simulation as agent_sim

        captured_kwargs = {}

        async def fake_run_simulation(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "transcript": [],
                "evaluation_results": [],
                "metrics": {},
                "stt_outputs": [],
                "tool_calls": [],
            }

        # Patch the symbol in calibrate.agent.__init__ namespace
        # (imported locally inside run_single)
        with patch(
            "calibrate.agent.run_simulation.run_simulation",
            side_effect=fake_run_simulation,
        ):
            await agent_sim.run_single(
                system_prompt="You are a user",
                language="english",
                gender="female",
                evaluation_criteria=[{"name": "x", "description": "y"}],
                output_dir="/tmp/any",
                judge=judge,
            )
        return captured_kwargs

    async def test_judge_model_propagates_to_stt_when_stt_model_missing(self):
        kwargs = await self._call_run_single({"model": "custom-judge-model"})
        self.assertEqual(kwargs.get("judge_model"), "custom-judge-model")
        # Critical: STT judge inherits from judge.model when stt_model is absent
        self.assertEqual(kwargs.get("stt_judge_model"), "custom-judge-model")

    async def test_explicit_stt_model_overrides_judge_model(self):
        kwargs = await self._call_run_single({
            "model": "primary-model",
            "stt_model": "stt-specific-model",
        })
        self.assertEqual(kwargs.get("judge_model"), "primary-model")
        self.assertEqual(kwargs.get("stt_judge_model"), "stt-specific-model")

    async def test_no_judge_dict_leaves_both_unset(self):
        kwargs = await self._call_run_single(None)
        self.assertNotIn("judge_model", kwargs)
        self.assertNotIn("stt_judge_model", kwargs)


# ---------------------------------------------------------------------------
# Bug 3: simulation_leaderboard `overall` must be unit-consistent across
# mixed binary/rating criteria
# ---------------------------------------------------------------------------


class TestSimulationLeaderboardOverallUnitConsistent(unittest.TestCase):

    def test_overall_uses_normalized_values_for_rating(self):
        """Binary % and rating raw means can't be averaged directly.
        The overall column should be computed from normalized (0-100) values:
        for rating, (mean - scale_min) / (scale_max - scale_min) * 100.
        """
        import pathlib
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            # Model A: binary 80%, rating mean 4/5 → normalized 75%.
            # Expected overall (normalized mean) = (80 + 75) / 2 = 77.5
            (base / "model-a").mkdir()
            (base / "model-a" / "metrics.json").write_text(
                json.dumps({
                    "accuracy": {"type": "binary", "mean": 0.8},
                    "fluency": {
                        "type": "rating",
                        "mean": 4.0,
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                })
            )
            # Model B: binary 100%, rating mean 1/5 → normalized 0%.
            # Expected overall = (100 + 0) / 2 = 50
            (base / "model-b").mkdir()
            (base / "model-b" / "metrics.json").write_text(
                json.dumps({
                    "accuracy": {"type": "binary", "mean": 1.0},
                    "fluency": {
                        "type": "rating",
                        "mean": 1.0,
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                })
            )

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            import pandas as pd
            df = pd.read_csv(save_dir / "simulation_leaderboard.csv")

            row_a = df[df["model"] == "model-a"].iloc[0]
            row_b = df[df["model"] == "model-b"].iloc[0]

            # Display values stay on their own scale (binary %, rating raw mean)
            self.assertAlmostEqual(row_a["accuracy"], 80.0)
            self.assertAlmostEqual(row_a["fluency"], 4.0)
            self.assertAlmostEqual(row_b["accuracy"], 100.0)
            self.assertAlmostEqual(row_b["fluency"], 1.0)

            # Overall is computed from normalized 0-100 values, so it's
            # unit-consistent. Adding the rating criterion can no longer
            # arbitrarily reorder models via unit-mixing.
            self.assertAlmostEqual(row_a["overall"], 77.5)
            self.assertAlmostEqual(row_b["overall"], 50.0)


# ---------------------------------------------------------------------------
# Bug 4: STT workbook exporter must tolerate numeric llm_judge_score
# ---------------------------------------------------------------------------


class TestSTTLeaderboardWorkbookRatingTolerant(unittest.TestCase):

    def test_numeric_llm_judge_score_does_not_break_workbook(self):
        """When a user renames their rating criterion to `llm_judge`,
        results.csv's llm_judge_score column is numeric. The STT leaderboard
        workbook exporter used to do `df = df[~df["llm_judge_score"]]`
        which is bitwise-NOT on ints — producing garbage or errors.
        The fix skips the boolean-negation filter for non-boolean columns.
        """
        import pathlib
        import pandas as pd
        from calibrate.stt.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            provider = base / "deepgram"
            provider.mkdir()
            (provider / "metrics.json").write_text(
                json.dumps({
                    "wer": 0.1,
                    "string_similarity": 0.9,
                    "llm_judge_score": 4.2,  # rating mean, not bool
                })
            )
            # results.csv with numeric llm_judge_score (rating scores 3-5)
            pd.DataFrame([
                {"id": 1, "gt": "hi", "pred": "hi", "llm_judge_score": 5},
                {"id": 2, "gt": "bye", "pred": "by", "llm_judge_score": 3},
                {"id": 3, "gt": "ok", "pred": "ok", "llm_judge_score": 5},
            ]).to_csv(provider / "results.csv", index=False)

            # Should not raise; should produce a valid workbook
            generate_leaderboard(str(base))

            xlsx = base / "leaderboard" / "stt_leaderboard.xlsx"
            self.assertTrue(xlsx.exists())

            # All 3 rows should appear in the per-provider sheet
            # (filter is skipped because column is numeric)
            provider_sheet = pd.read_excel(xlsx, sheet_name="deepgram")
            self.assertEqual(len(provider_sheet), 3)


# ---------------------------------------------------------------------------
# Bug 5: simulation metrics.json must persist scale_min/scale_max for rating
# ---------------------------------------------------------------------------


class TestSimulationMetricsPersistScaleBounds(unittest.IsolatedAsyncioTestCase):

    async def test_sdk_simulation_writes_scale_for_rating_criterion(self):
        """The simulation leaderboard normalizes rating means via
        (mean - scale_min) / (scale_max - scale_min) * 100. Without the
        bounds in metrics.json it falls back to 0..1, treating a 1-5 mean
        of 4.0 as 400% in the overall column. The SDK simulation writer
        must persist scale_min/scale_max for rating criteria.
        """
        from calibrate.llm import simulations

        async def fake_task(
            semaphore, config, persona_index, user_persona, scenario_index,
            scenario, output_dir, args, agent=None,
        ):
            async with semaphore:
                eval_results = [
                    {
                        "name": "fluency",
                        "type": "rating",
                        "value": 4.0,
                        "reasoning": "good",
                        "scale_min": 1,
                        "scale_max": 5,
                    },
                ]
                sim_metrics = {"name": f"sim_{persona_index}_{scenario_index}", "fluency": 4.0}
                return sim_metrics, eval_results

        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "calibrate.llm.run_simulation.run_single_simulation_task",
                 side_effect=fake_task,
             ):
            await simulations.run(
                system_prompt="...",
                tools=[],
                personas=[{"characteristics": "x", "language": "english"}],
                scenarios=[{"description": "x"}],
                evaluation_criteria=[{
                    "name": "fluency",
                    "type": "rating",
                    "scale_min": 1,
                    "scale_max": 5,
                    "description": "rate fluency",
                }],
                output_dir=tmp,
                model="gpt-4.1",
                provider="openai",
            )

            with open(os.path.join(tmp, "metrics.json")) as f:
                metrics = json.load(f)

        self.assertEqual(metrics["fluency"]["type"], "rating")
        self.assertEqual(metrics["fluency"]["scale_min"], 1)
        self.assertEqual(metrics["fluency"]["scale_max"], 5)
        self.assertAlmostEqual(metrics["fluency"]["mean"], 4.0)


class TestSimulationLeaderboardNormalizesWithPersistedBounds(unittest.TestCase):
    """End-to-end: a metrics.json written by the SDK with scale bounds
    should produce the right normalized overall in the leaderboard."""

    def test_overall_uses_persisted_bounds(self):
        import pathlib
        import pandas as pd
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            (base / "model-a").mkdir()
            (base / "model-a" / "metrics.json").write_text(json.dumps({
                "fluency": {
                    "type": "rating",
                    "mean": 4.0,
                    "scale_min": 1,
                    "scale_max": 5,
                },
            }))

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            df = pd.read_csv(save_dir / "simulation_leaderboard.csv")
            row = df[df["model"] == "model-a"].iloc[0]
            # Rating display column shows raw mean
            self.assertAlmostEqual(row["fluency"], 4.0)
            # Overall normalized: (4-1)/(5-1)*100 = 75.0
            # Without scale bounds the fallback would give 400.
            self.assertAlmostEqual(row["overall"], 75.0)


# ---------------------------------------------------------------------------
# Bug 6: tests_leaderboard chart should not mislabel rating bars
# ---------------------------------------------------------------------------


class TestTestsLeaderboardChartNormalizesRating(unittest.TestCase):

    def test_chart_renders_with_mixed_binary_and_rating(self):
        """Mixed configs should produce a chart without crashing and
        without clipping rating bars to a 0..105 'pass rate' axis."""
        import pathlib
        from calibrate.llm.tests_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            (base / "model-a").mkdir()
            (base / "model-a" / "metrics.json").write_text(json.dumps({
                "total": 4, "passed": 3,
                "criteria": {
                    "accuracy": {
                        "type": "binary",
                        "passed": 3, "total": 4, "pass_rate": 75.0,
                    },
                    "fluency": {
                        "type": "rating",
                        "mean": 4.0,
                        "min": 3, "max": 5, "count": 4,
                        "scale_min": 1, "scale_max": 5,
                    },
                },
            }))
            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            chart_path = save_dir / "llm_leaderboard.png"
            self.assertTrue(chart_path.exists())
            # File should be a non-trivial PNG
            self.assertGreater(chart_path.stat().st_size, 1000)


# ---------------------------------------------------------------------------
# Bug 7: simulation_leaderboard chart should not mislabel rating bars
# ---------------------------------------------------------------------------


class TestSimulationLeaderboardChartNormalizesRating(unittest.TestCase):

    def test_chart_renders_with_mixed_binary_and_rating(self):
        """Rating means on a 1-5 scale shouldn't be plotted on a 0-105
        Score % axis — bars would look artificially tiny or get clipped."""
        import pathlib
        from calibrate.llm.simulation_leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            (base / "model-a").mkdir()
            (base / "model-a" / "metrics.json").write_text(json.dumps({
                "tool_usage": {"type": "binary", "mean": 0.8},
                "fluency": {
                    "type": "rating",
                    "mean": 4.0,
                    "scale_min": 1,
                    "scale_max": 5,
                },
            }))

            save_dir = base / "leaderboard"
            generate_leaderboard(str(base), str(save_dir))

            chart_path = save_dir / "simulation_leaderboard.png"
            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
