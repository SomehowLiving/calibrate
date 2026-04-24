"""
Tests for STT/TTS multi-criteria judge aggregation.

Covers:
- get_llm_judge_score (STT): default single criterion still works, scores aggregated
- get_llm_judge_score (STT): multi-criteria produces per-criterion scores + per_row
- get_tts_llm_judge_score (TTS): same patterns

Run with:
    python -m pytest tests/test_stt_tts_multi_criteria.py -v
"""

import unittest
from unittest.mock import patch, AsyncMock


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


class TestSTTGetLLMJudgeScore(unittest.IsolatedAsyncioTestCase):
    async def test_default_criteria_single_llm_judge(self):
        from calibrate.stt import metrics as stt_metrics

        # Patch stt_llm_judge directly (it has @backoff + @observe decorators
        # so patching text_judge inside it is unreliable).
        # tqdm_asyncio.gather may not preserve input order, so return based on input.
        async def fake_judge(reference, prediction, model=None, criteria=None):
            match = reference == prediction
            return {"llm_judge": {"match": match, "reasoning": "ok" if match else "mismatch"}}

        with patch.object(stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_judge)):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello", "goodnight"],
                predictions=["hello", "goodbye"],  # first matches, second doesn't
            )

        self.assertEqual(result["criteria_names"], ["llm_judge"])
        self.assertEqual(result["scores"]["llm_judge"]["type"], "binary")
        self.assertEqual(result["scores"]["llm_judge"]["mean"], 0.5)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(len(result["per_row"]), 2)
        # Tally per_row matches: exactly one True and one False
        matches = [row["llm_judge"]["match"] for row in result["per_row"]]
        self.assertEqual(sorted(matches), [False, True])

    async def test_multi_criteria_per_row_and_aggregate(self):
        from calibrate.stt import metrics as stt_metrics

        custom_criteria = [
            {"name": "semantic_match", "description": "values match"},
            {"name": "completeness", "description": "nothing missing"},
        ]
        mock_stt_judge = AsyncMock(
            side_effect=[
                {
                    "semantic_match": {"match": True, "reasoning": "ok"},
                    "completeness": {"match": True, "reasoning": "all there"},
                },
                {
                    "semantic_match": {"match": True, "reasoning": "ok"},
                    "completeness": {"match": False, "reasoning": "missing word"},
                },
            ]
        )

        with patch.object(stt_metrics, "stt_llm_judge", mock_stt_judge):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello world", "foo bar"],
                predictions=["hello world", "foo"],
                criteria=custom_criteria,
            )

        self.assertEqual(
            set(result["criteria_names"]), {"semantic_match", "completeness"}
        )
        self.assertEqual(result["scores"]["semantic_match"]["mean"], 1.0)
        self.assertEqual(result["scores"]["completeness"]["mean"], 0.5)
        self.assertEqual(result["scores"]["semantic_match"]["type"], "binary")
        # Overall score is mean across criteria
        self.assertAlmostEqual(result["score"], 0.75)

    async def test_rating_criterion_aggregates_mean_score(self):
        from calibrate.stt import metrics as stt_metrics

        rating_criterion = {
            "name": "semantic_accuracy",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "description": "rate semantic accuracy",
        }

        async def fake_judge(reference, prediction, model=None, criteria=None):
            # Return score based on whether strings match: match=5, mismatch=2
            return {
                "semantic_accuracy": {
                    "reasoning": "ok",
                    "score": 5 if reference == prediction else 2,
                }
            }

        with patch.object(stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_judge)):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello", "world", "foo"],
                predictions=["hello", "word", "foo"],  # 2 match, 1 doesn't
                criteria=[rating_criterion],
            )

        self.assertEqual(result["scores"]["semantic_accuracy"]["type"], "rating")
        # Two 5s and one 2 → mean = 12/3 = 4.0
        self.assertAlmostEqual(result["scores"]["semantic_accuracy"]["mean"], 4.0)
        self.assertEqual(result["scores"]["semantic_accuracy"]["scale_min"], 1)
        self.assertEqual(result["scores"]["semantic_accuracy"]["scale_max"], 5)

    async def test_custom_criteria_passed_through(self):
        from calibrate.stt import metrics as stt_metrics

        custom_criteria = [{"name": "x", "description": "y"}]
        mock_stt_judge = AsyncMock(
            return_value={"x": {"match": True, "reasoning": "ok"}}
        )

        with patch.object(stt_metrics, "stt_llm_judge", mock_stt_judge):
            await stt_metrics.get_llm_judge_score(
                references=["ref"],
                predictions=["pred"],
                criteria=custom_criteria,
                model="custom-model",
            )

        # stt_llm_judge is called positionally for reference/prediction
        call_kwargs = mock_stt_judge.call_args.kwargs
        self.assertEqual(call_kwargs["criteria"], custom_criteria)
        self.assertEqual(call_kwargs["model"], "custom-model")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TestTTSGetLLMJudgeScore(unittest.IsolatedAsyncioTestCase):
    async def test_default_criteria_single_llm_judge(self):
        from calibrate.tts import metrics as tts_metrics

        # Patch tts_llm_judge directly (has @backoff + @observe decorators)
        mock_tts_judge = AsyncMock(
            side_effect=[
                {"llm_judge": {"match": True, "reasoning": "clear"}},
                {"llm_judge": {"match": False, "reasoning": "garbled"}},
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
            )

        self.assertEqual(result["criteria_names"], ["llm_judge"])
        self.assertEqual(result["scores"]["llm_judge"]["type"], "binary")
        self.assertEqual(result["scores"]["llm_judge"]["mean"], 0.5)
        self.assertEqual(result["score"], 0.5)

    async def test_multi_criteria_per_row_and_aggregate(self):
        from calibrate.tts import metrics as tts_metrics

        custom_criteria = [
            {"name": "intelligibility", "description": "clear"},
            {"name": "pronunciation", "description": "correct"},
        ]
        mock_tts_judge = AsyncMock(
            side_effect=[
                {
                    "intelligibility": {"match": True, "reasoning": "clear"},
                    "pronunciation": {"match": True, "reasoning": "good"},
                },
                {
                    "intelligibility": {"match": True, "reasoning": "clear"},
                    "pronunciation": {"match": False, "reasoning": "mispronounced"},
                },
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hello", "world"],
                criteria=custom_criteria,
            )

        self.assertEqual(
            set(result["criteria_names"]), {"intelligibility", "pronunciation"}
        )
        self.assertEqual(result["scores"]["intelligibility"]["mean"], 1.0)
        self.assertEqual(result["scores"]["pronunciation"]["mean"], 0.5)
        self.assertAlmostEqual(result["score"], 0.75)

    async def test_rating_criterion_aggregates_mean_score(self):
        from calibrate.tts import metrics as tts_metrics

        rating = {
            "name": "naturalness",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "description": "rate how natural the speech sounds",
        }
        mock_tts_judge = AsyncMock(
            side_effect=[
                {"naturalness": {"score": 5, "reasoning": "very natural"}},
                {"naturalness": {"score": 3, "reasoning": "okay"}},
                {"naturalness": {"score": 4, "reasoning": "good"}},
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav", "/tmp/c.wav"],
                reference_texts=["x", "y", "z"],
                criteria=[rating],
            )

        self.assertEqual(result["scores"]["naturalness"]["type"], "rating")
        self.assertAlmostEqual(result["scores"]["naturalness"]["mean"], 4.0)
        self.assertEqual(result["scores"]["naturalness"]["scale_min"], 1)
        self.assertEqual(result["scores"]["naturalness"]["scale_max"], 5)

    async def test_custom_criteria_passed_through(self):
        from calibrate.tts import metrics as tts_metrics

        custom_criteria = [{"name": "x", "description": "y"}]
        mock_tts_judge = AsyncMock(
            return_value={"x": {"match": True, "reasoning": "ok"}}
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav"],
                reference_texts=["text"],
                criteria=custom_criteria,
                model="custom-audio-model",
            )

        call_kwargs = mock_tts_judge.call_args.kwargs
        self.assertEqual(call_kwargs["criteria"], custom_criteria)
        self.assertEqual(call_kwargs["model"], "custom-audio-model")


if __name__ == "__main__":
    unittest.main()
