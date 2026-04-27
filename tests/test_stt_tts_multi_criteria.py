"""
Tests for STT/TTS multi-evaluator judge aggregation.

Covers:
- get_llm_judge_score (STT): default single evaluator still works, scores aggregated
- get_llm_judge_score (STT): multi-evaluator produces per-evaluator scores + per_row
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
    async def test_default_evaluator_single_judge(self):
        from calibrate.stt import metrics as stt_metrics

        # Patch stt_llm_judge directly (it has @backoff + @observe decorators
        # so patching text_judge inside it is unreliable).
        # tqdm_asyncio.gather may not preserve input order, so return based on input.
        async def fake_judge(reference, prediction, evaluators=None, fallback_model=None):
            match = reference == prediction
            return {
                "semantic_match": {
                    "match": match,
                    "reasoning": "ok" if match else "mismatch",
                }
            }

        with patch.object(stt_metrics, "stt_llm_judge", AsyncMock(side_effect=fake_judge)):
            result = await stt_metrics.get_llm_judge_score(
                references=["hello", "goodnight"],
                predictions=["hello", "goodbye"],  # first matches, second doesn't
            )

        self.assertEqual(list(result["scores"].keys()), ["semantic_match"])
        self.assertEqual(result["scores"]["semantic_match"]["type"], "binary")
        self.assertEqual(result["scores"]["semantic_match"]["mean"], 0.5)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(len(result["per_row"]), 2)
        # Tally per_row matches: exactly one True and one False
        matches = [row["semantic_match"]["match"] for row in result["per_row"]]
        self.assertEqual(sorted(matches), [False, True])

    async def test_multi_evaluators_per_row_and_aggregate(self):
        from calibrate.stt import metrics as stt_metrics

        custom_evaluators = [
            {
                "name": "semantic_match",
                "system_prompt": "values match",
                "judge_model": "openai/gpt-4.1",
            },
            {
                "name": "completeness",
                "system_prompt": "nothing missing",
                "judge_model": "openai/gpt-4.1",
            },
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
                evaluators=custom_evaluators,
            )

        self.assertEqual(
            set(result["scores"].keys()), {"semantic_match", "completeness"}
        )
        self.assertEqual(result["scores"]["semantic_match"]["mean"], 1.0)
        self.assertEqual(result["scores"]["completeness"]["mean"], 0.5)
        self.assertEqual(result["scores"]["semantic_match"]["type"], "binary")
        # Overall score is mean across evaluators
        self.assertAlmostEqual(result["score"], 0.75)

    async def test_rating_evaluator_aggregates_mean_score(self):
        from calibrate.stt import metrics as stt_metrics

        rating_evaluator = {
            "name": "semantic_accuracy",
            "system_prompt": "rate semantic accuracy",
            "judge_model": "openai/gpt-4.1",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
        }

        async def fake_judge(reference, prediction, evaluators=None, fallback_model=None):
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
                evaluators=[rating_evaluator],
            )

        self.assertEqual(result["scores"]["semantic_accuracy"]["type"], "rating")
        # Two 5s and one 2 → mean = 12/3 = 4.0
        self.assertAlmostEqual(result["scores"]["semantic_accuracy"]["mean"], 4.0)
        self.assertEqual(result["scores"]["semantic_accuracy"]["scale_min"], 1)
        self.assertEqual(result["scores"]["semantic_accuracy"]["scale_max"], 5)

    async def test_custom_evaluators_passed_through(self):
        from calibrate.stt import metrics as stt_metrics

        custom_evaluators = [
            {"name": "x", "system_prompt": "y", "judge_model": "openai/gpt-4.1"}
        ]
        mock_stt_judge = AsyncMock(
            return_value={"x": {"match": True, "reasoning": "ok"}}
        )

        with patch.object(stt_metrics, "stt_llm_judge", mock_stt_judge):
            await stt_metrics.get_llm_judge_score(
                references=["ref"],
                predictions=["pred"],
                evaluators=custom_evaluators,
                fallback_model="custom-model",
            )

        # stt_llm_judge is called positionally for reference/prediction
        call_kwargs = mock_stt_judge.call_args.kwargs
        self.assertEqual(call_kwargs["evaluators"], custom_evaluators)
        self.assertEqual(call_kwargs["fallback_model"], "custom-model")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TestTTSGetLLMJudgeScore(unittest.IsolatedAsyncioTestCase):
    async def test_default_evaluator_single_judge(self):
        from calibrate.tts import metrics as tts_metrics

        # Patch tts_llm_judge directly (has @backoff + @observe decorators)
        mock_tts_judge = AsyncMock(
            side_effect=[
                {"pronunciation": {"match": True, "reasoning": "clear"}},
                {"pronunciation": {"match": False, "reasoning": "garbled"}},
            ]
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            result = await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav", "/tmp/b.wav"],
                reference_texts=["hi", "bye"],
            )

        self.assertEqual(list(result["scores"].keys()), ["pronunciation"])
        self.assertEqual(result["scores"]["pronunciation"]["type"], "binary")
        self.assertEqual(result["scores"]["pronunciation"]["mean"], 0.5)
        self.assertEqual(result["score"], 0.5)

    async def test_multi_evaluators_per_row_and_aggregate(self):
        from calibrate.tts import metrics as tts_metrics

        custom_evaluators = [
            {
                "name": "intelligibility",
                "system_prompt": "clear",
                "judge_model": "openai/gpt-4o-audio-preview",
            },
            {
                "name": "pronunciation",
                "system_prompt": "correct",
                "judge_model": "openai/gpt-4o-audio-preview",
            },
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
                evaluators=custom_evaluators,
            )

        self.assertEqual(
            set(result["scores"].keys()), {"intelligibility", "pronunciation"}
        )
        self.assertEqual(result["scores"]["intelligibility"]["mean"], 1.0)
        self.assertEqual(result["scores"]["pronunciation"]["mean"], 0.5)
        self.assertAlmostEqual(result["score"], 0.75)

    async def test_rating_evaluator_aggregates_mean_score(self):
        from calibrate.tts import metrics as tts_metrics

        rating = {
            "name": "naturalness",
            "system_prompt": "rate how natural the speech sounds",
            "judge_model": "openai/gpt-4o-audio-preview",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
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
                evaluators=[rating],
            )

        self.assertEqual(result["scores"]["naturalness"]["type"], "rating")
        self.assertAlmostEqual(result["scores"]["naturalness"]["mean"], 4.0)
        self.assertEqual(result["scores"]["naturalness"]["scale_min"], 1)
        self.assertEqual(result["scores"]["naturalness"]["scale_max"], 5)

    async def test_custom_evaluators_passed_through(self):
        from calibrate.tts import metrics as tts_metrics

        custom_evaluators = [
            {"name": "x", "system_prompt": "y", "judge_model": "openai/gpt-4o-audio-preview"}
        ]
        mock_tts_judge = AsyncMock(
            return_value={"x": {"match": True, "reasoning": "ok"}}
        )
        with patch.object(tts_metrics, "tts_llm_judge", mock_tts_judge):
            await tts_metrics.get_tts_llm_judge_score(
                audio_paths=["/tmp/a.wav"],
                reference_texts=["text"],
                evaluators=custom_evaluators,
                fallback_model="custom-audio-model",
            )

        call_kwargs = mock_tts_judge.call_args.kwargs
        self.assertEqual(call_kwargs["evaluators"], custom_evaluators)
        self.assertEqual(call_kwargs["fallback_model"], "custom-audio-model")


if __name__ == "__main__":
    unittest.main()
