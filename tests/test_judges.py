"""
Unit tests for calibrate/judges.py — the unified judge module.

Covers:
- normalize_criteria (string → list, list passthrough)
- format_criteria_prompt
- build_criteria_output_model (dynamic pydantic model)
- text_judge calls LLM with criteria + returns per-criterion dict
- simulation_judge formats transcript with tool calls and agent instructions
- audio_judge builds audio message payload

Run with:
    python -m pytest tests/test_judges.py -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel

from calibrate.judges import (
    normalize_criteria,
    format_criteria_prompt,
    build_criteria_output_model,
    CriterionResult,
    text_judge,
    simulation_judge,
    audio_judge,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_STT_CRITERIA,
    DEFAULT_TTS_CRITERIA,
)


# ---------------------------------------------------------------------------
# normalize_criteria
# ---------------------------------------------------------------------------


class TestNormalizeCriteria(unittest.TestCase):
    def test_string_becomes_single_criterion(self):
        result = normalize_criteria("Agent should greet politely")
        self.assertEqual(
            result,
            [{"name": "criteria", "description": "Agent should greet politely"}],
        )

    def test_list_is_returned_as_is(self):
        criteria = [
            {"name": "accuracy", "description": "Correct info"},
            {"name": "tone", "description": "Polite"},
        ]
        self.assertEqual(normalize_criteria(criteria), criteria)

    def test_empty_list_is_returned_as_is(self):
        self.assertEqual(normalize_criteria([]), [])


# ---------------------------------------------------------------------------
# format_criteria_prompt
# ---------------------------------------------------------------------------


class TestFormatCriteriaPrompt(unittest.TestCase):
    def test_single_criterion(self):
        out = format_criteria_prompt([{"name": "a", "description": "b"}])
        self.assertEqual(out, "**a**: b")

    def test_multiple_criteria_joined_with_double_newline(self):
        out = format_criteria_prompt(
            [
                {"name": "accuracy", "description": "correct"},
                {"name": "tone", "description": "polite"},
            ]
        )
        self.assertIn("**accuracy**: correct", out)
        self.assertIn("**tone**: polite", out)
        self.assertIn("\n\n", out)

    def test_rating_criterion_includes_scale_hint(self):
        out = format_criteria_prompt([
            {
                "name": "fluency",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "description": "rate from 1 poor to 5 great",
            },
        ])
        self.assertIn("**fluency** (rating 1-5)", out)
        self.assertIn("rate from 1 poor to 5 great", out)


class TestIsRating(unittest.TestCase):
    def test_binary_criterion_is_not_rating(self):
        from calibrate.judges import is_rating
        self.assertFalse(is_rating({"name": "x", "description": "y"}))
        self.assertFalse(
            is_rating({"name": "x", "type": "binary", "description": "y"})
        )

    def test_rating_criterion(self):
        from calibrate.judges import is_rating
        self.assertTrue(
            is_rating({"name": "x", "type": "rating", "scale_min": 1, "scale_max": 5})
        )


# ---------------------------------------------------------------------------
# build_criteria_output_model
# ---------------------------------------------------------------------------


class TestBuildCriteriaOutputModel(unittest.TestCase):
    def test_model_has_field_per_criterion(self):
        Output = build_criteria_output_model([
            {"name": "accuracy", "description": "correct"},
            {"name": "tone", "description": "polite"},
        ])
        self.assertTrue(issubclass(Output, BaseModel))
        fields = Output.model_fields
        self.assertIn("accuracy", fields)
        self.assertIn("tone", fields)

    def test_binary_field_uses_criterion_result(self):
        Output = build_criteria_output_model(
            [{"name": "x", "description": "y"}]
        )
        instance = Output(x={"reasoning": "ok", "match": True})
        self.assertIsInstance(instance.x, CriterionResult)
        self.assertTrue(instance.x.match)
        self.assertEqual(instance.x.reasoning, "ok")

    def test_rating_field_accepts_score_in_range(self):
        Output = build_criteria_output_model([
            {
                "name": "fluency",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "description": "rate fluency",
            },
        ])
        instance = Output(fluency={"reasoning": "good", "score": 4})
        self.assertEqual(instance.fluency.score, 4)
        self.assertEqual(instance.fluency.reasoning, "good")

    def test_rating_field_rejects_score_out_of_range(self):
        from pydantic import ValidationError

        Output = build_criteria_output_model([
            {
                "name": "fluency",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 3,
                "description": "rate",
            },
        ])
        with self.assertRaises(ValidationError):
            Output(fluency={"reasoning": "x", "score": 5})

    def test_mixed_binary_and_rating_fields(self):
        Output = build_criteria_output_model([
            {"name": "correct", "description": "is correct"},
            {
                "name": "fluency",
                "type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "description": "rate",
            },
        ])
        instance = Output(
            correct={"reasoning": "ok", "match": True},
            fluency={"reasoning": "good", "score": 4},
        )
        self.assertEqual(instance.model_dump(), {
            "correct": {"reasoning": "ok", "match": True},
            "fluency": {"reasoning": "good", "score": 4},
        })


# ---------------------------------------------------------------------------
# text_judge
# ---------------------------------------------------------------------------


def _mock_openai_responses_parse(return_value: dict):
    """Helper to build a mock for AsyncOpenAI().responses.parse.

    The real client returns an object with .output_parsed (a Pydantic model).
    We fake that with a simple object whose .model_dump() returns the dict.
    """
    parsed = MagicMock()
    parsed.model_dump.return_value = return_value
    api_response = MagicMock()
    api_response.output_parsed = parsed

    client = MagicMock()
    client.responses.parse = AsyncMock(return_value=api_response)
    return client


class TestTextJudge(unittest.IsolatedAsyncioTestCase):
    async def test_returns_per_criterion_dict(self):
        expected = {
            "accuracy": {"reasoning": "good", "match": True},
            "tone": {"reasoning": "rude", "match": False},
        }
        client = _mock_openai_responses_parse(expected)

        with patch("calibrate.judges.AsyncOpenAI", return_value=client):
            result = await text_judge(
                criteria=[
                    {"name": "accuracy", "description": "correct"},
                    {"name": "tone", "description": "polite"},
                ],
                user_prompt="Some context to evaluate",
            )

        self.assertEqual(result, expected)

    async def test_uses_default_model_when_not_specified(self):
        client = _mock_openai_responses_parse(
            {"criteria": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate.judges.AsyncOpenAI", return_value=client):
            await text_judge(
                criteria=[{"name": "criteria", "description": "x"}],
                user_prompt="ctx",
            )
        call_kwargs = client.responses.parse.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_TEXT_JUDGE_MODEL)

    async def test_passes_custom_model(self):
        client = _mock_openai_responses_parse(
            {"criteria": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate.judges.AsyncOpenAI", return_value=client):
            await text_judge(
                criteria=[{"name": "criteria", "description": "x"}],
                user_prompt="ctx",
                model="custom-model",
            )
        self.assertEqual(
            client.responses.parse.call_args.kwargs["model"], "custom-model"
        )

    async def test_criteria_appear_in_user_prompt(self):
        client = _mock_openai_responses_parse(
            {"accuracy": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate.judges.AsyncOpenAI", return_value=client):
            await text_judge(
                criteria=[{"name": "accuracy", "description": "UNIQUE-CRITERION-TEXT"}],
                user_prompt="SOME-CTX-MARKER",
            )

        # Check the user message contains the context and criterion description
        messages = client.responses.parse.call_args.kwargs["input"]
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertIn("SOME-CTX-MARKER", user_msg["content"])
        self.assertIn("UNIQUE-CRITERION-TEXT", user_msg["content"])


# ---------------------------------------------------------------------------
# simulation_judge
# ---------------------------------------------------------------------------


class TestSimulationJudge(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_to_text_judge_with_formatted_transcript(self):
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        criteria = [{"name": "greeting", "description": "agent greets"}]

        mock_text_judge = AsyncMock(
            return_value={"greeting": {"reasoning": "ok", "match": True}}
        )

        with patch("calibrate.judges.text_judge", mock_text_judge):
            result = await simulation_judge(
                conversation=conversation,
                evaluation_criteria=criteria,
            )

        self.assertEqual(result, {"greeting": {"reasoning": "ok", "match": True}})
        call_kwargs = mock_text_judge.call_args.kwargs
        self.assertEqual(call_kwargs["criteria"], criteria)
        # User prompt includes conversation transcript
        self.assertIn("user: Hi", call_kwargs["user_prompt"])
        self.assertIn("assistant: Hello!", call_kwargs["user_prompt"])

    async def test_tool_calls_included_in_transcript(self):
        conversation = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}
                ],
            },
        ]
        mock_text_judge = AsyncMock(
            return_value={"x": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate.judges.text_judge", mock_text_judge):
            await simulation_judge(
                conversation=conversation,
                evaluation_criteria=[{"name": "x", "description": "y"}],
            )

        prompt = mock_text_judge.call_args.kwargs["user_prompt"]
        self.assertIn("[Tool Call] get_weather", prompt)

    async def test_agent_instructions_included_in_system_prompt(self):
        mock_text_judge = AsyncMock(
            return_value={"x": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate.judges.text_judge", mock_text_judge):
            await simulation_judge(
                conversation=[{"role": "user", "content": "Hi"}],
                evaluation_criteria=[{"name": "x", "description": "y"}],
                agent_system_prompt="You are a NURSE-ASSISTANT-X",
            )
        system_prompt = mock_text_judge.call_args.kwargs["system_prompt"]
        self.assertIn("NURSE-ASSISTANT-X", system_prompt)
        self.assertIn("agent_instructions", system_prompt)

    async def test_no_agent_instructions_section_when_no_prompt(self):
        mock_text_judge = AsyncMock(
            return_value={"x": {"reasoning": "ok", "match": True}}
        )
        with patch("calibrate.judges.text_judge", mock_text_judge):
            await simulation_judge(
                conversation=[{"role": "user", "content": "Hi"}],
                evaluation_criteria=[{"name": "x", "description": "y"}],
            )
        system_prompt = mock_text_judge.call_args.kwargs["system_prompt"]
        self.assertNotIn("<agent_instructions>", system_prompt)


# ---------------------------------------------------------------------------
# audio_judge
# ---------------------------------------------------------------------------


class TestAudioJudge(unittest.IsolatedAsyncioTestCase):
    async def test_builds_audio_message_and_returns_per_criterion(self):
        import tempfile
        import os

        expected = {
            "intelligibility": {"reasoning": "clear", "match": True},
            "pronunciation": {"reasoning": "good", "match": True},
        }

        # Fake audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"FAKE_WAV_BYTES")
            audio_path = f.name

        try:
            parsed = MagicMock()
            parsed.model_dump.return_value = expected

            # audio_judge uses instructor.apatch(AsyncOpenAI()) then chat.completions.create.
            # We mock instructor.apatch to return a client whose create() returns
            # a pydantic-ish object whose .model_dump() returns expected.
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=parsed)

            with patch("calibrate.judges.instructor.apatch", return_value=mock_client), \
                 patch("calibrate.judges.AsyncOpenAI", return_value=MagicMock()):
                result = await audio_judge(
                    criteria=[
                        {"name": "intelligibility", "description": "clear speech"},
                        {"name": "pronunciation", "description": "correct"},
                    ],
                    audio_path=audio_path,
                    reference_text="hello world",
                )

            self.assertEqual(result, expected)
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            # model defaults to DEFAULT_AUDIO_JUDGE_MODEL
            self.assertEqual(call_kwargs["model"], DEFAULT_AUDIO_JUDGE_MODEL)
            # User message contains reference text and criteria
            user_msg = next(m for m in call_kwargs["messages"] if m["role"] == "user")
            text_parts = [p for p in user_msg["content"] if p["type"] == "text"]
            self.assertTrue(any("hello world" in p["text"] for p in text_parts))
            self.assertTrue(any("intelligibility" in p["text"] for p in text_parts))
            audio_parts = [
                p for p in user_msg["content"] if p["type"] == "input_audio"
            ]
            self.assertEqual(len(audio_parts), 1)
        finally:
            os.unlink(audio_path)


# ---------------------------------------------------------------------------
# Default criteria sanity checks
# ---------------------------------------------------------------------------


class TestDefaultCriteria(unittest.TestCase):
    def test_stt_default_criteria_shape(self):
        self.assertEqual(len(DEFAULT_STT_CRITERIA), 1)
        self.assertEqual(DEFAULT_STT_CRITERIA[0]["name"], "llm_judge")
        self.assertTrue(DEFAULT_STT_CRITERIA[0]["description"])

    def test_tts_default_criteria_shape(self):
        self.assertEqual(len(DEFAULT_TTS_CRITERIA), 1)
        self.assertEqual(DEFAULT_TTS_CRITERIA[0]["name"], "llm_judge")
        self.assertTrue(DEFAULT_TTS_CRITERIA[0]["description"])


if __name__ == "__main__":
    unittest.main()
