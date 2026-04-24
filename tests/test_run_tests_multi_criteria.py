"""
Tests for multi-criteria LLM test evaluation.

Covers:
- run_test_external with list-of-criteria returns per-criterion judge_results
- run_test_external with string criteria falls back to flat {reasoning, match}
- "passed" is True iff ALL criteria match
- _aggregate_criteria aggregates pass rates correctly across test cases

Run with:
    python -m pytest tests/test_run_tests_multi_criteria.py -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


def _make_httpx_response(body: dict, status: int = 200):
    import httpx
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    if status >= 400:
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}", request=MagicMock(), response=mock
        )
    return mock


def _patch_httpx(response_body: dict, status: int = 200):
    mock_resp = _make_httpx_response(response_body, status)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


# ---------------------------------------------------------------------------
# run_test_external with multi-criteria
# ---------------------------------------------------------------------------


class TestRunTestExternalMultiCriteria(unittest.IsolatedAsyncioTestCase):

    async def _run(self, agent_response, criteria, judge_result):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        fake_body = {"response": agent_response, "tool_calls": []}
        evaluation = {"type": "response", "criteria": criteria}

        mock_judge = AsyncMock(return_value=judge_result)
        ctx, _ = _patch_httpx(fake_body)
        with ctx, patch(
            "calibrate.llm.run_tests.test_response_llm_judge", mock_judge
        ):
            return await run_test_external(
                chat_history=[{"role": "user", "content": "Hi"}],
                evaluation=evaluation,
                agent=agent,
            )

    async def test_all_criteria_match_passes(self):
        result = await self._run(
            agent_response="Hello, how can I help?",
            criteria=[
                {"name": "greeting", "description": "agent greets"},
                {"name": "helpful", "description": "offers help"},
            ],
            judge_result={
                "greeting": {"match": True, "reasoning": "greeted"},
                "helpful": {"match": True, "reasoning": "offered help"},
            },
        )
        self.assertTrue(result["metrics"]["passed"])
        self.assertIn("judge_results", result["metrics"])
        self.assertEqual(
            result["metrics"]["judge_results"]["greeting"]["match"], True
        )
        self.assertEqual(
            result["metrics"]["judge_results"]["helpful"]["match"], True
        )

    async def test_one_criterion_fails_overall_fails(self):
        result = await self._run(
            agent_response="Hello.",
            criteria=[
                {"name": "greeting", "description": "agent greets"},
                {"name": "helpful", "description": "offers help"},
            ],
            judge_result={
                "greeting": {"match": True, "reasoning": "greeted"},
                "helpful": {"match": False, "reasoning": "did not offer help"},
            },
        )
        self.assertFalse(result["metrics"]["passed"])
        # Reasoning surfaces the first failing criterion
        self.assertEqual(
            result["metrics"]["reasoning"], "did not offer help"
        )

    async def test_all_fail(self):
        result = await self._run(
            agent_response="go away",
            criteria=[
                {"name": "greeting", "description": "agent greets"},
                {"name": "helpful", "description": "offers help"},
            ],
            judge_result={
                "greeting": {"match": False, "reasoning": "no greeting"},
                "helpful": {"match": False, "reasoning": "not helpful"},
            },
        )
        self.assertFalse(result["metrics"]["passed"])

    async def test_single_string_criteria_still_works(self):
        """Backward compat: string criteria returns flat {reasoning, match}."""
        result = await self._run(
            agent_response="Hello!",
            criteria="Agent should greet",
            judge_result={"match": True, "reasoning": "greeted"},
        )
        self.assertTrue(result["metrics"]["passed"])
        self.assertEqual(result["metrics"]["reasoning"], "greeted")
        # When criteria was string, judge_results should NOT be set
        self.assertNotIn("judge_results", result["metrics"])

    async def test_all_pass_reasoning_message(self):
        result = await self._run(
            agent_response="Hello, how can I help?",
            criteria=[
                {"name": "greeting", "description": "agent greets"},
            ],
            judge_result={
                "greeting": {"match": True, "reasoning": "greeted"},
            },
        )
        self.assertEqual(
            result["metrics"]["reasoning"], "All criteria passed"
        )

    async def test_rating_criterion_does_not_affect_passed_flag(self):
        """Rating criteria are informational — they don't fail the test."""
        rating = {
            "name": "fluency",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "description": "rate",
        }
        result = await self._run(
            agent_response="Hello!",
            criteria=[rating],
            judge_result={"fluency": {"score": 2, "reasoning": "meh"}},
        )
        # Low rating score does NOT fail the test
        self.assertTrue(result["metrics"]["passed"])
        self.assertEqual(
            result["metrics"]["judge_results"]["fluency"]["score"], 2
        )

    async def test_mixed_binary_fails_overrides_rating(self):
        """A failing binary criterion fails the test even if rating is high."""
        binary = {"name": "accuracy", "description": "correct"}
        rating = {
            "name": "fluency",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "description": "rate",
        }
        result = await self._run(
            agent_response="wrong answer",
            criteria=[binary, rating],
            judge_result={
                "accuracy": {"match": False, "reasoning": "wrong"},
                "fluency": {"score": 5, "reasoning": "very fluent"},
            },
        )
        self.assertFalse(result["metrics"]["passed"])
        self.assertEqual(result["metrics"]["reasoning"], "wrong")


# ---------------------------------------------------------------------------
# _aggregate_criteria helper
# ---------------------------------------------------------------------------


class TestAggregateCriteria(unittest.TestCase):
    def test_empty_list_returns_empty_dict(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        self.assertEqual(_aggregate_criteria([]), {})

    def test_tool_call_tests_excluded(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        results = [
            {
                "metrics": {"passed": True},
                "test_case": {"evaluation": {"type": "tool_call"}},
            },
            {
                "metrics": {"passed": False},
                "test_case": {"evaluation": {"type": "tool_call"}},
            },
        ]
        self.assertEqual(_aggregate_criteria(results), {})

    def test_string_criteria_aggregates_under_criteria_key(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        results = [
            {
                "metrics": {"passed": True, "reasoning": "ok"},
                "test_case": {"evaluation": {"type": "response", "criteria": "X"}},
            },
            {
                "metrics": {"passed": False, "reasoning": "bad"},
                "test_case": {"evaluation": {"type": "response", "criteria": "Y"}},
            },
        ]
        agg = _aggregate_criteria(results)
        self.assertIn("criteria", agg)
        self.assertEqual(agg["criteria"]["passed"], 1)
        self.assertEqual(agg["criteria"]["total"], 2)
        self.assertEqual(agg["criteria"]["pass_rate"], 50.0)

    def test_multi_criteria_counted_independently(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        results = [
            {
                "metrics": {
                    "passed": False,
                    "reasoning": "x",
                    "judge_results": {
                        "accuracy": {"match": True, "reasoning": "ok"},
                        "tone": {"match": False, "reasoning": "rude"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [
                            {"name": "accuracy", "description": "a"},
                            {"name": "tone", "description": "t"},
                        ],
                    }
                },
            },
        ]
        agg = _aggregate_criteria(results)
        self.assertEqual(
            agg["accuracy"],
            {"type": "binary", "passed": 1, "total": 1, "pass_rate": 100.0},
        )
        self.assertEqual(
            agg["tone"],
            {"type": "binary", "passed": 0, "total": 1, "pass_rate": 0.0},
        )

    def test_rating_criterion_aggregates_mean(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        rating_criterion = {
            "name": "fluency",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "description": "rate",
        }
        results = [
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "All criteria passed",
                    "judge_results": {
                        "fluency": {"score": 4, "reasoning": "ok"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [rating_criterion],
                    }
                },
            },
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "All criteria passed",
                    "judge_results": {
                        "fluency": {"score": 2, "reasoning": "ok"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [rating_criterion],
                    }
                },
            },
        ]
        agg = _aggregate_criteria(results)
        self.assertEqual(
            agg["fluency"],
            {
                "type": "rating",
                "mean": 3.0,
                "min": 2,
                "max": 4,
                "count": 2,
                "scale_min": 1,
                "scale_max": 5,
            },
        )

    def test_mixed_binary_and_rating_criteria(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        binary = {"name": "accuracy", "description": "correct"}
        rating = {
            "name": "fluency",
            "type": "rating",
            "scale_min": 1,
            "scale_max": 5,
            "description": "rate",
        }
        results = [
            {
                "metrics": {
                    "passed": True,
                    "reasoning": "All criteria passed",
                    "judge_results": {
                        "accuracy": {"match": True, "reasoning": "ok"},
                        "fluency": {"score": 5, "reasoning": "ok"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [binary, rating],
                    }
                },
            },
        ]
        agg = _aggregate_criteria(results)
        self.assertEqual(agg["accuracy"]["type"], "binary")
        self.assertEqual(agg["accuracy"]["pass_rate"], 100.0)
        self.assertEqual(agg["fluency"]["type"], "rating")
        self.assertEqual(agg["fluency"]["mean"], 5.0)
        self.assertEqual(agg["fluency"]["scale_min"], 1)
        self.assertEqual(agg["fluency"]["scale_max"], 5)

    def test_mixed_single_and_multi_criteria(self):
        from calibrate.llm.run_tests import _aggregate_criteria
        results = [
            # Response test — string criteria (passes)
            {
                "metrics": {"passed": True, "reasoning": "ok"},
                "test_case": {"evaluation": {"type": "response", "criteria": "X"}},
            },
            # Response test — multi-criteria (one passes, one fails)
            {
                "metrics": {
                    "passed": False,
                    "reasoning": "rude",
                    "judge_results": {
                        "accuracy": {"match": True, "reasoning": "ok"},
                        "tone": {"match": False, "reasoning": "rude"},
                    },
                },
                "test_case": {
                    "evaluation": {
                        "type": "response",
                        "criteria": [
                            {"name": "accuracy", "description": "a"},
                            {"name": "tone", "description": "t"},
                        ],
                    }
                },
            },
            # Tool call test — skipped
            {
                "metrics": {"passed": True},
                "test_case": {"evaluation": {"type": "tool_call"}},
            },
        ]
        agg = _aggregate_criteria(results)
        self.assertEqual(set(agg.keys()), {"criteria", "accuracy", "tone"})
        self.assertEqual(agg["criteria"]["total"], 1)
        self.assertEqual(agg["accuracy"]["total"], 1)
        self.assertEqual(agg["tone"]["total"], 1)


if __name__ == "__main__":
    unittest.main()
