"""Tests for simple helpers in calibrate/agent/run_simulation.py."""

import asyncio
import socket
import unittest
from unittest.mock import patch, AsyncMock, MagicMock


class TestIsBenignGoogleSttIdleError(unittest.TestCase):
    def test_benign_match(self):
        from calibrate.agent.run_simulation import _is_benign_google_stt_idle_error

        self.assertTrue(_is_benign_google_stt_idle_error(
            "GoogleSTTService error: 409 Stream timed out after receiving no more client requests"
        ))

    def test_not_benign(self):
        from calibrate.agent.run_simulation import _is_benign_google_stt_idle_error

        self.assertFalse(_is_benign_google_stt_idle_error("Some other error"))
        self.assertFalse(_is_benign_google_stt_idle_error(
            "GoogleSTTService: different error"
        ))


class TestCountAgentMessageTurns(unittest.TestCase):
    def test_empty_messages(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        self.assertEqual(count_agent_message_turns([]), 0)

    def test_single_user_run(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},  # streaming fragment
        ]
        self.assertEqual(count_agent_message_turns(messages), 1)

    def test_alternating_turns(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        self.assertEqual(count_agent_message_turns(messages), 3)

    def test_skip_non_dict(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            "not a dict",
            {"role": "user", "content": "a"},
        ]
        self.assertEqual(count_agent_message_turns(messages), 1)

    def test_role_none_treated_as_separator(self):
        from calibrate.agent.run_simulation import count_agent_message_turns

        messages = [
            {"role": "user", "content": "a"},
            {"no_role": "x"},
            {"role": "user", "content": "b"},
        ]
        # 2 turns because no_role doesn't reset (role is None, falls through)
        self.assertEqual(count_agent_message_turns(messages), 1)


class TestFindAvailablePort(unittest.TestCase):
    def test_returns_port(self):
        from calibrate.agent.run_simulation import find_available_port

        port = find_available_port()
        self.assertGreater(port, 0)

    def test_os_error_raises(self):
        from calibrate.agent import run_simulation as RS

        with patch("socket.socket", side_effect=OSError("no port")):
            with self.assertRaises(RuntimeError):
                RS.find_available_port()


class TestMetricsLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_frame(self):
        from collections import defaultdict
        from calibrate.agent.run_simulation import MetricsLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ttft = defaultdict(list)
        proc_time = defaultdict(list)
        ctx = MagicMock()
        ctx.get_messages.return_value = [{"role": "user"}]

        logger = MetricsLogger(ttft, proc_time, ctx)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "metrics",
            "data": {
                "ttfb": [{"processor": "p1", "value": 0.5}, {"processor": "p2", "value": 0}],
                "processing": [{"processor": "p1", "value": 0.3}],
            },
        }

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(MetricsLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(ttft["p1"], [0.5])
        self.assertEqual(proc_time["p1"], [0.3])

    async def test_process_frame_no_context_messages(self):
        from collections import defaultdict
        from calibrate.agent.run_simulation import MetricsLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        ctx = MagicMock()
        ctx.get_messages.return_value = []  # empty context — skip
        logger = MetricsLogger(defaultdict(list), defaultdict(list), ctx)

        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {"label": "rtvi-ai", "type": "metrics", "data": {}}

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(MetricsLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)


class TestIOLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_tts_text_frame(self):
        from calibrate.agent.run_simulation import IOLogger
        from pipecat.frames.frames import TTSTextFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        logger = IOLogger()
        frame = MagicMock(spec=TTSTextFrame)
        frame.text = "hello"

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(IOLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)


class TestSimulatedUserTurnIndexHook(unittest.IsolatedAsyncioTestCase):
    async def test_marks_pending(self):
        from calibrate.agent.run_simulation import SimulatedUserTurnIndexHook
        from pipecat.frames.frames import LLMFullResponseStartFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        adapter = MagicMock()
        adapter._sim_user_turn_pending = False
        hook = SimulatedUserTurnIndexHook(adapter)
        frame = MagicMock(spec=LLMFullResponseStartFrame)

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(SimulatedUserTurnIndexHook, "push_frame", AsyncMock()):
            await hook.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertTrue(adapter._sim_user_turn_pending)


class TestSTTLogger(unittest.IsolatedAsyncioTestCase):
    async def test_process_frame_user_transcription(self):
        from calibrate.agent.run_simulation import STTLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        outputs = []
        adapter = MagicMock()
        adapter._stt_turn_index = 1
        logger = STTLogger(outputs, adapter)
        # logger sets last_turn_index=0, but adapter has turn=1 → append new
        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "user-transcription",
            "data": {"text": "hello", "final": True},
        }

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(STTLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(outputs[-1], "hello")

    async def test_process_frame_continues_turn(self):
        from calibrate.agent.run_simulation import STTLogger
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        outputs = []
        adapter = MagicMock()
        adapter._stt_turn_index = 0  # same turn
        logger = STTLogger(outputs, adapter)
        frame = MagicMock(spec=InputTransportMessageFrame)
        frame.message = {
            "label": "rtvi-ai",
            "type": "user-transcription",
            "data": {"text": "more", "final": True},
        }

        with patch.object(FrameProcessor, "process_frame", AsyncMock()), \
             patch.object(STTLogger, "push_frame", AsyncMock()):
            await logger.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Empty turn appends to outputs[-1] which is ""
        self.assertEqual(outputs[-1], "more")


class TestSilencePadder(unittest.IsolatedAsyncioTestCase):
    async def test_init(self):
        from calibrate.agent.run_simulation import SilencePadder

        padder = SilencePadder(silence_duration_ms=200, chunk_ms=20)
        self.assertEqual(padder._silence_duration_ms, 200)
        self.assertEqual(padder._chunk_ms, 20)


if __name__ == "__main__":
    unittest.main()
