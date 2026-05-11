"""Tests for RTVIMessageFrameAdapter helper methods that don't require pipecat runtime."""

import asyncio
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


def _make_adapter(**overrides):
    from calibrate.agent.run_simulation import RTVIMessageFrameAdapter

    ctx = MagicMock()
    ctx.get_messages.return_value = []

    audio_buffer = MagicMock()

    defaults = dict(
        context=ctx,
        audio_buffer=audio_buffer,
        interrupt_probability=0.0,
        tool_calls=[],
        stt_outputs=[],
        ttft=defaultdict(list),
        processing_time=defaultdict(list),
        output_dir="/tmp",
        audio_save_dir="/tmp",
        agent_speaks_first=True,
        max_turns=10,
    )
    defaults.update(overrides)
    return RTVIMessageFrameAdapter(**defaults)


class TestAssignNextTranscriptAudioLine(unittest.TestCase):
    def test_monotonic_increment(self):
        adapter = _make_adapter()
        line1 = adapter._assign_next_transcript_audio_line(role="bot")
        line2 = adapter._assign_next_transcript_audio_line(role="user")
        line3 = adapter._assign_next_transcript_audio_line(role="bot")
        self.assertEqual([line1, line2, line3], [1, 2, 3])


class TestBuildSerializedTranscript(unittest.TestCase):
    def test_empty(self):
        adapter = _make_adapter()
        result = adapter._build_serialized_transcript()
        self.assertEqual(result, [])

    def test_role_flipping(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "hi"},      # → assistant
            {"role": "assistant", "content": "hello"},  # → user
        ]
        adapter = _make_adapter(context=ctx)
        result = adapter._build_serialized_transcript()
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(result[0]["content"], "hi")
        self.assertEqual(result[1]["role"], "user")

    def test_merges_consecutive_same_role(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        adapter = _make_adapter(context=ctx)
        result = adapter._build_serialized_transcript()
        # Both became assistant, merged
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "a b")

    def test_with_end_reason(self):
        adapter = _make_adapter()
        result = adapter._build_serialized_transcript(end_reason="max_turns")
        self.assertEqual(result[-1]["role"], "end_reason")
        self.assertEqual(result[-1]["content"], "max_turns")

    def test_tool_calls_inserted(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "hi"},
        ]
        tool_calls = [{
            "position": 0,
            "data": {
                "tool_call_id": "call_1",
                "function_name": "foo",
                "args": {"x": 1},
            },
        }]
        adapter = _make_adapter(context=ctx, tool_calls=tool_calls)
        result = adapter._build_serialized_transcript()
        # First entry is tool_calls, then the message
        self.assertEqual(result[0]["role"], "assistant")
        self.assertIn("tool_calls", result[0])

    def test_tool_calls_after_messages(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            {"role": "user", "content": "hi"},
        ]
        tool_calls = [{
            "position": 5,
            "data": {
                "tool_call_id": "call_x",
                "function_name": "y",
                "args": {},
            },
        }]
        adapter = _make_adapter(context=ctx, tool_calls=tool_calls)
        result = adapter._build_serialized_transcript()
        # Tool call at position 5 (after all messages) appended
        self.assertEqual(result[-1]["role"], "assistant")
        self.assertIn("tool_calls", result[-1])

    def test_skip_non_dict_messages(self):
        ctx = MagicMock()
        ctx.get_messages.return_value = [
            "not a dict",
            {"role": "user", "content": "hi"},
        ]
        adapter = _make_adapter(context=ctx)
        result = adapter._build_serialized_transcript()
        self.assertEqual(len(result), 1)


class TestSaveTranscript(unittest.TestCase):
    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(output_dir=tmp)
            adapter._save_transcript([{"role": "assistant", "content": "hi"}])
            from calibrate.agent.run_simulation import TRANSCRIPT_FILE_NAME
            transcript_path = Path(tmp) / TRANSCRIPT_FILE_NAME
            self.assertTrue(transcript_path.exists())
            data = json.loads(transcript_path.read_text())
            self.assertEqual(data[0]["role"], "assistant")


class TestEnsureBotTranscriptLineForCurrentTurn(unittest.IsolatedAsyncioTestCase):
    async def test_not_awaiting_returns_early(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = False
        await adapter._ensure_bot_transcript_line_for_current_turn()  # No-op

    async def test_too_short_lexical_returns(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = ""
        await adapter._ensure_bot_transcript_line_for_current_turn(spoken_fragment="a")
        self.assertTrue(adapter._awaiting_first_bot_audio_chunk)

    async def test_no_alpha_returns(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = "123!"
        await adapter._ensure_bot_transcript_line_for_current_turn()
        self.assertTrue(adapter._awaiting_first_bot_audio_chunk)

    async def test_continues_bot_role(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = "hello"
        adapter._active_transcript_audio_role = "bot"
        adapter._active_transcript_audio_index = 5
        await adapter._ensure_bot_transcript_line_for_current_turn()
        self.assertFalse(adapter._awaiting_first_bot_audio_chunk)

    async def test_new_bot_line(self):
        adapter = _make_adapter()
        adapter._awaiting_first_bot_audio_chunk = True
        adapter._text_buffer = "hello"
        adapter._active_transcript_audio_role = "user"
        await adapter._ensure_bot_transcript_line_for_current_turn()
        self.assertEqual(adapter._stt_turn_index, 1)
        self.assertEqual(adapter._active_transcript_audio_role, "bot")


class TestFlushPendingBotAudio(unittest.IsolatedAsyncioTestCase):
    async def test_no_pending(self):
        adapter = _make_adapter()
        await adapter._flush_pending_bot_audio()
        self.assertEqual(adapter._pending_bot_audio_frames, [])

    async def test_flushes(self):
        from calibrate.agent import run_simulation as RS

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(audio_save_dir=tmp)
            adapter._active_transcript_audio_index = 1

            fake_frame = MagicMock()
            fake_frame.audio = b"\x00" * 100
            fake_frame.sample_rate = 16000
            fake_frame.num_channels = 1
            adapter._pending_bot_audio_frames = [fake_frame]

            with patch.object(RS, "save_audio_chunk", AsyncMock()):
                await adapter._flush_pending_bot_audio()
            self.assertEqual(adapter._pending_bot_audio_frames, [])


class TestResetBuffers(unittest.IsolatedAsyncioTestCase):
    async def test_clears_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _make_adapter(output_dir=tmp)
            adapter._text_buffer = "hi"
            adapter._heard_text_buffer = "hi"
            adapter._spoken_text_buffer = "hi"
            adapter._turn_index = 1
            await adapter._reset_buffers()
            self.assertEqual(adapter._text_buffer, "")
            self.assertEqual(adapter._spoken_text_buffer, "")


if __name__ == "__main__":
    unittest.main()
