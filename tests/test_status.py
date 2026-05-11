"""Tests for calibrate/status.py — provider checks and main()."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import httpx


def _mk_resp(body=None, status=200, content=b"audio_data" * 50):
    m = MagicMock()
    m.status_code = status
    m.raise_for_status = MagicMock()
    if status >= 400:
        m.raise_for_status.side_effect = httpx.HTTPStatusError(
            "fail", request=MagicMock(), response=m,
        )
    if body is not None:
        m.json = MagicMock(return_value=body)
    m.content = content
    return m


def _mk_client(post_resp=None):
    client = AsyncMock()
    if post_resp:
        client.post = AsyncMock(return_value=post_resp)
    return client


class TestSilenceWav(unittest.TestCase):
    def test_generate_silence(self):
        from calibrate.status import _generate_silence_wav

        wav = _generate_silence_wav(duration_s=0.1, sample_rate=8000)
        self.assertTrue(wav.startswith(b"RIFF"))


class TestCheckProviders(unittest.IsolatedAsyncioTestCase):
    async def test_check_openai_ok(self):
        from calibrate.status import _check_openai

        client = _mk_client(_mk_resp({"choices": [{"message": {"content": "Hi"}}]}))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            result = await _check_openai(client)
        self.assertEqual(result, "llm")

    async def test_check_openai_no_choices(self):
        from calibrate.status import _check_openai

        client = _mk_client(_mk_resp({}))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_openai(client)

    async def test_check_openrouter_ok(self):
        from calibrate.status import _check_openrouter

        client = _mk_client(_mk_resp({"choices": [{"message": {"content": "Hi"}}]}))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            result = await _check_openrouter(client)
        self.assertEqual(result, "llm")

    async def test_check_openrouter_no_choices(self):
        from calibrate.status import _check_openrouter

        client = _mk_client(_mk_resp({}))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_openrouter(client)

    async def test_check_groq_ok(self):
        from calibrate.status import _check_groq

        client = _mk_client(_mk_resp(content=b"x" * 200))
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            result = await _check_groq(client)
        self.assertEqual(result, "tts")

    async def test_check_groq_empty(self):
        from calibrate.status import _check_groq

        client = _mk_client(_mk_resp(content=b"x"))
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_groq(client)

    async def test_check_google_missing_creds(self):
        from calibrate.status import _check_google

        client = _mk_client()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await _check_google(client)

    async def test_check_google_missing_file(self):
        from calibrate.status import _check_google

        client = _mk_client()
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent.json"}):
            with self.assertRaises(FileNotFoundError):
                await _check_google(client)

    async def test_check_google_missing_project(self):
        from calibrate.status import _check_google

        client = _mk_client()
        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "creds.json"
            creds.write_text("{}")
            with patch.dict(os.environ,
                            {"GOOGLE_APPLICATION_CREDENTIALS": str(creds)},
                            clear=True):
                with self.assertRaises(ValueError):
                    await _check_google(client)

    async def test_check_elevenlabs_ok(self):
        from calibrate.status import _check_elevenlabs

        client = _mk_client(_mk_resp(content=b"x" * 200))
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}):
            result = await _check_elevenlabs(client)
        self.assertEqual(result, "tts")

    async def test_check_elevenlabs_empty(self):
        from calibrate.status import _check_elevenlabs

        client = _mk_client(_mk_resp(content=b""))
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_elevenlabs(client)

    async def test_check_cartesia_ok(self):
        from calibrate.status import _check_cartesia

        client = _mk_client(_mk_resp(content=b"x" * 200))
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}):
            result = await _check_cartesia(client)
        self.assertEqual(result, "tts")

    async def test_check_cartesia_empty(self):
        from calibrate.status import _check_cartesia

        client = _mk_client(_mk_resp(content=b""))
        with patch.dict(os.environ, {"CARTESIA_API_KEY": "k"}):
            with self.assertRaises(ValueError):
                await _check_cartesia(client)

    async def test_check_deepgram_ok(self):
        from calibrate.status import _check_deepgram

        client = _mk_client(_mk_resp())
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "k"}):
            result = await _check_deepgram(client)
        self.assertEqual(result, "stt")


class TestCheckSingleProvider(unittest.IsolatedAsyncioTestCase):
    async def test_missing_env_vars_skips(self):
        from calibrate.status import _check_single_provider

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}
        with patch.dict(os.environ, {}, clear=True):
            result = await _check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["missing_vars"], ["OPENAI_API_KEY"])

    async def test_ok_path(self):
        from calibrate import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": AsyncMock(return_value="llm")}):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["check_type"], "llm")

    async def test_timeout(self):
        from calibrate import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}

        async def slow(*a, **kw):
            await asyncio.sleep(100)
            return "llm"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": slow}), \
             patch("asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "error")
        self.assertIn("Timed out", result["error"])

    async def test_http_status_error(self):
        from calibrate import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}

        async def fails(*a, **kw):
            resp = MagicMock()
            resp.status_code = 401
            raise httpx.HTTPStatusError("auth fail", request=MagicMock(), response=resp)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": fails}):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "error")
        self.assertIn("HTTP 401", result["error"])

    async def test_other_exception(self):
        from calibrate import status as S

        provider = {"name": "openai", "types": ["llm"], "env_vars": ["OPENAI_API_KEY"]}

        async def fails(*a, **kw):
            raise RuntimeError("x" * 200)  # long error message

        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
             patch.dict(S._CHECK_FUNCTIONS, {"openai": fails}):
            result = await S._check_single_provider(provider, MagicMock())
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["error"].endswith("..."))


class TestPrintResults(unittest.TestCase):
    def test_mixed_statuses(self):
        from calibrate.status import _print_results

        results = [
            {"name": "p1", "types": ["llm"], "key_set": True, "missing_vars": [],
             "status": "ok", "check_type": "llm", "error": None, "latency_ms": 500},
            {"name": "p2", "types": ["stt"], "key_set": False, "missing_vars": ["K"],
             "status": "skipped", "check_type": None, "error": None, "latency_ms": None},
            {"name": "p3", "types": ["tts"], "key_set": True, "missing_vars": [],
             "status": "error", "check_type": None, "error": "boom", "latency_ms": 100},
        ]
        _print_results(results)


class TestMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_default(self):
        from calibrate import status as S

        async def fake_check(provider, client):
            return {"name": provider["name"], "types": provider["types"],
                    "key_set": False, "missing_vars": ["X"],
                    "status": "skipped", "check_type": None,
                    "error": None, "latency_ms": None}

        with patch.object(S, "_check_single_provider", fake_check):
            result = await S.main()
        self.assertIsInstance(result, dict)

    async def test_main_quiet(self):
        from calibrate import status as S

        async def fake_check(provider, client):
            return {"name": provider["name"], "types": provider["types"],
                    "key_set": True, "missing_vars": [],
                    "status": "ok", "check_type": "llm",
                    "error": None, "latency_ms": 200}

        with patch.object(S, "_check_single_provider", fake_check):
            result = await S.main(quiet=True)
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertEqual(v["status"], "pass")


if __name__ == "__main__":
    unittest.main()
