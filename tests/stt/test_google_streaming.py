"""Test _transcribe_google_streaming."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestGoogleStreaming(unittest.TestCase):
    def test_google_streaming_happy(self):
        from calibrate.stt import eval as E

        # Mock the response chunks
        result = MagicMock()
        result.alternatives = [MagicMock(transcript="hello world")]

        response = MagicMock()
        response.results = [result]

        empty_response = MagicMock()
        empty_response.results = [MagicMock(alternatives=[MagicMock(transcript="")])]

        fake_client = MagicMock()
        fake_client.streaming_recognize.return_value = iter([response, empty_response])

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT_ID": "proj"}), \
             patch.object(E, "SpeechClient", return_value=fake_client), \
             patch.object(E, "load_audio", return_value=b"\x00" * 100000), \
             patch("time.sleep"):  # Skip the pacing sleep
            result = E._transcribe_google_streaming(
                Path("/tmp/x.wav"), "en-US"
            )
        self.assertIn("hello world", result)


if __name__ == "__main__":
    unittest.main()
