"""Tests for voice note transcription."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from messaging.transcription import (
    MAX_AUDIO_SIZE_BYTES,
    transcribe_audio,
)


def test_transcribe_file_not_found_raises():
    """Non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        transcribe_audio(Path("/nonexistent/file.ogg"), "audio/ogg")


def test_transcribe_file_too_large_raises():
    """File exceeding max size raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"x" * (MAX_AUDIO_SIZE_BYTES + 1))
        path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="too large"):
            transcribe_audio(path, "audio/ogg")
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_local_success():
    """Local backend returns transcribed text."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg content")
        path = Path(f.name)
    try:
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), None)

        with patch(
            "messaging.transcription._get_local_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(path, "audio/ogg", whisper_model="large-v3-turbo")

        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_local_empty_segments_returns_no_speech():
    """Local backend with no speech returns placeholder."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg")
        path = Path(f.name)
    try:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), None)

        with patch(
            "messaging.transcription._get_local_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(path, "audio/ogg", whisper_model="large-v3-turbo")

        assert result == "(no speech detected)"
    finally:
        path.unlink(missing_ok=True)


def test_transcribe_local_import_error_raises():
    """Local backend when faster-whisper not installed raises ImportError."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(b"fake ogg")
        path = Path(f.name)
    try:
        with (
            patch(
                "messaging.transcription._get_local_model",
                side_effect=ImportError(
                    "Voice notes require the voice extra. Install with: uv sync --extra voice"
                ),
            ),
            pytest.raises(ImportError, match="voice extra"),
        ):
            transcribe_audio(path, "audio/ogg")
    finally:
        path.unlink(missing_ok=True)
