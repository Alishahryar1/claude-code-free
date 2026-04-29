from pathlib import Path
from unittest.mock import patch

import pytest

from messaging.voice import PendingVoiceRegistry, VoiceTranscriptionService


@pytest.mark.asyncio
async def test_pending_voice_registry_tracks_voice_and_status_ids():
    registry = PendingVoiceRegistry()

    await registry.register("chat", "voice-1", "status-1")

    assert await registry.is_pending("chat", "voice-1") is True
    assert await registry.cancel("chat", "status-1") == ("voice-1", "status-1")
    assert await registry.is_pending("chat", "voice-1") is False


@pytest.mark.asyncio
async def test_pending_voice_registry_complete_removes_entries():
    registry = PendingVoiceRegistry()

    await registry.register("chat", "voice-1", "status-1")
    await registry.complete("chat", "voice-1", "status-1")

    assert await registry.cancel("chat", "voice-1") is None


@pytest.mark.asyncio
async def test_voice_transcription_service_runs_backend():
    service = VoiceTranscriptionService()

    with patch("messaging.transcription.transcribe_audio", return_value="hello"):
        text = await service.transcribe(
            Path("audio.ogg"),
            "audio/ogg",
            whisper_model="base",
            whisper_device="cpu",
        )

    assert text == "hello"
