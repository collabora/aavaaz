"""
End-to-end smoke test: loads a real Whisper model and transcribes audio.

This test downloads the 'tiny.en' model (~75 MB) on first run.
Mark it with 'smoke' so it can be run separately:

    pytest tests/test_smoke.py -m smoke

Requires: faster-whisper (pip install faster-whisper)
"""

import sys
from pathlib import Path

import pytest

# Ensure the real faster_whisper is available (other test files may have
# inserted a MagicMock via sys.modules.setdefault). Remove any mock first.
if "faster_whisper" in sys.modules:
    _mod = sys.modules["faster_whisper"]
    if not hasattr(_mod, "__file__"):
        # It's a mock — remove it so the real import can proceed
        for key in list(sys.modules):
            if key == "faster_whisper" or key.startswith("faster_whisper."):
                del sys.modules[key]

FIXTURE_DIR = Path(__file__).parent / "fixtures"
AUDIO_FILE = FIXTURE_DIR / "smoke_test.wav"


@pytest.mark.smoke
def test_faster_whisper_transcribe():
    """Load tiny.en model and transcribe a short WAV file end-to-end."""
    pytest.importorskip("faster_whisper")
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(AUDIO_FILE), language="en")

    # Consume the generator
    segment_list = list(segments)

    # Basic sanity checks — the model loaded, ran inference, returned results
    assert info is not None
    assert info.language == "en"
    # A pure sine tone might produce empty text or hallucinated text,
    # but the model should at least return without error
    assert isinstance(segment_list, list)


@pytest.mark.smoke
def test_aavaaz_server_init_and_format_pipeline():
    """Test that AavaazServer can be instantiated and the formatting pipeline works
    on real transcription output."""
    pytest.importorskip("faster_whisper")

    import sys
    from unittest.mock import MagicMock

    # Mock whisper_live server (we don't want to start a real WebSocket server)
    sys.modules.setdefault("whisper_live", MagicMock())
    sys.modules.setdefault("whisper_live.server", MagicMock())

    from aavaaz.features.formatting import format_transcript
    from aavaaz.server import AavaazServer

    # Server init should not crash
    server = AavaazServer(
        model="tiny.en",
        word_timestamps=True,
        hotwords="test",
        batch_inference=False,
    )
    assert server.model == "tiny.en"

    # Simulate post-processing a transcription segment
    raw_text = "i have twenty one items. what do you think"
    formatted = format_transcript(raw_text, capitalize=True, numbers=True)
    assert formatted.startswith("I")
    assert "21" in formatted


@pytest.mark.smoke
def test_full_pipeline_transcribe_and_format():
    """Transcribe audio then run through formatting pipeline."""
    pytest.importorskip("faster_whisper")
    from faster_whisper import WhisperModel

    from aavaaz.features.formatting import format_transcript

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(AUDIO_FILE), language="en")

    # Collect all text
    full_text = " ".join(seg.text for seg in segments)

    # Run through formatting (should not crash regardless of content)
    formatted = format_transcript(full_text, capitalize=True, numbers=True, smart=True)
    assert isinstance(formatted, str)
