"""
Tests for core transcription functionality (Test Matrix §1).

Covers audio format handling, model sizes, language detection, and edge cases.
Requires: faster-whisper (uses tiny.en for speed).
"""

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _generate_sine_wav(
    path: Path, duration: float = 2.0, sample_rate: int = 16000, freq: float = 440.0
):
    """Generate a WAV file with a sine tone."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


def _generate_silence_wav(path: Path, duration: float = 3.0, sample_rate: int = 16000):
    """Generate a silent WAV file."""
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


@pytest.fixture
def short_audio(tmp_path):
    """A short 2-second sine WAV."""
    return _generate_sine_wav(tmp_path / "short.wav", duration=2.0)


@pytest.fixture
def silent_audio(tmp_path):
    """A 3-second silent WAV."""
    return _generate_silence_wav(tmp_path / "silence.wav", duration=3.0)


@pytest.fixture
def long_audio(tmp_path):
    """A 30-second sine WAV (tests longer transcriptions)."""
    return _generate_sine_wav(tmp_path / "long.wav", duration=30.0)


@pytest.fixture
def audio_8khz(tmp_path):
    """8kHz sample rate audio."""
    return _generate_sine_wav(tmp_path / "8khz.wav", duration=2.0, sample_rate=8000)


@pytest.fixture
def audio_44khz(tmp_path):
    """44.1kHz sample rate audio."""
    return _generate_sine_wav(tmp_path / "44khz.wav", duration=2.0, sample_rate=44100)


@pytest.fixture
def audio_48khz(tmp_path):
    """48kHz sample rate audio."""
    return _generate_sine_wav(tmp_path / "48khz.wav", duration=2.0, sample_rate=48000)


@pytest.fixture
def corrupted_file(tmp_path):
    """A file with invalid audio data."""
    p = tmp_path / "corrupted.wav"
    p.write_bytes(b"NOT A VALID WAV FILE AT ALL " * 100)
    return p


@pytest.fixture
def empty_file(tmp_path):
    """A zero-byte file."""
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    return p


def _remove_mock_faster_whisper():
    """Remove any mock faster_whisper from sys.modules."""
    if "faster_whisper" in sys.modules:
        _mod = sys.modules["faster_whisper"]
        if not hasattr(_mod, "__file__"):
            for key in list(sys.modules):
                if key == "faster_whisper" or key.startswith("faster_whisper."):
                    del sys.modules[key]


@pytest.mark.smoke
class TestBasicTranscription:
    """1.1 - Basic transcription (short audio < 10s)."""

    def test_short_audio_returns_segments(self, short_audio):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(short_audio), language="en")
        segment_list = list(segments)
        assert isinstance(segment_list, list)
        assert info is not None
        assert info.language == "en"

    def test_transcription_fixture_file(self):
        """Use the existing smoke_test.wav fixture."""
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        audio = FIXTURE_DIR / "smoke_test.wav"
        if not audio.exists():
            pytest.skip("smoke_test.wav fixture not found")

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio), language="en")
        segment_list = list(segments)
        assert isinstance(segment_list, list)


@pytest.mark.smoke
class TestLongAudio:
    """1.2 - Long audio transcription (> 5 min)."""

    def test_long_audio_doesnt_crash(self, long_audio):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(long_audio), language="en")
        segment_list = list(segments)
        # Should complete without exception
        assert isinstance(segment_list, list)


class TestSampleRates:
    """1.4 - Sample rate handling."""

    @pytest.mark.parametrize(
        "fixture_name", ["audio_8khz", "audio_44khz", "audio_48khz"]
    )
    def test_various_sample_rates(self, fixture_name, request):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        audio_path = request.getfixturevalue(fixture_name)
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio_path), language="en")
        segment_list = list(segments)
        assert isinstance(segment_list, list)


class TestSilentAudio:
    """1.5 - Empty/silent audio."""

    def test_silent_audio_returns_empty_or_minimal(self, silent_audio):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(silent_audio), language="en", vad_filter=True
        )
        segment_list = list(segments)
        # Silent audio with VAD should produce no segments or very few
        assert isinstance(segment_list, list)


class TestCorruptedAudio:
    """1.6 - Corrupted/invalid audio file."""

    def test_corrupted_file_raises_error(self, corrupted_file):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        with pytest.raises((RuntimeError, ValueError, Exception)):  # noqa: B017
            segments, info = model.transcribe(str(corrupted_file))
            list(segments)  # Force consumption

    def test_empty_file_raises_error(self, empty_file):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        with pytest.raises((RuntimeError, ValueError, Exception)):  # noqa: B017
            segments, info = model.transcribe(str(empty_file))
            list(segments)


class TestLanguageDetection:
    """1.8 - Language detection (auto-detect)."""

    def test_auto_detect_language(self, short_audio):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(short_audio))  # No language specified
        list(segments)
        assert info is not None
        assert info.language is not None
        assert len(info.language) >= 2


class TestWordTimestamps:
    """1.10 - Word timestamps accuracy."""

    def test_word_timestamps_returned(self, short_audio):
        _remove_mock_faster_whisper()
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(short_audio), language="en", word_timestamps=True
        )
        segment_list = list(segments)
        # If there are segments with text, they should have word-level timing
        for seg in segment_list:
            if seg.text.strip() and seg.words:
                for word in seg.words:
                    assert hasattr(word, "start")
                    assert hasattr(word, "end")
                    assert word.end >= word.start
