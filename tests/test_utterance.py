"""Integration tests for the utterance detection and paragraph segmentation module."""

from aavaaz.features.utterance import (
    Paragraph,
    ParagraphSegmenter,
    Utterance,
    UtteranceDetector,
    detect_utterances_from_segments,
)


class TestUtteranceDetector:
    def test_single_segment_flushes(self):
        det = UtteranceDetector()
        result = det.process_segment("hello", 0.0, 1.0)
        assert result == []
        final = det.flush()
        assert final is not None
        assert final.text == "hello"
        assert final.start == 0.0
        assert final.end == 1.0

    def test_pause_breaks_utterance(self):
        det = UtteranceDetector(min_pause_seconds=0.5)
        r1 = det.process_segment("hello", 0.0, 1.0)
        assert r1 == []
        # Gap of 1.0s > 0.5s threshold
        r2 = det.process_segment("world", 2.0, 3.0)
        assert len(r2) == 1
        assert r2[0].text == "hello"

    def test_sentence_end_with_short_pause(self):
        det = UtteranceDetector(min_pause_seconds=1.0, sentence_end_pause_seconds=0.3)
        det.process_segment("hello.", 0.0, 1.0)
        # Gap of 0.4s > sentence_end_pause but < min_pause
        result = det.process_segment("next", 1.4, 2.0)
        assert len(result) == 1
        assert result[0].text == "hello."

    def test_speaker_change_breaks(self):
        det = UtteranceDetector(min_pause_seconds=5.0)
        det.process_segment("hi", 0.0, 1.0, speaker="A")
        result = det.process_segment("hey", 1.1, 2.0, speaker="B")
        assert len(result) == 1
        assert result[0].speaker == "A"

    def test_max_duration_breaks(self):
        det = UtteranceDetector(min_pause_seconds=5.0, max_utterance_seconds=2.0)
        det.process_segment("start", 0.0, 1.0)
        # Total duration would be 3.0 > max 2.0
        result = det.process_segment("end", 2.9, 3.0)
        assert len(result) == 1

    def test_continuity_without_break(self):
        det = UtteranceDetector(min_pause_seconds=1.0)
        det.process_segment("hello", 0.0, 1.0)
        # Gap of 0.1s < threshold, no break
        result = det.process_segment("world", 1.1, 2.0)
        assert result == []
        final = det.flush()
        assert final.text == "hello world"

    def test_reset(self):
        det = UtteranceDetector()
        det.process_segment("hello", 0.0, 1.0)
        det.reset()
        assert det.flush() is None

    def test_flush_empty(self):
        det = UtteranceDetector()
        assert det.flush() is None


class TestParagraphSegmenter:
    def test_single_utterance(self):
        seg = ParagraphSegmenter()
        utts = [Utterance(text="hello", start=0.0, end=1.0)]
        paras = seg.segment(utts)
        assert len(paras) == 1
        assert paras[0].text == "hello"

    def test_pause_splits_paragraph(self):
        seg = ParagraphSegmenter(paragraph_pause_seconds=2.0)
        utts = [
            Utterance(text="first", start=0.0, end=1.0),
            Utterance(text="second", start=4.0, end=5.0),  # gap 3.0 > 2.0
        ]
        paras = seg.segment(utts)
        assert len(paras) == 2

    def test_speaker_change_splits(self):
        seg = ParagraphSegmenter(paragraph_pause_seconds=10.0, split_on_speaker_change=True)
        utts = [
            Utterance(text="hi", start=0.0, end=1.0, speaker="A"),
            Utterance(text="hey", start=1.1, end=2.0, speaker="B"),
        ]
        paras = seg.segment(utts)
        assert len(paras) == 2

    def test_max_sentences(self):
        seg = ParagraphSegmenter(paragraph_pause_seconds=100.0, max_sentences_per_paragraph=2)
        utts = [
            Utterance(text="one", start=0.0, end=1.0),
            Utterance(text="two", start=1.1, end=2.0),
            Utterance(text="three", start=2.1, end=3.0),
        ]
        paras = seg.segment(utts)
        assert len(paras) == 2
        assert len(paras[0].utterances) == 2
        assert len(paras[1].utterances) == 1

    def test_empty_input(self):
        seg = ParagraphSegmenter()
        assert seg.segment([]) == []


class TestDetectUtterancesFromSegments:
    def test_basic(self):
        segments = [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": 5.0, "end": 6.0},
        ]
        utts = detect_utterances_from_segments(segments, min_pause=0.7)
        assert len(utts) == 2
        assert utts[0].text == "hello"
        assert utts[1].text == "world"


class TestUtteranceDataclass:
    def test_duration(self):
        u = Utterance(text="hi", start=1.0, end=3.5)
        assert u.duration == 2.5

    def test_to_dict(self):
        u = Utterance(text="hi", start=1.0, end=2.0, speaker="A")
        d = u.to_dict()
        assert d["text"] == "hi"
        assert d["speaker"] == "A"


class TestParagraphDataclass:
    def test_text_property(self):
        p = Paragraph(utterances=[
            Utterance(text="hello", start=0.0, end=1.0),
            Utterance(text="world", start=1.1, end=2.0),
        ])
        assert p.text == "hello world"

    def test_to_dict(self):
        p = Paragraph(utterances=[
            Utterance(text="hello", start=0.0, end=1.0),
        ])
        d = p.to_dict()
        assert d["num_sentences"] == 1
