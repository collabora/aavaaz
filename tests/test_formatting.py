"""Integration tests for the formatting module."""

from aavaaz.features.formatting import (
    _capitalize_sentences,
    _collapse_whitespace,
    _replace_spoken_numbers,
    _words_to_number,
    format_transcript,
    smart_format,
)


class TestSpokenNumbers:
    def test_simple_number(self):
        assert _replace_spoken_numbers("I have twenty one apples") == "I have 21 apples"

    def test_large_number(self):
        assert _replace_spoken_numbers("one thousand two hundred thirty four") == "1234"

    def test_hundred(self):
        assert _replace_spoken_numbers("three hundred") == "300"

    def test_million(self):
        assert _replace_spoken_numbers("two million") == "2000000"

    def test_no_number_words(self):
        assert _replace_spoken_numbers("hello world") == "hello world"

    def test_number_with_punctuation(self):
        result = _replace_spoken_numbers("I have twenty one.")
        assert "21" in result

    def test_words_to_number_empty(self):
        assert _words_to_number([]) is None

    def test_words_to_number_and_only(self):
        assert _words_to_number(["and"]) is None


class TestCapitalization:
    def test_start_of_string(self):
        assert _capitalize_sentences("hello") == "Hello"

    def test_after_period(self):
        assert _capitalize_sentences("done. next") == "Done. Next"

    def test_after_question(self):
        assert _capitalize_sentences("what? really") == "What? Really"

    def test_empty(self):
        assert _capitalize_sentences("") == ""


class TestCollapseWhitespace:
    def test_multiple_spaces(self):
        assert _collapse_whitespace("hello    world") == "hello world"

    def test_leading_trailing(self):
        assert _collapse_whitespace("  hi  ") == "hi"


class TestFormatTranscript:
    def test_all_options(self):
        result = format_transcript("  hello world.  how  are you  ", capitalize=True, numbers=False)
        assert result.startswith("Hello")
        assert "  " not in result

    def test_numbers_enabled(self):
        result = format_transcript("I have twenty apples", numbers=True)
        assert "20" in result

    def test_empty(self):
        assert format_transcript("") == ""

    def test_none(self):
        assert format_transcript(None) is None


class TestSmartFormat:
    def test_currency(self):
        result = smart_format("50 dollars")
        assert "$50" in result

    def test_percentage(self):
        result = smart_format("50 percent")
        assert "50%" in result
