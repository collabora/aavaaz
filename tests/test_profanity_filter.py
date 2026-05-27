"""
Tests for profanity filter (Test Matrix §5.9).

Covers masking modes, custom word lists, and edge cases.
"""

from aavaaz.features.profanity_filter import (
    filter_profanity,
    get_default_profanity_words,
)


class TestPartialMasking:
    """Partial masking mode (default): f**k."""

    def test_basic_partial_mask(self):
        result = filter_profanity("What the fuck is this")
        assert "fuck" not in result
        assert "f**k" in result

    def test_short_word_fully_masked(self):
        """Words <= 2 chars should be fully masked even in partial mode."""
        result = filter_profanity("What the ass happened", mode="partial")
        # "ass" is 3 chars: a*s
        assert "ass" not in result.lower()

    def test_longer_word_partial(self):
        result = filter_profanity("That bullshit is bad")
        # "bullshit" → "b******t"
        assert "bullshit" not in result
        assert result.count("*") > 0

    def test_case_insensitive(self):
        result = filter_profanity("FUCK this SHIT")
        assert "FUCK" not in result
        assert "SHIT" not in result

    def test_no_profanity_unchanged(self):
        text = "This is a perfectly clean sentence"
        assert filter_profanity(text) == text


class TestFullMasking:
    """Full masking mode: ****."""

    def test_full_mask(self):
        result = filter_profanity("What the fuck", mode="full")
        assert "fuck" not in result
        # Should be 4 asterisks
        assert "****" in result

    def test_full_mask_preserves_spaces(self):
        result = filter_profanity("this shit is bad", mode="full")
        words = result.split()
        assert len(words) == 4  # Same word count


class TestRemoveMode:
    """Remove mode: word is deleted."""

    def test_remove_word(self):
        result = filter_profanity("What the fuck is this", mode="remove")
        assert "fuck" not in result
        # Should not have double spaces
        assert "  " not in result

    def test_remove_preserves_other_words(self):
        result = filter_profanity("Hello damn world", mode="remove")
        assert "damn" not in result
        assert "Hello" in result
        assert "world" in result


class TestCustomMaskChar:
    """Custom masking character."""

    def test_hash_mask(self):
        result = filter_profanity("What the fuck", mask_char="#")
        assert "f##k" in result

    def test_underscore_mask(self):
        result = filter_profanity("That shit", mask_char="_", mode="full")
        assert "____" in result


class TestCustomWordList:
    """Custom and extra word lists."""

    def test_custom_words_replace_default(self):
        # Only filter "badword", not standard profanity
        result = filter_profanity("This badword is fuck", custom_words={"badword"})
        assert "badword" not in result
        assert "fuck" in result  # Not in custom list

    def test_extra_words_extend_default(self):
        result = filter_profanity("This jerk is bad", extra_words={"jerk"})
        assert "jerk" not in result

    def test_empty_custom_list(self):
        result = filter_profanity("What the fuck", custom_words=set())
        # Empty custom list means nothing gets filtered
        assert "fuck" in result


class TestEdgeCases:
    """Edge cases."""

    def test_empty_text(self):
        assert filter_profanity("") == ""

    def test_only_profanity(self):
        result = filter_profanity("fuck", mode="remove")
        assert result == ""

    def test_profanity_in_word_not_matched(self):
        """'assassin' contains 'ass' but shouldn't be filtered (word boundary)."""
        result = filter_profanity("The assassin escaped")
        # Should NOT filter "assassin" (word boundary check)
        assert "assassin" in result

    def test_get_default_words(self):
        words = get_default_profanity_words()
        assert isinstance(words, set)
        assert "fuck" in words
        assert len(words) > 10

    def test_multiple_profanities(self):
        result = filter_profanity("fuck this shit damn")
        assert "fuck" not in result
        assert "shit" not in result
        assert "damn" not in result
