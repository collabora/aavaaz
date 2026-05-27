"""
Tests for PII redaction (Test Matrix §5.4-5.8).

Covers all PII types: SSN, credit cards, phone numbers, emails, IP addresses.
"""

from aavaaz.features.pii_redaction import (
    get_supported_pii_types,
    redact_pii,
)


class TestPIISSN:
    """5.4 - PII redaction: SSN."""

    def test_ssn_with_dashes(self):
        text = "My SSN is 123-45-6789 thanks"
        result = redact_pii(text, pii_types={"ssn"})
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result

    def test_ssn_with_spaces(self):
        text = "SSN: 123 45 6789"
        result = redact_pii(text, pii_types={"ssn"})
        assert "[SSN_REDACTED]" in result

    def test_ssn_no_separators(self):
        text = "Number is 123456789"
        result = redact_pii(text, pii_types={"ssn"})
        assert "[SSN_REDACTED]" in result

    def test_no_ssn_in_text(self):
        text = "No sensitive data here"
        result = redact_pii(text, pii_types={"ssn"})
        assert result == text


class TestPIICreditCard:
    """5.5 - PII redaction: credit card numbers."""

    def test_visa_card(self):
        text = "Card: 4111-1111-1111-1111"
        result = redact_pii(text, pii_types={"credit_card"})
        assert "4111" not in result
        assert "[CARD_REDACTED]" in result

    def test_card_with_spaces(self):
        text = "Pay with 4111 1111 1111 1111"
        result = redact_pii(text, pii_types={"credit_card"})
        assert "[CARD_REDACTED]" in result

    def test_card_no_separators(self):
        text = "Card number 4111111111111111"
        result = redact_pii(text, pii_types={"credit_card"})
        assert "[CARD_REDACTED]" in result

    def test_mastercard(self):
        text = "Card: 5500-0000-0000-0004"
        result = redact_pii(text, pii_types={"credit_card"})
        assert "[CARD_REDACTED]" in result


class TestPIIPhone:
    """5.6 - PII redaction: phone numbers."""

    def test_us_phone_with_dashes(self):
        text = "Call me at 555-123-4567"
        result = redact_pii(text, pii_types={"phone"})
        assert "555-123-4567" not in result
        assert "[PHONE_REDACTED]" in result

    def test_us_phone_with_area_code(self):
        text = "Phone: (555) 123-4567"
        result = redact_pii(text, pii_types={"phone"})
        assert "[PHONE_REDACTED]" in result

    def test_phone_with_country_code(self):
        text = "Number: +1-555-123-4567"
        result = redact_pii(text, pii_types={"phone"})
        assert "[PHONE_REDACTED]" in result

    def test_short_number_not_matched(self):
        text = "The year 2024 is not a phone"
        result = redact_pii(text, pii_types={"phone"})
        # Short numbers shouldn't be matched as phone numbers
        assert "2024" in result


class TestPIIEmail:
    """5.7 - PII redaction: email addresses."""

    def test_standard_email(self):
        text = "Email me at user@example.com"
        result = redact_pii(text, pii_types={"email"})
        assert "user@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_email_with_dots(self):
        text = "Contact: first.last@company.co.uk"
        result = redact_pii(text, pii_types={"email"})
        assert "[EMAIL_REDACTED]" in result

    def test_email_with_plus(self):
        text = "Send to user+tag@gmail.com"
        result = redact_pii(text, pii_types={"email"})
        assert "[EMAIL_REDACTED]" in result

    def test_no_email_in_text(self):
        text = "This has no email addresses"
        result = redact_pii(text, pii_types={"email"})
        assert result == text


class TestPIIIPAddress:
    """5.8 - PII redaction: IP addresses."""

    def test_ipv4_address(self):
        text = "Server IP: 192.168.1.100"
        result = redact_pii(text, pii_types={"ip_address"})
        assert "192.168.1.100" not in result
        assert "[IP_REDACTED]" in result

    def test_public_ip(self):
        text = "Connected from 203.0.113.42"
        result = redact_pii(text, pii_types={"ip_address"})
        assert "[IP_REDACTED]" in result

    def test_localhost(self):
        text = "Localhost is 127.0.0.1"
        result = redact_pii(text, pii_types={"ip_address"})
        assert "[IP_REDACTED]" in result

    def test_invalid_ip_not_matched(self):
        text = "Version 999.999.999.999 is not an IP"
        result = redact_pii(text, pii_types={"ip_address"})
        # Invalid octets (>255) should not match
        assert "999.999.999.999" in result


class TestPIIPipeline:
    """5.10 - Pipeline ordering and combined redaction."""

    def test_all_types_redacted(self):
        text = (
            "SSN 123-45-6789, card 4111-1111-1111-1111, "
            "email user@test.com, IP 10.0.0.1, phone 555-123-4567"
        )
        result = redact_pii(text)
        assert "[SSN_REDACTED]" in result
        assert "[CARD_REDACTED]" in result
        assert "[EMAIL_REDACTED]" in result
        assert "[IP_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result

    def test_selective_types(self):
        text = "SSN 123-45-6789 and email user@test.com"
        result = redact_pii(text, pii_types={"ssn"})
        assert "[SSN_REDACTED]" in result
        assert "user@test.com" in result  # Email NOT redacted

    def test_empty_text(self):
        assert redact_pii("") == ""

    def test_none_like_text(self):
        assert redact_pii("") == ""

    def test_get_supported_types(self):
        types = get_supported_pii_types()
        assert "ssn" in types
        assert "credit_card" in types
        assert "phone" in types
        assert "email" in types
        assert "ip_address" in types


class TestPIICustomPatterns:
    """Custom pattern support."""

    def test_custom_pattern(self):
        import re

        custom = {"badge": (re.compile(r"BADGE-\d+"), "[BADGE_REDACTED]")}
        text = "Employee BADGE-12345 reported"
        result = redact_pii(text, custom_patterns=custom)
        assert "[BADGE_REDACTED]" in result
        assert "BADGE-12345" not in result
