"""Integration tests for the translation relay module."""

import pytest

from aavaaz.features.translation_relay import RelayChannel, Subscriber, TranslationRelay


class TestTranslationRelay:
    def test_create_channel(self):
        relay = TranslationRelay()
        relay.create_channel("ch1", source_language="en")
        channels = relay.list_channels()
        assert len(channels) == 1
        assert channels[0]["channel_id"] == "ch1"
        assert channels[0]["source_language"] == "en"

    def test_duplicate_channel_raises(self):
        relay = TranslationRelay()
        relay.create_channel("ch1")
        with pytest.raises(ValueError):
            relay.create_channel("ch1")

    def test_remove_channel(self):
        relay = TranslationRelay()
        relay.create_channel("ch1")
        relay.remove_channel("ch1")
        assert relay.list_channels() == []

    def test_subscribe_and_publish(self):
        relay = TranslationRelay()
        relay.create_channel("ch1", source_language="en")

        received = []
        relay.subscribe("ch1", "fr", "sub1", callback=lambda seg: received.append(seg))
        relay.publish("ch1", {"text": "hello", "start": 0.0, "end": 1.0})

        assert len(received) == 1
        assert received[0]["text"] == "hello"

    def test_subscribe_nonexistent_channel(self):
        relay = TranslationRelay()
        assert relay.subscribe("nope", "fr", "sub1") is False

    def test_unsubscribe(self):
        relay = TranslationRelay()
        relay.create_channel("ch1")
        relay.subscribe("ch1", "fr", "sub1")
        relay.unsubscribe("ch1", "sub1")
        channel = relay.get_channel("ch1")
        assert channel.info()["subscribers"] == 0

    def test_publish_with_translator(self):
        relay = TranslationRelay()
        relay.create_channel("ch1", source_language="en")

        # Set a translator on the channel
        channel = relay.get_channel("ch1")
        channel.translator = lambda text, source_lang, target_lang: f"[{target_lang}]{text}"

        received = []
        relay.subscribe("ch1", "fr", "sub1", callback=lambda seg: received.append(seg))
        relay.publish("ch1", {"text": "hello", "start": 0.0, "end": 1.0})

        assert received[0]["text"] == "[fr]hello"
        assert received[0]["original_text"] == "hello"

    def test_publish_same_language_no_translation(self):
        relay = TranslationRelay()
        relay.create_channel("ch1", source_language="en")

        channel = relay.get_channel("ch1")
        channel.translator = lambda text, source_lang, target_lang: f"TRANSLATED:{text}"

        received = []
        # Subscribe for same language as source
        relay.subscribe("ch1", "en", "sub1", callback=lambda seg: received.append(seg))
        relay.publish("ch1", {"text": "hello", "start": 0.0, "end": 1.0})

        # Should NOT be translated (target == source)
        assert received[0]["text"] == "hello"


class TestSubscriber:
    def test_queue_mode(self):
        sub = Subscriber("s1", "fr")
        sub.deliver({"text": "hi"})
        sub.deliver({"text": "bye"})
        items = sub.drain()
        assert len(items) == 2
        assert sub.queue == []

    def test_callback_mode(self):
        received = []
        sub = Subscriber("s1", "fr", callback=lambda seg: received.append(seg))
        sub.deliver({"text": "hi"})
        assert len(received) == 1
        assert sub.queue == []


class TestRelayChannel:
    def test_info(self):
        ch = RelayChannel("test", source_language="en")
        ch.add_subscriber("s1", "fr")
        ch.add_subscriber("s2", "de")
        info = ch.info()
        assert info["subscribers"] == 2
        assert set(info["target_languages"]) == {"fr", "de"}

    def test_close_all(self):
        ch = RelayChannel("test")
        ch.add_subscriber("s1", "fr")
        ch.close_all()
        assert ch.info()["subscribers"] == 0
