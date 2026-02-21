"""ElevenLabs TTS client with thread-safe LRU cache.

Uses stdlib urllib only (no extra pip deps) so the Docker image
doesn't need a rebuild.
"""

import json
import os
import threading
from collections import OrderedDict
from urllib.request import Request, urlopen


class ElevenLabsTTS:
    _MAX_CACHE = 50

    def __init__(self) -> None:
        self._api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self._voice_id = os.environ.get(
            "ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"
        )
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def synthesize(self, text: str) -> bytes | None:
        """Return MP3 bytes for *text*, or None on any failure."""
        if not self._api_key:
            return None

        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]

        try:
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
            )
            body = json.dumps({
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
            }).encode()
            req = Request(
                url,
                data=body,
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                method="POST",
            )
            with urlopen(req, timeout=5) as resp:
                audio = resp.read()
        except Exception:
            return None

        with self._lock:
            self._cache[text] = audio
            if len(self._cache) > self._MAX_CACHE:
                self._cache.popitem(last=False)

        return audio
