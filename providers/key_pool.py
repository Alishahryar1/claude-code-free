"""
API Key Pool and Fallback Logic for Providers
"""

import threading


class ApiKeyInfo:
    def __init__(self, key: str, usage_limit: int):
        self.key = key
        self.usage_limit = usage_limit
        self.usage_count = 0
        self.lock = threading.Lock()
        self.exhausted = False
        self.failed = False

    def increment(self):
        with self.lock:
            self.usage_count += 1
            if self.usage_count >= self.usage_limit:
                self.exhausted = True

    def mark_failed(self):
        with self.lock:
            self.failed = True

    def is_available(self):
        with self.lock:
            return not self.exhausted and not self.failed


class ApiKeyPool:
    def __init__(self, keys: list[str], usage_limit: int):
        self.keys = [ApiKeyInfo(key, usage_limit) for key in keys]
        self.lock = threading.Lock()
        self.current_index = 0

    def get_next_key(self) -> str | None:
        with self.lock:
            # Search for the first available key starting from the current index
            for _ in range(len(self.keys)):
                key_info = self.keys[self.current_index]
                if key_info.is_available():
                    return key_info.key
                self.current_index = (self.current_index + 1) % len(self.keys)
            return None

    def mark_key_used(self, key: str):
        with self.lock:
            for key_info in self.keys:
                if key_info.key == key:
                    key_info.increment()
                    break

    def mark_key_failed(self, key: str):
        with self.lock:
            for key_info in self.keys:
                if key_info.key == key:
                    key_info.mark_failed()
                    break
