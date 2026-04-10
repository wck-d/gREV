# storage.py — Abstract base class + concrete implementations with multiple bugs
#
# Bug 1: BaseStorage defines abstract method `exists()` but MemoryStorage
#         doesn't implement it (inherits object behavior, doesn't override)
#
# Bug 2: FileStorage.read() opens file in binary mode ("rb") but writes were
#         done in text mode, causing decode mismatch
#
# Bug 3: CachingStorage.write() writes to underlying store but forgets to
#         invalidate (update) the cache — stale reads after writes
#
# Bug 4: BaseStorage.copy() calls self.write(dest_key, data) but the argument
#         order in write() is (data, key) — wrong argument order

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional


class BaseStorage(ABC):
    """Abstract base for key-value stores."""

    @abstractmethod
    def read(self, key: str) -> Optional[str]:
        """Return the value for key, or None if not found."""

    @abstractmethod
    def write(self, key: str, value: str) -> None:
        """Write value at key."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key. Return True if it existed."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if key exists."""

    def copy(self, src_key: str, dest_key: str) -> bool:
        """Copy src_key → dest_key. Return True if src existed."""
        data = self.read(src_key)
        if data is None:
            return False
        self.write(dest_key, data)  # BUG: argument order wrong — should be write(dest_key, data)
        return True


class MemoryStorage(BaseStorage):
    """In-memory key-value store."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def read(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def write(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    # BUG: exists() is NOT implemented — abstract method missing
    # (Python won't raise TypeError at class definition time because
    #  we inherit from ABC but don't call abstractmethod at runtime check)


class FileStorage(BaseStorage):
    """File-system backed storage. Each key is a file in base_dir."""

    def __init__(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        self._base_dir = base_dir

    def _path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("..", "")
        return os.path.join(self._base_dir, safe_key)

    def read(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:  # BUG: should be "r" (text mode), not "rb" (binary)
            return f.read()

    def write(self, key: str, value: str) -> None:
        with open(self._path(key), "w") as f:
            f.write(value)

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))


class CachingStorage(BaseStorage):
    """Write-through cache wrapping another BaseStorage instance."""

    def __init__(self, backing: BaseStorage):
        self._backing = backing
        self._cache: dict[str, str] = {}

    def read(self, key: str) -> Optional[str]:
        if key in self._cache:
            return self._cache[key]
        value = self._backing.read(key)
        if value is not None:
            self._cache[key] = value
        return value

    def write(self, key: str, value: str) -> None:
        self._backing.write(key, value)
        # BUG: cache not updated after write — subsequent reads return stale value
        # Should add: self._cache[key] = value

    def delete(self, key: str) -> bool:
        self._cache.pop(key, None)
        return self._backing.delete(key)

    def exists(self, key: str) -> bool:
        if key in self._cache:
            return True
        return self._backing.exists(key)
