import os
import tempfile
import pytest
from storage import BaseStorage, MemoryStorage, FileStorage, CachingStorage


# ── MemoryStorage tests ──────────────────────────────────────

def test_memory_write_read():
    s = MemoryStorage()
    s.write("k", "v")
    assert s.read("k") == "v"

def test_memory_read_missing():
    s = MemoryStorage()
    assert s.read("missing") is None

def test_memory_exists_true():
    s = MemoryStorage()
    s.write("x", "1")
    assert s.exists("x") is True

def test_memory_exists_false():
    s = MemoryStorage()
    assert s.exists("nope") is False

def test_memory_delete_returns_true():
    s = MemoryStorage()
    s.write("del", "val")
    assert s.delete("del") is True

def test_memory_delete_missing_returns_false():
    s = MemoryStorage()
    assert s.delete("ghost") is False

def test_memory_delete_removes_key():
    s = MemoryStorage()
    s.write("gone", "bye")
    s.delete("gone")
    assert s.read("gone") is None


# ── FileStorage tests ────────────────────────────────────────

@pytest.fixture
def tmpdir_storage():
    with tempfile.TemporaryDirectory() as d:
        yield FileStorage(d)

def test_file_write_read(tmpdir_storage):
    tmpdir_storage.write("a", "hello")
    assert tmpdir_storage.read("a") == "hello"

def test_file_read_missing(tmpdir_storage):
    assert tmpdir_storage.read("no_such_key") is None

def test_file_exists_after_write(tmpdir_storage):
    tmpdir_storage.write("b", "world")
    assert tmpdir_storage.exists("b") is True

def test_file_not_exists_before_write(tmpdir_storage):
    assert tmpdir_storage.exists("never_written") is False

def test_file_delete(tmpdir_storage):
    tmpdir_storage.write("c", "data")
    assert tmpdir_storage.delete("c") is True
    assert tmpdir_storage.exists("c") is False

def test_file_read_returns_str_not_bytes(tmpdir_storage):
    tmpdir_storage.write("strkey", "text value")
    result = tmpdir_storage.read("strkey")
    assert isinstance(result, str), f"Expected str, got {type(result)}"


# ── CachingStorage tests ─────────────────────────────────────

def test_caching_write_then_read():
    s = CachingStorage(MemoryStorage())
    s.write("k", "v")
    assert s.read("k") == "v"

def test_caching_write_updates_cache():
    """Read after write must return new value, not stale cache."""
    s = CachingStorage(MemoryStorage())
    s.write("key", "old")
    s.read("key")          # populate cache
    s.write("key", "new")  # update
    assert s.read("key") == "new"

def test_caching_exists_after_write():
    s = CachingStorage(MemoryStorage())
    s.write("e", "1")
    assert s.exists("e") is True

def test_caching_delete():
    s = CachingStorage(MemoryStorage())
    s.write("d", "data")
    assert s.delete("d") is True
    assert s.read("d") is None


# ── BaseStorage.copy() tests ─────────────────────────────────

def test_copy_existing_key():
    s = MemoryStorage()
    s.write("src", "payload")
    result = s.copy("src", "dst")
    assert result is True
    assert s.read("dst") == "payload"

def test_copy_missing_key():
    s = MemoryStorage()
    result = s.copy("ghost", "dst")
    assert result is False

def test_copy_does_not_delete_src():
    s = MemoryStorage()
    s.write("orig", "keep")
    s.copy("orig", "clone")
    assert s.read("orig") == "keep"
