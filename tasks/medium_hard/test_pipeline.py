import pytest
from pipeline import chunked_reader, normalize_record, fetch_record, process_batch


# ── chunked_reader tests ─────────────────────────────────────

def test_chunked_even_split():
    result = list(chunked_reader([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunked_uneven_includes_remainder():
    """Last partial chunk must be yielded."""
    result = list(chunked_reader([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_chunked_single_element():
    result = list(chunked_reader([42], 2))
    assert result == [[42]]

def test_chunked_empty():
    result = list(chunked_reader([], 2))
    assert result == []

def test_chunked_exact_size():
    result = list(chunked_reader([1, 2, 3], 3))
    assert result == [[1, 2, 3]]


# ── normalize_record tests ────────────────────────────────────

def test_normalize_strips_whitespace():
    record = {"name": "  alice  ", "role": " admin "}
    result = normalize_record(record)
    assert result["name"] == "alice"
    assert result["role"] == "admin"

def test_normalize_does_not_mutate_input():
    """Normalizing must not modify the original dict."""
    original = {"name": "  bob  "}
    original_copy = dict(original)
    normalize_record(original)
    assert original == original_copy

def test_normalize_preserves_non_string():
    record = {"id": 1, "score": 99.5, "name": " test "}
    result = normalize_record(record)
    assert result["id"] == 1
    assert result["score"] == 99.5


# ── fetch_record / retry decorator tests ────────────────────

def test_fetch_record_returns_dict():
    """retry decorator must not swallow the return value."""
    result = fetch_record(1)
    assert result is not None
    assert isinstance(result, dict)

def test_fetch_record_has_id():
    result = fetch_record(5)
    assert result["id"] == 5

def test_fetch_record_has_value():
    result = fetch_record(3)
    assert result["value"] == 30


# ── process_batch integration tests ──────────────────────────

def test_process_batch_count():
    results = process_batch([1, 2, 3])
    assert len(results) == 3

def test_process_batch_names_stripped():
    results = process_batch([1])
    assert results[0]["name"] == "record_1"

def test_process_batch_no_mutation():
    """process_batch should not corrupt records across calls."""
    r1 = process_batch([7])
    r2 = process_batch([7])
    assert r1[0]["name"] == r2[0]["name"]
