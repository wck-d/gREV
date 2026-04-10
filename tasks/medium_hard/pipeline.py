# pipeline.py — Broken data pipeline with 3 independent bugs
# Bug 1: retry decorator swallows the return value (returns None)
# Bug 2: chunked_reader generator doesn't yield the last partial chunk
# Bug 3: normalize_record mutates the input dict instead of a copy

import time
from typing import Generator


def retry(max_attempts: int = 3, delay: float = 0.0):
    """Retry decorator for flaky operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    return  # BUG: should be `return result`
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


def chunked_reader(data: list, chunk_size: int) -> Generator[list, None, None]:
    """Yield successive chunks from a list."""
    i = 0
    while i + chunk_size <= len(data):  # BUG: should be `i < len(data)`
        yield data[i:i + chunk_size]
        i += chunk_size


def normalize_record(record: dict) -> dict:
    """Return a normalized copy of the record with stripped string values."""
    for key in record:
        if isinstance(record[key], str):
            record[key] = record[key].strip()  # BUG: mutates input, should work on a copy
    return record


@retry(max_attempts=2, delay=0.0)
def fetch_record(record_id: int) -> dict:
    """Fetch a record by ID. Simulated — never actually fails."""
    return {"id": record_id, "name": f"  record_{record_id}  ", "value": record_id * 10}


def process_batch(record_ids: list[int]) -> list[dict]:
    """Full pipeline: fetch and normalize each record."""
    results = []
    for rid in record_ids:
        raw = fetch_record(rid)
        normalized = normalize_record(raw)
        results.append(normalized)
    return results
