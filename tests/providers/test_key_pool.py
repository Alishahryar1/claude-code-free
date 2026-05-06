"""
Tests for API Key Pool
"""

from providers.key_pool import ApiKeyPool


def test_key_pool_rotation():
    keys = ["key1", "key2", "key3"]
    pool = ApiKeyPool(keys, usage_limit=2)

    # First key
    assert pool.get_next_key() == "key1"
    pool.mark_key_used("key1")
    assert pool.get_next_key() == "key1"
    pool.mark_key_used("key1")

    # First key exhausted, should rotate
    assert pool.get_next_key() == "key2"
    pool.mark_key_used("key2")
    assert pool.get_next_key() == "key2"
    pool.mark_key_used("key2")

    # Second key exhausted
    assert pool.get_next_key() == "key3"
    pool.mark_key_used("key3")
    pool.mark_key_used("key3")

    # All exhausted
    assert pool.get_next_key() is None


def test_key_pool_failure():
    keys = ["key1", "key2"]
    pool = ApiKeyPool(keys, usage_limit=10)

    assert pool.get_next_key() == "key1"
    pool.mark_key_failed("key1")

    # Key1 failed, should move to Key2
    assert pool.get_next_key() == "key2"


def test_key_pool_manual_failure():
    keys = ["key1", "key2"]
    pool = ApiKeyPool(keys, usage_limit=10)

    assert pool.get_next_key() == "key1"
    pool.mark_key_failed("key1")
    assert pool.get_next_key() == "key2"
    pool.mark_key_failed("key2")
    assert pool.get_next_key() is None
