"""Tests for providers/nvidia_nim/request.py."""

from unittest.mock import MagicMock

import pytest

from config.nim import NimSettings
from providers.common.utils import set_if_not_none
from providers.nvidia_nim.request import (
    _set_extra,
    build_request_body,
)


@pytest.fixture
def req():
    r = MagicMock()
    r.model = "test"
    r.messages = [MagicMock(role="user", content="hi")]
    r.max_tokens = 100
    r.system = None
    r.temperature = None
    r.top_p = None
    r.stop_sequences = None
    r.tools = None
    r.tool_choice = None
    r.extra_body = None
    r.top_k = None
    return r


class TestSetIfNotNone:
    def test_value_not_none_sets(self):
        body = {}
        set_if_not_none(body, "key", "value")
        assert body["key"] == "value"

    def test_value_none_skips(self):
        body = {}
        set_if_not_none(body, "key", None)
        assert "key" not in body


class TestSetExtra:
    def test_key_in_extra_body_skips(self):
        extra = {"top_k": 42}
        _set_extra(extra, "top_k", 10)
        assert extra["top_k"] == 42

    def test_value_none_skips(self):
        extra = {}
        _set_extra(extra, "top_k", None)
        assert "top_k" not in extra

    def test_value_equals_ignore_value_skips(self):
        extra = {}
        _set_extra(extra, "top_k", -1, ignore_value=-1)
        assert "top_k" not in extra

    def test_value_set_when_valid(self):
        extra = {}
        _set_extra(extra, "top_k", 10, ignore_value=-1)
        assert extra["top_k"] == 10


class TestBuildRequestBody:
    def test_max_tokens_capped_by_nim(self, req):
        req.max_tokens = 100000
        nim = NimSettings(max_tokens=4096)
        body = build_request_body(req, nim)
        assert body["max_tokens"] == 4096

    def test_presence_penalty_included_when_nonzero(self, req):
        nim = NimSettings(presence_penalty=0.5)
        body = build_request_body(req, nim)
        assert body["presence_penalty"] == 0.5

    def test_include_stop_str_in_output_not_sent(self, req):
        body = build_request_body(req, NimSettings())
        assert "include_stop_str_in_output" not in body.get("extra_body", {})

    def test_parallel_tool_calls_included(self, req):
        nim = NimSettings(parallel_tool_calls=False)
        body = build_request_body(req, nim)
        assert body["parallel_tool_calls"] is False
