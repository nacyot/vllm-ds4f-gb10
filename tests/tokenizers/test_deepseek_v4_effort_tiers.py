# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 reasoning-effort tables (official three-level table by default,
DSPARK_EFFORT_TIERS=preview for the June-preview max-only preamble) and the
tool-turn preamble cap (DSPARK_TOOL_EFFORT_CAP)."""

import importlib

import pytest

from vllm.tokenizers import deepseek_v4 as wrapper
from vllm.tokenizers import deepseek_v4_encoding as enc


def _render(effort):
    msgs = [{"role": "user", "content": "hi"}]
    return enc.render_message(0, msgs, "thinking", True, effort)


def test_default_table_is_official():
    assert enc.REASONING_EFFORT_TIERS == "official"
    assert enc.REASONING_EFFORT_PROMPTS["low"] == ""
    assert enc.REASONING_EFFORT_PROMPTS["high"] == enc.REASONING_EFFORT_HIGH_TEXT
    assert enc.REASONING_EFFORT_PROMPTS["max"] == enc.REASONING_EFFORT_MAX_TEXT
    assert enc.REASONING_EFFORT_HIGH_TEXT.startswith(
        "Reasoning Effort: Absolute maximum"
    )
    assert enc.REASONING_EFFORT_MAX_TEXT.startswith("Reasoning Effort: Beyond maximum")


def test_render_official():
    assert _render("low") == _render(None)
    assert not _render("low").startswith("Reasoning Effort")
    assert _render("high").startswith(enc.REASONING_EFFORT_HIGH_TEXT)
    assert _render("max").startswith(enc.REASONING_EFFORT_MAX_TEXT)


def test_preview_table_via_env(monkeypatch):
    monkeypatch.setenv("DSPARK_EFFORT_TIERS", "preview")
    mod = importlib.reload(enc)
    try:
        assert mod.REASONING_EFFORT_TIERS == "preview"
        assert mod.REASONING_EFFORT_PROMPTS["high"] == ""
        assert mod.REASONING_EFFORT_PROMPTS["max"] == mod.REASONING_EFFORT_HIGH_TEXT
    finally:
        monkeypatch.delenv("DSPARK_EFFORT_TIERS", raising=False)
        importlib.reload(enc)


def test_unknown_env_value_falls_back_to_official(monkeypatch):
    monkeypatch.setenv("DSPARK_EFFORT_TIERS", "bogus")
    mod = importlib.reload(enc)
    try:
        assert mod.REASONING_EFFORT_TIERS == "official"
    finally:
        monkeypatch.delenv("DSPARK_EFFORT_TIERS", raising=False)
        importlib.reload(enc)


@pytest.mark.parametrize(
    "cap,effort,has_tools,thinking,expected",
    [
        (None, "high", True, True, "high"),  # no cap configured
        ("low", "high", True, True, "low"),
        ("low", "max", True, True, "low"),
        ("high", "max", True, True, "high"),
        ("high", "high", True, True, "high"),
        ("low", "low", True, True, "low"),
        ("low", "high", False, True, "high"),  # no tools: untouched
        ("low", "high", True, False, "high"),  # chat mode: untouched
        ("low", None, True, True, None),
        ("bogus", "max", True, True, "max"),  # unknown cap ignored
    ],
)
def test_tool_effort_cap(monkeypatch, cap, effort, has_tools, thinking, expected):
    if cap is None:
        monkeypatch.delenv("DSPARK_TOOL_EFFORT_CAP", raising=False)
    else:
        monkeypatch.setenv("DSPARK_TOOL_EFFORT_CAP", cap)
    assert wrapper.dspark_cap_tool_effort(effort, has_tools, thinking) == expected


def test_golden_matches_checkpoint_encoding():
    """Rebase guard: the table must equal the encoding shipped with the served
    checkpoint (skipped where the model directory is not present)."""
    import hashlib
    import os
    import re

    candidates = [
        os.path.expanduser(
            "~/models/DeepSeek-V4-Flash-Vision-Exp/encoding/encoding_dsv4.py"
        ),
        os.path.expanduser("~/models/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py"),
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        pytest.skip("no checkpoint encoding on this machine")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r"REASONING_EFFORT_PROMPTS.*?\n\}", src, re.S)
    assert m, "table not found in checkpoint encoding"
    ns: dict = {}
    exec("from typing import Dict\n" + m.group(0), ns)  # noqa: S102 - test-only
    table = ns["REASONING_EFFORT_PROMPTS"]
    for k in ("low", "high", "max"):
        a = hashlib.sha256(table[k].encode()).hexdigest()
        b = hashlib.sha256(
            enc._REASONING_EFFORT_TABLES["official"][k].encode()
        ).hexdigest()
        assert a == b, f"{k} preamble differs from the checkpoint encoding"
