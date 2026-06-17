from __future__ import annotations

from backend.agents.llm import should_use_llm


def test_should_use_llm_can_be_disabled_by_state_even_when_key_exists(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert should_use_llm({"use_llm": False}) is False


def test_should_use_llm_uses_dashscope_key_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TRAVEL_AGENT_USE_LLM", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert should_use_llm({}) is True


def test_should_use_llm_can_be_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("TRAVEL_AGENT_USE_LLM", "false")

    assert should_use_llm({}) is False
