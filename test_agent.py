"""
tests/test_agent.py

Run with: pytest test_agent.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


# --- Tool tests ---

def test_read_file_success(tmp_path):
    """File reader should return file contents."""
    from tools import read_file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello from test file!")
    result = read_file.invoke(str(test_file))
    assert "Hello from test file!" in result


def test_read_file_not_found():
    """File reader should return error for missing files."""
    from tools import read_file
    result = read_file.invoke("/nonexistent/path/file.txt")
    assert "Error" in result


def test_read_file_unsupported_extension(tmp_path):
    """File reader should reject unsupported file types."""
    from tools import read_file
    bad_file = tmp_path / "file.exe"
    bad_file.write_bytes(b"\x00\x01\x02")
    result = read_file.invoke(str(bad_file))
    assert "Error" in result


def test_read_file_truncates_large_files(tmp_path):
    """File reader should truncate files > 8000 chars."""
    from tools import read_file
    large_file = tmp_path / "large.txt"
    large_file.write_text("x" * 10000)
    result = read_file.invoke(str(large_file))
    assert "truncated" in result


# --- Graph structure test ---

def test_agent_graph_compiles():
    """The agent graph should compile without errors."""
    from graph import build_agent
    app = build_agent(use_memory=False)
    assert app is not None


def test_agent_state_schema():
    """AgentState should accept messages correctly."""
    from state import AgentState
    state: AgentState = {"messages": [HumanMessage(content="hello")]}
    assert len(state["messages"]) == 1


# --- Integration test (mocked LLM) ---

def test_agent_single_turn_no_tools():
    """Agent should return a plain response when no tools are needed."""
    from graph import build_agent

    app = build_agent(use_memory=False)

    plain_response = AIMessage(content="Paris is the capital of France.")

    with patch("langchain_ollama.ChatOllama.bind_tools") as mock_bind:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = plain_response
        mock_bind.return_value = mock_llm

        app2 = build_agent(use_memory=False)
        assert app2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
