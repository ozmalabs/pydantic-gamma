"""
Tests: GammaAgent — declared Γ on tools, gamma_spec(), session enforcement.

Uses pydantic-ai's TestModel to avoid real LLM calls.
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field as dc_field

from pydantic_gamma.agent import (
    GammaAgent,
    GammaSession,
    ToolGamma,
    tool_forbidden_after,
    tool_postcondition,
    tool_precondition,
    tool_produces_state,
    tool_requires_prior,
    tool_requires_state,
)
from pydantic_gamma.errors import GammaError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class Deps:
    gamma: GammaSession = dc_field(default_factory=GammaSession)


def make_agent(enforce: bool = True) -> GammaAgent:
    try:
        from pydantic_ai.models.test import TestModel
    except ImportError:
        pytest.skip("pydantic-ai not installed")
    return GammaAgent(TestModel(), output_type=str, enforce=enforce)


# ---------------------------------------------------------------------------
# gamma_spec() — declared Γ is introspected correctly
# ---------------------------------------------------------------------------

def test_gamma_spec_empty_when_no_decorators():
    agent = make_agent()

    @agent.tool_plain
    def plain_tool(query: str) -> str:
        return f"result: {query}"

    assert agent.gamma_spec() == {}


def test_gamma_spec_requires_prior():
    agent = make_agent()

    @agent.tool_plain
    @tool_requires_prior("search")
    def summarise(query: str) -> str:
        return "summary"

    spec = agent.gamma_spec()
    assert "summarise" in spec
    assert spec["summarise"]["requires_prior"] == ["search"]


def test_gamma_spec_full_decoration():
    agent = make_agent()

    @agent.tool_plain
    @tool_requires_state("active")
    @tool_produces_state("archived")
    @tool_requires_prior("create_item")
    @tool_forbidden_after("archive_item")
    @tool_precondition("Item must be active")
    @tool_postcondition("Item is now archived")
    def archive_item(item_id: int) -> str:
        return "archived"

    spec = agent.gamma_spec()
    assert "archive_item" in spec
    s = spec["archive_item"]
    assert s["requires_state"] == ["active"]
    assert s["produces_state"] == "archived"
    assert s["requires_prior"] == ["create_item"]
    assert s["forbidden_after"] == ["archive_item"]
    assert "Item must be active" in s["preconditions"]
    assert "Item is now archived" in s["postconditions"]


def test_gamma_for_returns_tool_gamma():
    agent = make_agent()

    @agent.tool_plain
    @tool_requires_prior("init")
    def do_thing(x: int) -> int:
        return x

    tg = agent.gamma_for("do_thing")
    assert tg is not None
    assert isinstance(tg, ToolGamma)
    assert tg.requires_prior == ["init"]


def test_gamma_for_unknown_returns_none():
    agent = make_agent()
    assert agent.gamma_for("nonexistent") is None


# ---------------------------------------------------------------------------
# GammaSession — session-level Γ tracking
# ---------------------------------------------------------------------------

def test_session_check_requires_prior_fails_when_not_called():
    session = GammaSession()
    tg = ToolGamma(tool_name="summarise", requires_prior=["search"])
    err = session.check(tg)
    assert err is not None
    assert isinstance(err, GammaError)
    assert err.violation == "requires_prior"
    assert "search" in err.missing_prior
    assert "summarise" in err.description
    assert "search" in err.description


def test_session_check_requires_prior_passes_after_call():
    session = GammaSession()
    session.called.add("search")
    tg = ToolGamma(tool_name="summarise", requires_prior=["search"])
    assert session.check(tg) is None


def test_session_check_forbidden_after_fails():
    session = GammaSession()
    session.called.add("archive_item")
    tg = ToolGamma(tool_name="archive_item", forbidden_after=["archive_item"])
    err = session.check(tg)
    assert err is not None
    assert err.violation == "forbidden_after"
    assert err.blocked_by == "archive_item"
    assert "archive_item" in err.description


def test_session_check_forbidden_after_passes_when_not_called():
    session = GammaSession()
    tg = ToolGamma(tool_name="archive_item", forbidden_after=["archive_item"])
    assert session.check(tg) is None


def test_session_check_requires_state_fails():
    session = GammaSession()
    session.resource_states["item:1"] = "draft"
    tg = ToolGamma(tool_name="publish", requires_state=["published"])
    err = session.check(tg, resource_key="item:1")
    assert err is not None
    assert err.violation == "state_violation"
    assert err.current_state == "draft"
    assert "published" in err.required_state


def test_session_check_requires_state_passes():
    session = GammaSession()
    session.resource_states["item:1"] = "published"
    tg = ToolGamma(tool_name="archive", requires_state=["published"])
    assert session.check(tg, resource_key="item:1") is None


def test_session_record_updates_called_and_state():
    session = GammaSession()
    tg = ToolGamma(tool_name="publish", produces_state="published")
    session.record(tg, resource_key="item:1")
    assert "publish" in session.called
    assert session.resource_states["item:1"] == "published"


def test_session_record_without_resource_key_still_marks_called():
    session = GammaSession()
    tg = ToolGamma(tool_name="search")
    session.record(tg)
    assert "search" in session.called
    assert session.resource_states == {}


# ---------------------------------------------------------------------------
# Multiple requires_prior
# ---------------------------------------------------------------------------

def test_session_requires_multiple_prior_all_missing():
    session = GammaSession()
    tg = ToolGamma(tool_name="checkout", requires_prior=["add_to_cart", "set_address"])
    err = session.check(tg)
    assert err is not None
    assert "add_to_cart" in err.missing_prior
    assert "set_address" in err.missing_prior


def test_session_requires_multiple_prior_one_missing():
    session = GammaSession()
    session.called.add("add_to_cart")
    tg = ToolGamma(tool_name="checkout", requires_prior=["add_to_cart", "set_address"])
    err = session.check(tg)
    assert err is not None
    assert "set_address" in err.missing_prior
    assert "add_to_cart" not in err.missing_prior


def test_session_requires_multiple_prior_all_satisfied():
    session = GammaSession()
    session.called.update(["add_to_cart", "set_address"])
    tg = ToolGamma(tool_name="checkout", requires_prior=["add_to_cart", "set_address"])
    assert session.check(tg) is None
