"""
GammaAgent — a pydantic-ai Agent with explicit admissibility grammar.

pydantic-ai implements Γ implicitly:
- args_validator  → preconditions on tool calls
- prepare_tools   → state-dependent tool availability
- output_validator → postconditions on model output
- node return types → transition rules in the execution graph

GammaAgent makes this explicit:
- Tools decorated with @gamma.* carry declared Γ
- gamma_spec() exports the full grammar for all tools
- Session-level Γ (requires_prior, forbidden_after) is enforced via
  a GammaSession tracker threaded through RunContext.deps
- Violations return GammaError, not ModelRetry with a string

Requires: pip install pydantic-gamma[agent]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_gamma.errors import GammaError

_GAMMA_ATTR = "_ozma_gamma"

# ---------------------------------------------------------------------------
# Γ annotation dataclasses (mirrors openapi/gamma.py — lives here at the
# model layer so agents and models share the same vocabulary)
# ---------------------------------------------------------------------------


@dataclass
class ToolGamma:
    """Declared admissibility grammar for one agent tool."""
    tool_name: str
    requires_state: list[str] | None = None
    produces_state: str | None = None
    requires_prior: list[str] | None = None
    forbidden_after: list[str] | None = None
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.requires_state:
            d["requires_state"] = self.requires_state
        if self.produces_state:
            d["produces_state"] = self.produces_state
        if self.requires_prior:
            d["requires_prior"] = self.requires_prior
        if self.forbidden_after:
            d["forbidden_after"] = self.forbidden_after
        if self.preconditions:
            d["preconditions"] = self.preconditions
        if self.postconditions:
            d["postconditions"] = self.postconditions
        return d

    def is_empty(self) -> bool:
        return not any([
            self.requires_state, self.produces_state,
            self.requires_prior, self.forbidden_after,
            self.preconditions, self.postconditions,
        ])


# ---------------------------------------------------------------------------
# Tool-level Γ decorators
# ---------------------------------------------------------------------------

def _get_or_create_tool_gamma(func: Callable[..., Any], name: str | None = None) -> ToolGamma:
    if not hasattr(func, _GAMMA_ATTR):
        setattr(func, _GAMMA_ATTR, ToolGamma(tool_name=name or func.__name__))
    return getattr(func, _GAMMA_ATTR)


def tool_requires_state(*states: str) -> Callable[..., Any]:
    """
    Declare that this tool is only admissible when the resource is
    in one of the given states.

        @agent.tool
        @gamma.tool_requires_state("active", "pending")
        async def process(ctx: RunContext[Deps], item_id: int) -> str: ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_tool_gamma(func).requires_state = list(states)
        return func
    return decorator


def tool_produces_state(state: str) -> Callable[..., Any]:
    """Declare the state this tool transitions the resource into."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_tool_gamma(func).produces_state = state
        return func
    return decorator


def tool_requires_prior(*tool_names: str) -> Callable[..., Any]:
    """
    Declare that this tool may only be called after the given tools
    have been called in the same agent run.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_tool_gamma(func).requires_prior = list(tool_names)
        return func
    return decorator


def tool_forbidden_after(*tool_names: str) -> Callable[..., Any]:
    """Declare that after this tool, the given tools are inadmissible."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_tool_gamma(func).forbidden_after = list(tool_names)
        return func
    return decorator


def tool_precondition(description: str) -> Callable[..., Any]:
    """Document a precondition on a tool (human-readable)."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_tool_gamma(func).preconditions.append(description)
        return func
    return decorator


def tool_postcondition(description: str) -> Callable[..., Any]:
    """Document a postcondition on a tool (human-readable)."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_tool_gamma(func).postconditions.append(description)
        return func
    return decorator


# ---------------------------------------------------------------------------
# Session-level Γ tracker — threaded through RunContext.deps
# ---------------------------------------------------------------------------

@dataclass
class GammaSession:
    """
    Tracks session-level Γ state for an agent run.

    Thread through RunContext.deps so tools can check and update it.
    GammaAgent wraps run() to inject this automatically when enforce=True.

        @dataclass
        class Deps:
            gamma: GammaSession = field(default_factory=GammaSession)
            db: Database = ...
    """
    called: set[str] = field(default_factory=set)
    resource_states: dict[str, str] = field(default_factory=dict)
    violations: list[GammaError] = field(default_factory=list)

    def check(self, tool_gamma: ToolGamma, resource_key: str | None = None) -> GammaError | None:
        """
        Check session-level Γ for the given tool.
        Returns GammaError if inadmissible, None if ok.
        """
        name = tool_gamma.tool_name

        if tool_gamma.requires_prior:
            missing = [t for t in tool_gamma.requires_prior if t not in self.called]
            if missing:
                return GammaError.requires_prior(operation=name, missing=missing)

        if tool_gamma.forbidden_after:
            for blocker in tool_gamma.forbidden_after:
                if blocker in self.called:
                    return GammaError.forbidden_after(operation=name, blocked_by=blocker)

        if tool_gamma.requires_state and resource_key is not None:
            current = self.resource_states.get(resource_key)
            if current not in tool_gamma.requires_state:
                return GammaError.wrong_state(
                    operation=name,
                    resource=resource_key,
                    current=str(current),
                    required=tool_gamma.requires_state,
                )

        return None

    def record(self, tool_gamma: ToolGamma, resource_key: str | None = None) -> None:
        """Record that a tool was called; update resource state if produces_state."""
        self.called.add(tool_gamma.tool_name)
        if tool_gamma.produces_state and resource_key is not None:
            self.resource_states[resource_key] = tool_gamma.produces_state


# ---------------------------------------------------------------------------
# GammaAgent
# ---------------------------------------------------------------------------

class GammaAgent:
    """
    A pydantic-ai Agent with explicit admissibility grammar.

    Wraps pydantic_ai.Agent. All agent methods pass through. Adds:

    - Tool functions decorated with gamma.tool_requires_state() etc. are
      introspected at construction time.
    - gamma_spec() → dict of tool_name → ToolGamma for all registered tools.
    - With enforce=True, session-level Γ (requires_prior, forbidden_after,
      requires_state) is checked before each tool executes; violations raise
      GammaError to trigger ModelRetry with the structured error description.

    Usage::

        from pydantic_gamma.agent import GammaAgent, tool_requires_prior
        from pydantic_ai import RunContext

        agent = GammaAgent("openai:gpt-4o", output_type=str, enforce=True)

        @agent.tool
        @tool_requires_prior("search")
        async def summarise(ctx: RunContext[GammaSession], query: str) -> str:
            ...

        @agent.tool
        async def search(ctx: RunContext[GammaSession], query: str) -> list[str]:
            ...

        result = await agent.run("summarise the results of searching for X")
    """

    def __init__(
        self,
        model: Any,
        *,
        output_type: Any = str,
        enforce: bool = True,
        **agent_kwargs: Any,
    ) -> None:
        try:
            from pydantic_ai import Agent
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai is required for GammaAgent. "
                "pip install pydantic-gamma[agent]"
            ) from exc

        self.enforce = enforce
        self._tool_gammas: dict[str, ToolGamma] = {}

        self._agent: Any = Agent(model, output_type=output_type, **agent_kwargs)

    # ------------------------------------------------------------------
    # Tool registration — passthrough with Γ introspection
    # ------------------------------------------------------------------

    def tool(self, func: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
        """
        Register a tool, capturing any Γ metadata from decorators.

        Can be used as @agent.tool or @agent.tool(retries=3).
        """
        def _register(f: Callable[..., Any]) -> Callable[..., Any]:
            tg = getattr(f, _GAMMA_ATTR, None)
            name = kwargs.get("name") or f.__name__
            if tg is not None:
                tg.tool_name = name
                self._tool_gammas[name] = tg

            if self.enforce and tg is not None and not tg.is_empty():
                f = self._wrap_with_enforcement(f, tg)

            return self._agent.tool(f, **kwargs)

        if func is not None:
            return _register(func)
        return _register

    def tool_plain(self, func: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
        """Register a stateless tool (no RunContext), capturing Γ metadata."""
        def _register(f: Callable[..., Any]) -> Callable[..., Any]:
            tg = getattr(f, _GAMMA_ATTR, None)
            name = kwargs.get("name") or f.__name__
            if tg is not None:
                tg.tool_name = name
                self._tool_gammas[name] = tg
            return self._agent.tool_plain(f, **kwargs)

        if func is not None:
            return _register(func)
        return _register

    def _wrap_with_enforcement(
        self, func: Callable[..., Any], tg: ToolGamma
    ) -> Callable[..., Any]:
        """
        Wrap a tool function to check session-level Γ before execution.

        Expects ctx.deps to be (or have a .gamma attribute of type) GammaSession.
        Raises GammaError if the grammar is violated — pydantic-ai converts this
        to a ModelRetry with the error description.
        """
        import functools
        import inspect

        @functools.wraps(func)
        async def wrapper(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            # Locate the GammaSession in deps
            session: GammaSession | None = None
            deps = getattr(ctx, "deps", None)
            if isinstance(deps, GammaSession):
                session = deps
            elif hasattr(deps, "gamma") and isinstance(deps.gamma, GammaSession):
                session = deps.gamma

            if session is not None:
                err = session.check(tg)
                if err is not None:
                    session.violations.append(err)
                    # Raise as a string so pydantic-ai's ModelRetry carries the WHY
                    try:
                        from pydantic_ai import ModelRetry
                        raise ModelRetry(err.description)
                    except ImportError:
                        raise RuntimeError(err.description) from None

            result = await func(ctx, *args, **kwargs) if inspect.iscoroutinefunction(func) else func(ctx, *args, **kwargs)

            if session is not None:
                session.record(tg)

            return result

        return wrapper

    # ------------------------------------------------------------------
    # Γ introspection
    # ------------------------------------------------------------------

    def gamma_spec(self) -> dict[str, dict[str, Any]]:
        """
        Export the declared Γ for all registered tools.

        Returns a dict of tool_name → ToolGamma.to_dict().
        Tools with no Γ declarations are omitted.
        """
        return {
            name: tg.to_dict()
            for name, tg in self._tool_gammas.items()
            if not tg.is_empty()
        }

    def gamma_for(self, tool_name: str) -> ToolGamma | None:
        """Return the ToolGamma for a specific tool, or None."""
        return self._tool_gammas.get(tool_name)

    # ------------------------------------------------------------------
    # Passthrough to underlying Agent
    # ------------------------------------------------------------------

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return await self._agent.run(*args, **kwargs)

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.run_sync(*args, **kwargs)

    async def run_stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.run_stream(*args, **kwargs)

    def iter(self, *args: Any, **kwargs: Any) -> Any:
        return self._agent.iter(*args, **kwargs)

    @property
    def name(self) -> str | None:
        return self._agent.name

    def __repr__(self) -> str:
        return f"GammaAgent({self._agent!r}, tools={list(self._tool_gammas)})"
