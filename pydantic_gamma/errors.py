"""
GammaError — a structured graph error. Returned, not raised.

Describes WHY an operation was inadmissible: the graph state at the
point of failure, the transition that was attempted, and what the
grammar required. Not just a status code and a string.

This is the base layer. fastapi-gamma and openapi-ozma-clients both
use this type — it lives here because it's a Pydantic model and the
violation vocabulary belongs at the model layer, not the API layer.
"""
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel
from starlette.responses import JSONResponse


class GammaError(BaseModel):
    """
    A Γ violation — returned from model updates, endpoint functions,
    and agent tool calls instead of raising an exception.

    Every instance carries:
    - violation: the category (maps to HTTP status if used in an API)
    - description: WHY it was inadmissible — the graph state in plain language
    - structured fields: current_state, required_state, missing_prior, etc.
      so callers can inspect the grammar programmatically, not just read the string

    Usage in a model::

        result = item.update(status="archived")
        if isinstance(result, GammaError):
            print(result.description)   # explains why

    Usage in a FastAPI endpoint (ozmalabs/fastapi intercepts this)::

        async def publish(id: int) -> Item | GammaError:
            item = db.get(id)
            return item.update(status="published")
    """

    violation: str
    description: str
    operation: str | None = None
    resource: str | None = None
    current_state: str | None = None
    required_state: list[str] | None = None
    missing_prior: list[str] | None = None
    blocked_by: str | None = None
    field: str | None = None

    _STATUS_CODES: ClassVar[dict[str, int]] = {
        "resource_not_found": 404,
        "state_violation": 409,
        "field_not_writable": 409,
        "forbidden_after": 409,
        "requires_prior": 409,
        "permission_denied": 403,
        "precondition_failed": 412,
        "schema_violation": 422,
    }

    def status_code(self) -> int:
        return self._STATUS_CODES.get(self.violation, 400)

    def to_response(self) -> JSONResponse:
        """Serialise to a Starlette JSONResponse at the correct HTTP status."""
        return JSONResponse(
            content=self.model_dump(exclude_none=True),
            status_code=self.status_code(),
        )

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def not_found(cls, resource: str, identifier: Any = None) -> "GammaError":
        id_part = f" with id {identifier!r}" if identifier is not None else ""
        return cls(
            violation="resource_not_found",
            description=f"{resource}{id_part} does not exist",
            resource=resource,
        )

    @classmethod
    def wrong_state(
        cls,
        *,
        operation: str,
        resource: str,
        current: str,
        required: list[str],
    ) -> "GammaError":
        required_desc = " or ".join(repr(s) for s in required)
        return cls(
            violation="state_violation",
            description=(
                f"{operation!r} requires {resource} to be in state {required_desc}, "
                f"but it is currently {current!r}"
            ),
            operation=operation,
            resource=resource,
            current_state=current,
            required_state=required,
        )

    @classmethod
    def field_not_writable(
        cls,
        *,
        field: str,
        resource: str,
        current_state: str,
        writable_in: list[str],
    ) -> "GammaError":
        writable_desc = " or ".join(repr(s) for s in writable_in)
        return cls(
            violation="field_not_writable",
            description=(
                f"field {field!r} on {resource} is not writable in state {current_state!r}; "
                f"it is only writable in state {writable_desc}"
            ),
            resource=resource,
            field=field,
            current_state=current_state,
            required_state=writable_in,
        )

    @classmethod
    def requires_prior(cls, *, operation: str, missing: list[str]) -> "GammaError":
        missing_desc = ", ".join(repr(m) for m in missing)
        return cls(
            violation="requires_prior",
            description=f"{operation!r} requires {missing_desc} to have been called first",
            operation=operation,
            missing_prior=missing,
        )

    @classmethod
    def forbidden_after(cls, *, operation: str, blocked_by: str) -> "GammaError":
        return cls(
            violation="forbidden_after",
            description=f"{operation!r} is not admissible after {blocked_by!r} has been called",
            operation=operation,
            blocked_by=blocked_by,
        )

    @classmethod
    def permission_denied(cls, *, operation: str, reason: str) -> "GammaError":
        return cls(
            violation="permission_denied",
            description=reason,
            operation=operation,
        )
