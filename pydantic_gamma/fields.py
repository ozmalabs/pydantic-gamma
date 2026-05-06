"""
gamma_field() — a Pydantic Field with Γ metadata.

Extends Field with writable_in: which states admit writes to this field.
The metadata is stored in json_schema_extra so it round-trips through
JSON Schema and is visible in OpenAPI specs.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic.fields import FieldInfo

_GAMMA_WRITABLE_IN = "gamma_writable_in"


def gamma_field(
    default: Any = ...,
    *,
    writable_in: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """
    A Pydantic Field that carries Γ admissibility metadata.

    Parameters
    ----------
    writable_in:
        List of states in which this field may be written.
        Writes attempted in any other state return GammaError.

    All other kwargs are passed to pydantic.Field unchanged.

    Example::

        class Item(GammaModel):
            __gamma_states__ = ["draft", "published", "archived"]
            __gamma_transitions__ = [("draft", "published"), ("published", "archived")]
            __gamma_state_field__ = "status"

            status: str = "draft"
            published_at: datetime | None = gamma_field(
                default=None, writable_in=["published", "archived"]
            )
            editorial_note: str = gamma_field(
                default="", writable_in=["draft"]
            )
    """
    extra: dict[str, Any] = kwargs.pop("json_schema_extra", None) or {}
    if writable_in is not None:
        extra[_GAMMA_WRITABLE_IN] = writable_in
    if default is ...:
        return Field(json_schema_extra=extra or None, **kwargs)
    return Field(default, json_schema_extra=extra or None, **kwargs)


def get_field_writable_in(field_info: FieldInfo) -> list[str] | None:
    """Extract gamma_writable_in from a FieldInfo's json_schema_extra."""
    extra = field_info.json_schema_extra
    if not extra or not isinstance(extra, dict):
        return None
    return extra.get(_GAMMA_WRITABLE_IN)
