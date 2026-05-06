"""
GammaModel — a Pydantic BaseModel with embedded admissibility grammar.

Declares a state machine on the class; enforces transition validity and
field-level write admissibility on update(). Returns GammaError instead
of raising on violations.

Usage::

    from pydantic_gamma import GammaModel, gamma_field, GammaError

    class Item(GammaModel):
        __gamma_states__ = ["draft", "published", "archived"]
        __gamma_transitions__ = [
            ("draft", "published"),
            ("published", "archived"),
        ]
        __gamma_state_field__ = "status"

        id: int
        title: str
        status: str = "draft"
        published_at: datetime | None = gamma_field(
            default=None, writable_in=["published", "archived"]
        )

    item = Item(id=1, title="hello")
    result = item.update(status="published")  # → Item
    result = item.update(status="archived")   # → GammaError (draft→archived not in transitions)
    result = item.update(published_at=now())  # → GammaError (not writable in draft)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel

from pydantic_gamma.errors import GammaError
from pydantic_gamma.fields import get_field_writable_in

if TYPE_CHECKING:
    pass


class GammaModel(BaseModel):
    """
    BaseModel subclass with embedded Γ.

    Class variables
    ---------------
    __gamma_states__:
        All valid states for the state machine.
    __gamma_transitions__:
        Allowed (from, to) pairs. Only listed transitions are admissible.
        Self-transitions (same state) are always allowed.
    __gamma_state_field__:
        Name of the field that holds the current state. Default "status".
    """

    __gamma_states__: ClassVar[list[str]] = []
    __gamma_transitions__: ClassVar[list[tuple[str, str]]] = []
    __gamma_state_field__: ClassVar[str] = "status"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, **kwargs: Any) -> "Self | GammaError":
        """
        Return a new instance with the given fields updated, enforcing Γ.

        Checks, in order:
        1. Transition validity — if the state field is being changed, is
           (current_state, new_state) in __gamma_transitions__?
        2. Field admissibility — if a field declares writable_in, is the
           current state (after any transition) in that list?

        Returns the updated model on success, GammaError on violation.
        Never raises.
        """
        state_field = self.__gamma_state_field__
        current_state: str | None = getattr(self, state_field, None)

        # 1. Transition validity
        if state_field in kwargs and self.__gamma_transitions__:
            new_state = kwargs[state_field]
            if new_state != current_state:
                if not self._gamma_transition_valid(current_state, new_state):
                    reachable = [t for f, t in self.__gamma_transitions__ if f == current_state]
                    return GammaError.wrong_state(
                        operation=f"update(status={new_state!r})",
                        resource=type(self).__name__,
                        current=str(current_state),
                        required=reachable if reachable else (
                            [new_state]  # show what was asked, grammar has no path
                        ),
                    )
            # Transition is valid — effective state for field checks is the new state
            effective_state = new_state
        else:
            effective_state = current_state

        # 2. Field-level admissibility
        model_fields = type(self).model_fields
        for field_name, _value in kwargs.items():
            if field_name == state_field:
                continue
            field_info = model_fields.get(field_name)
            if field_info is None:
                continue
            writable_in = get_field_writable_in(field_info)
            if writable_in is not None and effective_state not in writable_in:
                return GammaError.field_not_writable(
                    field=field_name,
                    resource=type(self).__name__,
                    current_state=str(effective_state),
                    writable_in=writable_in,
                )

        return self.model_copy(update=kwargs)

    def gamma_reachable_states(self) -> list[str]:
        """States directly reachable from the current state."""
        current: str | None = getattr(self, self.__gamma_state_field__, None)
        return [t for f, t in self.__gamma_transitions__ if f == current]

    def gamma_admissible_fields(self) -> list[str]:
        """
        Fields that are writable in the current state.

        Fields without writable_in metadata are always admissible.
        """
        current: str | None = getattr(self, self.__gamma_state_field__, None)
        result = []
        for name, field_info in type(self).model_fields.items():
            writable_in = get_field_writable_in(field_info)
            if writable_in is None or current in writable_in:
                result.append(name)
        return result

    def gamma_spec(self) -> dict[str, Any]:
        """
        Export the Γ for this model as a dict — the same shape as x-gamma in OpenAPI.

        Includes: states, transitions, current_state, admissible_fields.
        """
        current: str | None = getattr(self, self.__gamma_state_field__, None)
        spec: dict[str, Any] = {}
        if self.__gamma_states__:
            spec["states"] = self.__gamma_states__
        if self.__gamma_transitions__:
            spec["transitions"] = [
                {"from": f, "to": t} for f, t in self.__gamma_transitions__
            ]
        if current is not None:
            spec["current_state"] = current
            spec["reachable_states"] = self.gamma_reachable_states()
            spec["admissible_fields"] = self.gamma_admissible_fields()

        # Field-level writable_in metadata
        field_gamma: dict[str, Any] = {}
        for name, field_info in type(self).model_fields.items():
            writable_in = get_field_writable_in(field_info)
            if writable_in is not None:
                field_gamma[name] = {"writable_in": writable_in}
        if field_gamma:
            spec["fields"] = field_gamma

        return spec

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gamma_transition_valid(self, from_state: str | None, to_state: str) -> bool:
        if from_state == to_state:
            return True
        return (from_state, to_state) in self.__gamma_transitions__
