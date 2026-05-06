"""Tests: GammaModel — state machine transitions and field admissibility."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pydantic_gamma import GammaError, GammaModel, gamma_field


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
    editorial_note: str = gamma_field(default="", writable_in=["draft"])


def now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_valid_transition_draft_to_published():
    item = Item(id=1, title="hello")
    result = item.update(status="published")
    assert isinstance(result, Item)
    assert result.status == "published"


def test_valid_transition_published_to_archived():
    item = Item(id=1, title="hello", status="published")
    result = item.update(status="archived")
    assert isinstance(result, Item)
    assert result.status == "archived"


def test_self_transition_always_ok():
    item = Item(id=1, title="hello", status="published")
    result = item.update(status="published", title="updated")
    assert isinstance(result, Item)
    assert result.title == "updated"


def test_non_state_field_update_no_gamma():
    item = Item(id=1, title="hello")
    result = item.update(title="new title")
    assert isinstance(result, Item)
    assert result.title == "new title"


def test_field_writable_in_matching_state():
    item = Item(id=1, title="hello", status="published")
    result = item.update(published_at=now())
    assert isinstance(result, Item)
    assert result.published_at is not None


def test_editorial_note_writable_in_draft():
    item = Item(id=1, title="hello", status="draft")
    result = item.update(editorial_note="needs review")
    assert isinstance(result, Item)
    assert result.editorial_note == "needs review"


def test_update_state_and_field_together():
    """Transitioning to published and setting published_at in same call."""
    item = Item(id=1, title="hello", status="draft")
    result = item.update(status="published", published_at=now())
    assert isinstance(result, Item)
    assert result.status == "published"
    assert result.published_at is not None


# ---------------------------------------------------------------------------
# Transition violations
# ---------------------------------------------------------------------------

def test_invalid_transition_draft_to_archived():
    item = Item(id=1, title="hello", status="draft")
    result = item.update(status="archived")
    assert isinstance(result, GammaError)
    assert result.violation == "state_violation"
    assert result.current_state == "draft"
    # WHY: what is reachable from draft
    assert "published" in result.required_state
    assert "draft" in result.description
    assert "archived" in result.description


def test_invalid_transition_archived_to_published():
    item = Item(id=1, title="hello", status="archived")
    result = item.update(status="published")
    assert isinstance(result, GammaError)
    assert result.violation == "state_violation"
    assert result.current_state == "archived"


def test_invalid_transition_published_to_draft():
    item = Item(id=1, title="hello", status="published")
    result = item.update(status="draft")
    assert isinstance(result, GammaError)
    assert result.violation == "state_violation"
    assert result.current_state == "published"
    assert "archived" in result.required_state


def test_transition_error_carries_resource_name():
    item = Item(id=1, title="hello", status="draft")
    result = item.update(status="archived")
    assert isinstance(result, GammaError)
    assert result.resource == "Item"


# ---------------------------------------------------------------------------
# Field admissibility violations
# ---------------------------------------------------------------------------

def test_published_at_not_writable_in_draft():
    item = Item(id=1, title="hello", status="draft")
    result = item.update(published_at=now())
    assert isinstance(result, GammaError)
    assert result.violation == "field_not_writable"
    assert result.field == "published_at"
    assert result.current_state == "draft"
    assert "published" in result.required_state
    assert "archived" in result.required_state


def test_editorial_note_not_writable_after_publish():
    item = Item(id=1, title="hello", status="published")
    result = item.update(editorial_note="oops")
    assert isinstance(result, GammaError)
    assert result.violation == "field_not_writable"
    assert result.field == "editorial_note"
    assert result.current_state == "published"
    assert "draft" in result.required_state


def test_field_violation_checked_against_new_state():
    """
    Transitioning to published and trying to write editorial_note
    (which is only writable in draft) — field check uses the new state.
    """
    item = Item(id=1, title="hello", status="draft")
    result = item.update(status="published", editorial_note="this should fail")
    assert isinstance(result, GammaError)
    assert result.violation == "field_not_writable"
    assert result.current_state == "published"


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_gamma_reachable_states_from_draft():
    item = Item(id=1, title="hello", status="draft")
    assert item.gamma_reachable_states() == ["published"]


def test_gamma_reachable_states_from_archived():
    item = Item(id=1, title="hello", status="archived")
    assert item.gamma_reachable_states() == []


def test_gamma_admissible_fields_in_draft():
    item = Item(id=1, title="hello", status="draft")
    admissible = item.gamma_admissible_fields()
    assert "title" in admissible
    assert "editorial_note" in admissible
    assert "published_at" not in admissible


def test_gamma_admissible_fields_in_published():
    item = Item(id=1, title="hello", status="published")
    admissible = item.gamma_admissible_fields()
    assert "published_at" in admissible
    assert "editorial_note" not in admissible


def test_gamma_spec_shape():
    item = Item(id=1, title="hello", status="draft")
    spec = item.gamma_spec()
    assert spec["states"] == ["draft", "published", "archived"]
    assert {"from": "draft", "to": "published"} in spec["transitions"]
    assert spec["current_state"] == "draft"
    assert spec["reachable_states"] == ["published"]
    assert "published_at" in spec["fields"]
    assert spec["fields"]["published_at"]["writable_in"] == ["published", "archived"]


def test_update_is_immutable():
    """update() returns a new instance; the original is unchanged."""
    item = Item(id=1, title="hello", status="draft")
    result = item.update(status="published")
    assert isinstance(result, Item)
    assert item.status == "draft"
    assert result.status == "published"
