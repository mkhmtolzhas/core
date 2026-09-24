"""Generic list-query spec shared by handlers, use cases and repositories.

A caller builds a ``ListQuery`` from whitelisted column names; the base
repository executes it (equality, ranges, membership, text search, sort,
pagination). The repository rejects any column its model does not declare, so
an unchecked name cannot reach the database.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from src.utils.enums.sort_order import SortOrder


@dataclass(frozen=True, kw_only=True)
class RangeFilter:
    """An inclusive range on a column (either bound optional)."""

    column: str
    gte: Optional[Any] = None
    lte: Optional[Any] = None


@dataclass(frozen=True, kw_only=True)
class InFilter:
    """Membership: a column whose value is in a set (SQL ``IN``)."""

    column: str
    values: Tuple[Any, ...]


@dataclass(frozen=True, kw_only=True)
class SearchFilter:
    """A free-text search matched against a set of columns (all ORed together).

    ``term`` is matched case-insensitively (``ilike``) against ``columns``.
    ``exact`` adds ``(column, value)`` equality clauses to the same OR, so a
    non-text column can join the search, matching a typed value exactly where
    ``ilike`` would not apply.
    """

    columns: Tuple[str, ...]
    term: str
    exact: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True, kw_only=True)
class ListQuery:
    """A list request: filters, sort and pagination.

    Every field is optional, and an empty query is a valid one: it lists the
    most recent rows of a table, which is what a caller with no criteria wants.
    """

    equals: dict[str, Any] = field(default_factory=dict)
    ranges: Tuple[RangeFilter, ...] = ()
    ins: Tuple[InFilter, ...] = ()
    search: Optional[SearchFilter] = None
    sort_by: Optional[str] = None
    order: SortOrder = SortOrder.DESC
    limit: int = 50
    offset: int = 0
