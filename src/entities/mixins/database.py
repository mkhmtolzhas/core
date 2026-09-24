"""Database-backed entity mixin."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass(kw_only=True)
class DatabaseMixin:
    """Common database identity and timestamp fields for persisted entities.

    All three are optional because an entity exists before its row does: the
    repository drops the unset values on insert and the database fills them in.
    """

    id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
