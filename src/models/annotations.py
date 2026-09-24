"""Column types shared by every model, as reusable annotations.

Written against no particular dialect. `Uuid` and a Python-side default render
as a native `uuid` column on PostgreSQL and as `CHAR(32)` on SQLite, and the
timestamps are stamped by the application rather than by `now()`, which SQLite
does not have. The alternative — `PGUUID` and `server_default=text("now()")` —
reads more naturally in SQL but pins the schema to one database.
"""

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Uuid
from sqlalchemy.orm import mapped_column


def utcnow() -> datetime:
    """Current time, always aware and always UTC.

    SQLite stores a timestamp as text and drops the offset, so a naive local
    value would read back indistinguishable from a UTC one. Writing UTC makes
    that loss harmless: what comes back is naive but correct.
    """
    return datetime.now(timezone.utc)


IDPK = Annotated[
    UUID,
    mapped_column(Uuid(), primary_key=True, default=uuid4),
]
CreatedAt = Annotated[
    datetime,
    mapped_column(DateTime(timezone=True), default=utcnow),
]
UpdatedAt = Annotated[
    datetime,
    mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow),
]
