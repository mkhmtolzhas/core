from sqlalchemy.orm import DeclarativeBase, Mapped

from src.models.annotations import IDPK, CreatedAt, UpdatedAt


class BaseModel(DeclarativeBase):
    """
    Base model class for SQLAlchemy ORM.

    All models should inherit from this class.
    """

    __abstract__ = True

    id: Mapped[IDPK]
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]
