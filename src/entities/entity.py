from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


@dataclass(kw_only=True)
class Entity:
    """
    Base entity class.

    All entities should inherit from this class.
    Includes common fields like id, created_at, and updated_at.
    """

    def dict(self):
        return _serialize_value(asdict(self))
