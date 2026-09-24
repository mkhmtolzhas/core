from dataclasses import dataclass

from src.entities.entity import Entity


@dataclass(kw_only=True)
class Response(Entity):
    body: str
