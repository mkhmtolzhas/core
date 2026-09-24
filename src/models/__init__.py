"""Every mapped class, imported so `BaseModel.metadata` is complete.

The schema is created from that metadata, and a model the package never
imports is a table the setup script never creates. Importing them here is what
makes `create_all` mean *all*.
"""

from src.models.base_model import BaseModel
from src.models.message import MessageModel
from src.models.session import SessionModel

__all__ = ["BaseModel", "MessageModel", "SessionModel"]
