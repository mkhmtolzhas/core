"""Sort direction for list queries."""

from enum import Enum


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"
