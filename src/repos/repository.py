from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable, Generic, List, Type, TypeVar
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.connections.database.database_connection import DatabaseConnection
from src.entities.entity import Entity
from src.models.base_model import BaseModel
from src.utils.enums.sort_order import SortOrder
from src.utils.querying import ListQuery

T = TypeVar("T", bound=BaseModel)
E = TypeVar("E", bound=Entity)


class IRepository(ABC, Generic[E]):
    """
    Generic repository interface.

    Defines methods for basic CRUD operations.
    """

    @abstractmethod
    async def get(self, id: UUID) -> E:
        """
        Retrieve an entity by its ID.

        Parameters:
            id: int
                The ID of the entity to retrieve.

        Returns:
            E
                The entity.
        """

        pass

    @abstractmethod
    async def get_slice(self, limit: int = 10, offset: int = 0) -> List[E]:
        """
        Retrieve a list of entities with pagination support.

        Parameters:
            limit: int
                Maximum number of entities to retrieve.
            offset: int
                Number of entities to skip before starting to collect the result set.

        Returns:
            List[E]
                A list of entities.
        """

        pass

    @abstractmethod
    async def find(self, query: ListQuery) -> List[E]:
        """Return entities matching a filter/sort/pagination query."""

        pass

    @abstractmethod
    async def count_query(self, query: ListQuery) -> int:
        """Return the number of entities matching a query's filters."""

        pass

    @abstractmethod
    async def create(self, obj: E) -> UUID:
        """
        Create a new entity.

        Parameters:
            obj: E
                The entity to create.

        Returns:
            int
                The ID of the created entity.
        """

        pass

    @abstractmethod
    async def update(self, id: UUID, obj: E) -> bool:
        """
        Update an existing entity by its ID.

        Parameters:
            id: int
                The ID of the entity to update.
            obj: E
                The entity data to update.

        Returns:
            bool
                A boolean indicating success.
        """

        pass

    @abstractmethod
    async def patch(self, id: UUID, data: dict[str, Any]) -> bool:
        """
        Partially update an existing entity by its ID.

        Parameters:
            id: int
                The ID of the entity to update.
            data: dict[str, Any]
                The subset of entity fields to update.

        Returns:
            bool
                A boolean indicating success.
        """

        pass

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """
        Delete an entity by its ID.

        Parameters:
            id: int
                The ID of the entity to delete.

        Returns:
            bool
                A boolean indicating success.
        """

        pass


class Repository(IRepository, Generic[T, E]):
    """
    Generic repository implementation.

    Implements methods for basic CRUD operations using SQLAlchemy ORM.
    All methods are asynchronous and return appropriate types.
    Uses SQLAlchemy models for database operations and Entity as the base type for input and output.
    """

    def __init__(
        self, connection: DatabaseConnection, model: Type[T], entity: Type[E]
    ) -> None:
        """
        Initialize the repository with a database connection, model, and entity type.


        Parameters:
            connection: DatabaseConnection
                The database connection to use.
            model: Type[T]
                The SQLAlchemy model class.
            entity: Type[E]
                The entity class.
        """
        self.connection = connection
        self.connection_pool: Callable[
            ..., AbstractAsyncContextManager[AsyncSession]
        ] = lambda: self.connection.connection_factory()()

        self.model = model
        self.entity = entity

    def _model_to_entity(self, model: T) -> E:
        """Convert an ORM row into its domain entity.

        Concrete repositories override this to coerce typed columns (enums,
        value objects) the generic mapping cannot infer. The default maps the
        table columns onto the entity by name, so every read path produces the
        same entity regardless of which method loaded the row.
        """

        data = {
            column.name: getattr(model, column.name)
            for column in self.model.__table__.columns
        }
        return self.entity(**data)

    async def get(self, id: UUID) -> E:
        """
        Retrieve an entity by its ID.

        Parameters:
            id: int
                The ID of the entity to retrieve.

        Returns:
            E
                The entity.
        """

        async with self.connection_pool() as session:
            result = await session.get(self.model, id)

            if not result:
                raise ValueError(f"{self.model.__name__} with id {id} not found")

            return self._model_to_entity(result)

    async def get_slice(self, limit: int = 10, offset: int = 0) -> List[E]:
        """
        Retrieve a list of entities with pagination support.

        Parameters:
            limit: int
                Maximum number of entities to retrieve.
            offset: int
                Number of entities to skip before starting to collect the result set.

        Returns:
            List[E]
                A list of entities.
        """

        async with self.connection_pool() as session:
            query = (
                select(self.model)
                .order_by(self.model.id.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(query)
            items = result.scalars().all()

            entities = []
            for item in items:
                entities.append(self._model_to_entity(item))
            return entities

    def _column(self, name: str):
        if not hasattr(self.model, name):
            raise ValueError(f"{self.model.__name__} has no column '{name}'")
        return getattr(self.model, name)

    def _apply_filters(self, stmt, query: ListQuery):
        for column, value in query.equals.items():
            stmt = stmt.where(self._column(column) == value)
        for rng in query.ranges:
            col = self._column(rng.column)
            if rng.gte is not None:
                stmt = stmt.where(col >= rng.gte)
            if rng.lte is not None:
                stmt = stmt.where(col <= rng.lte)
        for membership in query.ins:
            if membership.values:
                stmt = stmt.where(
                    self._column(membership.column).in_(membership.values)
                )
        if query.search is not None:
            clauses = []
            if query.search.term:
                term = f"%{query.search.term}%"
                clauses.extend(
                    self._column(column).ilike(term) for column in query.search.columns
                )
            clauses.extend(
                self._column(column) == value for column, value in query.search.exact
            )
            if clauses:
                stmt = stmt.where(or_(*clauses))
        return stmt

    async def find(self, query: ListQuery) -> List[E]:
        async with self.connection_pool() as session:
            stmt = self._apply_filters(select(self.model), query)
            sort_col = self._column(query.sort_by) if query.sort_by else self.model.id
            stmt = stmt.order_by(
                sort_col.asc() if query.order is SortOrder.ASC else sort_col.desc()
            )
            stmt = stmt.offset(query.offset).limit(query.limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._model_to_entity(row) for row in rows]  # type: ignore

    async def count_query(self, query: ListQuery) -> int:
        async with self.connection_pool() as session:
            stmt = self._apply_filters(
                select(func.count()).select_from(self.model), query
            )
            return int((await session.execute(stmt)).scalar_one())

    async def create(self, obj: E) -> UUID:
        """
        Create a new entity.

        Parameters:
            obj: E
                The entity to create.

        Returns:
            int
                The ID of the created entity.
        """

        try:
            async with self.connection_pool() as session:
                payload = {
                    key: value for key, value in obj.dict().items() if value is not None
                }
                new_obj = self.model(**payload)
                session.add(new_obj)
                await session.commit()
                await session.refresh(new_obj)
                id = getattr(new_obj, "id", None)
                if id is None:
                    raise ValueError(
                        f"Failed to retrieve id for the created {self.model.__name__}"
                    )
                return id
        except IntegrityError as e:
            raise ValueError(f"Integrity error: {e.orig}")

    async def update(self, id: UUID, obj: E) -> bool:
        """
        Update an existing entity by its ID.

        Parameters:
            id: int
                The ID of the entity to update.
            obj: E
                The entity data to update.

        Returns:
            bool
                A boolean indicating success.
        """

        async with self.connection_pool() as session:
            existing_obj = await session.get(self.model, id)
            if not existing_obj:
                raise ValueError(f"{self.model.__name__} with id {id} not found")

            for key, value in obj.dict().items():
                if key in {"id", "created_at", "updated_at"}:
                    continue
                setattr(existing_obj, key, value)

            session.add(existing_obj)
            await session.commit()
            await session.refresh(existing_obj)

            return True

    async def patch(self, id: UUID, data: dict[str, Any]) -> bool:
        """
        Partially update an existing entity by its ID.

        Parameters:
            id: int
                The ID of the entity to update.
            data: dict[str, Any]
                The subset of entity fields to update.

        Returns:
            bool
                A boolean indicating success.
        """

        protected_fields = {"id", "created_at", "updated_at"}
        update_data = {
            key: value for key, value in data.items() if key not in protected_fields
        }

        async with self.connection_pool() as session:
            existing_obj = await session.get(self.model, id)
            if not existing_obj:
                raise ValueError(f"{self.model.__name__} with id {id} not found")

            for key, value in update_data.items():
                if not hasattr(existing_obj, key):
                    raise ValueError(f"{self.model.__name__} has no attribute '{key}'")
                setattr(existing_obj, key, value)

            session.add(existing_obj)
            await session.commit()
            await session.refresh(existing_obj)

            return True

    async def delete(self, id: UUID) -> bool:
        """
        Delete an entity by its ID.

        Parameters:
            id: int
                The ID of the entity to delete.

        Returns:
            bool
                A boolean indicating success.
        """

        async with self.connection_pool() as session:
            obj = await session.get(self.model, id)
            if obj:
                await session.delete(obj)
                await session.commit()
                return True
            return False
