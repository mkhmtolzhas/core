import asyncio

from src.core.app.lifespan import lifespan
from src.handler.cli.handler import get_cli_handler


async def _session() -> None:
    async with lifespan():
        await get_cli_handler().run()


def main() -> None:
    try:
        asyncio.run(_session())
    except KeyboardInterrupt:
        # Ctrl+C outside a run: leave quietly instead of dumping a traceback.
        pass


if __name__ == "__main__":
    main()
