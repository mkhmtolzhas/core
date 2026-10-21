import asyncio

from src.core.app.lifespan import lifespan
from src.handler.cli.handler import get_cli_handler
from src.services.agent import prompts


async def _session() -> None:
    async with lifespan():
        # Prompts are fetched while the user types, not when they press Enter.
        warm_up = asyncio.create_task(prompts.warm_up())
        try:
            await get_cli_handler().run()
        finally:
            warm_up.cancel()


def main() -> None:
    try:
        asyncio.run(_session())
    except KeyboardInterrupt:
        # Ctrl+C outside a run: leave quietly instead of dumping a traceback.
        pass


if __name__ == "__main__":
    main()
