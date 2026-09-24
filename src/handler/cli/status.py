"""Making an in-flight agent run abortable."""

import asyncio
import signal
from typing import AsyncIterator, TypeVar

T = TypeVar("T")


class Interrupted(Exception):
    """The user aborted the run before the agent finished."""


async def interruptible(events: AsyncIterator[T]) -> AsyncIterator[T]:
    """Relay `events` until they end or the user presses `Ctrl+C`.

    `Ctrl+C` is captured through the event loop rather than as a
    `KeyboardInterrupt`: the exception would surface outside the coroutine and
    tear the whole session down, while here it only ends the current run and
    returns the user to the prompt.

    Each step waits on the next event and on the abort signal together, because
    an agent that is thinking produces nothing to react to — without the second
    waiter the interrupt would only be noticed once the model finally answered,
    which is the moment it stops being useful.
    """
    loop = asyncio.get_running_loop()
    aborted = asyncio.Event()

    try:
        loop.add_signal_handler(signal.SIGINT, aborted.set)
        handles_sigint = True
    except (NotImplementedError, RuntimeError):
        # Windows, or a loop running off the main thread.
        handles_sigint = False

    iterator = events.__aiter__()

    try:
        while True:
            following = asyncio.ensure_future(iterator.__anext__())
            abort = asyncio.ensure_future(aborted.wait())

            await asyncio.wait(
                {following, abort},
                return_when=asyncio.FIRST_COMPLETED,
            )
            abort.cancel()

            if aborted.is_set():
                following.cancel()
                # Let the cancellation unwind before returning to the prompt,
                # otherwise the agent keeps running behind the next input box.
                await asyncio.gather(following, abort, return_exceptions=True)
                raise Interrupted()

            try:
                event = following.result()
            except StopAsyncIteration:
                return

            yield event
    finally:
        if handles_sigint:
            loop.remove_signal_handler(signal.SIGINT)

        closer = getattr(events, "aclose", None)
        if closer is not None:
            await closer()
