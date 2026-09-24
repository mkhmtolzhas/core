from enum import Enum


class Provider(str, Enum):
    """Which service serves the chat model.

    Named after the service, not after where it runs: `llama.cpp` and LM Studio
    are local too, so `LOCAL` would already be ambiguous the day a second local
    backend appears.

    The `str` mixin comes first and `Enum` last — the reverse order is a
    `TypeError` at import — and it is what lets `PROVIDER=openai` in the
    environment be read straight into a member.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
