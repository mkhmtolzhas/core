"""Keeping credentials out of traces.

The agent reads files for a living, and `.env` is a file. A tool result is
traced verbatim, so without this the provider key and every Langfuse secret
would be one `read_file` away from the trace store, readable by anyone with
access to the project.
"""

import json
import re
from typing import Any

from langfuse._utils.serializer import EventSerializer

REDACTED = "[REDACTED]"

_TOKENS = re.compile(
    "|".join(
        (
            r"sk-ant-[A-Za-z0-9_\-]{20,}",  # Anthropic
            r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}",  # OpenAI
            r"sk-lf-[A-Za-z0-9\-]+",  # Langfuse secret key
            r"gh[pousr]_[A-Za-z0-9]{36,}",  # GitHub
            r"glpat-[A-Za-z0-9_\-]{20,}",  # GitLab
            r"AKIA[0-9A-Z]{16}",  # AWS access key id
            r"xox[abprs]-[A-Za-z0-9\-]{10,}",  # Slack
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
        )
    )
)
"""Credentials recognisable by their shape alone, wherever they appear."""

_ASSIGNMENTS = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Z][A-Z0-9_]*(?:SECRET|PASSWORD|PASSWD|TOKEN|KEY|SALT|CREDENTIALS?)[A-Z0-9_]*"
    r"[ \t]*[=:][ \t]*[\"']?)"
    r"[^\s\"']+"
)
"""`NAME=value` where the name says the value is secret.

Catches what has no recognisable shape — a database password, a salt — as it
appears in `.env`, compose files and shell exports. Upper-case names only, so
code that merely mentions a key (`api_key: str`) stays readable; the line
prefixes `read_file` and `grep` add are allowed by anchoring on a non-word
character rather than on the start of the line.
"""


def _redact(text: str) -> str:
    text = _ASSIGNMENTS.sub(lambda match: match.group(1) + REDACTED, text)
    return _TOKENS.sub(REDACTED, text)


def _walk(data: Any) -> Any:
    if isinstance(data, str):
        return _redact(data)
    if isinstance(data, dict):
        return {key: _walk(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_walk(value) for value in data]
    return data


def mask_secrets(*, data: Any, **_: Any) -> Any:
    """The `mask` hook Langfuse applies to every input, output and metadata.

    The value is first flattened with the SDK's own serializer, so the walk
    sees exactly the JSON that would be sent. Walking the raw value instead
    would stop at the first object it does not know — a LangChain message, a
    LangGraph `Command` — and send whatever is inside it untouched.
    """
    try:
        plain = json.loads(json.dumps(data, cls=EventSerializer))
    except (TypeError, ValueError):
        return _redact(str(data))
    return _walk(plain)
