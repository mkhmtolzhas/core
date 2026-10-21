from pathlib import Path
from typing import Optional, Tuple

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.enums.provider import Provider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Anchor for every default path, so nothing depends on the current directory."""

_DB_FILE_NAME = "core.db"
_HISTORY_FILE_NAME = "core.history"
_LOG_FILE_NAME = "core.log"


def _default_state_root() -> Path:
    """Where the agent keeps what must survive between runs.

    The project root, in plain sight. State a person is expected to inspect,
    back up or throw away should not hide in `~/.local/state` under a name they
    never chose — and putting it beside the code scopes it to the checkout, so
    what the agent learned about this project stays with this project.
    """
    return PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- LLM ---
    provider: Provider = Provider.OLLAMA
    """Which service serves `model`."""

    api_key: Optional[str] = None
    """Credential for a hosted `provider`. Unused, and unset, for Ollama."""

    model: str
    """Model tag, in whatever form `provider` names its models.

    `qwen3:4b` for Ollama, `gpt-4o` for OpenAI, `claude-opus-5` for Anthropic.
    """

    fallback_models: Tuple[str, ...] = ()
    """Provider-prefixed ids (`ollama:qwen3:8b`) tried when the primary model fails."""

    embedding_model: str
    """Ollama embedding model tag, e.g. `bge-m3:latest`.

    A separate model from `model`: a generative tag cannot produce embeddings,
    so this one has to be pulled on its own.
    """

    num_ctx: int = 8192
    """Context window handed to Ollama, in tokens.

    Ollama otherwise defaults to 4096, of which the system prompt and the tool
    schemas already claim about 2900 — leaving barely a thousand for the whole
    conversation. The ceiling here is video memory, not the model: past this
    value the KV cache pushes the embedding model off the GPU, and every
    long-term memory search then pays for reloading it.
    """

    embedding_dims: int = 1024
    """Vector width of `embedding_model` (1024 for `bge-m3`).

    Must match the model exactly. The value is baked into the vector column
    when the store is set up, so a mismatch is not a silent quality loss but a
    hard failure on insert — and changing it later needs the vectors rebuilt.
    """

    # --- Workspace ---
    workspace_root: Path = Field(default=PROJECT_ROOT)
    """Root the filesystem/shell tools are confined to."""

    # --- Storage ---
    state_root: Path = Field(
        default_factory=_default_state_root, validate_default=True
    )
    """Single directory owning every file the agent keeps between runs.

    Override it to move the memory somewhere else — a scratch directory for an
    experiment, or a shared location if several checkouts should remember the
    same things.
    """

    db_busy_timeout_ms: int = 5000
    """How long a contended write waits before raising `database is locked`."""

    # --- Budget (ModelCallLimitMiddleware) ---
    thread_call_limit: int = 30
    run_call_limit: int = 15

    # --- Context management ---
    context_edit_trigger_tokens: int = 5000
    context_edit_keep_tool_results: int = 3
    summarization_trigger_tokens: int = 6000
    summarization_keep_messages: int = 20
    """Both triggers must stay below `num_ctx`, with room left for a reply.

    Above it they never fire: Ollama silently drops the oldest tokens long
    before the threshold is reached, so the history is lost instead of being
    summarized."""

    # --- Fault tolerance ---
    model_max_retries: int = 3
    tool_max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_initial_delay: float = 1.0

    # --- Capability toggles ---
    enable_todos: bool = True
    enable_filesystem: bool = True
    filesystem_writable: bool = False
    """Adds write_file/edit_file/delete to the filesystem toolset."""

    enable_file_search: bool = False
    """Standalone ripgrep tools. Redundant when `enable_filesystem` is on."""

    enable_shell: bool = False
    """Runs arbitrary commands on the host. Opt in deliberately."""

    enable_subagents: bool = True
    """Lets the agent hand a self-contained job to a fresh agent.

    Worth its cost mainly for keeping a noisy search out of the conversation:
    the parent gets the conclusion instead of every tool result along the way.
    """

    # --- Observability (Langfuse) ---
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    """Project key pair. Tracing is on exactly when both are set."""

    langfuse_base_url: str = "http://localhost:3000"
    """Langfuse server. The default is the instance `docker-compose.yml` starts."""

    langfuse_tracing_environment: str = "development"
    """Keeps local runs apart from production traces in the same project.

    Lowercase letters, digits, `-` and `_` only; Langfuse drops anything else.
    """

    langfuse_release: Optional[str] = None
    """Version of the code behind a trace, e.g. a git SHA.

    What pins a regression in the dashboards to the deploy that caused it.
    """

    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    """Share of turns traced. Sampling is per trace, so a kept turn is kept whole."""

    @model_validator(mode="after")
    def _require_api_key(self) -> "Settings":
        """Refuse to start a hosted provider without a credential.

        The alternative is finding out on the first prompt, after the banner,
        the database and the model have all been set up — by which point the
        error arrives as a rejected HTTP call rather than as the missing line
        in `.env` that it is.
        """
        if self.provider is not Provider.OLLAMA and not self.api_key:
            raise ValueError(
                f"PROVIDER={self.provider.value} needs API_KEY to be set in .env."
            )
        return self

    @field_validator("state_root")
    @classmethod
    def _prepare_state_root(cls, value: Path) -> Path:
        """Pin the directory to one absolute path and create it once, here.

        Everything below derives its location from this value, so creating it
        in a single place spares every consumer an `mkdir` of its own. It also
        has to happen before first use: `aiosqlite.connect` creates the database
        file but none of the directories above it, so a fresh machine would
        otherwise fail on the very first run.
        """
        expanded = value.expanduser()
        if not expanded.is_absolute():
            expanded = PROJECT_ROOT / expanded
        resolved = expanded.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def db_path(self) -> Path:
        """SQLite file holding short-term checkpoints and long-term memory."""
        return self.state_root / _DB_FILE_NAME

    def db_async_url(self) -> str:
        """`db_path()` as a SQLAlchemy URL, driven by aiosqlite.

        The same file the raw connection opens: the ORM tables and LangGraph's
        own tables are neighbours in one database, so a session row and the
        checkpoints it refers to can never end up on different disks.
        """
        return f"sqlite+aiosqlite:///{self.db_path()}"

    def log_path(self) -> Path:
        """File the structured logger appends to."""
        return self.state_root / _LOG_FILE_NAME

    def history_path(self) -> Path:
        """File the terminal handler appends submitted prompts to."""
        return self.state_root / _HISTORY_FILE_NAME

    @field_validator("workspace_root")
    @classmethod
    def _absolute_workspace_root(cls, value: Path) -> Path:
        """Pin the root to one absolute path.

        A value from `.env` arrives as a raw string: `~` is not expanded and a
        relative path stays relative, so the tools would otherwise resolve it
        against whatever the current working directory happens to be.
        """
        expanded = value.expanduser()
        if not expanded.is_absolute():
            expanded = PROJECT_ROOT / expanded
        return expanded.resolve()


settings = Settings()  # type: ignore[call-arg]
