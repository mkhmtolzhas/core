<!--
Sync Impact Report
Version change: 1.0.0 → 1.1.0
Bump rationale: Principle I gained a rule it did not state. v1.0.0 described the direction
  imports travel and presented it as the direction layers depend, which is the reverse of the
  truth under DIP. The layer-dependency rule is now stated in its own right alongside the
  import rule, so the principle carries an obligation it did not carry before.
Modified principles:
  I. Dependencies Point Inward — layer dependency and import direction separated; the
    use-case layer named as innermost; entities restated as shared kernel rather than a layer
Added sections: none
Removed sections: none
Deferred items:
  TODO(PORT_OWNERSHIP): `AgentService` lives in `src/services/agent/service.py`, so the port
    is owned by the plugin that implements it rather than by the use case that needs it. The
    inversion holds between classes but not between packages. Left as observed practice
    pending an explicit decision.

Ratification history:
  1.0.0 (2026-09-07) — initial ratification from the unfilled scaffold.
-->

# CORE Constitution

## Core Principles

### I. Dependencies Point Inward

Two things travel between layers, and they travel in opposite directions. Confusing them is
the most common way to describe this architecture wrongly, so both are stated here.

**Layer dependency** — what a layer is built upon. `usecase` is the innermost layer: it holds
what the application does and depends on nothing but entities. `handler`, `service` and
`repository` are plugins into it. Each exists to serve a use case and MUST be replaceable
without the use case changing, so every dependency arrow points inward, at `usecase`. A layer
MUST NOT be given knowledge that makes an inner layer depend on it.

**Imports** — the mechanical direction the source reads: `handler` → `usecase` → `service` /
`repository` → `entities`. A module MUST NOT import from a layer further out than its own.

That these run opposite ways is not a contradiction; it is what the Dependency Inversion
Principle buys. `AgentUseCaseImpl` names `AgentService`, the abstraction, and never
`AgentServiceImpl`. Both sides depend on the abstract class, so control flows outward while
the concrete dependency points back inward at the contract.

Across a layer boundary exactly two things may be imported: the abstract base class and the
dependency-injection function returning an instance of it. Importing a concrete `*Impl` across
a boundary is forbidden — it is the single import that collapses an inverted dependency back
into a real one.

`src/entities` is the shared kernel, not a layer in this ordering: plain dataclasses carrying
no behaviour, the vocabulary every layer speaks. It MUST NOT import from any other part of the
project and MUST NOT import a framework — no LangChain, no Rich, no Pydantic.

Rationale: reading the import direction as the dependency direction would say the use case is
built on the agent service. The truth is the reverse — the service is built to fit the use
case, and only the abstraction crosses between them. That is why `handler` knowing just
`AgentUseCase` and `get_agent_use_case` lets the terminal be swapped for an HTTP API by
touching one directory, and why the damage from breaking the rule is quiet: the day `usecase`
imports `BaseChatModel` merely to name a type, the innermost layer has acquired an
LLM-framework dependency it can never shed.

### II. Configuration Stops at the Composition Root

`src.core.config.settings` MUST be read only inside a composition root — the
`@lru_cache(maxsize=1)` factory of a layer, or a helper invoked solely from it.

Configuration MUST travel downward as individual named values. Passing `Settings`, or any
configuration object, into a constructor or builder is forbidden, however many parameters
the honest alternative requires.

Every configuration field MUST have a docstring stating what changing it costs, and a
commented entry in `.env`. Invalid configuration MUST be rejected by a validator at load
time, never discovered at first use.

Rationale: a component holding `settings` can read anything, so its real dependencies stop
being visible in its signature and stop being substitutable in a test. A twenty-argument
`build_middlewares` is not verbosity — it is the dependency list, written down.

### III. Every Seam Is an Abstract Base Class

A replaceable boundary MUST be expressed as an `ABC` whose abstract methods
`raise NotImplementedError()`. A bare `pass` in an abstract method body is forbidden: it
turns a missing implementation into a silent `None`.

Each abstract method MUST carry a docstring describing the contract. Implementations are
reached through a module-level DI function or singleton, never constructed by their callers.

Rationale: the ABC is the only thing an outer layer is allowed to see, so it carries the
entire contract. If the contract is not written there, it is not written anywhere.

### IV. Connections Are Owned by the Lifespan

Every external resource MUST be reached through the `Connection` abstraction: the `Connection`
ABC (`connect` / `close`), a per-kind specialisation such as `DatabaseConnection`, and a
concrete class exposed as a module-level singleton.

`lifespan()` opens and closes connections, and shutdown MUST run in a `finally` block.
Consumers receive an open connection through a factory; they MUST NOT open one themselves.

Rationale: shutdown is not tidiness. `aiosqlite` drives SQLite from a non-daemon worker
thread, so a connection left open keeps the interpreter alive and the CLI never returns to
the shell — a crash would hang instead of reporting.

### V. Fix the Cause, Not the Symptom

When a defect can be corrected at its source or masked further downstream, the source fix is
required. Post-processing output to hide behaviour, or special-casing a value to route around
a bug, is forbidden.

A fix MUST be verified by executing the affected path. Reporting MUST distinguish what was
observed from what was inferred: an unreproduced defect MUST be described as unreproduced.

Rationale: the `<final_answer>` tags were absent from our code, the middleware prompts and the
model template. The cause was a missing output contract in the system prompt, and stripping
the tags in the renderer would have hidden that — while also mangling any answer that quoted
such a tag on purpose.

## Code Style Constraints

Optionality is written `Optional[X]`, never `X | None`. Generic aliases come from `typing`:
`List`, `Tuple`, `Sequence`, `Mapping`. Ruff rules UP006, UP035 and UP045 contradict this and
MUST NOT be applied.

Functions MUST be declared at module level. Nested function definitions are forbidden; a
closure that exists only to capture arguments MUST become a module-level function that takes
them explicitly. Decorators are the sole exception.

The same logic MUST NOT exist in two places.

The handler layer MUST NOT contain hardcoded presentation values. Thresholds, markers, limits
and labels arrive as parameters from `get_cli_handler`.

`pathlib` MUST NOT appear in the agent service layer; paths reach it as resolved absolute
strings. It belongs in `src/core/config.py` and in the handler layer, which is where paths are
actually normalised.

Dependencies MUST be added with `uv add`. Hand-editing `[project.dependencies]` is forbidden,
so that `uv.lock` can never disagree with `pyproject.toml`.

`uv run pyright --pythonpath .venv/bin/python src scripts` and `uv run ruff check src scripts`
MUST pass before a change is considered done.

## Development Workflow

For any non-trivial change, the approach MUST be described and agreed before code is written.
The question the description answers is whether the change fits the architecture, not whether
it works.

Whoever makes a change verifies it by running it, and reports what the run actually produced.
Claiming a mechanism was confirmed when only its source was read is a violation of Principle V.

Every change is reviewed by the project owner. Concerns found but not acted upon MUST be
stated explicitly rather than silently fixed or silently dropped — narrowing the requested
scope is the owner's decision.

## Governance

This constitution supersedes other practices in this repository. Where a habit and this
document disagree, this document wins until it is amended.

Amendments are made by editing this file, bumping the version and updating the amendment date
in the same change. Versioning is semantic:

- MAJOR — a principle is removed or redefined in a way that invalidates existing code.
- MINOR — a principle or section is added, or its guidance materially expanded.
- PATCH — clarification, wording, or a correction that changes no obligation.

Every review MUST check the change against these principles. Complexity that violates one is
permitted only when the alternative is documented and shown to be worse; the justification
belongs in the code, next to the thing it explains.

**Version**: 1.1.0 | **Ratified**: 2026-09-07 | **Last Amended**: 2026-09-07
