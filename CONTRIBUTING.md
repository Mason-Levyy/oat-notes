# Contributing

Thanks for looking. oat-notes is a small, on-device tool with one hard rule:
nothing captured — audio, transcripts, dictated text, voice embeddings — ever
leaves the machine or reaches a durable log. Every change is judged against
that first.

## Setup

Windows 10/11, [Git](https://git-scm.com/downloads/win) and
[uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs the
pinned Python 3.12 itself.

```powershell
git clone https://github.com/Mason-Levyy/oat-notes.git
cd oat-notes
uv sync --group dev
uv run oat-notes --ui
```

`uv sync --group dev --extra openvino` adds the OpenVINO backend.

## The gate

Every pull request must pass:

```powershell
uv run ruff check src tests
uv run pytest
```

`scripts\check.ps1` runs both and prints the advisory mypy count
(`uvx mypy src/oat_notes`), which should only go down. CI runs the same gate
on `windows-latest` with `pytest -m "not slow"`; tests marked `slow` load a
real model and run locally only.

Tests use inline fakes — no real audio, model or network. `tests/test_local_only.py`
is the privacy gate: a change that makes it fail is the wrong change.

## Code standards

- Validate at the boundary: JSON bodies are parsed in `webapp/requests.py`;
  handlers raise `ApiError` with a real HTTP status. Settings are parsed once
  in `AppSettings.from_dict`.
- Errors are values: no `except: pass`. Log failures through `logging` with
  `error_kind(error)` — the type only — anywhere a message could carry captured
  speech.
- One home per rule: when logic appears a second time, move it to its module
  (`roster.py`, `embeddings.py`, `unknown_turns.py`, ...).
- Types tell the truth: `Literal` aliases for string states, `Channel` rather
  than `"mic"`/`"loopback"`, callbacks typed by what they take.
- No comments: names and docstrings carry the meaning. A comment survives only
  for an external constraint the code cannot express (a Win32 quirk, a
  library's threading rule, a privacy invariant).
- Callbacks from `Session`, `Pipeline`, the hotkey hook and the enrollment
  recorder fire on their own threads; shared state goes through `LineFeed` or
  `state.lock`. Queues drop rather than block, and every drop is reported.
- Files are CRLF (`core.autocrlf=true`).

## Branches and pull requests

- Branch from `main`, one concern per branch. The maintainer uses
  [Graphite](https://graphite.dev) stacks; plain `git` branches and PRs are
  equally welcome.
- Commit messages follow `type(scope): summary` — `fix`, `feat`, `refactor`,
  `chore`, `docs`, `test`.
- PRs are rebase-merged; keep history linear and each commit passing the gate.
- Fill in the PR template: what changed, why, and how you verified it.

## Reporting bugs

Open an issue with the steps to reproduce, the `oat-notes` version (the
commit, or the installer name) and the Windows build.
Never paste a transcript, dictated text or audio into an issue; describe the
shape of the problem instead. Security concerns go through [SECURITY.md](SECURITY.md).
