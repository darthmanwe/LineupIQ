# `lineupiq` — modelling and data package

The Python half of [LineupIQ](../../README.md): ingestion, stint reconstruction,
the shot model, the evaluation harness, and the exporters that hand precomputed
values to the Worker.

## Quickstart

```bash
uv sync --extra dev
uv run pytest          # offline, free, no API key, no network
uv run lineupiq seasons
```

A bare `pytest` is deliberately offline and free. Tests that hit the network,
spend money, or need a Snowflake account are behind markers and are deselected
by default:

```bash
uv run pytest -m net        # public data mirrors
uv run pytest -m repro      # refit and assert committed metrics
uv run pytest -m snowflake  # operator-only, needs credentials
```

## Layout

| Path | What |
|---|---|
| `src/lineupiq/seasons.py` | Season identity. The declared coverage lives here and nowhere else. |
| `src/lineupiq/hashing.py` | Order-invariant lineup hash, verified against DuckDB. |
| `src/lineupiq/paths.py` | Marker-based repo root discovery and the data layout. |
| `src/lineupiq/config.py` | Settings. Every credential defaults to absent. |
| `src/lineupiq/cli.py` | The `lineupiq` command. Unbacked commands refuse rather than print zeros. |

## Two rules that shape the code

**Nothing is fabricated.** If a number cannot be computed it is absent, not
invented. Commands that have no data behind them yet exit non-zero and name what
will back them, the same way an unbacked API route returns 501.

**Published numbers are generated, never typed.** Metrics in the README and the
model cards are rendered from run logs by `lineupiq report render`, and CI fails
if a committed block is stale.
