# Snowflake adapter — versioned DDL

**Nothing in this directory is required to run LineupIQ.** The demo path, the tests, and
`lineupiq verify` all work with no Snowflake account. This is an optional adapter.

## State

Empty. The DDL lands with **M7**, generated from the same `TableSpec` registry that drives
the local parquet layer — so the two schemas cannot drift apart by hand-editing one of them.

Planned files:

| File                        | Contents                                                                    |
| --------------------------- | --------------------------------------------------------------------------- |
| `000_account_setup.sql`     | Database, schemas, warehouses, roles, grants, resource monitor              |
| `010_bronze.sql`            | All-VARCHAR landing tables plus `_loaded_at` / `_source_file` / `_row_hash` |
| `020_silver.sql`            | `EVENTS_TYPED`, `SUBSTITUTIONS`, `STINTS`, `EVENTS_ENRICHED`                |
| `030_gold.sql`              | `SHOT_FACTS`, `PLAY_TYPE_FACTS`, `LINEUP_DOCS`, dimensions                  |
| `040_ml.sql`                | `EP_PREDICTIONS`, `OPTIMAL_PLAYS`, `DEFENSIVE_RATINGS`, `TRADE_*`           |
| `050_eval.sql`              | Evaluation harness tables                                                   |
| `060_stage_and_formats.sql` | Internal stage and a PARQUET file format                                    |
| `090_teardown.sql`          | `DROP` everything — what makes the cost story credible                      |

## How this stays verifiable without an account

Four levels, each stated for what it is:

1. **`snowflake-sql-lint` (CI)** — `sqlfluff lint --dialect snowflake`. Every file parses as
   genuine Snowflake SQL on every push, with no account and no cost. This is the
   load-bearing one.
2. **`schema-parity` (CI)** — the DDL's columns and types must match the committed parquet
   contract. Catches the failure where the SQL is well-formed but describes different data.
3. **`lineupiq snowflake load --dry-run`** — prints the exact `PUT` and `COPY INTO`
   statements without opening a connection, so the SQL that _would_ run is reviewable.
4. **Credential-gated integration test** — loads a small slice and asserts row counts and
   `lineup_hash` parity against Python. Operator-only; skipped without credentials.

`SNOWFLAKE_PROVENANCE.md` will record which statements have actually been executed against
a live account versus which are only designed. A reviewer trusts "designed, not yet run"
far more than an unverifiable claim.

## One known risk, recorded before it bites

Snowflake's `ARRAY_SORT` over a VARIANT array of numbers must sort **numerically**, not
lexicographically. Python and DuckDB agreement is already asserted offline in
`services/ml/tests/test_hashing.py`; the Snowflake side is unverified because it needs an
account. If it turns out to sort lexicographically, the fix is to canonicalise player ids as
zero-padded fixed-width strings on both sides. See `services/ml/src/lineupiq/hashing.py`.
