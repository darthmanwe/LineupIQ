# Snowflake adapter — versioned DDL

**Nothing in this directory is required to run LineupIQ.** The demo path, the tests, and
`lineupiq verify` all work with no Snowflake account. This is an optional adapter, kept
because the project began as a Snowflake-native design and the schema is genuinely portable.

## What is here, and what is not

**No statement in this directory has ever been executed against a live Snowflake account.**
That is stated first because it is the thing a reviewer most needs to know, and because
"designed, not run" is worth more than an unverifiable claim.

| File                    | State   | Contents                                                     |
| ----------------------- | ------- | ------------------------------------------------------------ |
| `000_account_setup.sql` | present | Database, schemas, warehouse, role, grants, resource monitor |
| `030_gold.sql`          | present | `SHOT_FACTS`, `POSSESSION_FACTS`, `STINTS`, `DIM_PLAYER`     |
| `040_ml.sql`            | present | `PLAYER_RAPM`                                                |
| `090_teardown.sql`      | present | `DROP` everything — what makes the cost story credible       |
| `010_bronze.sql`        | absent  | Landing tables. Not built: ingestion never runs in Snowflake |
| `020_silver.sql`        | absent  | Typed/derived layer. Same reason                             |
| `050_eval.sql`          | absent  | Evaluation tables. The eval harness reads parquet            |
| `060_stage_and_formats` | absent  | Stage + PARQUET format. Needed only by the loader, which is not built |

The gold tables are the ones worth having: they are the grain the API serves and the grain a
`SELECT` against either backend has to agree on.

## How this stays verifiable without an account

Two checks, both running in CI on every push, and both real:

1. **`sqlfluff lint --dialect snowflake`** — every file parses as genuine Snowflake SQL. No
   account, no cost. This is the load-bearing one: it is the difference between SQL and
   SQL-shaped text.
2. **`lineupiq snowflake check`** — the DDL's columns and types must match the committed
   parquet data contracts. This catches the failure that actually happens: a gold table
   gaining a column while the SQL does not. The DDL is *generated* by
   `lineupiq snowflake ddl` from the same `TableSpec` registry that drives the local layer,
   so the two schemas cannot drift by hand-editing one of them — and this check asserts the
   generated file was regenerated.

**Two further levels were designed and are not built**, and are named here rather than left
to look like oversights:

- `lineupiq snowflake load --dry-run`, printing the exact `PUT` and `COPY INTO` without
  opening a connection. There is no loader, so there is nothing to dry-run.
- A credential-gated integration test loading a slice and asserting `lineup_hash` parity.
  The `snowflake` pytest marker and the `snowflake` dependency extra exist; no test uses
  them yet.

An earlier version of this file claimed all four levels and described this directory as
"Empty. The DDL lands with M7". Both were wrong in opposite directions — the DDL landed, and
half the verification did not.

## One known risk, recorded before it bites

Snowflake's `ARRAY_SORT` over a VARIANT array of numbers must sort **numerically**, not
lexicographically. Sorting the string forms puts `1630552` before `201143` and produces a
different lineup hash — the same failure the parity fixture exists to catch between Python
and TypeScript.

Python/DuckDB agreement is asserted offline in `services/ml/tests/test_hashing.py`. The
Snowflake side is **unverified**, because verifying it needs an account. If it turns out to
sort lexicographically, the fix is to canonicalise player ids as zero-padded fixed-width
strings on both sides. See `services/ml/src/lineupiq/hashing.py`.
