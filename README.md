# Football data normalizer

A working first version of the intake layer for our professional player profiling system. It prepares data for analysis; it does not assign playing-style labels yet.

## What works now

- A native **StatsBomb Open Data importer**: match metadata, teams, players, lineup/position records, events, and optional 360 snapshots.
- A **configurable importer for CSV, JSON, and JSONL**: map another source's fields to our schema without changing the analysis code. Flat event and tracking exports are supported. Nested proprietary formats require a small provider adapter or preprocessing step; this is not automatic support for every provider.
- Stable provider-scoped identities, explicit reviewed identity overrides, coordinate and clock conversion, missing-value preservation, raw-source archives and SHA-256 checksums.
- Validation of required fields, duplicates, references, coordinates, periods, chronology, tracking identities and observation flags. Tracking gaps over one second are reported.
- Coverage and field-completeness reports. Missing data is not interpreted as zero activity.
- JSONL output with optional compressed Parquet export.
- Rejected imports stay in a separate staging directory. Existing outputs are never intentionally overwritten.

Python 3.11+ is required. The normalizer itself uses only Python's standard library. Parquet export requires the optional `pyarrow` dependency.

## Try the supplied examples

Run these from this project directory; the included match and tracking examples are **synthetic**, not real player assessments.

```sh
python3 -m football_normalizer statsbomb \
  --match-id demo-1 \
  --match examples/statsbomb/match.json \
  --lineups examples/statsbomb/lineups.json \
  --events examples/statsbomb/events.json \
  --snapshots examples/statsbomb/360.json \
  --output normalized-demo

python3 -m football_normalizer mapped \
  --manifest examples/mapped/manifest.json \
  --output normalized-tracking-demo

python3 -m football_normalizer validate normalized-demo
```

Choose a new output directory for each run. Exit code `0` means accepted; `2` means rejected. The command prints the accepted output location or the quarantine location and error. `--help` describes each command.

## Import a real StatsBomb match

Download an authorized local copy of the [StatsBomb Open Data repository](https://github.com/hudl/open-data), then pass its files:

```sh
python3 -m football_normalizer statsbomb \
  --match-id 3895302 \
  --match /absolute/path/to/open-data/data/matches/9/281.json \
  --lineups /absolute/path/to/open-data/data/lineups/3895302.json \
  --events /absolute/path/to/open-data/data/events/3895302.json \
  --snapshots /absolute/path/to/open-data/data/three-sixty/3895302.json \
  --output normalized-match-3895302
```

The matches file may be a season-level array or one match object. Omit `--snapshots` if unavailable. If you do not have match metadata, explicitly supply `--home-team-id` and `--away-team-id`; lineup ordering is never used to guess home/away.

StatsBomb's standardized 120×80 coordinates are scaled to a declared 105×68 reference pitch unless you provide both `--pitch-length` and `--pitch-width`. Actual pitch dimensions remain null when unknown, and the manifest records the assumption. The reference pitch must not be mistaken for measured physical distances. Event coordinates remain in the provider's attacking-left-to-right frame; they are **not** silently converted to stadium-fixed coordinates.

The raw event payload is retained in `provider_attributes` and the complete input is archived. For passes, StatsBomb's missing outcome is specifically normalized as `complete`; for other event types an absent outcome stays unknown. Absent pressure flags remain null. Lineup position intervals retain their provider match-clock strings and periods; minutes and exact period-relative intervals are not invented.

## Add another source using a mapping manifest

Start from `examples/mapped/manifest.json`. Each input declares a dataset, local file, field mappings, and constants. Paths resolve relative to the manifest. The destination names are **our schema fields**, and values are source paths. Dot paths support nested objects and array indexes (for example, `player.id` and `location.0`). A top-level JSON wrapper can be selected with `root`. CSV delimiter defaults to a comma; override it with `delimiter`.

```json
{
  "provider": "my_provider",
  "inputs": [{
    "dataset": "tracking",
    "path": "tracking.csv",
    "constants": {
      "match_id": "provider-match-123",
      "object_type": "player"
    },
    "fields": {
      "frame_id": "frame",
      "period": "half",
      "period_time_ms": "time_seconds",
      "object_id": "player_id",
      "x_m": "x",
      "y_m": "y",
      "observed": "is_detected"
    },
    "clock": {"basis": "period", "unit": "s"},
    "coordinates": {
      "system": "centered_metres",
      "length_m": 105,
      "width_m": 68,
      "frame": "stadium_fixed"
    }
  }]
}
```

Supply the related metadata tables as additional inputs whenever possible. Without them, reference checks are reported as unavailable. An ID that cannot be found in a **supplied** reference table is an error.

Coordinate systems:

| `system` | Source meaning | Extra settings |
| --- | --- | --- |
| `metres` | Coordinates already measured from the declared origin | None |
| `scaled` | Source has a known rectangular scale | `x_max`, `y_max`; use 1/1 for normalized coordinates |
| `centered_metres` | Source origin is pitch centre | Half the pitch dimensions are added |

Set `flip_x` and/or `flip_y` to reverse an axis. In the target frame, origin is the top-left corner of the pitch diagram, x increases along length and y along width. The source orientation must be established from provider documentation. Coordinates are not clipped. Declare `dimensions_assumed: true` when using reference dimensions.

Every event/tracking input must explicitly declare its coordinates. `frame` is `stadium_fixed` or `team_attacking_left_to_right`. Direction-normalized events and fixed-frame tracking are not spatially synchronized by this importer; a later alignment step needs team directions for each period.

Clock units: `ms`, `s`, or `timestamp` (`HH:MM:SS.sss` or `MM:SS.sss`). For `basis: match`, provide explicit `period_offsets_ms`, for example `{"1": 0, "2": 2700000}` **only if the provider's second-half clock actually starts at 45:00**. Never use the first-half final timestamp as a guessed second-half offset. Frame number is not interpreted as match time. Input records must be chronological within each match/period.

Boolean values must be true/false or 1/0. Null and empty strings are missing. In numeric, clock and boolean fields, NaN, NA, None and null string markers also become null; textual fields and IDs preserve those strings to avoid corrupting legitimate values such as a country code of NA. Missing coordinates are allowed and measured in the coverage report. Incomplete coordinate pairs and non-finite numeric coordinates are rejected.

Use per-input `value_maps` for provider vocabularies, for example `"value_maps": {"type": {"Pass": "pass", "Shot": "shot"}}`. When a value map is declared, unknown non-null values reject the import rather than silently guessing a meaning. Map outcome codes explicitly in the same way.

The mapped adapter expects one record per entity/event/object observation. Wide Metrica CSV, nested SkillCorner frames, EPTS XML, and undocumented proprietary schemas need dedicated adapters that unpack them first. Raw snapshots can be imported through the native StatsBomb adapter; mapped snapshots currently require schema-ready nested `players` and `visible_area` fields, including canonical nested player identities and already-converted coordinates.

## Identity reconciliation

IDs are deterministically generated using UUIDv5 from the provider, entity kind, source ID and appropriate scope. Players/teams are provider-scoped; seasons are also competition-scoped; event and appearance IDs are match-scoped. Names are never used to merge players.

Pass `--identity-map reviewed-identities.json` for manual, reviewed reconciliation:

```json
{
  "player": {"1234": "player_reviewed_001"},
  "team": {"56": "team_reviewed_001"},
  "match": {"789": "match_reviewed_001"},
  "season": {"competition123/season456": "season_reviewed_001"}
}
```

Separate manifests from different providers may map their IDs to the same canonical ID. An override is an assertion by the operator, not a model prediction. The override file is archived with the import. Keep these IDs stable when integrating into a database.

## Outputs and data contract

See [SCHEMA.md](SCHEMA.md) for the format. Each accepted bundle contains:

- One `<dataset>.jsonl` file per supplied dataset.
- `raw/`: unmodified input files, including configuration, named by content hash and original filename.
- `manifest.json`: schema version, provider, import time, source checksums, row counts, explicit null counts, and assumption warnings.
- `validation.json`: acceptance status, errors, warnings, issue counts, up to 200 issue examples, field presence/non-null counts, and a sample of tracking gaps.
- `status.json`: accepted or rejected. A quarantined bundle is never an accepted dataset even if some files are readable.

Coverage describes **what was supplied**, not an entire league, season, or player's playing time. A substitute on the bench may have a lineup entry without participating. A few sample tracking frames do not prove complete match coverage. No off-camera positions, tactical instructions, or playing-style labels are inferred.

## Optional Parquet output

Use an isolated environment, install this project with its optional export dependency, and add `--parquet` to either import command:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[parquet]'
.venv/bin/python -m football_normalizer mapped \
  --manifest examples/mapped/manifest.json \
  --output normalized-parquet-demo \
  --parquet
```

JSONL remains the full-fidelity interchange format. Parquet exports scalar columns with explicit types; nested fields such as `source`, `players`, and `provider_attributes` are encoded as JSON strings under their original column names. Decode them when reading. Exports scan the source twice and write batches of 10,000 rows to avoid holding tracking tables in memory. JSONL/CSV inputs stream; source JSON arrays currently load in memory.

## Verification

```sh
python3 -m unittest discover -s tests -v
```

Tests cover conversions, period clocks, missing values, identity mappings, duplicate detection, broken references, rejected import isolation, output preservation, and optional Parquet round trips. See [VERIFICATION.md](VERIFICATION.md) for the real-match integration result.

## Scope and next integration steps

This is a local intake library and command-line tool, not the final scouting interface or a hosted ingestion service. It imports one StatsBomb match per run and supports multi-match mapped input. It does not write to PostgreSQL, download feeds automatically, calculate playing-style labels, or synchronize separate providers' clocks. Batch scheduling, database upserts and provider-specific adapters can be added around this contract.

Public availability does not by itself establish commercial redistribution rights. Keep provider licences and attribution alongside the dataset when selecting production sources. No supplier credentials or paid data are included.

## Midfield assessment dashboard

Open `dashboard/index.html` directly in a browser. Use **Open JSON report** to
load output from `python3 -m football_normalizer.midfield`. See
[MIDFIELD_ARCHITECTURE.md](MIDFIELD_ARCHITECTURE.md) for generation, comparison
rules, and the validation plan.

Generated reports and the dashboard with embedded match data are local artifacts.
To recreate the two-dataset dashboard from your reports, run from the project root:

```sh
python3 dashboard/build.py reports/bundesliga-relative-role-report.json \
  reports/premier-league-relative-role-report.json
```

Open `dashboard/local.html` for the built-in dataset selector. The tracked
`dashboard/index.html` works directly with **Open JSON report**.

## Video analyzer

Open [Video analysis](video/index.html) to review existing footage, pitch
tracks and movement evidence. See [VIDEO_ANALYZER.md](VIDEO_ANALYZER.md) to
start the local app and track either a short clip or a full video from the browser.
