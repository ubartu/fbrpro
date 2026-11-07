# Common intake format — version 1.0.0

All tables use UTF-8 JSON Lines: one object per line. Every record includes `schema_version` and `source`:

```json
{
  "schema_version": "1.0.0",
  "source": {
    "provider": "statsbomb",
    "record_id": "original-provider-id",
    "file_sha256": "hash-of-archived-input",
    "row_number": 1
  }
}
```

IDs are nonempty strings. Dates are retained as provider date/ISO strings, without inventing an unknown timezone. Unknown values are null. Additional provider-specific attributes are retained; downstream analytics must use documented normalized fields rather than guessing units in extension fields.

## Tables

| Dataset | Required fields | Other standard fields |
| --- | --- | --- |
| competitions | `competition_id`, `name` | `country` |
| seasons | `season_id`, `competition_id`, `name` | start/end dates if supplied |
| teams | `team_id`, `name` | provider extensions |
| players | `player_id`, `name` | `nickname`, `country`, provider extensions |
| matches | `match_id`, `home_team_id`, `away_team_id` | `competition_id`, `season_id`, `date`, `pitch_length_m`, `pitch_width_m`, coordinate dimensions/frame, `dimensions_assumed`, `attacking_directions` |
| appearances | `appearance_id`, `match_id`, `team_id`, `player_id` | `shirt_number`, `starter`, `participated`, `minutes_played`, `position_intervals` |
| events | `event_id`, `match_id`, `period`, `period_time_ms`, `type` | `team_id`, `player_id`, `outcome`, start/end coordinates, `recipient_id`, `replacement_id`, `under_pressure`, `duration_ms`, possession metadata, coordinate dimensions/frame, `provider_attributes`, `related_event_ids` |
| tracking | `tracking_id`, `match_id`, `period`, `period_time_ms`, `frame_id`, `object_id`, `object_type` | `player_id`, `team_id`, `x_m`, `y_m`, `z_m`, `observed`, `estimated`, coordinate dimensions/frame |
| snapshots | `snapshot_id`, `match_id`, `event_id`, `players` | `visible_area`, coordinate dimensions/frame |
| video_references | `video_id`, `match_id`, `reference` | provider-defined synchronisation metadata; this version does not apply offsets |

Event, tracking and snapshot records additionally require valid `coordinate_frame`, `coordinate_length_m` and `coordinate_width_m`. Supported frames are `stadium_fixed` and `team_attacking_left_to_right`.

## Coordinates

- Metres on a declared pitch rectangle; x increases along length, y along width from the upper-left diagram corner.
- `pitch_length_m`/`pitch_width_m`: actual physical dimensions, null if unknown.
- `coordinate_length_m`/`coordinate_width_m`: dimensions used for coordinate conversion, including reference dimensions when declared assumed.
- Start coordinates: `x_m`, `y_m`; end coordinates: `end_x_m`, `end_y_m`.
- Coordinate pairs are both numeric or both null. Outside-pitch observations remain intact and are flagged. Camera visible-area polygons may extend beyond pitch boundaries.
- Physical velocities and distances should not be inferred from reference-pitch scaling as if they were measured tracking.

## Time

- `period`: 1–4 for playing periods, 5 for penalty shootout.
- `period_time_ms`: integer milliseconds from that period's clock origin, including stoppage time. It resets each period.
- `frame_id`: integer source frame index; it is not a substitute for timestamps.
- StatsBomb provider minute/second and lineup interval strings remain available in explicitly marked provider fields; they do not override period time.

## Tracking

One row per player/ball per frame. `(match_id, period, frame_id, object_id)` must be unique. All rows in one frame must share a timestamp.

- `object_type`: `player` or `ball`.
- For a player, `object_id` equals canonical `player_id`.
- For the ball, `object_id` is `ball`, and `player_id` is null.
- `observed`/`estimated`: true, false, or unknown. They cannot both be true. The mapped adapter derives `estimated = not observed` only when an observed flag is explicitly supplied and estimated is unknown. Use this mapping only when the provider's false detection flag really means estimation.
- Missing positions stay missing. The importer does not interpolate trajectories.

## Snapshots

`players` is an array of objects containing `player_id` (nullable), `x_m`, `y_m`, `teammate`, `actor`, and `keeper`. Anonymous players are not assigned invented identities. `visible_area` is an array of `[x_m, y_m]` polygon vertices. Snapshot positions are relative to the linked event's coordinate frame.

## Events and appearances

Event types and outcomes are lower snake case in the StatsBomb adapter. Generic mappings must supply this vocabulary themselves; there is no universal outcome inference. Original payloads remain archived. Provider-only distances and angles inside `provider_attributes` retain their original units.

StatsBomb `position_intervals` retains the position name/ID, `from`/`to` strings, periods, start/end reasons, and `clock_basis: provider_match_clock`. Exact playing minutes and derived tactical roles are not calculated by intake.

## Identity scopes

Player and team IDs use the provider namespace; they do not imply cross-provider reconciliation. Season identities include competition ID. Events, snapshots, appearances and video references include match ID. The CLI archives any reviewed identity override map.

## Acceptance and completeness

Required-field, malformed-type, duplicate, clock-order and broken-reference failures reject an import. Missing reference datasets and out-of-pitch coordinates produce warnings. `validation.json` supplies field presence and non-null counts separately, so missing fields can be distinguished from explicit nulls. Warnings do not prove the data is suitable for every analytical label; label-specific eligibility checks come later.
