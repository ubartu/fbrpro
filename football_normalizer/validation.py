"""Disk-backed referential checks keep large tracking imports memory bounded."""
import json
import hashlib
import math
import sqlite3
import tempfile
from pathlib import Path

from .core import BOOL_FIELDS, FLOAT_FIELDS, INT_FIELDS, PRIMARY, REQUIRED, SCHEMA_VERSION, TABLES, dumps

REFERENCES = {"competition_id": "competitions", "season_id": "seasons", "team_id": "teams",
              "home_team_id": "teams", "away_team_id": "teams", "player_id": "players",
              "recipient_id": "players", "replacement_id": "players", "possession_team_id": "teams",
              "match_id": "matches", "event_id": "events"}


def validate(directory):
    directory = Path(directory)
    counts = {"errors": 0, "warnings": 0}
    examples = []
    issue_counts = {}
    coverage = {}
    present = {t for t in TABLES if (directory / f"{t}.jsonl").exists()}

    def issue(severity, code, table, line, detail):
        counts[severity] += 1
        issue_counts[code] = issue_counts.get(code, 0) + 1
        if len(examples) < 200:
            examples.append({"severity": severity, "code": code, "table": table, "line": line, "detail": detail})

    with tempfile.TemporaryDirectory(prefix="football-validate-") as tmp:
        db = sqlite3.connect(str(Path(tmp) / "index.sqlite"))
        db.execute("CREATE TABLE ids (dataset TEXT, id TEXT, PRIMARY KEY(dataset,id))")
        db.execute("CREATE TABLE refs (dataset TEXT, line INTEGER, target TEXT, id TEXT)")
        db.execute("CREATE TABLE objects (match_id TEXT, period INT, frame INT, object_id TEXT, PRIMARY KEY(match_id,period,frame,object_id))")
        db.execute("CREATE TABLE frames (match_id TEXT, period INT, frame INT, stamp INT, PRIMARY KEY(match_id,period,frame))")
        db.execute("CREATE TABLE clocks (dataset TEXT, match_id TEXT, period INT, stamp INT, PRIMARY KEY(dataset,match_id,period))")
        for table in TABLES:
            path = directory / f"{table}.jsonl"
            if not path.exists():
                continue
            cov = {"records": 0, "fields_present": {}, "fields_non_null": {}, "missing_coordinates": 0,
                   "anonymous_snapshot_players": 0, "min_period_time_ms": None, "max_period_time_ms": None}
            coverage[table] = cov
            with path.open(encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, 1):
                    try:
                        row = json.loads(raw)
                        if not isinstance(row, dict):
                            raise ValueError("record must be an object")
                    except (ValueError, TypeError) as exc:
                        issue("errors", "invalid_json", table, line_number, str(exc))
                        continue
                    cov["records"] += 1
                    for field, val in row.items():
                        cov["fields_present"][field] = cov["fields_present"].get(field, 0) + 1
                        if val is not None:
                            cov["fields_non_null"][field] = cov["fields_non_null"].get(field, 0) + 1
                    for field in REQUIRED[table]:
                        if row.get(field) is None or row.get(field) == "":
                            issue("errors", "required_field", table, line_number, field)
                        elif field not in ("period", "period_time_ms", "frame_id", "players") and (not isinstance(row[field], str) or not row[field].strip()):
                            issue("errors", "required_string", table, line_number, field)
                    if row.get("schema_version") != SCHEMA_VERSION:
                        issue("errors", "schema_version", table, line_number, "unsupported schema version")
                    source = row.get("source")
                    if not isinstance(source, dict) or not isinstance(source.get("provider"), str) or not source.get("provider"):
                        issue("errors", "provenance", table, line_number, "source provider is required")
                    for field in BOOL_FIELDS:
                        value = row.get(field)
                        if value is not None and not isinstance(value, bool):
                            issue("errors", "invalid_boolean", table, line_number, field)
                    for field in INT_FIELDS | FLOAT_FIELDS | {"coordinate_length_m", "coordinate_width_m", "minutes_played"}:
                        value = row.get(field)
                        if value is None:
                            continue
                        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                        if not valid or (field in INT_FIELDS and not isinstance(value, int)):
                            issue("errors", "invalid_number", table, line_number, field)
                        elif field not in ("x_m", "y_m", "end_x_m", "end_y_m", "z_m") and value < 0:
                            issue("errors", "negative_number", table, line_number, field)
                    period, stamp = row.get("period"), row.get("period_time_ms")
                    if period is not None and (not isinstance(period, int) or isinstance(period, bool) or period not in range(1, 6)):
                        issue("errors", "invalid_period", table, line_number, "expected 1..5")
                    if row.get("object_type") is not None and row["object_type"] not in ("player", "ball"):
                        issue("errors", "object_type", table, line_number, "expected player or ball")
                    if table == "tracking" and row.get("object_type") == "player" and row.get("object_id") != row.get("player_id"):
                        issue("errors", "tracking_identity", table, line_number, "player object_id must equal player_id")
                    if row.get("observed") is True and row.get("estimated") is True:
                        issue("errors", "contradictory_flags", table, line_number, "observed and estimated both true")
                    if table == "matches":
                        if row.get("home_team_id") == row.get("away_team_id"):
                            issue("errors", "same_teams", table, line_number, "home and away teams must differ")
                        for field in ("pitch_length_m", "pitch_width_m", "coordinate_length_m", "coordinate_width_m"):
                            value = row.get(field)
                            if isinstance(value, (int, float)) and value <= 0:
                                issue("errors", "pitch_dimensions", table, line_number, field)
                    if table in ("tracking", "events", "snapshots"):
                        if row.get("coordinate_frame") not in ("stadium_fixed", "team_attacking_left_to_right"):
                            issue("errors", "coordinate_frame", table, line_number, "declare stadium_fixed or team_attacking_left_to_right")
                        for field in ("coordinate_length_m", "coordinate_width_m"):
                            value = row.get(field)
                            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                                issue("errors", "coordinate_dimensions", table, line_number, field)
                    coordinate_pairs = [(row.get("x_m"), row.get("y_m")), (row.get("end_x_m"), row.get("end_y_m"))]
                    if table == "snapshots":
                        players = row.get("players")
                        if not isinstance(players, list):
                            issue("errors", "snapshot_players", table, line_number, "players must be an array")
                        else:
                            for p in players:
                                if not isinstance(p, dict):
                                    issue("errors", "snapshot_player", table, line_number, "player must be an object")
                                    continue
                                coordinate_pairs.append((p.get("x_m"), p.get("y_m")))
                                if p.get("player_id") is None:
                                    cov["anonymous_snapshot_players"] += 1
                                else:
                                    db.execute("INSERT INTO refs VALUES(?,?,?,?)", (table, line_number, "players", p["player_id"]))
                                for field in ("teammate", "actor", "keeper"):
                                    if p.get(field) is not None and not isinstance(p[field], bool):
                                        issue("errors", "snapshot_boolean", table, line_number, field)
                        area = row.get("visible_area", [])
                        if not isinstance(area, list) or any(not isinstance(p, list) or len(p) != 2 for p in area):
                            issue("errors", "visible_area", table, line_number, "visible area must be an array of XY pairs")
                        else:
                            # A camera view polygon can legitimately extend beyond the pitch.
                            for p in area:
                                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
                                    issue("errors", "visible_area", table, line_number, "polygon requires finite numeric coordinates")
                    if table in ("events", "tracking") and row.get("x_m") is None:
                        cov["missing_coordinates"] += 1
                    for x, y in coordinate_pairs:
                        if (x is None) != (y is None):
                            issue("errors", "partial_coordinates", table, line_number, "both coordinates required together")
                        elif x is not None:
                            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)):
                                issue("errors", "invalid_coordinates", table, line_number, "finite numbers required")
                            elif all(isinstance(row.get(f), (int, float)) and not isinstance(row.get(f), bool) and math.isfinite(row[f]) and row[f] > 0 for f in ("coordinate_length_m", "coordinate_width_m")):
                                if not (0 <= x <= row["coordinate_length_m"] and 0 <= y <= row["coordinate_width_m"]):
                                    issue("warnings", "outside_pitch", table, line_number, "coordinate preserved outside pitch; may be a legitimate out-of-play observation")
                    primary = row.get(PRIMARY[table])
                    if primary is not None:
                        if not isinstance(primary, str):
                            issue("errors", "invalid_id", table, line_number, "IDs must be strings")
                        else:
                            try:
                                db.execute("INSERT INTO ids VALUES (?,?)", (table, primary))
                            except sqlite3.IntegrityError:
                                issue("errors", "duplicate_id", table, line_number, primary)
                    for field, target in REFERENCES.items():
                        if field == PRIMARY[table] or row.get(field) is None:
                            continue
                        value = row[field]
                        if not isinstance(value, str):
                            issue("errors", "invalid_reference", table, line_number, field)
                        else:
                            db.execute("INSERT INTO refs VALUES(?,?,?,?)", (table, line_number, target, value))
                    if table in ("events", "tracking") and isinstance(stamp, int) and isinstance(period, int):
                        cov["min_period_time_ms"] = stamp if cov["min_period_time_ms"] is None else min(stamp, cov["min_period_time_ms"])
                        cov["max_period_time_ms"] = stamp if cov["max_period_time_ms"] is None else max(stamp, cov["max_period_time_ms"])
                        key = (table, row.get("match_id"), period)
                        last = db.execute("SELECT stamp FROM clocks WHERE dataset=? AND match_id=? AND period=?", key).fetchone()
                        if last and stamp < last[0]:
                            issue("errors", "time_order", table, line_number, "records must be in chronological order within each match/period")
                        db.execute("INSERT OR REPLACE INTO clocks VALUES(?,?,?,?)", (*key, stamp))
                        if table == "tracking":
                            framekey = (row.get("match_id"), period, row.get("frame_id"))
                            existing = db.execute("SELECT stamp FROM frames WHERE match_id=? AND period=? AND frame=?", framekey).fetchone()
                            if existing and existing[0] != stamp:
                                issue("errors", "frame_clock", table, line_number, "one frame has inconsistent timestamps")
                            db.execute("INSERT OR IGNORE INTO frames VALUES(?,?,?,?)", (*framekey, stamp))
                            try:
                                db.execute("INSERT INTO objects VALUES(?,?,?,?)", (*framekey, row.get("object_id")))
                            except sqlite3.IntegrityError:
                                issue("errors", "duplicate_frame_object", table, line_number, "same object repeated within frame")
            db.commit()
        absent_warned = set()
        for table, line, target, ident in db.execute("SELECT r.* FROM refs r LEFT JOIN ids i ON i.dataset=r.target AND i.id=r.id WHERE i.id IS NULL"):
            if target not in present:
                if target not in absent_warned:
                    issue("warnings", "reference_dataset_unavailable", table, line, f"{target} not supplied; identities cannot be verified")
                    absent_warned.add(target)
            else:
                issue("errors", "unresolved_reference", table, line, f"{target}: {ident}")
        tracking_gaps = []
        previous = None
        for match, period, stamp in db.execute("SELECT DISTINCT match_id, period, stamp FROM frames ORDER BY match_id,period,stamp"):
            if previous and previous[:2] == (match, period) and stamp - previous[2] > 1000:
                if len(tracking_gaps) < 100:
                    tracking_gaps.append({"match_id": match, "period": period, "from_ms": previous[2], "to_ms": stamp})
            previous = (match, period, stamp)
        if tracking_gaps:
            issue("warnings", "tracking_gaps", "tracking", None, "gaps over one second found; sample included in report")
        db.close()
    if not coverage or sum(c["records"] for c in coverage.values()) == 0:
        issue("errors", "empty_import", None, None, "no normalized records")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            for source in manifest.get("sources", []):
                archive = (directory / source["archive"]).resolve()
                if not archive.is_relative_to(directory.resolve()):
                    issue("errors", "archive_path", None, None, "archive must be inside bundle")
                    continue
                digest = hashlib.sha256()
                with archive.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != source["sha256"]:
                    issue("errors", "archive_checksum", None, None, source["archive"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            issue("errors", "invalid_manifest", None, None, str(exc))
    report = {"schema_version": SCHEMA_VERSION, "valid": counts["errors"] == 0,
              **counts, "issues": examples, "issues_truncated": sum(counts.values()) > len(examples),
              "coverage": coverage, "issue_counts": issue_counts, "tracking_gaps_sample": tracking_gaps}
    (directory / "validation.json").write_text(dumps(report) + "\n", encoding="utf-8")
    return report
