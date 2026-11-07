from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

SCHEMA_VERSION = "1.0.0"
TABLES = ("competitions", "seasons", "teams", "players", "matches", "appearances",
          "events", "tracking", "snapshots", "video_references")
REQUIRED = {
    "competitions": ("competition_id", "name"),
    "seasons": ("season_id", "competition_id", "name"),
    "teams": ("team_id", "name"),
    "players": ("player_id", "name"),
    "matches": ("match_id", "home_team_id", "away_team_id"),
    "appearances": ("appearance_id", "match_id", "team_id", "player_id"),
    "events": ("event_id", "match_id", "period", "period_time_ms", "type"),
    "tracking": ("tracking_id", "match_id", "period", "period_time_ms", "frame_id", "object_id", "object_type"),
    "snapshots": ("snapshot_id", "event_id", "match_id", "players"),
    "video_references": ("video_id", "match_id", "reference"),
}
PRIMARY = {table: fields[0] for table, fields in REQUIRED.items()}
BOOL_FIELDS = {"under_pressure", "observed", "estimated", "starter", "participated", "dimensions_assumed", "teammate", "actor", "keeper"}
INT_FIELDS = {"period", "period_time_ms", "frame_id", "start_time_ms", "end_time_ms", "start_period", "end_period", "shirt_number"}
FLOAT_FIELDS = {"x_m", "y_m", "end_x_m", "end_y_m", "z_m", "pitch_length_m", "pitch_width_m", "duration_ms", "minutes_played", "coordinate_length_m", "coordinate_width_m"}
ID_KINDS = {"competition_id": "competition", "season_id": "season", "team_id": "team",
            "home_team_id": "team", "away_team_id": "team", "player_id": "player",
            "recipient_id": "player", "replacement_id": "player", "possession_team_id": "team", "match_id": "match",
            "event_id": "event", "appearance_id": "appearance", "snapshot_id": "snapshot",
            "tracking_id": "tracking", "video_id": "video"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def missing(value, numeric=False):
    markers = ("", "null", "nan", "none", "na") if numeric else ("",)
    return value is None or (isinstance(value, str) and value.strip().lower() in markers)


def number(value):
    if missing(value, numeric=True):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite number")
    return result


def integer(value):
    result = number(value)
    if result is not None and not result.is_integer():
        raise ValueError(f"expected integer, got {value!r}")
    return None if result is None else int(result)


def boolean(value):
    if missing(value, numeric=True):
        return None
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"invalid boolean {value!r}")


def time_ms(value, unit="ms"):
    """Explicit time units. HH:MM:SS(.sss) and MM:SS(.sss) supported."""
    if missing(value, numeric=True):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean timestamp")
    try:
        if unit == "timestamp":
            parts = str(value).split(":")
            if len(parts) not in (2, 3):
                raise ValueError("timestamp must be MM:SS or HH:MM:SS")
            values = [Decimal(p) for p in parts]
            if any(v < 0 or not v.is_finite() for v in values):
                raise ValueError("invalid timestamp")
            if any(v != v.to_integral_value() for v in values[:-1]):
                raise ValueError("only seconds may contain a decimal fraction")
            if values[-1] >= 60 or (len(values) == 3 and values[-2] >= 60):
                raise ValueError("invalid timestamp component")
            seconds = sum(v * (60 ** i) for i, v in enumerate(reversed(values)))
            result = seconds * 1000
        elif unit in ("ms", "s"):
            result = Decimal(str(value)) * (1000 if unit == "s" else 1)
        else:
            raise ValueError(f"unsupported time unit {unit!r}")
        if not result.is_finite() or result < 0:
            raise ValueError("time must be finite and nonnegative")
        return int(result.quantize(Decimal("1")))
    except InvalidOperation as exc:
        raise ValueError(f"invalid time {value!r}") from exc


def point(x, y, config, length, width):
    """No clipping: outside-pitch observations remain visible to validation."""
    x, y = number(x), number(y)
    if (x is None) != (y is None):
        raise ValueError("coordinate pair is incomplete")
    if x is None:
        return None, None
    system = config.get("system", "metres")
    if system == "scaled":
        sx, sy = number(config["x_max"]), number(config["y_max"])
        if not sx or not sy or sx < 0 or sy < 0:
            raise ValueError("coordinate scale must be positive")
        x, y = x / sx * length, y / sy * width
    elif system == "centered_metres":
        x, y = x + length / 2, y + width / 2
    elif system != "metres":
        raise ValueError(f"unsupported coordinate system {system!r}")
    if config.get("flip_x", False):
        x = length - x
    if config.get("flip_y", False):
        y = width - y
    return round(x, 6), round(y, 6)


def get_path(value, path):
    if path in ("", None):
        return value
    for part in str(path).split("."):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return value


def read_records(path, root=None, delimiter=","):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = get_path(json.loads(path.read_text(encoding="utf-8-sig")), root)
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: expected array; set root for a nested array")
        yield from data
    else:
        raise ValueError(f"unsupported input extension {path.suffix}")


class Identity:
    def __init__(self, provider, overrides=None):
        self.provider = provider
        self.overrides = overrides or {}

    def resolve(self, kind, value, scope=None):
        if missing(value):
            return None
        key = str(value)
        mapping_key = f"{scope}/{key}" if scope else key
        override = self.overrides.get(kind, {}).get(mapping_key)
        if override:
            return str(override)
        return f"{kind}_{uuid5(NAMESPACE_URL, dumps([self.provider, kind, scope, key])).hex}"


class Bundle:
    """Staged import. Large tracking tables stream to disk, not a Python list."""
    def __init__(self, output, provider, identity_overrides=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.identity = Identity(provider, identity_overrides)
        self.files = {}
        self.counts = {t: 0 for t in TABLES}
        self.nulls = {t: {} for t in TABLES}
        self.sources = []
        self.warnings = []

    def archive(self, path):
        path = Path(path)
        digest = hashlib.sha256()
        raw = self.output / "raw"
        raw.mkdir(exist_ok=True)
        temp = raw / f".copy-{len(self.sources)}"
        with path.open("rb") as source, temp.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                dest.write(chunk)
        sha = digest.hexdigest()
        relative = Path("raw") / f"{sha}-{path.name}"
        temp.replace(self.output / relative)
        entry = {"name": path.name, "sha256": sha, "archive": str(relative)}
        if entry not in self.sources:
            self.sources.append(entry)
        return entry

    def emit(self, table, row, original_id=None, source=None, row_number=None):
        if table not in TABLES:
            raise ValueError(f"unsupported dataset {table!r}")
        row = dict(row)
        row["schema_version"] = SCHEMA_VERSION
        row["source"] = {"provider": self.provider, "record_id": original_id,
                         "file_sha256": source["sha256"] if source else None,
                         "row_number": row_number}
        if table not in self.files:
            self.files[table] = (self.output / f"{table}.jsonl").open("w", encoding="utf-8")
        self.files[table].write(dumps(row) + "\n")
        self.counts[table] += 1
        for key, val in row.items():
            if val is None:
                self.nulls[table][key] = self.nulls[table].get(key, 0) + 1

    def close(self):
        for handle in self.files.values():
            handle.close()

    def finish(self):
        self.close()
        manifest = {"schema_version": SCHEMA_VERSION, "provider": self.provider,
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "sources": self.sources, "counts": self.counts,
                    "null_counts": self.nulls, "warnings": self.warnings,
                    "coverage_note": "Counts describe supplied records, not full league coverage. Missing tables are unavailable, not zero activity."}
        (self.output / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
