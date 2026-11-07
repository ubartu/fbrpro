"""Declarative adapter for documented CSV/JSON/JSONL provider exports."""
import json
from pathlib import Path

from .core import (BOOL_FIELDS, FLOAT_FIELDS, ID_KINDS, INT_FIELDS, PRIMARY,
                   TABLES, boolean, get_path, integer, missing, number, point,
                   read_records, time_ms)


def normalize(manifest_path, bundle):
    manifest_path = Path(manifest_path)
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle.archive(manifest_path)
    inputs = config.get("inputs", [])
    if not inputs:
        raise ValueError("manifest needs at least one input")
    for spec in inputs:
        table = spec["dataset"]
        if table not in TABLES:
            raise ValueError(f"unknown dataset {table!r}")
        path = manifest_path.parent / spec["path"]
        source = bundle.archive(path)
        clock = spec.get("clock", {"basis": "period", "unit": "ms"})
        if clock.get("basis", "period") not in ("period", "match"):
            raise ValueError("clock basis must be period or match")
        coordinate = spec.get("coordinates")
        if table in ("events", "tracking") and not coordinate:
            raise ValueError("events/tracking require an explicit coordinates declaration")
        for index, original in enumerate(read_records(path, spec.get("root"), spec.get("delimiter", ",")), 1):
            try:
                if not isinstance(original, dict):
                    raise ValueError("each source record must be an object")
                row = dict(spec.get("constants", {}))
                for dest, source_path in spec.get("fields", {}).items():
                    value = get_path(original, source_path)
                    row[dest] = None if missing(value) else value
                for dest, values in spec.get("value_maps", {}).items():
                    if row.get(dest) is not None:
                        key = str(row[dest])
                        if key not in values:
                            raise ValueError(f"unmapped {dest} value {key!r}")
                        row[dest] = values[key]
                for name, value in list(row.items()):
                    if name in BOOL_FIELDS:
                        row[name] = boolean(value)
                    elif name in INT_FIELDS and name != "period_time_ms":
                        row[name] = integer(value)
                    elif name in FLOAT_FIELDS:
                        row[name] = number(value)
                raw_match = row.get("match_id")
                raw_competition = row.get("competition_id")
                original_id = row.get(PRIMARY[table])
                if table in ("events", "tracking"):
                    stamp = time_ms(row.get("period_time_ms"), clock.get("unit", "ms"))
                    if stamp is not None and clock.get("basis", "period") == "match":
                        offsets = clock.get("period_offsets_ms", {})
                        if str(row.get("period")) not in offsets:
                            raise ValueError("match clock requires explicit offset for each period")
                        stamp -= integer(offsets[str(row["period"])])
                        if stamp < 0:
                            raise ValueError("timestamp precedes period offset")
                    row["period_time_ms"] = stamp
                    length, width = number(coordinate["length_m"]), number(coordinate["width_m"])
                    if length is None or width is None or length <= 0 or width <= 0:
                        raise ValueError("positive coordinate dimensions required")
                    row["x_m"], row["y_m"] = point(row.get("x_m"), row.get("y_m"), coordinate, length, width)
                    if "end_x_m" in row or "end_y_m" in row:
                        row["end_x_m"], row["end_y_m"] = point(row.get("end_x_m"), row.get("end_y_m"), coordinate, length, width)
                    row["coordinate_frame"] = coordinate.get("frame", "stadium_fixed")
                    row["coordinate_length_m"], row["coordinate_width_m"] = length, width
                    if coordinate.get("dimensions_assumed", False):
                        bundle.warnings.append("Coordinate dimensions are declared assumed; do not treat them as measured pitch dimensions.")
                for field, kind in ID_KINDS.items():
                    if field in row:
                        scope = str(raw_match) if kind in ("event", "appearance", "snapshot", "tracking", "video") else str(raw_competition) if kind == "season" and raw_competition is not None else None
                        row[field] = bundle.identity.resolve(kind, row[field], scope)
                if table == "tracking":
                    kind = row.get("object_type")
                    if kind == "player":
                        raw_object = row.get("object_id")
                        row["object_id"] = bundle.identity.resolve("player", raw_object)
                        row["player_id"] = row["object_id"]
                    elif kind == "ball":
                        row["object_id"] = "ball"
                        row["player_id"] = None
                    if not row.get("tracking_id"):
                        row["tracking_id"] = bundle.identity.resolve("tracking", f"{row.get('period')}:{row.get('frame_id')}:{row.get('object_id')}", str(raw_match))
                    if row.get("estimated") is None and row.get("observed") is not None:
                        row["estimated"] = not row["observed"]
                bundle.emit(table, row, original_id, source, index)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path.name}, record {index}: {exc}") from exc
    bundle.warnings = list(dict.fromkeys(bundle.warnings))
