"""Optional bounded-memory Parquet export. Nested fields are JSON strings."""
import json

from .core import TABLES, dumps


def parquet(directory):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("Parquet export requires: pip install '.[parquet]'") from exc
    for table in TABLES:
        path = directory / f"{table}.jsonl"
        if not path.exists():
            continue
        types = {}
        nested = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                for key, value in json.loads(line).items():
                    if isinstance(value, (dict, list)):
                        nested.add(key)
                    kind = "null" if value is None else "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "float" if isinstance(value, float) else "str"
                    previous = types.get(key, "null")
                    if previous == "null":
                        types[key] = kind
                    elif kind not in (previous, "null"):
                        types[key] = "float" if {previous, kind} <= {"int", "float"} else "str"
        arrow_types = {"null": pa.string(), "str": pa.string(), "bool": pa.bool_(), "int": pa.int64(), "float": pa.float64()}
        schema = pa.schema([(key, pa.string() if key in nested else arrow_types[kind]) for key, kind in sorted(types.items())])

        def convert(row):
            return {key: None if row.get(key) is None else dumps(row[key]) if key in nested else str(row[key]) if types[key] == "str" and not isinstance(row[key], str) else row[key] for key in types}

        with pq.ParquetWriter(directory / f"{table}.parquet", schema, compression="zstd") as writer:
            batch = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    batch.append(convert(json.loads(line)))
                    if len(batch) >= 10000:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
