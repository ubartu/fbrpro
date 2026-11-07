import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import mapped, statsbomb
from .core import Bundle, dumps
from .validation import validate


def parser():
    p = argparse.ArgumentParser(description="Normalize football exports into an auditable common schema.")
    commands = p.add_subparsers(dest="command", required=True)
    sb = commands.add_parser("statsbomb", help="StatsBomb Open Data JSON")
    sb.add_argument("--events", type=Path, required=True)
    sb.add_argument("--lineups", type=Path, required=True)
    sb.add_argument("--match-id", required=True)
    sb.add_argument("--match", type=Path, help="match object or competition-season matches array")
    sb.add_argument("--snapshots", type=Path, help="optional three-sixty JSON")
    sb.add_argument("--home-team-id")
    sb.add_argument("--away-team-id")
    sb.add_argument("--pitch-length", type=float)
    sb.add_argument("--pitch-width", type=float)
    mp = commands.add_parser("mapped", help="CSV/JSON/JSONL with an explicit mapping manifest")
    mp.add_argument("--manifest", type=Path, required=True)
    for cmd in (sb, mp):
        cmd.add_argument("--output", type=Path, required=True, help="new directory; existing paths are never overwritten")
        cmd.add_argument("--identity-map", type=Path, help="optional reviewed provider ID to canonical ID mappings")
        cmd.add_argument("--parquet", action="store_true", help="also export Parquet (requires optional pyarrow)")
    check = commands.add_parser("validate", help="recheck a normalized bundle")
    check.add_argument("directory", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    stage = None
    bundle = None
    try:
        if args.command == "validate":
            report = validate(args.directory)
            print(dumps({"valid": report["valid"], "errors": report["errors"], "warnings": report["warnings"]}))
            return 0 if report["valid"] else 2
        if args.output.exists():
            raise ValueError("output already exists; choose a new directory")
        if args.parquet:
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise ValueError("Parquet export requires: pip install '.[parquet]'") from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
        overrides = json.loads(args.identity_map.read_text()) if args.identity_map else None
        provider = "statsbomb" if args.command == "statsbomb" else json.loads(args.manifest.read_text())["provider"]
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a nonempty string")
        bundle = Bundle(stage, provider, overrides)
        if args.identity_map:
            bundle.archive(args.identity_map)
        if args.command == "statsbomb":
            statsbomb.normalize(args, bundle)
        else:
            mapped.normalize(args.manifest, bundle)
        bundle.finish()
        report = validate(stage)
        if not report["valid"]:
            raise ValueError(f"validation failed with {report['errors']} errors; see validation.json")
        if args.parquet:
            from .export import parquet
            parquet(stage)
        (stage / "status.json").write_text(dumps({"status": "accepted"}) + "\n")
        # A stage on the same filesystem is published only after validation succeeds.
        if args.output.exists():
            raise ValueError("output was created during import; refusing to replace it")
        stage.rename(args.output)
        print(dumps({"status": "accepted", "output": str(args.output.resolve()), "counts": bundle.counts,
                     "warnings": report["warnings"] + len(bundle.warnings)}))
        return 0
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, OSError) as exc:
        if bundle:
            bundle.close()
        details = {"status": "rejected", "error": str(exc)}
        if stage and stage.exists():
            if bundle and not (stage / "manifest.json").exists():
                bundle.finish()
            (stage / "status.json").write_text(dumps(details) + "\n")
            # Retain isolated partial files and raw input for diagnosis; never publish them as accepted.
            details["quarantine"] = str(stage.resolve())
        print(dumps(details), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
