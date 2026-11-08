"""Embed two locally generated reports in a standalone dashboard."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundesliga", type=Path)
parser.add_argument("premier_league", type=Path)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("local.html"))
args = parser.parse_args()

source = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
for id_, path in (("sample-data", args.bundesliga), ("epl-data", args.premier_league)):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "midfield-v0.1" or not data.get("comparison_version"):
        parser.error(f"{path}: expected a relative midfield report")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    marker = f'<script id="{id_}" type="application/json">'
    placeholder = marker + 'null</script>'
    if placeholder not in source:
        parser.error(f"Missing dashboard placeholder {id_}")
    source = source.replace(placeholder, marker + payload + '</script>', 1)
with args.output.open("x", encoding="utf-8") as handle:
    handle.write(source)
print(args.output.resolve())
