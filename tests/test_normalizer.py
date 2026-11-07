import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from football_normalizer.cli import main
from football_normalizer.core import Identity, boolean, integer, missing, number, point, time_ms
from football_normalizer.validation import validate

ROOT = Path(__file__).resolve().parents[1]


def records(directory, name):
    return [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(list(map(str, args)))

    def sb(self, output, extra=()):
        ex = ROOT / "examples/statsbomb"
        return self.run_cli("statsbomb", "--events", ex / "events.json", "--lineups", ex / "lineups.json",
                            "--match", ex / "match.json", "--match-id", "demo-1", "--snapshots", ex / "360.json",
                            "--output", output, *extra)

    def test_statsbomb_preserves_unknowns_period_clock_and_anonymous_players(self):
        out = self.base / "sb"
        self.assertEqual(self.sb(out), 0)
        events = records(out, "events")
        self.assertEqual(events[0]["period_time_ms"], 754200)
        self.assertEqual(events[-1]["period_time_ms"], 3000)
        self.assertEqual(events[0]["x_m"], 52.5)
        self.assertEqual(events[0]["y_m"], 34)
        self.assertIsNone(events[0]["under_pressure"])
        self.assertIs(events[1]["under_pressure"], False)
        self.assertEqual(events[0]["outcome"], "complete")
        self.assertEqual(events[1]["outcome"], "incomplete")
        self.assertIsNone(records(out, "snapshots")[0]["players"][0]["player_id"])
        self.assertTrue(json.loads((out / "validation.json").read_text())["valid"])
        match = records(out, "matches")[0]
        self.assertIsNone(match["pitch_length_m"])
        self.assertTrue(match["dimensions_assumed"])
        self.assertTrue((out / "raw").exists())

    def test_mapped_csv_tracking_flags_and_coordinates(self):
        out = self.base / "mapped"
        self.assertEqual(self.run_cli("mapped", "--manifest", ROOT / "examples/mapped/manifest.json", "--output", out), 0)
        rows = records(out, "tracking")
        self.assertEqual(rows[0]["x_m"], 52.5)
        self.assertEqual(rows[0]["y_m"], 34)
        self.assertTrue(rows[0]["observed"])
        self.assertFalse(rows[0]["estimated"])
        self.assertFalse(rows[1]["observed"])
        self.assertTrue(rows[1]["estimated"])
        self.assertEqual(rows[-1]["period_time_ms"], 100)

    def test_deterministic_and_explicit_cross_provider_identities(self):
        a, b = Identity("a"), Identity("b")
        self.assertEqual(a.resolve("player", 1), a.resolve("player", "1"))
        self.assertNotEqual(a.resolve("player", 1), b.resolve("player", 1))
        self.assertNotEqual(a.resolve("event", 1, "m1"), a.resolve("event", 1, "m2"))
        self.assertEqual(Identity("a", {"player": {"1": "reviewed_player_123"}}).resolve("player", 1), "reviewed_player_123")

    def test_missing_markers_do_not_erase_legitimate_text_ids(self):
        self.assertFalse(missing("NA"))
        self.assertIsNotNone(Identity("provider").resolve("team", "NA"))
        self.assertIsNone(number("NA"))

    def test_coordinates_no_silent_clipping(self):
        self.assertEqual(point(1, 1, {"system": "scaled", "x_max": 1, "y_max": 1}, 105, 68), (105, 68))
        self.assertEqual(point(-52.5, -34, {"system": "centered_metres"}, 105, 68), (0, 0))
        self.assertEqual(point(110, 70, {"system": "metres"}, 105, 68), (110, 70))
        with self.assertRaises(ValueError):
            point(1, None, {}, 105, 68)

    def test_boolean_and_time_parsing(self):
        self.assertFalse(boolean("false"))
        self.assertIsNone(boolean(""))
        self.assertEqual(time_ms("00:47:12.350", "timestamp"), 2832350)
        self.assertEqual(time_ms("0.125", "s"), 125)
        for bad in ("maybe", "yes"):
            with self.assertRaises(ValueError):
                boolean(bad)
        self.assertIsNone(time_ms("NaN"))
        for bad in ("Infinity", "-1", "01:70:00"):
            with self.assertRaises(ValueError):
                time_ms(bad, "timestamp" if ":" in bad else "ms")
        with self.assertRaises(ValueError):
            integer("1.2")

    def test_duplicate_events_rejected(self):
        out = self.base / "sb"
        self.assertEqual(self.sb(out), 0)
        path = out / "events.jsonl"
        with path.open("a") as f:
            f.write(path.read_text().splitlines()[0] + "\n")
        report = validate(out)
        self.assertFalse(report["valid"])
        self.assertIn("duplicate_id", {i["code"] for i in report["issues"]})

    def test_broken_player_reference_rejected(self):
        out = self.base / "sb"
        self.assertEqual(self.sb(out), 0)
        rows = records(out, "events")
        rows[0]["player_id"] = "unknown_player"
        (out / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        report = validate(out)
        self.assertIn("unresolved_reference", {i["code"] for i in report["issues"]})

    def test_existing_output_is_never_overwritten(self):
        out = self.base / "sb"
        out.mkdir()
        (out / "keep.txt").write_text("keep")
        self.assertEqual(self.sb(out), 2)
        self.assertEqual((out / "keep.txt").read_text(), "keep")

    def test_bad_input_is_quarantined_not_published(self):
        path = self.base / "bad.json"
        path.write_text(json.dumps({"provider": "test", "inputs": [{"dataset": "tracking", "path": "rows.csv", "fields": {}}]}))
        (self.base / "rows.csv").write_text("id\n1\n")
        out = self.base / "accepted"
        self.assertEqual(self.run_cli("mapped", "--manifest", path, "--output", out), 2)
        self.assertFalse(out.exists())
        quarantines = list(self.base.glob(".accepted-*"))
        self.assertEqual(len(quarantines), 1)
        self.assertEqual(json.loads((quarantines[0] / "status.json").read_text())["status"], "rejected")

    def test_match_clock_requires_explicit_offsets(self):
        config = json.loads((ROOT / "examples/mapped/manifest.json").read_text())
        # Use absolute references when relocating the manifest.
        for spec in config["inputs"]:
            spec["path"] = str(ROOT / "examples/mapped" / spec["path"])
            if spec["dataset"] == "tracking":
                spec["clock"] = {"basis": "match", "unit": "s"}
        path = self.base / "manifest.json"
        path.write_text(json.dumps(config))
        self.assertEqual(self.run_cli("mapped", "--manifest", path, "--output", self.base / "bad"), 2)
        config["inputs"][-1]["clock"]["period_offsets_ms"] = {"1": 0}
        path.write_text(json.dumps(config))
        self.assertEqual(self.run_cli("mapped", "--manifest", path, "--output", self.base / "good"), 0)

    def test_contradictory_tracking_flags_rejected(self):
        out = self.base / "mapped"
        self.assertEqual(self.run_cli("mapped", "--manifest", ROOT / "examples/mapped/manifest.json", "--output", out), 0)
        rows = records(out, "tracking")
        rows[0]["observed"] = rows[0]["estimated"] = True
        (out / "tracking.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        report = validate(out)
        self.assertIn("contradictory_flags", {i["code"] for i in report["issues"]})

    def test_source_archive_tampering_detected(self):
        out = self.base / "sb"
        self.assertEqual(self.sb(out), 0)
        manifest = json.loads((out / "manifest.json").read_text())
        (out / manifest["sources"][0]["archive"]).write_text("changed")
        report = validate(out)
        self.assertFalse(report["valid"])
        self.assertIn("archive_checksum", report["issue_counts"])

    def test_tracking_gap_reported_without_inventing_observations(self):
        out = self.base / "mapped"
        self.assertEqual(self.run_cli("mapped", "--manifest", ROOT / "examples/mapped/manifest.json", "--output", out), 0)
        rows = records(out, "tracking")
        rows[1]["period_time_ms"] = 2000
        (out / "tracking.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        report = validate(out)
        self.assertTrue(report["valid"])
        self.assertIn("tracking_gaps", report["issue_counts"])
        self.assertEqual(report["coverage"]["tracking"]["records"], 2)

    def test_same_frame_object_is_rejected_even_with_different_row_id(self):
        out = self.base / "mapped"
        self.assertEqual(self.run_cli("mapped", "--manifest", ROOT / "examples/mapped/manifest.json", "--output", out), 0)
        rows = records(out, "tracking")
        duplicate = dict(rows[-1], tracking_id="different-id")
        with (out / "tracking.jsonl").open("a") as handle:
            handle.write(json.dumps(duplicate) + "\n")
        report = validate(out)
        self.assertFalse(report["valid"])
        self.assertIn("duplicate_frame_object", report["issue_counts"])

    def test_mapped_json_events_with_vocabulary_and_identity_conversion(self):
        config = json.loads((ROOT / "examples/mapped/manifest.json").read_text())
        config["inputs"] = config["inputs"][:-1]
        for spec in config["inputs"]:
            spec["path"] = str(ROOT / "examples/mapped" / spec["path"])
        event = {"id": "e1", "player": {"id": "10"}, "time": "00:00:01.250", "kind": "Pass", "xy": [0.5, 0.5]}
        (self.base / "events.json").write_text(json.dumps({"items": [event]}))
        config["inputs"].append({"dataset": "events", "path": "events.json", "root": "items",
                                 "constants": {"match_id": "m1", "team_id": "h", "possession_team_id": "h", "period": 1},
                                 "fields": {"event_id": "id", "player_id": "player.id", "type": "kind", "period_time_ms": "time", "x_m": "xy.0", "y_m": "xy.1"},
                                 "value_maps": {"type": {"Pass": "pass"}},
                                 "clock": {"basis": "period", "unit": "timestamp"},
                                 "coordinates": {"system": "scaled", "x_max": 1, "y_max": 1, "length_m": 105, "width_m": 68}})
        path = self.base / "manifest.json"
        path.write_text(json.dumps(config))
        out = self.base / "json-events"
        self.assertEqual(self.run_cli("mapped", "--manifest", path, "--output", out), 0)
        row = records(out, "events")[0]
        self.assertEqual(row["type"], "pass")
        self.assertEqual(row["period_time_ms"], 1250)
        self.assertEqual(row["x_m"], 52.5)
        self.assertEqual(row["possession_team_id"], row["team_id"])
        event["kind"] = "Unknown"
        (self.base / "events.json").write_text(json.dumps({"items": [event]}))
        self.assertEqual(self.run_cli("mapped", "--manifest", path, "--output", self.base / "bad-vocab"), 2)

    def test_parquet_when_available(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("optional pyarrow not installed")
        out = self.base / "parquet"
        self.assertEqual(self.sb(out, ("--parquet",)), 0)
        rows = pq.read_table(out / "events.parquet").to_pylist()
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[0]["under_pressure"])
        self.assertIsInstance(json.loads(rows[0]["source"]), dict)


if __name__ == "__main__":
    unittest.main()
