from copy import deepcopy
import unittest

from football_normalizer.video_events import align_predictions, analyze


def tracking():
    frames = []
    for i, x in enumerate((99, 103, 140, 176, 210, 240)):
        frames.append({"video_time_ms": i*100, "period": 1, "clip_id": "c", "objects": [
            {"track_id": "passer", "object_type": "player", "observed": True, "on_pitch": True,
             "bbox_xyxy": [90, 60, 110, 100]},
            {"track_id": "ball", "object_type": "ball", "observed": True, "bbox_xyxy": [x-2, 97, x+2, 101]},
        ]})
    return {"version": "fbrpro-video-tracking-v1", "sample_fps": 10, "coordinate_length_m": 105,
            "coordinate_width_m": 68, "video_id": "v", "frames": frames}


def predictions():
    return {"UrlLocal": "fixture", "predictions": [
        {"half": "1", "gameTime": "1 - 00:30", "position": "30140", "label": "Pass", "confidence": "0.82"},
        {"half": "1", "gameTime": "1 - 00:30", "position": "30510", "label": "Shot", "confidence": "0.72"},
        {"half": "2", "gameTime": "2 - 00:30", "position": "30140", "label": "High Pass", "confidence": "0.9"},
    ]}


ALIGN = {"half": 1, "source_ms": 30000, "video_ms": 0}


class VideoEventsTests(unittest.TestCase):
    def test_soccernet_half_alignment_and_separate_pass_count(self):
        source = tracking()
        before = deepcopy(source)
        report = analyze(source, predictions=predictions(), alignment=ALIGN)
        self.assertEqual(source, before)
        self.assertEqual(report["video_id"], "v")
        self.assertEqual(report["predictions_in_scope"], 2)
        p, shot = report["events"]
        self.assertEqual((p["t_ms"], p["actor_sample_ms"], p["track_id"]), (140, 100, "passer"))
        self.assertEqual(p["action_score"], .82)
        self.assertEqual(shot["type"], "shot")
        self.assertIsNone(shot["track_id"])
        self.assertEqual(report["player_pass_attempts"], {"passer": 1})

    def test_only_predictions_in_the_selected_half_and_scope_are_imported(self):
        source = tracking()
        report = analyze(source, {"period_intervals": [{"start_ms": 0, "end_ms": 600, "period": 2}], "end_ms": 300},
                         predictions(), ALIGN)
        self.assertEqual(report["predictions_in_scope"], 0)
        self.assertEqual(report["predictions_outside_scope"], 2)
        report = analyze(source, {"end_ms": 300}, predictions(), ALIGN)
        self.assertEqual(report["predictions_in_scope"], 1)

    def test_tracking_only_reports_possible_pass_without_counting_an_attempt(self):
        report = analyze(tracking())
        self.assertEqual(report["player_pass_attempts"], {})
        self.assertEqual([(e["t_ms"], e["status"], e["track_id"]) for e in report["events"]], [(150, "candidate", "passer")])

    def test_short_teammate_transfer_survives_one_missing_ball_sample(self):
        source = tracking()
        for i, (frame, x) in enumerate(zip(source["frames"], (99, 102, 111, 121, 123, 124), strict=True)):
            passer, ball = frame["objects"]
            passer["inferred_team"] = "kit:team_b"
            ball.update(bbox_xyxy=[x-2, 97, x+2, 101], observed=i != 2)
            frame["objects"].append({**passer, "track_id": "receiver", "bbox_xyxy": [112, 60, 132, 100]})
        report = analyze(source)
        event, = report["events"]
        self.assertEqual((event["t_ms"], event["track_id"], event["recipient_candidate_track_id"]), (200, "passer", "receiver"))
        self.assertEqual(report["player_pass_attempts"], {})
        for negative in ("opponent", "cut", "duplicate player box"):
            with self.subTest(negative=negative):
                altered = deepcopy(source)
                if negative == "opponent":
                    for frame in altered["frames"]:
                        frame["objects"][-1]["inferred_team"] = "kit:team_a"
                elif negative == "cut":
                    altered["frames"][2]["camera_cut_suspected"] = True
                else:
                    for frame in altered["frames"]:
                        frame["objects"].append({**frame["objects"][0], "track_id": "duplicate"})
                self.assertEqual(analyze(altered)["events"], [])

    def test_one_touch_redirection_can_support_a_short_pass(self):
        source = tracking()
        positions = [(100, 68), (100, 79), (100, 99), (111, 99), (122, 99), (123, 99)]
        for frame, (x, y) in zip(source["frames"], positions, strict=True):
            passer, ball = frame["objects"]
            passer["inferred_team"] = "kit:team_b"
            ball["bbox_xyxy"] = [x-2, y-2, x+2, y+2]
            frame["objects"].append({**passer, "track_id": "receiver", "bbox_xyxy": [112, 60, 132, 100]})
        event, = analyze(source)["events"]
        self.assertEqual((event["t_ms"], event["track_id"], event["recipient_candidate_track_id"]), (250, "passer", "receiver"))

    def test_occlusion_cut_and_competing_players_leave_actor_unknown(self):
        for reason in ("occlusion", "camera cut", "overlapping players"):
            with self.subTest(reason=reason):
                source = tracking()
                if reason == "occlusion":
                    source["frames"][1]["objects"][1]["observed"] = False
                elif reason == "camera cut":
                    source["frames"][2]["camera_cut_suspected"] = True
                else:
                    for frame in source["frames"]:
                        frame["objects"].append({**frame["objects"][0], "track_id": "other"})
                report = analyze(source, predictions=predictions(), alignment=ALIGN)
                self.assertIsNone(report["events"][0]["track_id"])
                self.assertEqual(report["player_pass_attempts"], {})

    def test_nearby_duplicate_predictions_share_one_release(self):
        labels = predictions()
        labels["predictions"].append({"half": "1", "position": "30170", "label": "Pass", "confidence": "0.94"})
        report = analyze(tracking(), predictions=labels, alignment=ALIGN)
        self.assertEqual(report["player_pass_attempts"], {"passer": 1})
        self.assertIsNone(report["events"][1]["track_id"])

    def test_rejects_labels_and_inconsistent_prediction_clock(self):
        with self.assertRaisesRegex(ValueError, "results_spotting"):
            align_predictions({"annotations": []}, ALIGN)
        labels = predictions()
        labels["predictions"][0]["half"] = "2"
        with self.assertRaisesRegex(ValueError, "must agree"):
            align_predictions(labels, ALIGN)
