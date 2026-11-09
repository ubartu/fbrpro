import unittest

from football_normalizer.video_roles import annotate_video_roles


def person(track_id, hue, *, x=45, y=31, **changes):
    return {"track_id": track_id, "object_type": "player", "kit_color_hint": hue,
            "x_m": x, "y_m": y, "projection_status": "verified", "observed": True,
            "estimated": False, "visibility": "observed", "team_id": None, "player_id": None, **changes}


def team_frames(*, first="red", second="white", total=9):
    return [{"clip_id": "clip-a", "video_time_ms": i*200, "calibration_status": "verified",
             "objects": [person(f"a-{j}", first, team_id="reviewed-team", player_id=f"player-{j}") for j in range(3)]
                        + [person(f"b-{j}", second) for j in range(3)]}
            for i in range(total)]


def annotate(frames, **options):
    return annotate_video_roles(frames, coordinate_length_m=105, coordinate_width_m=68, **options)["clips"][0]["shots"]


class VideoRoleTests(unittest.TestCase):
    def test_two_stable_kits_dont_force_official_into_team_or_change_reviewed_identity(self):
        frames = team_frames()
        for frame in frames:
            frame["objects"].append(person("official", "yellow"))
        shot, = annotate(frames)
        self.assertEqual(shot["team_status"], "inferred")
        self.assertEqual({k["inferred_team"] for k in shot["kit_groups"]}, {"kit:red", "kit:white"})
        a, _, _, b, _, _, official = frames[0]["objects"]
        self.assertEqual((a["inferred_team"], a["team_id"], a["player_id"]), ("kit:red", "reviewed-team", "player-0"))
        self.assertEqual((b["inferred_team"], b["team_id"]), ("kit:white", None))
        self.assertIsNone(official["inferred_team"])
        self.assertIsNone(official["inferred_role"])

    def test_three_recurring_kits_and_non_overlapping_kits_are_ambiguous(self):
        frames = team_frames()
        for frame in frames:
            frame["objects"].extend(person(f"third-{j}", "black") for j in range(3))
        shot, = annotate(frames)
        self.assertEqual(shot["team_status"], "multiple_recurring_kits")
        self.assertTrue(all(row["inferred_team"] is None for f in frames for row in f["objects"]))

        frames = team_frames()
        for i, frame in enumerate(frames):
            frame["objects"] = frame["objects"][:3] if i < 4 else frame["objects"][3:]
        shot, = annotate(frames)
        self.assertEqual(shot["team_status"], "two_co_visible_kits_unconfirmed")

    def test_one_goalkeeper_candidate_per_end_requires_verified_pitch_and_repeated_position(self):
        frames = team_frames()
        for frame in frames:
            frame["objects"].extend([person("left", "yellow", x=4, y=33),
                                     person("right", "green", x=101, y=35),
                                     person("sideline", "yellow", x=4, y=3)])
        shot, = annotate(frames)
        self.assertEqual({(k["track_id"], k["inferred_goal_end"]) for k in shot["goalkeepers"]},
                         {("left", "low_x"), ("right", "high_x")})
        left, right, sideline = frames[-1]["objects"][-3:]
        self.assertEqual((left["inferred_role"], right["inferred_role"]), ("probable_goalkeeper", "probable_goalkeeper"))
        self.assertIsNone(left["inferred_team"])
        self.assertIsNone(sideline["inferred_role"])

        shot, = annotate(frames, dimensions_assumed=True)
        self.assertEqual(shot["goalkeeper_status"], "inferred")
        for frame in frames:
            frame["calibration_status"] = "uncalibrated"
        shot, = annotate(frames, dimensions_assumed=True)
        self.assertEqual(shot["goalkeeper_status"], "verified_pitch_required")
        self.assertIsNone(left["inferred_role"])
        for frame in frames[:2]:
            frame["calibration_status"] = "verified"
        for frame in frames[2:]:
            frame["calibration_status"] = "rejected"
        shot, = annotate(frames)
        self.assertEqual(shot["goalkeepers"], [])

    def test_a_camera_cut_and_a_gap_cannot_establish_keeper_continuity(self):
        frames = team_frames(total=8)
        for frame in frames:
            frame["objects"].append(person("keeper", "green", x=4, y=33))
        frames[4]["camera_cut_suspected"] = True
        self.assertTrue(all(s["goalkeepers"] == [] for s in annotate(frames)))

        frames[4]["camera_cut_suspected"] = False
        frames[4]["objects"][-1].update(observed=False, visibility="missing")
        shot, = annotate(frames)
        self.assertEqual(shot["goalkeepers"], [])

    def test_multiple_plausible_tracks_at_same_goal_abstain(self):
        frames = team_frames()
        for frame in frames:
            frame["objects"].extend([person("one", "yellow", x=4, y=33), person("two", "cyan", x=6, y=35)])
        shot, = annotate(frames)
        self.assertEqual(shot["ambiguous_goal_ends"], ["low_x"])
        self.assertEqual(shot["goalkeepers"], [])

    def test_disagreeing_reviewed_team_ids_or_unstable_jersey_abstain(self):
        frames = team_frames()
        for frame in frames:
            frame["objects"][1]["team_id"] = "other-team"
        shot, = annotate(frames)
        self.assertEqual(shot["team_status"], "reviewed_teams_conflict_with_kits")

        frames = team_frames()
        for i, frame in enumerate(frames):
            if i % 2:
                frame["objects"][1]["kit_color_hint"] = "orange"
        shot, = annotate(frames)
        self.assertEqual(shot["team_status"], "two_co_visible_kits_unconfirmed")


if __name__ == "__main__":
    unittest.main()
