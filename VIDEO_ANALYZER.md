# FBRPRO video analyzer

Open [Video analysis](video/index.html) in a browser. Load a video and its
tracking report to watch the footage and pitch replay together. Tracking boxes
can be turned on over the footage. The timeline highlights sampled windows.
Add a clip plan to review individual opportunities, or add a movement report
to jump from a measure to the footage and see why other episodes were excluded.
The [Midfield Lab](dashboard/index.html) can open the same movement report next
to an existing event-based report. Video evidence and role comparisons use
separate results.

## Run a fixture

Use an authorized wide-angle recording and an accepted normalized event bundle
for the same match. Install the local tracking dependencies with
`python -m pip install -e '.[tracking]'`. Install `.[tracking,parquet]` to
enable optional Parquet export. Obtain the 640×640 YOLOX-s ONNX export from the
[OpenCV Zoo model directory](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox).

1. In **Align video to the match clock**, enter a fixture ID and a video ID.
   Record two known events in each uninterrupted live segment, and a third
   independent check event within it. Match seconds reset at each half.
   Export the alignment. Split around edits, replays, and missing footage.
   A check error above 0.5 seconds excludes the span. Old alignments without
   checks remain usable for preview, with movement and canonical export gated.
2. Select review windows from the event feed. The planner samples opposition
   possessions and optional own possessions, and queues candidate regains.
   The vision stage decodes selected windows, so a fixture need not be
   processed from kickoff to full time.

   ```sh
   python -m football_normalizer.video normalized-match --match MATCH_ID \
     --alignment video-alignment.json --team home --build-up-limit 10 \
     --in-possession-limit 10 --output clip-plan.json
   ```

3. Prepare camera landmarks in `calibration.json`. Use the image's native
   pixels and stadium-fixed ground coordinates in metres, with one shared
   origin and orientation for the fixture. Use at least four noncollinear
   points, plus separate check marks omitted from fitting. This is the
   [format of the sample calibration](#calibration-format).
4. Run the tracker. Load the source video and the resulting
   `video-tracking.json` in the review workspace. Import the clip plan to
   confirm track IDs against either team's roster. Export the identity map
   after inspecting the relevant clips. Unknown people remain anonymous.

   ```sh
   python -m football_normalizer.video_run match.mp4 --plan clip-plan.json \
     --model object_detection_yolox_2022nov.onnx --calibration calibration.json \
     --output video-runs/fixture
   ```

5. Review the phase and any missing context. Rerun the same command with
   `--identity-map video-identities.json --context context.json --bundle normalized-match`
   to generate movement evidence. Add `--parquet` to export nonempty canonical
   tracking in Parquet. Load the `video-metrics.json` in either workspace.

The output paths are printed and recorded in `run.json`. Tracking is reused
when the footage, model, plan, calibration, identity map, code, dependencies
and sampling options match. Changing the reviewed movement context reruns the
measurements without rerunning detection. Changing calibration or identity
assignments creates a separate tracking result. Keep the source video, plans,
review files and run folder together outside Git.

### Calibration format

```json
{
  "video_id": "fixture-wide",
  "coordinate_length_m": 105,
  "coordinate_width_m": 68,
  "dimensions_assumed": true,
  "segments": [{
    "segment_id": "first-half-live",
    "video_start_ms": 10000,
    "video_end_ms": 14000,
    "image_points": [[410,304],[596,367],[104,654],[1276,341],[1280,603]],
    "pitch_points": [[0,0],[16.5,13.84],[16.5,54.16],[52.5,0],[52.5,43.15]],
    "check_image_points": [[1277,450]],
    "check_pitch_points": [[52.5,24.85]],
    "follow_camera": true
  }]
}
```

Coordinates are illustrative. For a fixed shot use `follow_camera: false`.
With camera following enabled, include the reference frame at the beginning of
the selected window. The tracker discards calibration when landmark tracking
or independent geometry checks fail. Use `--max-error-m` to set the
projection check tolerance (default 2 metres). Assumed pitch dimensions
produce reference-scale coordinates. Review optical flow drift on long pans.

### Reviewed context format

Use canonical IDs from the plan. The optional accepted event bundle derives
known midfield playing intervals and tagged regains. Specify attacking
direction for the selected team in each period (`1` toward increasing X,
`-1` toward decreasing X). Review phase boundaries and supply the players
and geometry each measure needs.

```json
{
  "match_id": "MATCH_ID",
  "video_id": "fixture-wide",
  "plan_id": "PLAN_ID",
  "attacking_direction": {"1": 1, "2": -1},
  "players": ["PLAYER_ID"],
  "episodes": [{
    "opportunity_id": "opportunity-1",
    "phase": "settled_defence",
    "player_ids": ["PLAYER_ID"],
    "position_groups": {"PLAYER_ID": "defensive"},
    "defender_ids": ["DEFENDER_1", "DEFENDER_2"],
    "ground_ball_reviewed": true,
    "opponent_team_id": "OPPOSITION_ID",
    "ball_carrier_id": "CARRIER_ID"
  }]
}
```

| Measure | Additional reviewed evidence |
| --- | --- |
| Central screening time | Own defensive line and an observable ground ball |
| Deep-to-advanced support | A regain anchor and a continuous track for the player |
| Deep passing outlet | Ground ball and complete lane coverage, with `opponent_ids` and `lane_context_complete` |
| Between-line availability | `opponent_midfielder_ids` and `opponent_defender_ids` from both lines |
| Between-line reception | Both lines and period-clock `receptions`, each with `player_id` and `period_time_ms`, plus `receptions_complete` |
| Closing down | The opposing `ball_carrier_id` |

Optional `start_ms` and `end_ms` can narrow a reviewed episode within its
opportunity, for example to the point of a turnover. Match scope and evidence
IDs stay with every observation. A missing measure shows as unavailable, and
its exclusions remain visible. Geometric tolerances define the measurement,
while player comparison uses the measured rates within the selected scope.

Pool match reports from one competition and season using:

```sh
python -m football_normalizer.video_metrics --pool match-1.json match-2.json \
  --output pooled-video-metrics.json
```

To compare with human review, have two analysts independently export
observations for the same clips in **Record an observation**, then run:

```sh
python -m football_normalizer.video_validate video-metrics.json \
  --review analyst-a.json analyst-b.json --output episode-validation.json
```

The comparison reports agreement, binary episode precision and recall where
the reviewers agree and the episode is observable. Partial and missing
results, and analyst disagreements, stay visible separately. It does not
measure timing overlap or per-player rate error.

## Real-video pilot

The [Roboflow sample setup](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh#L21)
references the real 30-second DFL clip `121364_0.mp4`. The pilot decoded and
tracked two four-second windows at 5 samples per second. It checked individual
frames manually and retained only the pitch frames that passed the projection
check. The MP4 and weights are local inputs and remain outside Git.

The sample supports image detection, clip seeking, and limited anonymous
pitch replay. It has no verified fixture clock or player identities in this
project. The generic detector also missed the visible ball in the manually
reviewed frames. The movement report therefore contains no player scores.
Validated tactical measurements require synchronized footage, confirmed
rosters, and independent movement reviews on held-out fixtures. The current
tracker still relies on manually seeded calibration and analyst-confirmed
identity; it has no full-match automated replay or lineup recognition.
