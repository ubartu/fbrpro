# FBRPRO video analyzer

Open the running local app for automatic tracking, or open
[Video analysis](video/index.html) to review existing files directly. Load a
video and its tracking report to watch the footage and pitch replay together.
Tracking boxes can be turned on over the footage. The timeline highlights sampled windows.
Add a clip plan to review individual opportunities, or add a movement report
to jump from a measure to the footage and see why other episodes were excluded.
The [Midfield Lab](dashboard/index.html) can open the same movement report next
to an existing event-based report. Video evidence and role comparisons use
separate results.

## Start the local app

From the project checkout, install `python -m pip install -e '.[tracking]'`
in your Python environment and obtain the YOLOX-s ONNX model linked below.

```sh
python -m football_normalizer.video_ui --model /path/to/object_detection_yolox_2022nov.onnx
```

On a Mac, install the optional acceleration dependency with
`python -m pip install -e '.[tracking,acceleration]'`. The browser workbench
uses Core ML through ONNX Runtime when available and falls back to OpenCV
if a model cannot run through Core ML. It reuses loaded models for later
runs in the same local server. Pass `--inference-backend opencv` to compare
against the CPU pipeline. Tracking reports record the backend for each model.

To enable automatic pitch replay, download the 640 px FP32
[`football-pitch-detection.onnx` landmark export](https://huggingface.co/SzymonKulpinski/football-pitch-detection-onnx)
and put it beside the person detector, or pass `--pitch-model /path/to/football-pitch-detection.onnx`.
The model card lists the checkpoint under AGPL-3.0. The model runs locally
and is loaded only when a sampled frame needs an automatic estimate.

For semantic field localization, use the published checkpoint and
`resources/mean.npy` and `resources/std.npy` from
[SoccerNet calibration](https://github.com/SoccerNet/sn-calibration).
Export in an environment with PyTorch, Torchvision and ONNX installed.

```sh
python -m football_normalizer.video_export_pitch_lines soccer_pitch_segmentation.pth \
  --mean mean.npy --std std.npy --output football-pitch-lines.onnx
```

Put the export beside the person detector or pass
`--pitch-lines-model /path/to/football-pitch-lines.onnx`. It takes precedence
over the landmark model for automatic estimates. The model identifies
semantic pitch markings. The fitter uses straight lines and checks additional
markings before accepting a ground-plane mapping. When that fit is unavailable,
the landmark model can supply a mapping if the detected semantic markings
independently support it. Across adjacent frames,
camera motion can carry that mapping for up to two samples while current
painted lines still support it. Cuts, long gaps and failed checks force a new
estimate. Reviewed manual calibration takes precedence.
Export reviewed identity maps from a run using the same pitch model and
sampling rate as the run that applies them.

To compare with the landmark model on SoccerNet's labeled images, use the
benchmark below. It reports ground-marking point errors and availability,
including the paired subset where both methods returned a mapping.

```sh
python -m football_normalizer.video_calibration_benchmark /path/to/labeled-images \
  --landmark-model football-pitch-detection.onnx --line-model football-pitch-lines.onnx \
  --output reports/video-pilot/calibration-validation.json
```

For dedicated ball tracking, export the
[`ball11n.pt` football checkpoint](https://huggingface.co/boxerab/gst-python-ml-football)
as a 1280 px single-class ONNX detector. The checkpoint is listed under
AGPL-3.0. In a separate export environment with Ultralytics and ONNX:

```sh
yolo export model=ball11n.pt format=onnx imgsz=1280 opset=17 simplify=False nms=False
```

Pass `--ball-model /path/to/ball11n.onnx` to the local app, or name the export
`football-ball-detection.onnx` beside the person model. The ball detector runs
locally through the same inference backend. If the full-frame detector misses
a recently tracked ball, it checks a crop around the predicted position.
Crop detections fill individual gaps while the full-frame detections steer
the trajectory.
The track resets after camera cuts and leaves gaps where the ball cannot be
located confidently. Turn on the footage overlay to see its
yellow ring, or use **Jump to first ball detection**. On the pitch replay,
the marker is the ball's ground-plane projection. An airborne ball can
appear away from its true position.

Keep that process open and visit [Video Analysis on localhost](http://127.0.0.1:8765/video/).
Choose **Match video**, then choose a tracking mode:

- **Track full video** starts at the beginning and follows the decoder to the
  end. It targets 10 frames per second and keeps the same tracker
  throughout, resetting at suspected camera cuts. Follow the progress on the
  page and use **Stop tracking** to end a run.
  The app accepts up to 4 GiB and up to six hours of footage, subject to local
  temporary storage.
- **Track the next 8 seconds** samples a short window from the playhead.
  This mode accepts videos up to 256 MB.

Both modes and the tracking commands default to 10 samples per second,
limited by the frames available in the source. Pass `--sample-fps 5` when
starting the local app or running the tracking commands to use the previous
rate. Reviewed identity maps must be reapplied at the rate used to export them.

Both modes draw the pitch outline and on-pitch detections over the footage and
can save a tracking report. **Movement Evidence** shows how many sampled
frames from the current video contain people, the ball and usable player pitch
positions. Jump from each card to its first matching frame. Open
`video-metrics.json` from the aligned fixture workflow to review per-player
behaviour measures. With the
landmark model available, the app estimates positions using the visible pitch
markings. It checks the spread
and agreement of the landmarks and leaves the pitch empty where the view
does not support a mapping. **Jump to mapped frame** opens a usable frame.
Estimates use reference pitch dimensions by default. For reviewed positions,
choose a [calibration JSON for that exact recording](#calibration-format) in
**Add camera calibration** before tracking. A verified manual mapping takes
precedence. Use full video tracking for a moving manually calibrated shot so
the tracker can read its reference frame. The local app removes temporary
video files when the run ends. For individual movement rates, follow the
aligned fixture workflow below.

Add `--demo-video /path/to/sample.mp4 --demo-tracking /path/to/video-tracking.json
--demo-plan /path/to/clip-plan.json` to enable **Load real match demo**.
The demo loads the matching footage, tracking and clip markers together.
Add `--demo-calibration /path/to/calibration.json` to reuse the camera
calibration when tracking that exact demo file again.

### Pitch, shirts and goalkeeper hints

- A person's foot point must be within the pitch before the detection can
  enter a track. Verified camera calibration supplies the boundary when
  available. Otherwise, each frame uses the visible turf and markings to
  estimate the boundary. When neither is available, tracking waits for a
  usable view. The estimated outline can miss part of the field or include
  runoff when a touchline is hard to see.
- A second look at overlapping, enlarged parts of the pitch helps detect
  smaller players in wide shots. Color agreement can help preserve a track.
  Occlusion, camera cuts and very small players can still produce gaps.
- Several clear torso crops describe each player's shirt, including dark and
  mixed colours. Two distinct groups that repeatedly appear together establish
  **Team A** and **Team B**. Those kit profiles apply throughout the recording,
  including close views before and after the wide shot. Shirt grouping uses
  people detected on the pitch, with camera projection checked separately.
  Unclear tracks and officials can remain
  unassigned. The groups are separate from the reviewed lineup.
- **GK?** marks a probable goalkeeper when a distinct shirt and a continuous
  run of verified positions near one goal support the hint. An ordinary
  outline does not establish where the goals are. Upload previews without
  camera calibration can show kit groups and boxes. Add a calibration file
  before running tracking to enable goalkeeper hints and pitch positions.
- Ball observations carry a track ID and detection confidence. When a
  dedicated model is used, the report records ball availability per frame.
  Player identity maps from before dedicated ball tracking must be exported
  again for runs with that mode enabled.

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

## Compare tracking backends

The tracker estimates camera motion from background features outside the
detected people. When those features agree, it moves the previous track
positions with the camera before associating them with the next detections.
The report marks the frames where it used compensation. Reviewed identity
maps from earlier tracker versions need a fresh track export.

For research evaluation, use extracted MOT sequences from
[SoccerNet Tracking](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023).
Use a separate environment for the evaluation packages.

```sh
python -m pip install 'opencv-python>=4.11,<5' 'trackeval==1.3.0' 'supervision==0.25.1'
python -m football_normalizer.video_benchmark /path/to/SNMOT-116 /path/to/SNMOT-134 /path/to/SNMOT-187 \
  --model /path/to/object_detection_yolox_2022nov.onnx --output benchmark-yolox
python -m football_normalizer.video_benchmark /path/to/SNMOT-116 /path/to/SNMOT-134 /path/to/SNMOT-187 \
  --detector football --model /path/to/football.onnx --output benchmark-football
```

The second command uses the 640 px FP32
[Collabora football detector export](https://huggingface.co/boxerab/gst-python-ml-football).
It also supports `--coreml` with ONNX Runtime and the CoreML provider. Both
detectors use a confidence floor of 0.23. The default stride scores every
fifth source frame. The evaluation applies the same estimated image-space
pitch mask to both the annotations and detections, and excludes ball boxes.
It exports sampled MOT text files and reports TrackEval HOTA, DetA, AssA,
and ID switches. These are scoped comparisons against FBRPRO's processing
settings and differ from the official challenge protocol.

## Testing ball recognition around match actions

The [SoccerNet spotting kit](https://github.com/SoccerNet/sn-spotting) marks the
time and type of an action. Use those times to select review windows and
[SoccerNet Tracking](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023)
boxes to measure localization. The benchmark compares the same detector and
frames with crop recovery enabled and disabled. It reports matches within
12 source-image pixels, unmatched predictions, and the cost of extra crops.
It reports results across the clip and within two seconds on either side of
each action.

```sh
python -m football_normalizer.video_ball_benchmark /path/to/SNMOT-135 \
  --model /path/to/object_detection_yolox_2022nov.onnx \
  --ball-model /path/to/football-ball-detection.onnx \
  --output reports/tracking-benchmark/ball-spotting-validation.json
```

The benchmark defaults to the action timestamp in each tracking sequence.
To use SoccerNet's `Labels-ball.json` for a single aligned clip, pass
`--spotting-labels /path/to/Labels-ball.json --half 1 --clip-start-ms 816000`
with the actual half and clip start on that broadcast's video timeline.
