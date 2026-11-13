# Video statistics

Open the local video workbench, load a video tracking report or finish a tracking
run, then use **Video statistics → Calculate statistics**. This is a separate
analysis request: it does not rerun detection, replace tracks, move the playhead,
change team labels or add work to replay frames. Existing stable tracking reports
are supported; rerunning a video just to obtain statistics is unnecessary.

**Players** starts with a single frame containing simultaneously visible,
team-assigned players. Filter by team or choose **Team unconfirmed** to correct
an assignment. **Use current frame** follows the playhead, and **All track
fragments** reveals the complete list. Saved team corrections apply when you
calculate statistics again.

In **Actions to review**, pause on an action and choose the player at the
playhead. Select the action and its outcome where available, then save it.
Calculate statistics again to see the reviewed observations on that player's
profile. **Export review context** preserves tags and team corrections for
another session.

## Available measurements

| Category | Implemented outputs | Evidence needed |
| --- | --- | --- |
| Positioning | Average position, 6×4 heatmap, 3×3 zone occupancy, position width/depth, zone transitions | Contiguous usable pitch positions; unknown boundary time is separate |
| Movement | Distance and rate as a share of the pitch, sideways displacement, forward/backward displacement | Continuous mapped positions with movement above mapping uncertainty. Forward and backward also need consistent orientation and a confirmed attacking direction |
| Movement in metres | Observed distance, forward/backward/lateral displacement, run direction time | Reviewed live play, orientation, scale/calibration and low-error continuous windows. Team direction for directional outputs |
| Physical estimates | Mean/window-maximum speed, high-speed distance/time, sprint bouts/distance, window acceleration/deceleration maxima and bouts | The same physical quality gates; at least one second/five samples per window |
| Relationships | Distance to ground ball, nearest teammate/opponent, own/opposition defensive line | Coincident usable observations; ground-ball review for ball distance; reviewed line membership for line distances |
| Observed team shape | Centroid, width, length, mean radius/compactness, convex-hull area, observed player count, line spacing | At least three simultaneously visible teammates; reviewed lines for line spacing |
| Off-ball evidence | Support runs, overlaps, recovery runs, passing lanes, pressure approaches, central screening, deep-to-advanced support, deep outlets, between-line availability/receptions | Reviewed phases, possession team/carrier/lines as applicable, live play and direction; reception arrival events for receptions |
| Reviewed actions | Pass attempts/completions/rate, receptions, touches, tackles/wins, interceptions, shots, press attempts/success | A reviewed action at an observed track sample provides a partial count. Complete totals require confirmed identity, full event-type coverage and outcomes where applicable |
| Possession | Team possession time/share and review coverage | Reviewed possession intervals; proximity does not establish possession |

Off-ball movements and ball-transfer proximity patterns are **candidates for
review**, not recognized intent, confirmed passes or successful actions. There is
no newly trained video event-recognition model. Counting reviewed actions and
spotting geometric candidates are distinct operations. Individual possession is
unavailable without evidence of actual individual ball control.

## Review and interpretation

Position maps and observed team geometry are provisional by default. Physical
values require live play, verified actual pitch dimensions, consistent pitch
orientation and reviewed pitch alignment. Enter the stadium dimensions in the
scope controls when available. Use the scope controls to exclude
replays and close-ups. Confirm ground-ball positions only where applicable.
Set attacking directions separately by team and period. For video-only full
recordings, enter the actual second-half start in video seconds; the selector
for a half's direction does not itself change the period assigned to a frame.
Extra-time boundaries can be supplied in `period_intervals`.

An unchecked live-play box means unreviewed, allowing provisional position maps.
An explicitly imported `live_play: false` means non-live and excludes those
observations. The UI preserves that explicit assertion. Advanced context files
can review only selected live intervals.

Every player profile reports usable time, contiguous observed track time and
selected-scope coverage. These denominators are different. Off-camera time,
missing calibration, cuts, gaps, identity changes, duplicate timestamps and
implausible calibration/player jumps are never filled as zero movement. Anonymous
track IDs remain separate; they are not shirt numbers, roster identities or an
automatic full-match re-identification system.

Heatmap and zone percentages divide resolved cell time by usable observed time.
Boundary-uncertain/crossing intervals are separate, so cell totals can be below
100%. Team geometry describes the visible subset, not an unseen complete team.
Defensive and midfield lines are not guessed from anonymous positional clusters.

Observed distance is the sum of robust, approximately one-second displacement
windows; short unresolved tails are not extrapolated. Speed and acceleration
maxima are window estimates, not instantaneous GPS peaks. Movement below the
error proxy is unresolved rather than proof of standing still. The default
high-speed/sprint definitions are 5.5/7 m/s, minimum sprint duration 1,000 ms,
and acceleration/deceleration bout magnitude 2.5 m/s²; context can override them.
These are measurement definitions, not player-role thresholds.

When stored point sensitivity is absent, analysis uses frame calibration
fit/check residual plus a 0.25 m proxy. The analysis adds disagreement between
the fresh and stabilized mappings at each observed position, and breaks motion
intervals when the mapping is re-established. This does not establish real-world
positional accuracy or a confidence interval. Automatic calibration can contain
systematic error; a checkbox records analyst review, not independent validation.

Event totals remain unavailable without complete review of the selected scope.
Partial confirmed observations are retained separately from totals. A reviewed
pass's release time is not reused as a reception time; between-line receptions
need explicit arrival annotations and reception coverage.

## Advanced context example

All timestamps below are **video milliseconds**. Replace the example IDs with
actual track and roster IDs, and supply only assertions verified from footage.
The example is a format illustration, not annotations for the included demo.

```json
{
  "match_id": "MATCH_ID",
  "video_id": "VIDEO_ID",
  "plan_id": "PLAN_ID",
  "start_ms": 0,
  "end_ms": 3000,
  "live_play": true,
  "calibration_reviewed": true,
  "orientation_confirmed": true,
  "pitch_dimensions_verified": true,
  "ground_ball_reviewed": true,
  "directions": {"1": {"kit:team_a": 1, "kit:team_b": -1}},
  "identities": {
    "track-1": {"player_id": "roster-player-1", "team_id": "kit:team_a"},
    "track-2": {"player_id": "roster-player-2", "team_id": "kit:team_a"}
  },
  "defensive_lines": {"kit:team_b": ["defender-1", "defender-2"]},
  "team_lines": {"kit:team_b": [["defender-1", "defender-2"], ["midfielder-1", "midfielder-2"]]},
  "episodes": [{
    "start_ms": 0, "end_ms": 3000, "period": 1,
    "phase": "build_up", "possession_team_id": "kit:team_a",
    "opponent_team_id": "kit:team_b", "ball_carrier_id": "track-1",
    "opponent_ids": ["defender-1", "defender-2", "midfielder-1", "midfielder-2"],
    "opponent_midfielder_ids": ["midfielder-1", "midfielder-2"],
    "opponent_defender_ids": ["defender-1", "defender-2"],
    "lane_context_complete": true, "ground_ball_reviewed": true
  }],
  "events": [
    {"id": "pass-1", "type": "pass", "t_ms": 1000, "period": 1, "track_id": "track-1", "recipient_track_id": "track-2", "outcome": "complete", "reviewed": true},
    {"id": "reception-1", "type": "reception", "t_ms": 1400, "period": 1, "track_id": "track-2", "reviewed": true}
  ],
  "event_coverage": [{"start_ms": 0, "end_ms": 3000, "period": 1, "types": ["pass", "reception"], "complete": true}],
  "possession_intervals": [{"start_ms": 0, "end_ms": 3000, "period": 1, "team_id": "kit:team_a", "reviewed": true}]
}
```

Supported phase names: `attack`, `build_up`, `transition`, `settled_defence`,
`recovery`. Other optional fields: `live_intervals`, `ground_ball_intervals`,
`period_intervals`, `pitch_length_m`, `pitch_width_m`, `flip_pitch`, and `config`.
Intervals use `start_ms`, `end_ms`, `period`. Ground-ball/possession intervals
need `reviewed: true`; live intervals use `live_play`. Global pitch flipping is
an analysis transform only and never rewrites source tracking coordinates.

## Architecture and commands

```text
Stable video tracking → saved tracking JSON (unchanged)
                                  ↓ explicit Calculate statistics
                  context validation + scope/quality preparation
                         ↙                         ↘
           positions/physical/relationships    off-ball/reviewed events
                         ↘                         ↙
                       fbrpro-video-stats-v1 JSON
                         ↓
                player maps, team view, evidence + export
```

`video_stats_common.py` validates scopes, identities, periods and calibration
compatibility, excludes ambiguous samples and detects cohort coordinate jumps.
`video_stats_physical.py` measures interval/window geometry.
`video_stats_behaviors.py` evaluates contextual opportunities and reviewed events.
`video_statistics.py` combines results and fingerprints input data and context.
`video/statistics.js` renders the separate dashboard. No statistics module is
called by `video_tracking.run` or by the per-frame replay renderer.

```sh
python -m football_normalizer.video_statistics tracking.json --output video-statistics.json
python -m football_normalizer.video_statistics tracking.json --context context.json --output video-statistics-reviewed.json
python -m unittest discover -s tests
node tests/video_statistics_ui.cjs
```

Local API: `POST /api/statistics`, header `X-FBRPRO-Video: 1`, JSON body
`{"tracking": {...}, "context": {...}}`. Requests are rejected while a tracking
job is active, and computation never invokes model inference. Imported reports
are read separately from the active tracking report; mismatched provenance
disables evidence seeking. Statistics and context can both be exported.

Real-video smoke reports and validation results are saved in
`reports/video-statistics`. Those checks establish computation, coverage and
source immutability, not independent positional, physical or event accuracy.
