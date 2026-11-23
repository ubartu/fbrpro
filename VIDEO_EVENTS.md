# Automatic video events

## Test pass detection on your footage

Open your match clip and choose **Analyze match actions**. After inference, the workbench automatically compares the results with matching local SoccerNet labels. It matches the exact video by SHA-256, so renaming the clip works. For other videos, choose **Open ground-truth labels** and select a JSON file for that exact edit.

```json
{"start_ms": 0, "end_ms": 60000, "events": [{"type": "pass", "t_ms": 12500}]}
```

Set the interval to the part of the video exhaustively labeled for pass attempts. Times are relative to the uploaded video, and the end of the interval is exclusive. Each pass label marks the release, regardless of outcome. An optional `source_sha256` ties the labels to the exact source file.

The test pairs events one-to-one within the timing tolerance, initially 480 ms. It reports timing matches, unmatched labels, unmatched detections, and scores against the imported labels inside the labeled interval. Select a timestamp to jump to it in the video. Save the automatic results or the comparison as JSON. Optional pass fields `passer_player_id`, `receiver_player_id` and `outcome` add attribution and outcome scores. IDs must use the same roster IDs in both files. Outcomes use `complete`, `incomplete` or `unknown`. Accuracy is calculated on resolved, labeled matches, with coverage reported alongside it.

## Review automatic events

After tracking, choose **Calculate statistics**, then **Actions to review → Automatic match actions**. **Analyze actions** finds possible ball releases from the saved tracks. To classify spotted actions and estimate player pass attempts, choose a SoccerNet `results_spotting.json` for the same broadcast and run the analysis again.

Choose the half in the prediction file and enter the same moment on both video timelines. If the open video starts 30 seconds into that half, use `30` for the SoccerNet video and `0` for the open video. The importer reads millisecond `position` values and uses `gameTime` only to identify or verify the half. Match the broadcast and timing before using the results. One analysis aligns one prediction half.

The timeline shows the SoccerNet score and the evidence for attributing a release. **Review in footage** jumps to the event. Use **Export event timeline** to retain the alignment, event evidence, unresolved actors and attributed pass counts. Reviewed events remain in the existing review context.

## Match action timeline and possession

**Analyze match actions** tracks the whole upload, then opens the clickable action timeline. Filter by action or switch to the possession ledger. **Export action report** saves the automatic events, the reviewed actions and the possession ledger. In the API report, `events` carries passes, linked receptions and spotting predictions, `ball_actions` carries the additional tracking estimates, and `reviewed_events` carries independent observations supplied in the context.

- Touches use observed ball movement relative to a player's feet or head. A continuous control episode is counted once. A brief redirection tied to a pass release can indicate a one-touch play. Other isolated redirections remain possible deflections or headers.
- Shots require a fast release and a calibrated trajectory toward a goal mouth. A missing ball or uncertain projection can leave a shot undetected.
- A tackle estimate needs a close challenge on the previous carrier and observed opposition control. An interception needs an observed pass in transit before opposition control. Each estimated defensive action records the next known track and team.
- A press estimate links a calibrated closing run to an observed opposing carrier. The report follows possession for five seconds after the press ends. Set `press_outcome_window_ms` in the analysis context to use a different window. A pressing-team regain within the window is marked successful and links to the regain time and player. A cut or lost evidence yields an unknown outcome.
- Possession separates player control, transit, contested play, dead ball and unobservable time. Dead-ball intervals come from `dead_ball_intervals` in context or aligned spotting evidence for both an out event and a restart. Missing footage, video cuts and ambiguous observations remain unobservable. Team shares use known team-control time. Player control can still be credited to an anonymous track.

These are tracking-based estimates. Player names require roster links. Automatic event counts and possession times are separate from analyst-reviewed totals.

## Event definitions

All times are video milliseconds. Record an approximate time or interval when the exact frame is unclear. Attribute actions to the visible player first and link the player to a roster identity where evidence supports it. Record ambiguous actors and outcomes as unknown.

| Event | Annotation rule |
| --- | --- |
| Pass attempt | Deliberate release directed to a teammate or into space for a teammate. Anchor it at release. Crosses and high passes count as pass types. A teammate's controlled reception completes it. A clear turnover or ball out before that makes it incomplete. Unseen arrivals stay unknown. |
| Reception | First deliberate control after a pass, including a controlled one-touch action. Anchor it at arrival and link it to the pass when identifiable. |
| Touch | A distinct intentional ball contact. A contact can also be a pass, shot, or reception. Count the same physical contact once in the touch total. Record incidental deflections separately. |
| Shot | An intentional attempt to score, anchored at ball contact. Record the outcome where visible. |
| Tackle | A direct challenge for the ball against an opponent in control. Record the challenger and whether their team secured control, play stopped, or the outcome remained unresolved. |
| Interception | A defender deliberately cuts out an opponent's ball in transit. Record the contact and whether the defending team gained control. |
| Press and outcome | A player actively closes the ball carrier to limit their options. Record the interval and who applied pressure. The initial outcome convention is a controlled team regain within five seconds of the press ending during the same uninterrupted live sequence. This measures a regain after pressure. |
| Possession | Mark intervals of player control, ball in transit, contested play, dead ball and unobservable play. Use explicit team control intervals for team possession and player control intervals for individual time. |

The [reviewed examples](examples/video-events/reviewed-clips.json) identify real pass releases, receptions and an incomplete attempt on two saved clips. They include a real player overlap, an external shot annotation and a constructed camera-cut case. Their labels cover the listed examples. Match-level scoring needs densely labeled, held-out footage.

## Current implementation

- An imported pass, high pass or cross enters the predicted timeline. Other imported labels are visible for review.
- Passer attribution uses sustained separation or a transfer between teammates. The transfer check compares the ball near each player's feet, including short passes with little separation from the passer. It can use a brief observed change of direction for one-touch contacts, or recover over one brief ball observation gap in the same shot. The analysis checks gaps, cuts, clips and competing tracks. It uses image-space distances relative to player height.
- Tracking alone proposes releases for review. Attributed predicted passes provide a separate estimated count by track. A completed pass links to an estimated reception. The first observed control by a teammate or opponent gives an estimated completed or incomplete pass respectively. A break in ball evidence, unresolved team, or unobserved arrival leaves the outcome unknown. Completion rate uses resolved outcomes and is reported with outcome coverage. The analyzer does not infer ball-out outcomes from disappearance.

```sh
python -m football_normalizer.video_events tracking.json --output video-events.json
python -m football_normalizer.video_events tracking.json --predictions results_spotting.json \
  --half 1 --source-ms 30000 --video-ms 0 --output video-events.json
```

See [SoccerNet Ball Action Spotting](https://www.soccer-net.org/tasks/ball-action-spotting) and the [Recokick prediction writer](https://github.com/recokick/ball-action-spotting/blob/master/src/ball_action/annotations.py) for the input format and spotting task.

## Batch validation

Run `python -m football_normalizer.video_pass_benchmark manifest.json --output benchmark.json` on cached tracking, with labels supplied separately. A manifest contains a `matches` array. Each entry specifies `match_id`, `tracking` and `labels` paths relative to the manifest. Optional fields are:

- `context` for reviewed tracking context and roster links.
- `predictions` and `alignment` for SoccerNet spotting-only and combined scores. The alignment object uses `half`, `source_ms` and `video_ms`.
- `footpass` with `predictions`, `game` (for example `game_18_H1`), `source_start_ms` and optional `score_min` (default 0.15). The importer evaluates published TAAD pass predictions and uses `team:jersey` for player IDs.

Reports show the matches each method covered. Compare methods on the same intervals. Timing matches are one-to-one, and receiver or outcome accuracy is available when the reference labels include those fields. To measure identity continuity, include `identity_cases` in the labels with `kind` (`overlap`, `reentry` or `camera_cut`), `same_player`, and `before` and `after` objects containing exact tracking sample `t_ms` and reviewed `bbox_xyxy`. Track-ID and roster-ID results are scored separately.

The workbench can track and find pass candidates in one run. For selected-clip jobs, add `--pass-events` to `python -m football_normalizer.video_run` to write the event report alongside cached tracking. Reviewed roster links flow into event analysis without rerunning detection when using the workbench's statistics context.


## Train the FBRPRO pass detector

Provide a manifest with a `matches` array containing `tracking` and `labels` paths for each labeled interval. Each label file must include the SHA-256 of the exact video. Use fully labeled intervals so unlisted moments can supply negative training examples. The classifier learns from short windows of observed ball movement relative to the closest player's feet. Windows with missing observations or a cut are skipped. Use separate matches to measure performance on unseen footage.

```sh
python -m football_normalizer.video_pass_model training-manifest.json --output pass-model.json
```

Add `--pass-model pass-model.json` when launching `football_normalizer.video_ui`. Upload a video and choose **Analyze match actions**. The trained detector runs before any labels are opened. For batch evaluation, add `pass_model` to the evaluation manifest entry. Evaluation rejects intervals that overlap training on the same source.


## Add video to midfield role reports

Create a normal `midfield-v0.1` report from accepted event bundles. Then supply a manifest with a `videos` list, one entry per selected match. Paths are relative to the manifest.

```json
{"videos":[{"tracking":"match-a-tracking.json","context":"match-a-context.json"},{"tracking":"match-b-tracking.json","context":"match-b-context.json"}]}
```

```sh
python -m football_normalizer.video_midfield midfield.json --manifest videos.json --output midfield-with-video.json
```

Open the output with **Open JSON report** in the [Midfield Lab](http://127.0.0.1:8765/dashboard/). The join checks canonical match, player and team IDs, verifies the video-to-match clock, and selects only the provider's midfield position intervals. Context supplies confirmed attacking directions. If those are missing the report can still show the provider signals.

The combined comparison ranks players within the same match and position group. Each role uses its own set of provider event rates and mapped video measures. For example, the deep-distribution pattern combines its event rates with time in the own-third central zone. End-third occupancy balance and relative movement support the box-to-box comparison. Match cells report video seconds and event opportunities per signal. Across-match cells show how often the pattern was scorable or strongest. Estimated events from the video timeline stay outside these validated event rates. Role comparisons describe observed patterns. A holding-midfielder pattern in this report uses positioning and event evidence, with direct screening evidence available separately in the movement report.
