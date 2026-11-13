# Automatic video events

After tracking, choose **Calculate statistics**, then **Actions to review → Automatic pass analysis**. **Analyze actions** finds possible ball releases from the saved tracks. To classify spotted actions and estimate player pass attempts, choose a SoccerNet `results_spotting.json` for the same broadcast and run the analysis again.

Choose the half in the prediction file and enter the same moment on both video timelines. If the open video starts 30 seconds into that half, use `30` for the SoccerNet video and `0` for the open video. The importer reads millisecond `position` values and uses `gameTime` only to identify or verify the half. Match the broadcast and timing before using the results. One analysis aligns one prediction half.

The timeline shows the SoccerNet score and the evidence for attributing a release. **Review in footage** jumps to the event. Use **Export event timeline** to retain the alignment, event evidence, unresolved actors and attributed pass counts. Reviewed events remain in the existing review context.

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
- Tracking alone proposes releases for review. Attributed predicted passes provide a separate estimated count by track. Linked receptions and automatic pass outcomes are the next milestone.

```sh
python -m football_normalizer.video_events tracking.json --output video-events.json
python -m football_normalizer.video_events tracking.json --predictions results_spotting.json \
  --half 1 --source-ms 30000 --video-ms 0 --output video-events.json
```

See [SoccerNet Ball Action Spotting](https://www.soccer-net.org/tasks/ball-action-spotting) and the [Recokick prediction writer](https://github.com/recokick/ball-action-spotting/blob/master/src/ball_action/annotations.py) for the input format and spotting task.
