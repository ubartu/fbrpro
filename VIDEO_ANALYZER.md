# Video analyzer: clip review milestone

Open [video/index.html](video/index.html) in a browser. This is the first
implementation slice of the video architecture: prepare aligned footage,
select clips automatically from the event feed, and export human observations.
The event-driven moment catcher needs no video model and does not scan the
full video. Automatic player tracking, camera calibration, pitch replay and
automatic movement measurements are later milestones.

## Review a fixture

1. Open an authorized local match video. In **Align video to the match clock**,
   enter the fixture ID and a video ID. Pick two visible anchor events in each
   uninterrupted segment and enter their period-relative match seconds. Use
   the playhead buttons to capture the video times. Export the alignment.
   Each period clock resets at kickoff. Use separate segments for cuts,
   missing footage or edits. Verify an independent anchor before trusting sync.
2. Generate the clip plan from the normalized bundle:

   ```sh
   python3 -m football_normalizer.video /path/to/normalized-match \
     --match MATCH_ID --alignment /path/to/video-alignment.json \
     --team home --build-up-limit 10 --output clip-plan.json
   ```

3. Load the plan into the review page and choose a clip. Use **Play selected
   clip**, choose the opportunity and player, then tag the behaviour as
   observed, visible but not observed, or unobservable. Export observations
   before closing the page. Files are processed locally.

## Selection and evidence

- Candidate regains are changes to the selected team's possession in the
  same half, during recorded open play, with no more than four seconds since
  the previous tagged possession's last event. The candidate includes up to
  three seconds of lead-in and up to twelve seconds after the regain.
- For screening review, the planner samples longer opposition possessions
  across the aligned timeline. It does not require a tackle or interception
  and does not claim the selected player screened. Sampling is deterministic.
- The planner only seeks clips within a single aligned segment; it records
  unaligned and deliberately unsampled opportunities separately. Overlapping
  clips are combined for playback while preserving each original opportunity.
- A plan estimates <em>video selected</em>, not usable visibility. Tags are
  analyst evidence for selected clips, not a complete match denominator.
  They are not yet included in the role scorer.
- The versioned alignment and observation JSON retain match, video, period
  and clip identifiers. Keep the correct footage with its exported alignment.
  Do not reuse video IDs for different edits or camera angles.

Next: verify with one real tactical clip and independent clock anchors,
then add player detections, track association, pitch calibration and identity
review. Use the exported observations as validation examples before calculating
screening time or transition runs automatically.
