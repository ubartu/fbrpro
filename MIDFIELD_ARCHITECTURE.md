# fbrpro midfield pilot: architecture and operation

**Current implementation:** see [match role percentages](#current-scoring-match-role-percentages). Earlier candidate and match rules below are historical pilot designs.

The pilot profiles defensive, central and attacking midfielders in selected StatsBomb matches. It separates observations, role suggestions and analyst judgment so that incomplete data does not turn into a confident player label.

## Data flow

| Layer | Input | Responsibility and output |
|---|---|---|
| 1. Intake | StatsBomb match metadata, lineups, events, optional 360 | The existing normalizer emits a validated bundle with canonical IDs and source provenance. |
| 2. Match and position selection | Accepted bundles and optional match IDs | Select the requested matches. Split each player's provider position intervals by half and group them into defensive, central or attacking midfield. Use half-end events to preserve stoppage time. |
| 3. Feature extraction | Match events inside the player's intervals | Calculate counts, opportunity denominators, coordinate/link coverage and supporting event IDs. Calculate execution, such as progressive-pass completion, separately. |
| 4. Profile aggregation | Per-match feature records | Group by player, team, competition-season and midfield group. Add numerators and denominators across matches before computing pooled rates. Keep the per-match evidence. |
| 5. Candidate rules | Pooled features and optional independent reference report | Compare with comparable midfield peers. Three roles can receive provisional candidate labels from events. Two need review of movement. |
| 6. Report | Profiles, metrics and candidate statuses | Write an auditable JSON report for analyst review or a later interface. |

The normalizer and pilot run independently. The pilot reads JSONL from accepted bundles; it does not require a database, web server or third-party dependency. See [the label dictionary](LABEL_DICTIONARY.md) for role definitions and [the measurement guide](ROLE_METRICS.md) for the roles deferred beyond midfield.

## Run the pilot

From the project folder:

```sh
python3 -m football_normalizer.midfield /path/to/normalized-bundle \
  --match 3895302 --output midfield-report.json
```

Pass multiple bundle paths to pool matches. Repeat `--match` for multiple selections or omit it to use all matches in those bundles. Selectors can be canonical IDs or source match IDs. The command refuses an existing output file.

For a peer comparison, first generate a report from a separate reference set, then pass `--baseline reference-report.json` when generating the target report. The reference matches must be disjoint. Comparison is within the same competition, season and midfield group; the target player is excluded, and a peer can contribute once. A profile needs five matches and 450 estimated position minutes. Each feature needs 30 eligible opportunities and 90% coverage and at least 30 eligible peers to compute its midrank. A candidate needs each required feature at or above the 75th percentile, with at least three recorded instances across two or more matches per feature. These are pilot settings to calibrate, not validated probabilities or a ranking of ability.

## Role rules

| Role | Evidence used by the pilot | Interpretation |
|---|---|---|
| Deep-lying playmaker | Share of team deep central passing; progressive attempts from deep central starts. | Event-based candidate after sample and peer checks. |
| Advanced playmaker | Linked shot assists per eligible advanced central pass; share of team advanced central passing. | Event-based candidate after sample and peer checks. |
| Ball-winning midfielder | Pressures and middle-third tackle/interception activity per 100 opponent possessions. | Event-based candidate after sample, peer and defensive-feed checks. |
| Holding midfielder | Deep interventions and deep passing as supporting evidence. | Unassessed without analyst video or tracking of defensive positioning and screening. |
| Box-to-box midfielder | Deep interventions, progressive carries and attacking-box shots/carries as supporting evidence. | Unassessed without analyst video or tracking of movement across phases. |

Pressures and interventions are counts within recorded opponent possessions, not verified regains or causal effects. Use `--confirm-defensive-coverage` on both target and reference runs only after checking the feed's logging of pressure, tackles and interceptions. Positional coverage in the report describes eligible feature fields, not overall match coverage. Open play is conservatively restricted to the provider's Regular Play and From Counter patterns. This excludes some legitimate play after restarts, so comparisons must use the same rule. Position minutes are estimates based on provider clocks rounded to whole seconds. The pilot handles regulation matches; it does not infer tactics from ambiguous interval data or use 360 snapshots as continuous trajectories.

## Verification and expansion gate

One real Bundesliga match (StatsBomb ID 3895302) produced 10 midfield profiles. All role labels remained unassessed because the sample is too small or the role requires evidence not available to the pilot. This verifies ingestion and feature calculation on a real bundle, not classification accuracy.

Before expanding beyond midfield, collect a comparable multi-match reference sample. Have analysts independently review examples across all five roles, including players whose labels overlap and cases near the pilot thresholds. Compare candidate precision, missed roles, analyst agreement, coverage and label stability on matches held out from calibration. Choose acceptable thresholds with the analysts before evaluating the holdout. If the sample or analyst agreement is insufficient, leave the roles unassessed and improve data or definitions before extending the system.

## Explainable assessment

The dashboard at `dashboard/index.html` opens the included sample or an uploaded
`midfield-v0.1` report locally. Compare two profiles and open **See assessment
checks** to inspect the sample, measured signals, and unmet conditions. Match
selection shows observations without assigning a persistent player role.

`football_normalizer.role_assessment` owns the rules. Each role compares all
required signals against the same eligible peer cohort. A peer must satisfy
sample requirements and provide usable values for every measure in that role.
If defensive coverage is unconfirmed in either the target or baseline, the
ball-winning role cannot be assessed. Holding and box-to-box roles retain
event proxies and require tracking or analyst video. The JSON adds diagnostics
to the existing versioned report shape; older reports still open in the UI.

Next: build a disjoint reference cohort, collect independent analyst labels,
then evaluate precision, missed labels, coverage and stability on held-out
matches before changing thresholds or expanding to other positions.

## Match-level role rules

The match role timeline is computed per player, match and midfield position group.
It uses `midfield-match-v1` rules, independently of club and league comparisons.
For each event-based role, require at least 30 estimated minutes and at least
90% coverage for both required measures. The table gives provisional minima
for each measure as **opportunities / actions / rate per 100**.

| Role | Required measure 1 | Required measure 2 |
| --- | --- | --- |
| Deep-lying playmaker | Team deep central pass share: 12 / 6 / 15 | Progressive attempts from deep: 6 / 2 / 15 |
| Advanced playmaker | Team advanced central pass share: 10 / 4 / 15 | Advanced central passes linked to shots: 6 / 2 / 8 |
| Ball-winning midfielder | Pressures per opponent possession: 12 / 5 / 20 | Middle-third interventions per opponent possession: 12 / 2 / 8 |

Both measures must pass. If minutes, field coverage, opportunities, or required
feed coverage are missing, the role is unassessed. Otherwise it is supported
or below the match rule. A match can support several roles, and a role falling
below the rule does not establish that the player did not perform it. The
defensive rule requires confirmed event-feed coverage. Holding and box-to-box
require video or tracking. These cutoffs are engineering hypotheses to validate
against independently reviewed match clips, not learned role probabilities.

## Current scoring: match role percentages

This supersedes the match pass/fail rules above. For a feature with observed
rate r and an illustrative reference rate a, its fit is **100 × r / (r + a)**.
The role fit is the geometric mean of its feature fits: both behaviours
contribute, with no pass/fail threshold or winner-takes-all label. An anchor
sets the rate that maps to 50%; it is not a requirement. Scores are independent
and do not sum to 100. They are not probabilities, execution ratings or
proof of the role.

| Role | Feature anchors (per 100 opportunities) |
| --- | --- |
| Deep-lying playmaker | Deep central distribution share: 15; deep progressive pass attempts: 15 |
| Advanced playmaker | Advanced central distribution share: 15; advanced passes linked to shots: 8 |
| Ball-winning midfielder | Pressures: 20; middle-third tackles/interceptions: 8 |

Evidence strength is reported separately. For each feature use
`100 × field coverage × opportunities / (opportunities + scale)`. Take the
geometric mean, then multiply by `minutes / (minutes + 30)`. Opportunity
scales in the table's feature order are 12/6, 10/6, and 12/12. These scales
and the anchors are provisional and require calibration with reviewed clips.
No minimum minutes or event counts determine the match fit. A true zero
with opportunities is a score of zero; missing opportunities or coverage
leave a role unscored. The defensive score requires confirmed defensive-feed
coverage. Holding and box-to-box still require tracking or video.

The report retains each match's fit and strength, and averages each across
scored matches weighted by minutes. It shows scored match counts so a mean
does not hide missing coverage. League percentiles remain optional reference
metrics, never a role assignment.

## Current dashboard: relative roles

The relative comparison replaces absolute scoring in the dashboard. A match
compares players from both teams in the same midfield group. The report view
pools counts and opportunities first, then compares profiles within the same
competition, season and position group. It does not average match ranks.
For each role, both measures are compared against the same available players.
The score averages the share of other players beaten on each measure, counting
ties as half. A strongest-pattern label is assigned only where a measured
rate exceeds another player's; equal top patterns may share the label. There
are no minimum minute, action or percentile gates. Counts and opportunities
are displayed so users can judge small samples. Missing observations and
movement-dependent roles remain unscored.

- **Deep build-up distributor:** team deep central passing share and progression
  from deep. These are event-based clues to deep-lying playmaking.
- **Advanced chance provider:** advanced central involvement and passes directly
  linked to shots. This does not establish receiving between the lines.
- **Midfield disruptor:** pressure and tackle/interception activity. This requires
  confirmed defensive-feed coverage and does not measure successful regains.
- **Holding and box-to-box:** require evidence of positioning or movement.

Data sources: [StatsBomb Open Data](https://github.com/hudl/open-data).
The Bundesliga report contains Leverkusen's 34 league games. The Premier League
report uses rounds 1–3 from the full 2015/16 index (380 matches, 20 clubs).
The 2015/16 Serie A index likewise contains 380 matches and is a further native
input option. [Public Wyscout 2017/18 event data](https://www.nature.com/articles/s41597-019-0247-7)
covers five domestic leagues and requires a dedicated provider adapter before
using these role measures. Do not pool competitions or providers as one cohort.

Some historical StatsBomb receptions have clocks that wrap or overrun at
half-end. The native importer excludes specifically identified derived
receptions linked to a preceding pass, logs their IDs, and retains source files.
