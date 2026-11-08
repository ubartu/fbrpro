# fbrpro: match statistics for all 26 player roles

This extends the [midfield label dictionary](LABEL_DICTIONARY.md) to every role in Table 5 of the [2023 systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10443261/). The five midfield roles from [Aalbers & Van Haaren, §§3–4](https://arxiv.org/html/1809.05173) are included in those 26. We keep one label per role regardless of flank and store left/right separately. The proposed measurements below are our design, not thresholds published by either paper.

## Which match data to extract

| Input | What we need |
|---|---|
| Match and player intervals | Selected match IDs, teams, player IDs, playing intervals, position intervals, substitutions, period clocks, and pitch orientation. Add formation and score state where verified. |
| Events and possession | Action type, actor/team, timestamp, possession and possession team, open play or restart, start/end locations, recipient, and related event IDs. |
| Provider details | Pass height, crosses, through balls, shot-assist links, shot xG/body part, dribble and duel types/outcomes, keeper actions, pressure coverage. Verify these fields in provider payloads before deriving stats. |
| Positional evidence | Event-linked 360 frames for visible space and identifiable players at an instant; continuous tracking or timestamped analyst clips for runs, marking, and positioning between events. |

Calculate counts and rates for **only the requested matches and position intervals**, then retain a per-match breakdown. Pool by summing numerators and denominators, not by averaging match percentages. Report raw counts alongside rates. Use the following denominators where the role table does not specify a narrower one:

- **Passing, carrying and dribbling tendencies:** per 100 relevant eligible attempts, restricted to the phase or zone in question. For deep distribution or link involvement also show the player's share of eligible team actions while on the pitch.
- **Attacking involvements:** per 100 team possessions while the player is on the pitch; also show zone share among the player's own eligible actions.
- **Defensive actions:** per 100 opponent possessions during the same interval; for flank or box defence also use opponent entries into the relevant zone when available.
- **Movement and positioning:** per 100 eligible sequences or as a share of observed eligible frames. Report the observed sample and how much is missing. Count each run or sequence once. Speed and distances require a calibrated tracking feed.
- **Goalkeeping:** saves per on-target shot faced, claims per eligible cross faced, and distribution patterns per eligible keeper pass or restart. Save percentage and claim success are execution measures; shot and cross exposure primarily describe opportunity.

Use the dictionary's attack-relative pitch convention: own goal at low x, opponent goal at high x, and documented central/flank bands. Retain action maps alongside summaries. The penalty areas should use known pitch markings or verified provider geometry. Keep corners and free kicks separate from open play. Team possession counts do not measure whether a particular pass, tackle or run was available.

**E** = event metrics can support a role candidate if coverage is verified. **T** = tracking or analyst video is needed to establish the defining movement or positioning; events still provide screening evidence. Neither status confirms the coach's instructions.

## Goalkeepers (3)

| Role | Stats to extract from selected matches | Defining check |
|---|---|---|
| Sweeper keeper | Outside-box interventions per opponent through-ball or ball-in-behind opportunity; locations of clearances and interceptions; timing of keeper action relative to the opponent pass. | **T:** keeper starting depth and movement to intercept behind the line; distinguish sweeping from routine collection. |
| Ball-playing keeper | Short/long pass mix split by open play and restarts; share of build-up possessions involving the keeper; forward/progressive pass share and recipient zones; known pressure at distribution. | **E:** compare distribution choices across similar build-up opportunities. Show completion under known pressure separately. |
| Line keeper | Shot locations and on-target shots faced; save locations, claims and rush-outs per relevant opportunity; distribution profile. | **T:** depth from goal line while defending and starting position at shots. Saves alone show shot exposure and execution. |

## Centre-backs (3)

| Role | Stats to extract from selected matches | Defining check |
|---|---|---|
| Ball-playing defender | Share of team passes from deep zones; progressive pass attempt share, midfield-entry passes, switches and forward carries; open-play and restart splits. | **E:** distinguish repeated build-up involvement from long clearances. Track completion separately. |
| Stopper | Tackle, interception and aerial-duel attempts by zone; actions initiated ahead of the back line where observable; clearances and confirmed regains separately. | **E:** flag the intervention pattern; tracking/video verifies stepping out rather than simply defending under sustained pressure. |
| Sweeper / libero | Cover clearances and interceptions behind a teammate’s challenge where identifiable; defensive action depths and recovery locations. | **T:** depth relative to the other centre-backs and frequency of covering behind them. Deep event locations alone do not establish this. |

## Full-backs and wing-backs (3)

| Role | Stats to extract from selected matches | Defining check |
|---|---|---|
| Defensive full-back | Defensive flank tackles and interceptions per opponent flank entry; share of deep wide receptions and passes; attacking-third involvements. | **T:** verify staying connected to the back line and defending the flank. Few crosses alone are insufficient. |
| Offensive wing-back | Wide final-third receptions, carries and crosses; open-play box deliveries; wide attacking involvement relative to eligible team possessions. | **E:** flag advanced wide involvement from a back/wing-back slot. Use tracking/video to verify overlaps and sustained width. |
| Inverted wing-back | Central build-up receptions and passes from a back slot; flank-to-centre carries; central involvement split by possession phase. | **T:** verify moving inside without the ball and joining midfield. An isolated inward carry does not establish inversion. |

## Central and attacking midfielders (8)

| Role | Stats to extract from selected matches | Defining check |
|---|---|---|
| Deep-lying playmaker | Share of team passes from deep central areas, progressive pass attempts from those starts, switches, and midfield-entry passes. | **E:** depth of creation separates this candidate from an advanced playmaker; show completion separately. |
| Holding midfielder | Deep central interceptions, recoveries and distribution; tracked time in the central space ahead of the back line. | **T:** verify screening and spacing across opponent possessions. Low pressure volume is not evidence of screening. |
| Ball-winning midfielder | Pressures, tackle attempts and interceptions per 100 opponent possessions; location of interventions; confirmed regains and tackle outcomes separately. | **E:** compare disruption activity against peers and the team’s defending exposure. |
| Box-to-box midfielder | Deep defensive actions plus progressive carries and attacking box involvements in the same match intervals; sequences connecting both ends. | **T:** verify repeated movement between phases and timing of box arrivals. Disconnected events are only proxies. |
| Advanced playmaker | Shot assists, through balls and box-entry pass attempts from advanced central areas; share of team advanced central passes. | **E:** creation from high central zones supports a candidate; verify receiving between lines separately. |
| Trequartista | Advanced central receptions, take-ons, carries into the box, shots and shot assists; share of the team’s advanced involvement. | **T:** verify freedom to roam and combine creation with box runs; an event profile can overlap with advanced playmaker. |
| False attacking midfielder | From a central attacking slot, share of receptions or touches in deeper and wider zones; passes following those involvements. | **T:** verify repeated departures from the central slot and who fills the space. Broad event maps alone cannot establish rotation. |
| Wide midfielder / mezzala | From an interior midfield slot, flank receipts, carries, passes and recoveries; share of attacks joined in the near-side wide channel. | **T:** verify lateral shifts from midfield and support of flank overloads. Compare with a winger who starts and stays wide. |

## Wide attackers (4)

| Role | Stats to extract from selected matches | Defining check |
|---|---|---|
| Inside forward | Wide-to-central carries, box entries from the flank, and central/box shots after wide involvement. | **E:** flag inward attacking actions from a wide slot. Verify diagonal off-ball runs separately. |
| Winger | Wide receptions, flank take-ons, byline carries and crosses per eligible wide attacking actions; crossing destinations. | **E:** wide delivery and one-on-one involvement distinguish the candidate from a primarily inward attacker. Verify off-ball width separately. |
| Wide playmaker | Shot assists from wide areas, through balls or cutbacks from the flank, and share of team chance-creating passes from that side. | **E:** compare creation and delivery mix with winger and inside-forward profiles; keep cross completion and chance quality separate. |
| Defensive wing forward | Wide pressures and tackles per opponent flank possession; recoveries in the defensive half; attacking involvement retained as context. | **T:** verify tracking back and helping the full-back, including sequences with no logged action. |

## Centre-forwards (5)

| Role | Stats to extract from selected matches | Defining check |
|---|---|---|
| Poacher | Share of shots and receptions in the central box, one-touch shots if tagged, and rebound attempts when reliably linked. Show shots per team box entry. | **E:** flag concentrated close-range involvement. Goals and conversion describe execution, not this role on their own. |
| Mobile striker | Reception spread across channels; wide carries, combinations and receptions behind the line where verified; runs into space per eligible attack. | **T:** measure repeated off-ball runs and lane changes. Diverse touch locations do not establish mobility. |
| Second striker | Receptions behind a verified partner striker; combinations with the lead forward, shot assists and box arrivals; separate starts by formation. | **T:** confirm support positioning relative to another striker and timing of arrivals. Formation alone is insufficient. |
| Target man | Share of team direct passes aimed at the player when intended recipient is known; aerial-duel attempts, layoffs and retention after direct reception. | **E:** direct service plus link play supports a candidate. Video verifies back-to-goal hold-up; height alone is not a role measure. |
| False nine | From a verified centre-forward slot, deep central receptions and link passes, followed by teammate runs into the striker space when observed. | **T:** verify recurring drops away from the opposition back line and the attacking rotation they create. A forward’s deep touches alone are insufficient. |

## What the report should show

For each role and selected window, report: **position interval, match IDs, minutes, feature counts and denominators, data coverage, comparison group if available, and evidence event IDs or clip timestamps**. Add execution measures such as completion, duel win rate or shot outcomes in separate columns.

A single match can produce a **match observation** or a **role candidate**; do not apply the dictionary's multi-match pilot gates to it as though it were a season profile. A role can be **unassessed** even when its event proxies are available, if its defining tracking/video evidence is missing. A genuine zero requires a covered event family and an eligible opportunity. Multiple role candidates can coexist.

Implementation order: derive coordinates, intervals and possession exposures first. Then compute passes, carries, receptions, shots and defensive actions by phase and zone. Add provider-specific mappings and clip/tracking checks before using the movement-dependent roles. This is a feature specification; it does not compute stats or assign player labels yet.
