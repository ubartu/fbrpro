# fbrpro player label dictionary v0.1

Status: research-informed pilot specification. No player has been classified using these rules yet.

## Research basis

- [Plakias et al. (2023), *Identifying Soccer Players’ Playing Styles: A Systematic Review*, Figure 3 and Table 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10443261/#sec4-jfmk-08-00104): role vocabulary and descriptions. The review synthesizes 26 styles across 12 studies of adult professional men’s football. Its player examples are illustrative.
- [Aalbers & Van Haaren (2018), *Distinguishing Between Roles of Football Players in Play-by-play Match Event Data*, §§3–5](https://arxiv.org/html/1809.05173): five central midfield roles, action location and type as features, and normalization relative to player and team activity. The study proposed 21 roles but detailed and demonstrated these five. Its expert training labels captured a primary role; fbrpro will allow multiple independently supported labels.

The definitions below are paraphrased or adapted. Metric formulas, gates, zones, and review rules are **fbrpro proposals**. They are not published cutoffs or a reproduction of either paper’s model.

## Behaviour dictionary

Use normalized event types and eligible position intervals. For zones, set `u = x / coordinate_length_m` and `v = y / coordinate_width_m` after confirming the actor’s team attacks toward increasing x. Deep = `u < 0.50`; advanced = `u >= 2/3`; central = `0.25 <= v <= 0.75`. These are pilot zones. Skip events with invalid or ambiguous coordinates; report their count. Open play excludes restart actions and any set-piece phases that cannot be reliably separated from open play.

| ID | Label and meaning | Primary style measure | Separate execution evidence |
|---|---|---|---|
| `progressive_passer` | Regularly attempts to move possession forward by passing. | Percentage of eligible open-play passes with end u minus start u ≥ 0.10. Include completed and unsuccessful attempts when both locations are known. | Completion among qualifying attempts with known outcomes; show outcome coverage. |
| `progressive_carrier` | Regularly advances possession while carrying. | Percentage of eligible open-play carries with end u minus start u ≥ 0.10. | Retention after the carry if the next action can be reliably linked. |
| `deep_build_up_distributor` | Takes a visible share of the team’s passing from deep central areas. | Player’s open-play pass attempts starting deep and central divided by the team’s eligible attempts from that zone during the player’s assessed midfield minutes. | Completion and progression from that zone, shown separately. |
| `open_play_chance_creator` | Supplies passes directly linked to teammates’ shots. | Linked shot assists per 100 eligible player open-play pass attempts; use explicit provider shot links or verified shot-assist flags. Count each pass once. | Shot xG per linked pass when available. This describes the resulting chance, not the passer’s isolated contribution. |
| `active_presser` | Frequently applies recorded pressure to opposition ball holders. | Pressure events per 100 opponent possessions during assessed minutes, counting unique match/period/possession IDs. Requires confirmed pressure coverage. | Regain after a pressure only with reliable possession sequences; temporal association is not proof of causation. |
| `interception_and_tackle_activity` | Frequently intervenes via tackles and interceptions. | Deduplicated tackle attempts and interceptions per 100 opponent possessions; retain event counts and locations. | Outcomes by action type when known. Record confirmed regains separately; an intervention need not win possession. |

The advancement rule measures longitudinal progress on the declared pitch scale. It does not measure passes between defensive lines, pressure resistance, or physical speed. Each provider needs a verified mapping of action types, outcomes, phases, and coverage before use.

## Composite midfield roles

Use the same player, team and match window for all evidence within a role. A player can receive several labels. Do not infer the coach’s instructions from a role label.

| ID | Role | Positive evidence | What distinguishes it or still needs review |
|---|---|---|---|
| `deep_lying_playmaker` | Organizes attacks from deeper midfield. | Elevated deep distribution and progressive passing **from deep central starts**. | Check distribution across matches. Whole-pitch progression alone does not distinguish this role from an advanced creator. |
| `advanced_playmaker` | Creates chances from higher midfield areas. | Elevated shot-assist rate **from advanced starts**, with repeated advanced central passing involvement. | Event locations show where actions happened. Use video or tracking to verify receiving between the lines. |
| `ball_winning_midfielder` | Actively disrupts opposition possession. | Elevated recorded pressure and tackle/interception activity, supported by midfield intervention locations. | Show confirmed regains separately. High activity alone may reflect a team spending longer without the ball. |
| `holding_midfielder` | Stays connected to and protects the back line. | Deep central positioning across defensive sequences; interceptions in front of the defence are supporting evidence. | Requires tracking or analyst video review of screening and spacing. Few pressures alone do not indicate a holding role. |
| `box_to_box_midfielder` | Connects deeper defending with advanced attacking involvement. | Interventions in deeper zones and repeat progression or box involvement within the same match windows. | Event evidence can flag a candidate. Tracking or video should verify sustained movement and late arrivals. |

For role-specific rates, recompute the behaviour within the relevant zone and use its own eligible denominator. Do not apply a whole-pitch percentile to a zone-specific feature.

## Assignment and evidence rules

- **Unit:** player × team × competition-season × midfield position group. Assign events to verified position intervals. Store match-level evidence and split windows if the role changes. The current normalizer can preserve provider position intervals without calculating exact minutes; resolve interval timing before using time-based gates.
- **Pilot gate:** test 450 verified position minutes across five matches and at least 30 eligible denominator opportunities for each rate. Require at least 90% coverage for the fields needed by that metric in the candidate sample. Provider-level absence of an event family makes it unavailable, even if every supplied row has a valid value.
- **Peer baseline:** compare within the same competition-season and position group, with compatible provider coverage. Use at least 30 eligible peers for an initial percentile. Record the cohort and split by position where supported; avoid pooling incompatible leagues or positions just to reach the minimum.
- **Pilot threshold:** flag a behaviour at or above its peer 75th percentile, with at least three qualifying actions across at least two matches. For composite roles use the listed evidence jointly, compare zone-specific rates against zone-specific peers, and confirm through analyst review. These settings are starting points to test and tune.
- **Status:** `unassessed` if required data, sample or peers are missing; `below_pilot_threshold` if assessed and below the rule; `candidate` if the rule is met or event proxies warrant review; `reviewed` when an analyst confirms the label. Holding and box-to-box roles cannot be confirmed by event proxies alone.
- **Store:** label ID/version, status, period and position group, numerator and denominator, matches and verified minutes, coverage, peer cohort, percentile, separate execution measure, and supporting event IDs or clip references. A percentile is a comparison with peers, not a probability or a quality score.
- **Validate before deployment:** have two analysts independently review matched and borderline examples, including similar roles that can be confused. Tune the pilot gates on reviewed development matches; check agreement, false positives and repeatability on held-out matches. Do not treat agreement with famous example players as validation.

## Build order

Implement progression and deep distribution first, using the coordinates and event fields already in the intake format. Add chance creation once shot links and open-play coding are checked, then the two defensive behaviours once pressure, tackles and possessions are verified. Use the resulting evidence to review the three event-supported role candidates. Add tracking or video review for holding and box-to-box.
