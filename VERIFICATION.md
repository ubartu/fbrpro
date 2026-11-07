# Verification — 7 November 2026

## Automated checks

17 tests passed using Python 3.14 and the optional PyArrow 26.0.1 export dependency. File-resource warnings were treated as errors. The tests also support running without PyArrow, in which case the optional Parquet test is skipped.

Covered: StatsBomb import, mapped CSV tracking, mapped nested-field JSON events, vocabulary mappings, missing versus false values, legitimate text identifiers, coordinate conversions, period-relative and explicit match clocks, stable and reviewed identities, duplicates, reference integrity, contradictory observation flags, source-checksum integrity, tracking gaps, rejected-import isolation, existing-output preservation, and Parquet round trips.

## Real-match integration

Imported the public StatsBomb files for Bundesliga match **3895302**, using the event JSON, lineup JSON, 2023/24 match metadata, and corresponding 360 JSON.

| Output | Records |
| --- | ---: |
| Competitions | 1 |
| Seasons | 1 |
| Teams | 2 |
| Players | 38 |
| Appearance/lineup records | 38 |
| Matches | 1 |
| Events | 4,223 |
| 360 snapshots | 3,525 |
| Continuous tracking | Not supplied |

Validation returned **zero errors**. It also flagged 370 out-of-pitch coordinate observations in snapshots; these were preserved rather than clipped. This is a data-quality warning, not evidence that every observation is erroneous. The importer separately records the assumed 105×68 reference pitch and the distinction between snapshots and tracking.

Exported all eight supplied tables to Parquet, read them back, decoded nested JSON fields, and compared every field of every record against JSONL. All comparisons passed.

The downloaded match data remains in the local work area and is not redistributed in this source package. The bundled examples and demonstration output are synthetic. This checks ingestion compatibility with one real match, not suitability across every provider revision or the accuracy of future player labels.

Source: [StatsBomb Open Data](https://github.com/hudl/open-data).
