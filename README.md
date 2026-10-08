# DC policing analyses

The analysis code behind [Policing in DC](https://residentscount.com/series/policing-dc/), a five-part series in [Residents Count](https://residentscount.com). Every analysis reads the Metropolitan Police Department's public records from DC Open Data and the Census Bureau's population estimates.

| Article | What it tests | Code |
|---|---|---|
| [Stops and arrests by police district](https://residentscount.com/reports/dc-stops-arrests/) | The Black share of people stopped and arrested against each police district's Black share of residents, overall and by year | `database/disparity.py`, `database/arrests.py`, `database/district_population.py` |
| [Who DC police search, and what they find](https://residentscount.com/reports/dc-searches/) | Pat-down rates per person stopped; hit rates by search type (the outcome test of Knowles, Persico and Todd, 2001); the Bayesian threshold test of Simoiu, Corbett-Davies and Goel (2017) | `database/disparity.py`, `scripts/threshold_model.py`, `scripts/fit_threshold_test.py`, `database/thresholds.py` |
| [Where enforcement lands, against reported crime](https://residentscount.com/reports/dc-enforcement-vs-crime/) | Stops, frisks and arrests per reported crime, against all victim crime and violent crime alone; officer-written against camera tickets | `database/crime_baseline.py`, `database/tickets.py` |
| [The daylight test](https://residentscount.com/reports/dc-daylight-test/) | The veil-of-darkness test (Grogger and Ridgeway, 2006): Mantel-Haenszel odds ratios across clock-time strata, with district and seasonal checks | `database/veil.py` |
| [Downgraded crime reports and the public data](https://residentscount.com/reports/dc-crime-classification/) | Interrupted time series at the 2013 and August 2023 boundaries with placebo dates; a district difference-in-differences with a pre-trend test, its interpretation written down before it ran | `database/classification_mix.py`, `database/theft_exit.py` |

## Data

All public, all loaded by the scripts here:

| Source | Window | Loader |
|---|---|---|
| MPD reported incidents (nine Part I offenses) | 2008 on | `database/ingest.py` |
| MPD stops of persons (NEAR Act data) | July 2019 on | `scripts/ingest_stops.py` (`--layer 45` for 2019 to 2022, the default layer 41 for 2023 on) |
| MPD adult arrests | 2013 to 2025 | `scripts/ingest_arrests.py` |
| Moving violations, camera and officer | 2023 to 2025 | `scripts/ingest_ticket_counts.py` |
| Population by census tract (American Community Survey, 2019 to 2023), assigned to police districts | | `scripts/build_district_population.py`, then `scripts/load_district_population.py` |

Records come from DC Open Data (maps2.dcgis.dc.gov) and the Census API. The population builder uses `CENSUS_API_KEY` if it is set and a keyless DC GIS layer otherwise.

## Rebuild

Needs Python 3.11 or later and PostgreSQL 14 or later (PostGIS optional). Each loader creates its own tables.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://localhost/dcpolicing

python database/ingest.py                     # reported incidents
python scripts/ingest_stops.py --layer 45     # stops, July 2019 to 2022
python scripts/ingest_stops.py                # stops, 2023 on
python scripts/ingest_arrests.py
python scripts/ingest_ticket_counts.py
python scripts/build_district_population.py
python scripts/load_district_population.py
PYTHONPATH=. python scripts/fit_threshold_test.py   # the threshold model, by MCMC; slow
python scripts/run_disparity.py               # prints the stop and search tables
```

Every analysis is an async function that takes a database connection, for example:

```python
import asyncio
from database import db
from database.veil import veil_of_darkness

async def main():
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        print(await veil_of_darkness(conn))

asyncio.run(main())
```

The others are `district_representation_index` and `citywide_disparity_summary` (`database/disparity.py`), `arrest_representation_index` (`database/arrests.py`), `enforcement_vs_crime` (`database/crime_baseline.py`), `enforcement_skew` (`database/tickets.py`), `latest_threshold_fit` (`database/thresholds.py`), `classification_mix` (`database/classification_mix.py`) and `theft_exit_did` (`database/theft_exit.py`).

MPD revises its records, and DC Open Data publishes new releases, so a rebuild reflects the data as of the day it runs. The figures in the articles were computed in 2026 on data through 2025.

## Tests

```bash
pytest
```

212 tests, none of which need a database.

## Where this came from

These analyses first ran on Justice Lens, a Specta site, in 2026. This repository holds only the analysis code. The web pages, accounts, newsletter and alert emails are left out, and `database/db.py` and `config/settings.py` are trimmed to match: where the site emailed an administrator about an unusual batch of record corrections, this code logs a warning. Tests of the web pages are removed, and two tests of that alert are adapted to the warning.

Copyright 2026 Specta LLC. Shared so the analyses can be checked.
