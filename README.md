# Data

This directory contains a 10,000-row sample from the Calgary Solar Energy
Production dataset, taken from the **Calgary Fire Hall Headquarters** PV site.
This is the exact subset used to produce all results in the graduation report
and the IEEE Access journal submission.

## Files

- `solar_sample_10k.csv` — 10,000 hourly kWh readings (≈1.9 MB)

## Full dataset

The complete dataset (301,231 rows, 11 sites, ~61 MB) is publicly available
from the **City of Calgary Open Data Portal**:

> https://data.calgary.ca/Environment/Solar-Energy-Production/eyzg-jzbf

To run experiments on the full dataset:

1. Download `Solar_Energy_Production.csv` from the link above
2. Place it in this directory
3. In the notebook, change the data-loading cell to read the full CSV

## Columns

| Column | Description |
|---|---|
| `name` | PV site name |
| `id` | Site numeric ID |
| `address` | Physical address |
| `date` | Timestamp (YYYY/MM/DD HH:MM:SS AM/PM) |
| `kWh` | Energy produced in that hour |
| `public_url` | SolarEdge public monitoring URL |
| `installationDate` | When the panel was installed |
| `uid` | Unique row identifier |

For this project, only the `kWh` column is used as the entropy source.
