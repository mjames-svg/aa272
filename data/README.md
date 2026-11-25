# Local data directory

Raw input files are not included in the GitHub source release. Obtain each dataset from its original provider and observe the original licensing and access requirements. Place the required files here using the names below.

Core inputs used by different analyses:

- `Raw.csv`, `Fix.csv` — recorded Pixel Watch GNSS observations and Android fixes.
- `Sta23062.sp3` — GNSS precise ephemeris used by measured Doppler analysis.
- `input_2023_w01.csv` — NFL tracking input used by the simulation and sampling analyses.
- `PPK_expe_AG.rg`, `PPK_expe_AG.rss` — supplementary NovAtel/Waypoint trajectory/reference files, where available.
- SnapperGPS datasets — use `phase4_fetch_snapper_subset.py` for the subset retrieval workflow; additional reference tracks may be needed.
- GEDVM smartphone logs and GNSS/INS references — extract the GEDVM data set into `gedvm_dataset-main/` at repository root for `phase13_gedvm_validation.py`.

The work uses multiple independent datasets. The measured driving-speed comparison, walking-position comparison, and NFL reference motion are distinct experiments. Actual NFL satellite observations and stadium roof measurements are *not* in these inputs; those are simulated.
