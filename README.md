# AA 272 — GNSS Velocity Estimation and High-Speed Football Movements

Python analysis for Doppler-derived velocity estimation, satellite fault detection and exclusion (FDE), robust weighted least squares (WLS), quality-control rejection, causal Kalman filtering, and NFL high-speed event detection.

## Study scope

This repository contains the project implementation and tests. Real NFL 10-Hz player trajectories are used as **reference motion**; satellite Doppler observations associated with NFL movement and stadium roof masks are **simulated**. Separate recordings from a Pixel Watch provide measured pseudorange-rate and signal-quality context. GEDVM smartphone driving recordings provide an independent processed-speed comparison with GNSS/INS reference velocity. SnapperGPS walking recordings provide position-accuracy context. These sources are not interchangeable.

## Setup (Visual Studio Code, Python 3.10+)

1. Open this repository folder in VS Code.
2. Create and activate a Python virtual environment. On Windows PowerShell:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Obtain the required raw files separately and follow `data/README.md`. **The full analysis cannot be rerun from this source-only repository without those inputs.**

## Major scripts

| Script | Purpose |
| --- | --- |
| `real_gnss_analysis.py` | Raw measured-watch pseudorange-rate estimation and quality-control methods |
| `phase2_nfl_simulation.py` | Initial NFL movement-based synthetic satellite observations |
| `phase5_event_integration.py` | High-speed interval definitions, detection recall, false alerts |
| `phase10_geometry_stadium.py` | Paired simulated stadium sky masks and WLS/FDE/Huber comparisons |
| `phase11_kalman_stadium.py` | Causal constant-velocity and constant-acceleration Kalman filters and matched four-satellite WLS controls |
| `phase12_verified_fault_injection.py` | Fault injection into measured Pixel Watch Doppler records |
| `phase13_gedvm_validation.py` | Android processed speed against GNSS/INS reference from driving recordings |
| `phase4_snapper_reference.py` | Walking-position accuracy against RTK reference |
| `paper_tools/make_publication_figures.py` | Rebuilds selected manuscript figures when the required project data and figure assets are present |

Additional `phase*`, `run_*`, and `test_*` scripts document supporting experiments, sensitivity tests, and verification steps. `paper_tools/` contains figure-generation scripts from the accompanying manuscript; it expects the manuscript figure inputs and is not runnable from this source-only archive alone. Filenames preserve the implementation dependencies of the study.

## Running experiments

With the original inputs in the documented directories:

```powershell
python phase10_geometry_stadium.py --seeds 6
python phase11_kalman_stadium.py --seeds 6
python test_phase11.py
python verify_phase11_results.py
python phase12_verified_fault_injection.py
python phase13_gedvm_validation.py --dataset gedvm_dataset-main
```

The six-seed numerical experiments can be computationally expensive. Phase 11 depends on supporting modules and the matching real NFL and measured-watch input files. Some supporting validation scripts also use outputs from earlier stages; see each script and `data/README.md`.

The `results/` directory contains a limited selection of existing summary tables for inspection. It **does not** contain the full epoch-by-epoch outputs needed by every verification or plot-generation script. The source-only package deliberately does not claim to reproduce the full report immediately after extraction.

## Interpretation

- Speed RMSE is computed on accepted velocity estimates.
- Availability records when a valid satellite-based update exists; a Kalman prediction is distinguished from a GNSS measurement.
- Event recall evaluates whether an observed NFL high-speed interval is detected at the sampled times using a 7 m/s threshold.
- For the paired roof simulation, the matched four-satellite WLS control helps distinguish the effect of accepting more satellite configurations from the benefit of temporal filtering.

## Data and citation

Input datasets are distributed by their original providers. Relevant sources are identified in the accompanying AA 272 paper, including NFL Big Data Bowl tracking data, the measured-watch GNSS dataset, SnapperGPS outdoor positioning recordings, and GEDVM driving records. Dataset download permissions and terms remain the responsibility of the person retrieving them.
