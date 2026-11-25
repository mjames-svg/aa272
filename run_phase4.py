"""Phase 4 local runner. Does not download data or rerun older Monte Carlo studies."""
from pathlib import Path
from phase4_nfl_sampling_audit import audit
from phase4_nfl_uncertainty import run as bootstrap_sampling
from phase4_snapper_reference import inspect_root,run as snapper_run
from phase4_snapper_summary import run as snapper_summary

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    audit(root/'data/input_2023_w01.csv',root/'results/phase4')
    bootstrap_sampling(root/'results/phase4/nfl_high_speed_event_sampling_detail.csv',root/'results/phase4')
    for candidate in [root/'external/snapper_reference_subset',root/'external/snapshot-gnss-data-2-main',root/'external/snapshot-gnss-data-2']:
        try:
            inspect_root(candidate)
            snapper_run(candidate,root/'results/phase4/snapper')
            snapper_summary(root/'results/phase4/snapper')
            break
        except FileNotFoundError:
            continue
    else:
        print('SnapperGPS files not installed. To add: python phase4_fetch_snapper_subset.py')
        print('Then: python run_phase4.py')
