"""Offline entry point for the reproducible Phase 1 analysis."""
from pathlib import Path
from real_gnss_analysis import run_analysis
from plot_real_results import run as plot_real
from nfl_data_audit import run as audit_nfl

root = Path(__file__).resolve().parent
run_analysis(root/'data'/'Raw.csv',root/'data'/'Fix.csv',root/'data'/'Sta23062.sp3',root/'results')
plot_real(root/'results')
if (root/'data'/'input_2023_w01.csv').exists():
    audit_nfl(root/'data'/'input_2023_w01.csv',root/'results')
print('Phase 1 finished. See results/ and README.md.')
