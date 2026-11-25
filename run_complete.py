"""Reproduce the supplemental measured-data analysis and verify the final project.

Use --full-simulation to regenerate the slower six-seed held-out NFL experiment.
All scripts run locally and offline with the bundled inputs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(script: str, *args: str):
    print(f'\n==> {script} {" ".join(args)}',flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args],cwd=ROOT,check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-simulation', action='store_true', help='Rerun the full expensive six-seed NFL simulation')
    parser.add_argument('--kalman', action='store_true', help='Rerun the paired Phase 11 Kalman extension and its tests')
    args = parser.parse_args()
    if args.full_simulation:
        call('run_final_analysis.py','--seeds','6')
        call('final_uncertainty_and_figures.py')
        call('refresh_final_sensitivity.py')
    call('phase7_novatel_rg.py')
    call('phase8_real_reference_robustness.py')
    call('phase9_stadium_context.py')
    call('write_final_report.py')
    if args.kalman:
        call('phase11_kalman_stadium.py','--seeds','6')
        call('test_phase11.py')
        call('verify_phase11_results.py')
    for test in ('self_check.py','test_phase2.py','test_phase3.py','test_phase4.py',
                 'test_phase5.py','test_phase7.py','test_phase8.py','test_phase9.py','test_final_analysis.py',
                 'verify_final_package.py'):
        call(test)
    print('\nCompleted the project rebuild and all verification checks.')


if __name__ == '__main__':
    main()
