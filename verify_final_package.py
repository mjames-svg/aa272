"""Source integrity and scientific sanity checks for the AA272 final archive."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'SOURCE_FILE_HASHES.json'
items=json.loads(MANIFEST.read_text())
for item in items:
    p=ROOT/item['file']
    assert p.is_file(),f'Missing bundled input: {item["file"]}'
    assert p.stat().st_size==item['bytes'],f'Unexpected input file size: {item["file"]}'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],f'Checksum mismatch: {item["file"]}'
needed=['FINAL_ANALYSIS_REPORT.md','NOVATEL_RG_VALIDATION.md','PHASE8_REFERENCE_SENSITIVITY.md',
        'FINAL_RESULTS_AT_A_GLANCE.md', 'results/phase8/real_reference_robustness.png',
        'results/phase8/paired_block_bootstrap.csv', 'results/final/main_method_comparison.csv',
        'results/phase7/novatel_rg_inferred_trajectory.csv']
for f in needed:
    assert (ROOT/f).is_file(),f'Missing output: {f}'
rg=pd.read_csv(ROOT/'results/phase7/novatel_rg_inferred_trajectory.csv')
matched=pd.read_csv(ROOT/'results/phase7/matched_watch_rg_velocity_epochs.csv')
real=pd.read_csv(ROOT/'results/phase8/real_device_speed_summary.csv')
nfl=pd.read_csv(ROOT/'results/final/main_method_comparison.csv')
assert len(rg)==14580
assert len(matched)==694 and int((matched.gps_sats>=7).sum())==674
assert len(real)==4
assert set(nfl.method)=={'wls','fde','robust','wls_epoch_gate'}
print(f'Verified {len(items)} original source files, {len(needed)} output files, 14,580 recovered trajectory records, and main results.')
