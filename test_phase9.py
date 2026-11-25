"""Sanity checks for the supplementary football-field / obstruction study."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'phase9'
real = pd.read_csv(OUT/'real_environment_track_comparison.csv').set_index('track')
sim = pd.read_csv(OUT/'exploratory_obstruction_stress_test.csv').set_index('scenario')
main = pd.read_csv(OUT/'main_six_seed_event_reference.csv').set_index('scenario')
assert len(real)==3 and len(sim)==3 and len(main)==2
assert int(real.loc['w02','matched_fixes'])==480
assert abs(real.loc['w02','median_error_m']-9.757192991963313)<1e-6
assert int(sim.loc['observed_like','epochs'])==827
assert sim.loc['observed_like','retained_pct']>sim.loc['reduced_satellites','retained_pct']
assert sim.loc['observed_like','high_speed_epoch_recall_pct']>sim.loc['combined_degradation','high_speed_epoch_recall_pct']
assert main.loc['observed_like','event_recall_pct']>main.loc['combined_degradation','event_recall_pct']
for png in ('football_field_w02_real_rtk.png','real_field_vs_urban_position_errors.png','nfl_obstruction_analogue_stress_test.png'):
    assert (OUT/png).is_file()
assert (ROOT/'PHASE9_STADIUM_CONTEXT.md').is_file()
print('Phase 9 source consistency, metrics and figures verified.')
