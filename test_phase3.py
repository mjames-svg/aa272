import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase3_reference_audit import parse_rss
from phase3_temporal_sensitivity import fde_adjustable, robust_adjustable, step_markov
from phase2_nfl_simulation import solve

root=Path(__file__).resolve().parent
log=parse_rss(root/'data/PPK_expe_AG.rss')
assert len(log)==271
assert log.utc.is_monotonic_increasing
assert str(log.utc.min().date())=='2024-03-19'

rng=np.random.default_rng(4)
H=np.column_stack([rng.normal(size=(10,3)),np.ones(10)])
truth=np.array([1.5,-.8,.3,.2])
sigma=np.ones(10)*.3
y=H@truth
for est in [solve(H,y,sigma),fde_adjustable(H,y,sigma)[0],robust_adjustable(H,y,sigma,2.2)]:
    assert np.allclose(est['x'],truth,atol=1e-8)
y[2]+=12
fde,removed=fde_adjustable(H,y,sigma,z_cut=4)
assert 2 in removed
assert np.linalg.norm(fde['x'][:3]-truth[:3])<np.linalg.norm(solve(H,y,sigma)['x'][:3]-truth[:3])

rng=np.random.default_rng(272)
blocked=rng.random(2000)<.3
means=[]
for _ in range(400):
    blocked=step_markov(rng,blocked,.3,.85)
    means.append(blocked.mean())
assert .26 < np.mean(means) < .34

out=root/'results/phase3'
for p in ['novatel_reference_audit.json','real_orbit_model_audit.json','phase3_temporal_summary.json',
          'phase3_repeatability_metadata.json','heldout_monte_carlo_summary.csv','heldout_paired_bootstrap.csv']:
    assert (out/p).exists(),p
rep=pd.read_csv(out/'heldout_monte_carlo_summary.csv')
assert rep.n_seeds.eq(6).all()
meta=json.loads((out/'phase3_split.json').read_text())
assert not set(meta['train_games'])&set(meta['test_games'])
results=pd.read_csv(out/'nfl_temporal_threshold_epoch_results.csv')
assert results.groupby(['model','scenario','split','trial','game','play','player','frame']).method.size().eq(7).all()
assert not results[results.accepted].estimated_speed_mps.isna().any()
print('Phase 3 synthetic fault recovery, reference-log parsing, Markov occupancy, game-holdout and output consistency checks passed.')
