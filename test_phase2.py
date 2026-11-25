import numpy as np
from phase2_nfl_simulation import solve, solve_fde, solve_robust, select_trajectories, event_capture
from pathlib import Path

rng=np.random.default_rng(272)
H=np.column_stack((rng.normal(size=(10,3)),np.ones(10)))
x=np.array([3.,-1.,2.,0.15]);y=H@x
sigma=np.ones(10)*0.3
for method in [solve(H,y,sigma),solve_fde(H,y,sigma)[0],solve_robust(H,y,sigma)]:
    assert np.allclose(method['x'],x,atol=1e-9),method

y_bad=y.copy();y_bad[2]+=12
naive=solve(H,y_bad,sigma)
fixed,removed=solve_fde(H,y_bad,sigma)
assert removed>=1
assert np.linalg.norm(fixed['x'][:3]-x[:3]) < np.linalg.norm(naive['x'][:3]-x[:3])
assert event_capture(np.array([1,7.5,8,3,7.2,2]),np.arange(6),np.array([0,2,4]))==(2,2)
tracks=select_trajectories(Path(__file__).resolve().parent/'data/input_2023_w01.csv',high_n=3,other_n=2)
assert len(tracks)==5
assert all(np.allclose(np.linalg.norm(t['vel_enu'],axis=1),t['speed']) for t in tracks)
print('Phase 2 synthetic known-truth, fault-recovery, reference-motion and sprint-event tests passed.')
