"""Numerical self-check of WLS and two quality-control implementations."""
import numpy as np
import pandas as pd
from real_gnss_analysis import solve_wls, solve_fde, solve_robust

rng = np.random.default_rng(42)
u = rng.normal(size=(9,3))
u /= np.linalg.norm(u,axis=1,keepdims=True)
h = np.column_stack((u,np.ones(9)))
truth = np.array([2.2,-1.6,0.3,115.0])
y = h @ truth
a = pd.DataFrame({'hx':u[:,0], 'hy':u[:,1], 'hz':u[:,2],
                  'y':y,'sigma':np.full(9,.2),'sv':[f'G{i:02d}' for i in range(1,10)]})
sol = solve_wls(a)
assert np.allclose(sol['est'],truth,atol=1e-8)
rob,_ = solve_robust(a)
assert np.allclose(rob['est'],truth,atol=1e-8)
sol_fde, excl = solve_fde(a)
assert not excl and np.allclose(sol_fde['est'],truth,atol=1e-8)
a.loc[3,'y'] += 10
sol_fde, excl = solve_fde(a)
assert excl and np.linalg.norm(sol_fde['est'][:3]-truth[:3]) < 0.01
print('Numerical WLS / robust / FDE self-checks PASSED')
