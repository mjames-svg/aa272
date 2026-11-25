from pathlib import Path
import numpy as np
from phase11_kalman_stadium import DopplerKF, FilterConfig, DEFAULT_Q, SIGMA_LIMIT, PREDICTION_LIMIT_S, wls_matched_phase10

rng=np.random.default_rng(272)
xyz=rng.standard_normal((9,3))
xyz=xyz/np.linalg.norm(xyz,axis=1)[:,None]
H=np.column_stack([xyz,np.ones(len(xyz))])
sigma=np.full(len(H),.35)
truth=np.array([3.2,1.8,0.,.45])
y=H@truth
rotation=np.eye(3)

cv=DopplerKF(FilterConfig('unit_cv',6,DEFAULT_Q),rotation)
assert cv.step(0,H[:3],y[:3],sigma[:3])[2]=='unavailable'
init=cv.step(10,H,y,sigma)
assert init[0] and init[2]=='initialized'
miss1=cv.step(20,H[:0],y[:0],sigma[:0])
assert miss1[0] and miss1[2]=='prediction' and abs(miss1[4]-1)<1e-12
miss2=cv.step(30,H[:0],y[:0],sigma[:0])
assert miss2[0] and miss2[2]=='prediction' and abs(miss2[4]-2)<1e-12
miss3=cv.step(40,H[:0],y[:0],sigma[:0])
assert not miss3[0] and miss3[2]=='prediction_withheld'

cv4=DopplerKF(FilterConfig('unit_cv4',4,DEFAULT_Q),rotation)
assert cv4.step(10,H,y,sigma)[2]=='initialized'
update=cv4.step(20,H[:4],y[:4],sigma[:4])
assert update[0] and update[2]=='gnss_update' and update[5]==4
cv6=DopplerKF(FilterConfig('unit_cv6',6,DEFAULT_Q),rotation)
assert cv6.step(10,H,y,sigma)[2]=='initialized'
assert cv6.step(20,H[:4],y[:4],sigma[:4])[2]=='prediction'

ca=DopplerKF(FilterConfig('unit_ca',4,3.,'ca'),rotation)
assert ca.step(10,H,y,sigma)[2]=='initialized'
assert ca.n==7 and ca.x.shape==(7,) and ca.P.shape==(7,7)
assert ca.step(20,H[:4],y[:4],sigma[:4])[0]
assert np.allclose(ca.P,ca.P.T,atol=1e-10)
assert np.linalg.eigvalsh(ca.P).min()>-1e-10

qual6=wls_matched_phase10(H[:4],y[:4],sigma[:4],rotation,min_sats=6)
qual4=wls_matched_phase10(H[:4],y[:4],sigma[:4],rotation,min_sats=4)
assert not qual6[0] and qual4[0]

# The filter consumes only current data: changing the next observation must
# not change previous outputs or states.
a=DopplerKF(FilterConfig('test_no_future',4,DEFAULT_Q),rotation)
b=DopplerKF(FilterConfig('test_no_future',4,DEFAULT_Q),rotation)
for frame in (10,20,30):
    va=a.step(frame,H,y,sigma)
    vb=b.step(frame,H,y,sigma)
    assert va==vb and np.allclose(a.x,b.x) and np.allclose(a.P,b.P)
a.step(40,H,y+5,sigma)
b.step(40,H,y,sigma)
assert np.allclose(a.P,a.P.T) and np.allclose(b.P,b.P.T)

print('Phase 11 tests passed: initialization, 4-satellite update, outages, CA model, WLS controls, causal filtering, covariance stability')
