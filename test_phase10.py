"""Validate the complete paired stadium visibility experiment and its outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from phase10_geometry_stadium import sky_geometry,SKIES,SIGNALS

R=Path(__file__).resolve().parent/'results'/'phase10'
e=pd.read_csv(R/'paired_epoch_results.csv')
g=pd.read_csv(R/'paired_roof_geometry.csv')
s=pd.read_csv(R/'stadium_method_comparison.csv')
r=pd.read_csv(R/'reference_events.csv')
b=pd.read_csv(R/'paired_game_cluster_bootstrap.csv')
assert set(e.seed.unique())=={272,847,1927,3117,9872,11234}
assert set(e.method.unique())=={'wls','fde','robust'}
assert set(s.method.unique())=={'wls','fde','robust','wls_epoch_gate'}
assert set(e.sky.unique())==set(SKIES)
assert set(e.signal.unique())==set(SIGNALS)
assert len(r)==287
assert e.groupby(['seed','game','play','player','frame','sky','signal']).method.nunique().eq(3).all()
assert e.loc[~e.accepted,'est_speed_mps'].isna().all()
assert e.loc[e.accepted,'n_satellites'].ge(6).all()
assert e.loc[e.accepted,'baseline_formal_sigma'].le(4).all()
assert e.loc[e.accepted,'est_speed_mps'].le(15).all()
paired=e[e.method.eq('wls')]
keys=['seed','game','play','player','frame']
assert paired.groupby(keys).scenario.nunique().eq(8).all()
assert paired.groupby(keys).true_speed_mps.nunique().eq(1).all()
assert paired.groupby(keys).roof_orientation_deg.nunique().eq(1).all()
assert paired.groupby(keys+['sky']).n_satellites.nunique().eq(1).all()
geo=g.pivot(index=['seed','game','play','player'],columns='sky',values='retained_n_satellites')
assert ((geo.open>=geo.partial_canopy)&(geo.partial_canopy>=geo.deep_canopy)&(geo.deep_canopy>=geo.sealed_roof)).all()
assert geo.sealed_roof.eq(0).all()
assert paired[paired.sky.eq('sealed_roof')].accepted.eq(False).all()
assert s[s.scenario.str.startswith('sealed_roof')].mean_speed_rmse_on_accepted_mps.isna().all()
assert s[s.scenario.str.startswith('sealed_roof')].event_recall_pct.eq(0).all()
assert s[s.scenario.str.startswith('sealed_roof')].mean_epoch_availability_pct.eq(0).all()
assert len(s)==32
assert len(b)==24
assert b.lower_95_pp.le(b.point_delta_recall_pp).all()
assert b.upper_95_pp.ge(b.point_delta_recall_pp).all()
rot=np.eye(3)
H=np.column_stack((-np.array([[.2,.6,.8],[.1,.4,.91],[.5,.1,.86]]),np.ones(3)))
_,_,m=sky_geometry(H,rot,5)
assert np.all(m['deep_canopy']<=m['partial_canopy']) and np.all(m['partial_canopy']<=m['open'])
print('Phase 10 passed: six seeds, 287 reference events, 32 scenario-method summaries, paired sampling, nested masks, accepted-epoch constraints, and valid uncertainty intervals.')
