"""Scientific consistency checks for reproducible Phase 11 outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from phase11_kalman_stadium import OUT, METHODS, PREDICTION_LIMIT_S, validate_paired_wls

p=OUT/'kalman_paired_epoch_results.csv'
df=pd.read_csv(p)
summary=pd.read_csv(OUT/'kalman_method_comparison.csv')
events=pd.read_csv(OUT/'reference_events.csv')
assert set(df.seed)=={272,847,1927,3117,9872,11234}
assert len(events)==287
assert set(df.method)==set(METHODS)
assert not df.duplicated(['seed','model','scenario','game','play','player','frame','method']).any()
assert df.loc[~df.accepted,'est_speed_mps'].isna().all()
assert not df.loc[df.accepted,'est_speed_mps'].isna().any()
assert df.loc[df.solution_type=='prediction','accepted'].all()
assert (df.loc[df.solution_type=='prediction','seconds_since_update']<=PREDICTION_LIMIT_S+1e-8).all()
assert df.loc[df.solution_type=='prediction','satellites_used_in_update'].eq(0).all()
assert df.loc[df.solution_type=='gnss_update','satellites_used_in_update'].ge(4).all()
assert not df[df.scenario.str.startswith('sealed_roof__')].accepted.any()
assert (summary.event_recall_pct<=summary.sampling_ceiling_pct+1e-8).all()
assert (summary.prediction_only_pct+summary.gnss_update_pct<=summary.mean_epoch_availability_pct+1e-7).all()
assert (summary.mean_epoch_availability_pct<=100+1e-8).all()
assert (summary.gnss_update_pct>=0).all()
assert (summary.prediction_only_pct>=0).all()
ref=validate_paired_wls(df,strict=True)
assert ref['matched_rows']==87168

# A fixed complete blackout cannot bootstrap a filter without external aiding.
fully_blocked=summary[summary.scenario.str.startswith('sealed_roof__')]
assert (fully_blocked.event_recall_pct==0).all()
assert (fully_blocked.gnss_update_pct==0).all()

# Four-satellite WLS is a distinct control, not mislabeled as a KF prediction.
assert df[df.method=='wls4'].solution_type.isin(['gnss_update','unavailable']).all()
assert (df[(df.method=='wls4')&df.accepted].n_satellites>=4).all()
assert (df[(df.method=='wls')&df.accepted].n_satellites>=6).all()

# Predictions can bridge outages but should not be claimed as direct satellite solutions.
case=summary[summary.scenario.eq('open__measured_proxy__outage2')].set_index('method')
assert case.loc['kf_cv4','prediction_only_pct']>40
assert case.loc['kf_cv4','gnss_update_pct']<case.loc['kf_cv4','mean_epoch_availability_pct']

print(json.dumps({'result':'PASS','epochs':len(df),'reference_events':len(events),
 'scenarios':int(summary.scenario.nunique()),'methods':len(METHODS),
 'seeds':6,'phase10_wls_rows_exactly_reproduced':ref['matched_rows']},indent=2))
