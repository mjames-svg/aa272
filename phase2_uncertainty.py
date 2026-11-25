"""Cluster-bootstrap uncertainty for simulated matched-epoch differences and real C/N0 signal relationship."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED=92872


def compare_methods(df, out, iters=1200):
    rng=np.random.default_rng(SEED)
    df=df[df.rate_hz==1].copy()
    idx=['scenario','trial','game','play','player','frame']
    base=df[df.method=='wls'].set_index(idx)
    comparisons=[]
    for scenario in sorted(df.scenario.unique()):
        for method in ['fde','robust','wls_epoch_gate']:
            alt=df[df.method==method].set_index(idx)
            merged=base.loc[scenario].join(alt.loc[scenario],lsuffix='_base',rsuffix='_alt',how='inner').reset_index()
            common=merged[(merged.retained_base)&(merged.retained_alt)].copy()
            if not len(common):continue
            common['sq_base']=(common.est_speed_mps_base-common.true_speed_mps_base)**2
            common['sq_alt']=(common.est_speed_mps_alt-common.true_speed_mps_alt)**2
            g=common.groupby(['trial','game','play','player'],sort=True)
            totals=g.agg(n=('sq_base','size'),base=('sq_base','sum'),alt=('sq_alt','sum'))
            arr=totals[['n','base','alt']].to_numpy()
            rmse0=np.sqrt(arr[:,1].sum()/arr[:,0].sum())
            rmse1=np.sqrt(arr[:,2].sum()/arr[:,0].sum())
            boot_idxs=rng.integers(0,len(arr),size=(iters,len(arr)))
            sums=arr[boot_idxs].sum(axis=1)
            diffs=np.sqrt(sums[:,2]/sums[:,0])-np.sqrt(sums[:,1]/sums[:,0])
            low,high=np.percentile(diffs,[2.5,97.5])
            comparisons.append(dict(scenario=scenario,other_method=method,
                 common_accepted_epochs=len(common), clusters=len(arr),
                 rmse_wls_common_mps=float(rmse0),rmse_other_common_mps=float(rmse1),
                 rmse_difference_other_minus_wls_mps=float(rmse1-rmse0),
                 bootstrap_95ci_low_mps=float(low),bootstrap_95ci_high_mps=float(high),
                 interpretation='negative difference favors other method; 95% CI is a descriptive clustered bootstrap across simulated player trajectories'))
    result=pd.DataFrame(comparisons)
    result.to_csv(out/'paired_common_epoch_bootstrap.csv',index=False)
    return result


def bootstrap_cno(out, iters=700):
    rng=np.random.default_rng(SEED+1)
    a=pd.read_csv(out/'empirical_leave_one_out_residuals.csv')
    a['block']=(a.epoch//15).astype(int)
    blocks=[b[['cn0_dbhz','loo_prediction_residual_mps']].to_numpy() for _,b in a.groupby('block')]
    observed=float(spearmanr(a.cn0_dbhz,np.abs(a.loo_prediction_residual_mps)).statistic)
    sampled=[]
    for _ in range(iters):
        ids=rng.integers(0,len(blocks),size=len(blocks))
        x=np.vstack([blocks[i] for i in ids])
        sampled.append(float(spearmanr(x[:,0],np.abs(x[:,1])).statistic))
    result=dict(observed_spearman_rho=observed,
        block_bootstrap_95ci_rho=[float(x) for x in np.percentile(sampled,[2.5,97.5])],
        block_size=15, bootstrap_replicates=iters,
        note='C/N0 relation is descriptive from leave-one-out Doppler residual proxies, not independently known measurement error; 15 epoch blocks account partly for local temporal correlation.')
    (out/'cn0_block_bootstrap.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    out=Path(__file__).resolve().parent/'results/phase2'
    d=pd.read_csv(out/'nfl_simulated_epoch_results.csv')
    results=compare_methods(d,out)
    uncertainty=bootstrap_cno(out)
    print(results.to_string(index=False))
    print(json.dumps(uncertainty,indent=2))
