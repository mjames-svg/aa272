"""Fixed-parameter Monte Carlo repeats on held-out NFL games.

This studies sensitivity to simulated random draws, not real NFL validation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phase2_validation import prepare
from phase2_nfl_simulation import select_trajectories, matrix_rotation, solve
from phase3_temporal_sensitivity import simulate_track, fde_adjustable, robust_adjustable

SEEDS=[272,847,1927,3117,9872,11234]


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/phase3';out.mkdir(parents=True,exist_ok=True)
    split=json.loads((out/'phase3_split.json').read_text())
    test_games=set(split['test_games'])
    tracks=select_trajectories(root/'data/input_2023_w01.csv',high_n=60,other_n=30)
    tracks=[t for t in tracks if t['key'][0] in test_games]
    real=prepare(root,root/'results/phase2')
    rot=matrix_rotation(real.epoch_info)
    scenarios={'observed_like':dict(noise=1.,fault_rate=0.,block=0.),
               'injected_faults':dict(noise=1.,fault_rate=.05,block=0.),
               'combined_degradation':dict(noise=2.,fault_rate=.05,block=.30)}
    rows=[]
    for seed in SEEDS:
      for model_id,model in enumerate(('independent','persistent')):
        for sc_id,(scenario,setup) in enumerate(scenarios.items()):
          rng=np.random.default_rng(seed*100003+model_id*7001+sc_id*1009)
          for t in tracks:
            phase=int(rng.integers(10))
            templ=real.templates[int(rng.integers(len(real.templates)))]
            for idx,H,y,sig,_ in simulate_track(rng,t,templ,rot,real.bank,setup,model,phase):
                truth=float(t['speed'][idx]);base=solve(H,y,sig) if len(y)>=4 else None
                common=(len(y)>=6 and base is not None and base['formal_sigma']<=4)
                estimates={'wls':base,'fde':fde_adjustable(H,y,sig,z_cut=5.)[0] if common else None,
                           'robust':robust_adjustable(H,y,sig,3.5) if common else None}
                for method,est in estimates.items():
                    accepted=False;speed=np.nan
                    if common and est is not None:
                        speed=float(np.linalg.norm((rot@est['x'][:3])[:2]))
                        accepted=bool(est['dof']>0 and est['formal_sigma']<=4 and np.isfinite(speed) and speed<=15)
                    rows.append(dict(seed=seed,model=model,scenario=scenario,
                                     game=t['key'][0],play=t['key'][1],player=t['key'][2],frame=int(t['frames'][idx]),
                                     method=method,accepted=accepted,true_speed_mps=truth,
                                     est_speed_mps=speed if accepted else np.nan,
                                     n_satellites=len(y),
                                     baseline_chi2=float(base['chi2']) if base is not None else np.nan,
                                     baseline_dof=int(base['dof']) if base is not None else 0,
                                     baseline_formal_sigma=float(base['formal_sigma']) if base is not None else np.nan))
    data=pd.DataFrame(rows);data.to_csv(out/'heldout_monte_carlo_epoch_results.csv',index=False)
    metrics=[]
    for (seed,model,scenario,method),a in data.groupby(['seed','model','scenario','method']):
        good=a[a.accepted]; rmse=float(np.sqrt(np.mean((good.est_speed_mps-good.true_speed_mps)**2))) if len(good) else np.nan
        metrics.append(dict(seed=seed,model=model,scenario=scenario,method=method,total_epochs=len(a),
                           accepted_epochs=len(good),availability_pct=100*len(good)/len(a),rmse_accepted_mps=rmse))
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'heldout_monte_carlo_seed_metrics.csv',index=False)
    paired=[]
    for (seed,model,scenario),d in data.groupby(['seed','model','scenario']):
        base=d[d.method=='wls'].set_index(['game','play','player','frame'])
        for method in ('fde','robust'):
            alt=d[d.method==method].set_index(['game','play','player','frame'])
            j=base.join(alt,how='inner',lsuffix='_w',rsuffix='_a')
            j=j[j.accepted_w & j.accepted_a]
            if not len(j):continue
            delta=(np.sqrt(np.mean((j.est_speed_mps_a-j.true_speed_mps_a)**2))-
                   np.sqrt(np.mean((j.est_speed_mps_w-j.true_speed_mps_w)**2)))
            paired.append(dict(seed=seed,model=model,scenario=scenario,method=method,paired_epoch_count=len(j),delta_rmse_mps=float(delta)))
    paired=pd.DataFrame(paired);paired.to_csv(out/'heldout_monte_carlo_paired_differences.csv',index=False)
    summary=paired.groupby(['model','scenario','method']).delta_rmse_mps.agg(['mean','min','max','std']).reset_index()
    summary['n_seeds']=len(SEEDS)
    summary.to_csv(out/'heldout_monte_carlo_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5.3))
    settings=[('injected_faults','robust'),('combined_degradation','fde'),('combined_degradation','robust')]
    x=np.arange(len(SEEDS))
    for sc,method in settings:
      for model in ('independent','persistent'):
        group=paired[(paired.scenario==sc)&(paired.method==method)&(paired.model==model)].set_index('seed').reindex(SEEDS)
        ax.plot(x,group.delta_rmse_mps,marker='o',label=f'{method}, {sc}, {model}')
    ax.axhline(0,color='gray',lw=.8)
    ax.set_xticks(x,[str(s) for s in SEEDS]);ax.set(xlabel='Independent simulation seed',
        ylabel='Matched-epoch speed RMSE difference vs WLS (m/s)',
        title='Held-out NFL games: Monte Carlo repeatability (negative favors QC)')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'heldout_monte_carlo_repeatability.png',dpi=180);plt.close(fig)
    metadata={'seed_count':len(SEEDS),'seeds':SEEDS,'heldout_games':sorted(test_games),
              'heldout_trajectories':len(tracks),
              'fixed_fde_standardized_residual_threshold':5.0,
              'fixed_huber_tuning_k':3.5,
              'notes':['Multiple seeds reduce Monte Carlo dependence but not model bias.',
                       'Same held-out games reused at all seeds; sampling realizations change.',
                       'All comparisons use identical accepted epochs within each pair.',
                       'Satellite errors are measurement residual proxies, not known ground truth.']}
    (out/'phase3_repeatability_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2));print(summary.to_string(index=False))
    return summary

if __name__=='__main__':run()
