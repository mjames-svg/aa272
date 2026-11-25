"""Paired event-level checks and conservative uncertainty summaries for Phase 5."""
from pathlib import Path
import numpy as np
import pandas as pd


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/phase5'
    events=pd.read_csv(out/'matched_event_outcomes.csv')
    key=['seed','model','scenario','game','play','player','event_ordinal']
    b=events.pivot(index=key,columns='method',values='detected').reset_index()
    rows=[]
    for (model,scenario),g in b.groupby(['model','scenario']):
        for method in ('fde','robust','wls_epoch_gate'):
            w=g.wls.to_numpy(bool);alt=g[method].to_numpy(bool)
            assert len(g)>0
            rows.append(dict(model=model,scenario=scenario,method=method,
                reference_event_trials=len(g),
                recovered_by_alternative=int(np.sum((~w)&alt)),
                lost_by_alternative=int(np.sum(w&(~alt))),
                same_result=int(np.sum(w==alt)),
                net_delta_events=int(alt.sum()-w.sum()),
                delta_event_recall_percentage_points=100*float(np.mean(alt)-np.mean(w))))
    pairs=pd.DataFrame(rows)
    pairs.to_csv(out/'paired_event_differences.csv',index=False)
    # Report variability across independent Monte Carlo seeds, rather than
    # treating 6 repeats of the same games as independent field experiments.
    seed_rates=events.groupby(['seed','model','scenario','method']).detected.mean().reset_index()
    piv=seed_rates.pivot(index=['seed','model','scenario'],columns='method',values='detected').reset_index()
    spread=[]
    for (model,scenario),g in piv.groupby(['model','scenario']):
        for method in ('fde','robust','wls_epoch_gate'):
            dd=100*(g[method]-g.wls)
            spread.append(dict(model=model,scenario=scenario,method=method,
                n_seeds=len(g),mean_delta_percentage_points=float(dd.mean()),
                min_seed_delta_pp=float(dd.min()),max_seed_delta_pp=float(dd.max()),
                seeds_favoring_alternative=int((dd>0).sum()),
                seeds_favoring_wls=int((dd<0).sum()),
                ties=int((dd==0).sum())))
    spread=pd.DataFrame(spread)
    spread.to_csv(out/'paired_seed_sensitivity.csv',index=False)
    print(pairs.to_string(index=False))
    print(spread.to_string(index=False))
    return pairs,spread


if __name__=='__main__':run()
