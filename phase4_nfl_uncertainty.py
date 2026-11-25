"""Game-cluster bootstrap uncertainty for phase-enumerated NFL event coverage.

Bootstrapping full NFL games prevents the false independence assumption that every
100-ms frame or every player episode is an independent observation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def run(event_csv,out_dir,nboot=2000,seed=272):
    out=Path(out_dir);out.mkdir(exist_ok=True,parents=True)
    events=pd.read_csv(event_csv)
    if events.empty:raise ValueError('No events found in sampling audit')
    rng=np.random.default_rng(seed)
    rows=[]
    for (threshold,mindur),subset in events.groupby(['threshold_mps','minimum_event_duration_s']):
        games=sorted(subset.game_id.unique())
        for hz in (1,2,5):
            col=f'capture_probability_{hz}hz'
            pergame=subset.groupby('game_id')[col].agg(['sum','size']).reindex(games)
            sums=pergame['sum'].to_numpy(float);counts=pergame['size'].to_numpy(float)
            original=sums.sum()/counts.sum()
            samples=rng.integers(0,len(games),size=(nboot,len(games)))
            draws=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
            low,high=np.percentile(draws,[2.5,97.5])
            rows.append({'threshold_mps':threshold,'minimum_event_duration_s':mindur,'rate_hz':hz,
                         'event_count':int(counts.sum()),'games':len(games),
                         'event_weighted_capture_fraction':float(original),
                         'game_cluster_bootstrap_low_95':float(low),'game_cluster_bootstrap_high_95':float(high)})
    report=pd.DataFrame(rows)
    report.to_csv(out/'nfl_sampling_game_bootstrap.csv',index=False)
    (out/'nfl_sampling_uncertainty_method.json').write_text(json.dumps({
        'resampling_unit':'entire NFL game', 'bootstrap_repetitions':nboot,'seed':seed,
        'interpretation':'Descriptive sampling uncertainty across these 16 games; not uncertainty of GNSS error or independent validation of NFL truth.'},indent=2))
    print(report[(report.threshold_mps==7.0)&(report.minimum_event_duration_s==0.1)].to_string(index=False))
    return report

if __name__=='__main__':
    root=Path(__file__).resolve().parent/'results/phase4'
    run(root/'nfl_high_speed_event_sampling_detail.csv',root)
