from pathlib import Path
from phase2_validation import prepare
from phase2_nfl_simulation import run_simulation

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    out=root/'results/phase2'
    data=prepare(root,out)
    print('Real GNSS analysis:', data.validation['watch_epoch_count'], 'epochs; empirical residual proxies:', data.validation['empirical_residual_count'],flush=True)
    metrics,meta=run_simulation(root,out,data)
    print('NFL simulation:',meta['selected_tracks'],'trajectories,',meta['total_simulated_gnss_cases'],'simulated GNSS epochs',flush=True)
    print(metrics.to_string(index=False))
