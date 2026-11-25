"""Final held-out NFL satellite-level velocity simulation and high-speed event study.

All NFL GNSS observations are synthetic; actual player kinematics are 10 Hz
NFL tracking data. External watch velocity ground truth remains unavailable.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from phase2_validation import prepare
from phase2_nfl_simulation import select_trajectories, matrix_rotation, solve
from phase3_temporal_sensitivity import simulate_track, fde_adjustable, robust_adjustable
from phase5_event_integration import plot_results, plot_failure_breakdown
from phase6_event_metrics import summarize_fast
from phase6_event_metrics import vectorized_events

SEEDS=(272,847,1927,3117,9872,11234)
SCENARIOS={
    'observed_like': dict(noise=1.,fault_rate=0.,block=0.),
    'injected_faults': dict(noise=1.,fault_rate=.05,block=0.),
    'combined_degradation': dict(noise=2.,fault_rate=.05,block=.30),
}
MODELS=('independent','persistent')
METHODS=('wls','fde','robust')


def choose_tracks(root:Path,negative_per_game:int=55):
    split=json.loads((root/'results/phase3/phase3_split.json').read_text())
    heldout=set(split['test_games'])
    df=pd.read_csv(root/'data/input_2023_w01.csv',usecols=['game_id','play_id','nfl_id','frame_id','x','y','s','dir'])
    df=df[df.game_id.isin(heldout)].copy()
    df=df.sort_values(['game_id','play_id','nfl_id','frame_id'])
    lengths=df.groupby(['game_id','play_id','nfl_id'],sort=False).agg(frames=('s','size'),max_speed=('s','max'),first_frame=('frame_id','min'),last_frame=('frame_id','max')).reset_index()
    lengths=lengths[(lengths.frames>=23)&((lengths.last_frame-lengths.first_frame+1)==lengths.frames)].copy()
    lengths['is_high']=lengths.max_speed*0.9144>=7.0
    # All high-speed tracks, plus a reproducible, game-stratified background set.
    chosen=[]
    for game,g in lengths.groupby('game_id',sort=True):
        highs=g[g.is_high]
        lows=g[~g.is_high]
        sampler=np.random.default_rng(int(game)+2026)
        keep=lows.iloc[sampler.choice(len(lows),min(negative_per_game,len(lows)),replace=False)]
        chosen.append(pd.concat([highs,keep]))
    keys=pd.concat(chosen)[['game_id','play_id','nfl_id']]
    sample=df.merge(keys,on=['game_id','play_id','nfl_id'],validate='many_to_one')
    tracks=[]
    for key,seg in sample.groupby(['game_id','play_id','nfl_id'],sort=True):
        seg=seg.sort_values('frame_id')
        theta=np.deg2rad(seg['dir'].to_numpy(float))
        speeds=seg['s'].to_numpy(float)*0.9144
        vel=np.column_stack((speeds*np.sin(theta),speeds*np.cos(theta),np.zeros(len(speeds))))
        tracks.append(dict(key=tuple(int(v) for v in key),frames=seg.frame_id.to_numpy(int),speed=speeds,vel_enu=vel))
    cohort=dict(heldout_games=sorted(heldout),candidate_eligible_tracks=int(len(lengths)),
                included_tracks=len(tracks),included_high_speed_tracks=int(sum(t['speed'].max()>=7 for t in tracks)),
                included_non_high_speed_tracks=int(sum(t['speed'].max()<7 for t in tracks)),
                background_sampling='At most 55 non-high-speed tracks per game chosen with fixed seed; all eligible >=7m/s tracks included.',
                frame_filter='At least 23 consecutive frames; original 10 Hz NFL pre-pass tracking only.')
    return tracks,cohort


def generate(root:Path,out:Path,seeds=SEEDS, negative_per_game=55):
    out.mkdir(exist_ok=True,parents=True)
    tracks,cohort=choose_tracks(root,negative_per_game)
    real=prepare(root,root/'results/phase2')
    rot=matrix_rotation(real.epoch_info)
    rows=[]; tick=time.monotonic(); accepted_count=0
    for seed in seeds:
        for model_idx,model in enumerate(MODELS):
            for scenario_idx,(scenario_name,scenario) in enumerate(SCENARIOS.items()):
                for t in tracks:
                    game,play,player=t['key']
                    # Sample phase and satellite geometry identical across error models and scenarios.
                    phase_rng=np.random.default_rng(np.random.SeedSequence([seed,game,play,player,731]))
                    phase=int(phase_rng.integers(0,10))
                    template=real.templates[int(phase_rng.integers(len(real.templates)))]
                    err_rng=np.random.default_rng(np.random.SeedSequence([seed,game,play,player,model_idx,scenario_idx,932]))
                    for idx,H,y,sig,faults in simulate_track(err_rng,t,template,rot,real.bank,scenario,model,phase):
                        baseline=solve(H,y,sig) if len(y)>=4 else None
                        common=(len(y)>=6 and baseline is not None and baseline['formal_sigma']<=4.)
                        if common:
                            fde,removed=fde_adjustable(H,y,sig,z_cut=5.0)
                            robust=robust_adjustable(H,y,sig,huber_k=3.5)
                        else:
                            fde=robust=None
                            removed=[]
                        for method,sol in (('wls',baseline),('fde',fde),('robust',robust)):
                            accepted=False;speed=np.nan
                            if common and sol is not None:
                                speed=float(np.linalg.norm((rot@sol['x'][:3])[:2]))
                                accepted=bool(sol['dof']>0 and sol['formal_sigma']<=4. and
                                    np.isfinite(speed) and speed<=15.)
                            rows.append((seed,model,scenario_name,game,play,player,int(t['frames'][idx]),method,accepted,
                                speed if accepted else np.nan,float(t['speed'][idx]),len(y),
                                float(baseline['chi2']) if baseline is not None else np.nan,
                                int(baseline['dof']) if baseline is not None else 0,
                                float(baseline['formal_sigma']) if baseline is not None else np.nan,
                                int(faults.sum()),len(removed) if method=='fde' else 0))
                            accepted_count+=accepted
                print(f'Seed {seed}; {model}; {scenario_name}: {len(rows):,} method-epoch rows total; {time.monotonic()-tick:.1f}s',flush=True)
    cols=['seed','model','scenario','game','play','player','frame','method','accepted','est_speed_mps',
          'true_speed_mps','n_satellites','baseline_chi2','baseline_dof','baseline_formal_sigma',
          'deliberately_faulted_satellites','fde_removed_satellites']
    df=pd.DataFrame.from_records(rows,columns=cols)
    df.to_csv(out/'heldout_expanded_epoch_results.csv',index=False)
    (out/'expanded_cohort.json').write_text(json.dumps({**cohort,'seeds':list(seeds),
        'scenarios':SCENARIOS,'models':list(MODELS),'method_settings':
        'WLS, FDE z=5, robust Huber k=3.5; common >=6 satellites, formal sigma <=4 m/s, speed <=15 m/s.'},indent=2))
    return df,cohort


def run(root:Path=None,seeds=SEEDS,negative_per_game=55):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/final'
    epoch,cohort=generate(root,out,seeds,negative_per_game)
    ev,matched,epstats,seedstats,summary=vectorized_events(root,epoch,threshold=7.0)
    ev.to_csv(out/'reference_events.csv',index=False)
    matched.to_csv(out/'event_outcomes.csv',index=False)
    epstats.to_csv(out/'epoch_classification_counts.csv',index=False)
    seedstats.to_csv(out/'seed_level_results.csv',index=False)
    summary.to_csv(out/'main_method_comparison.csv',index=False)
    for kind in MODELS: plot_results(summary,out,kind)
    plot_failure_breakdown(summary,out)
    # Sensitivities reuse exactly the same simulated satellite observations.
    sensitivity=[]
    for threshold in (5.5,7.0):
        if threshold==7.0:
            base_events,base_rows,base_epochs=ev,matched,epstats
        else:
            base_events,base_rows,base_epochs,_,_=vectorized_events(root,epoch,threshold=threshold)
        for minimum in (1,3,5):
            filtered=base_rows[base_rows.duration_s >= minimum/10.-1e-8]
            if filtered.empty:continue
            _,s=summarize_fast(filtered,base_epochs)
            s['speed_threshold_mps']=threshold
            s['minimum_duration_s']=minimum/10
            sensitivity.append(s)
    pd.concat(sensitivity,ignore_index=True).to_csv(out/'event_definition_sensitivity.csv',index=False)
    print('Cohort',json.dumps(cohort))
    print('Reference events',len(ev),'by game',ev.groupby('game').size().to_dict())
    print(summary[['model','scenario','method','event_recall_pct','mean_epoch_availability_pct',
                   'mean_speed_rmse_on_accepted_mps','mean_false_alerts_per_100_negative_samples']].to_string(index=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=len(SEEDS),help='First N fixed independent simulation seeds (1-6)')
    parser.add_argument('--background-per-game',type=int,default=55)
    args=parser.parse_args()
    run(seeds=SEEDS[:args.seeds],negative_per_game=args.background_per_game)
