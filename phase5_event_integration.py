"""Phase 5: joint 10-Hz NFL event and simulated 1-Hz Doppler QC evaluation.

This analysis uses precomputed Phase 3 held-out Monte Carlo observations. Every
NFL satellite measurement remains simulated; no measured NFL GNSS is implied.
Event truth is a contiguous 10-Hz NFL high-speed interval. Events are identified
only if a sampled, accepted GNSS speed crosses the same threshold within the
reference event. This is a conservative, instantaneous event-detection definition.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

METERS_PER_YARD=0.9144
SIM_METHODS=('wls','fde','robust','wls_epoch_gate')


def reference_events(tracks:pd.DataFrame, threshold:float, minimum_frames:int=1)->pd.DataFrame:
    events=[]
    for (game,play,player),g in tracks.groupby(['game_id','play_id','nfl_id'],sort=True):
        g=g.sort_values('frame_id')
        frames=g.frame_id.to_numpy(dtype=int)
        speed=g.s.to_numpy(float)*METERS_PER_YARD
        flagged=(speed>=threshold)
        # A discontinuity in frame ID separates otherwise consecutive intervals.
        starts=np.flatnonzero(flagged & ~np.r_[False,flagged[:-1] & (np.diff(frames)==1)])
        ends=np.flatnonzero(flagged & ~np.r_[flagged[1:] & (np.diff(frames)==1),False])
        for ordinal,(start,end) in enumerate(zip(starts,ends)):
            n=int(end-start+1)
            if n<minimum_frames:continue
            events.append(dict(game=int(game),play=int(play),player=int(player),
                event_ordinal=int(ordinal),start_frame=int(frames[start]),end_frame=int(frames[end]),
                duration_s=n/10.,peak_reference_mps=float(speed[start:end+1].max()),
                touches_track_edge=bool(start==0 or end==len(frames)-1)))
    return pd.DataFrame(events,columns=['game','play','player','event_ordinal','start_frame',
        'end_frame','duration_s','peak_reference_mps','touches_track_edge'])


def read_reference(root:Path, simulated:pd.DataFrame)->pd.DataFrame:
    cols=['game_id','play_id','nfl_id','frame_id','s']
    raw=pd.read_csv(root/'data/input_2023_w01.csv',usecols=cols)
    keys=simulated[['game','play','player']].drop_duplicates()
    tracks=raw.merge(keys,left_on=['game_id','play_id','nfl_id'],
                     right_on=['game','play','player'],how='inner',validate='many_to_one')
    return tracks[cols].copy()


def analyze(root:Path, threshold=7.0, min_frames=1, data=None):
    src=(root/'results/phase3/heldout_monte_carlo_epoch_results.csv')
    df=pd.read_csv(src) if data is None else data.copy()
    df['accepted']=df.accepted.astype(bool)
    required={'seed','model','scenario','game','play','player','frame','method','accepted','est_speed_mps','true_speed_mps'}
    assert required.issubset(set(df.columns)), 'Missing expected Phase 3 columns'
    assert set(df.method)=={'wls','fde','robust'}, 'Phase 3 method list changed'
    needed={'n_satellites','baseline_chi2','baseline_dof','baseline_formal_sigma'}
    assert needed.issubset(df.columns),'Re-run phase3_repeatability.py to get baseline diagnostics for gate'
    gate=df[df.method=='wls'].copy()
    good=gate.accepted & (gate.n_satellites>=6) & (gate.baseline_formal_sigma<=2.0) & \
        (gate.baseline_dof>0) & (gate.baseline_chi2<=chi2.ppf(.99,gate.baseline_dof.clip(lower=1)))
    gate['accepted']=good
    gate.loc[~good,'est_speed_mps']=np.nan
    gate['method']='wls_epoch_gate'
    df=pd.concat([df,gate],ignore_index=True)
    idcols=['seed','model','scenario','game','play','player','frame']
    counts=df.groupby(idcols).method.agg(['nunique','count'])
    assert bool(((counts['count']==4)&(counts['nunique']==4)).all()),'Each sampled epoch must have each of four methods exactly once'
    assert df.loc[~df.accepted,'est_speed_mps'].isna().all(),'Rejected estimates must be missing'
    refs=read_reference(root,df)
    events=reference_events(refs,threshold,min_frames)
    if len(events)==0:
        raise ValueError('No reference events at specified threshold/duration')
    # Independent sample phases across scenario/seeds, but common phases for
    # the three algorithms inside each scenario/seed/track.
    reference_lookup={tuple(map(int,k)):g.sort_values('frame_id').set_index('frame_id').s*METERS_PER_YARD
                      for k,g in refs.groupby(['game_id','play_id','nfl_id'])}
    by_track_events={tuple(map(int,k)):g for k,g in events.groupby(['game','play','player'])}
    records=[]; epo=[]
    for (seed,model,scenario),block in df.groupby(['seed','model','scenario'],sort=True):
        for (game,play,player),t in block.groupby(['game','play','player'],sort=True):
            key=(int(game),int(play),int(player))
            truth=reference_lookup[key]
            base=t[t.method=='wls'].sort_values('frame')
            selected=base.frame.to_numpy(int)
            assert np.all(np.isin(selected,truth.index.to_numpy()))
            assert np.max(np.abs(base.true_speed_mps.to_numpy()-truth.loc[selected].to_numpy()))<1e-8
            for method,sub in t.groupby('method'):
                sub=sub.sort_values('frame')
                assert np.array_equal(selected,sub.frame.to_numpy(int)),f'Sampling not paired: {key}'
                accepted=sub.accepted.to_numpy(bool)
                estimates=sub.est_speed_mps.to_numpy(float)
                detected=accepted & (estimates>=threshold)
                real_positive=(truth.loc[selected].to_numpy()>=threshold)
                # False detection is a sampled instantaneous alert during
                # reference-negative motion, not an entire invented NFL event.
                epo.append(dict(seed=int(seed),model=model,scenario=scenario,method=method,
                    game=key[0],play=key[1],player=key[2],sampled_epochs=len(selected),
                    accepted_epochs=int(accepted.sum()),reference_negative_epochs=int((~real_positive).sum()),
                    false_high_speed_epochs=int(np.sum(detected & ~real_positive)),
                    positive_predictions=int(detected.sum()),
                    accepted_sq_error_sum=float(np.sum((estimates[accepted]-truth.loc[selected].to_numpy()[accepted])**2))))
                for ev in (by_track_events[key].itertuples(index=False) if key in by_track_events else []):
                    in_event=(selected>=ev.start_frame)&(selected<=ev.end_frame)
                    sampled=bool(in_event.any())
                    kept=bool((in_event & accepted).any())
                    hit=bool((in_event & detected).any())
                    outcome=('not_sampled' if not sampled else
                             'sampled_no_valid_solution' if not kept else
                             'detected' if hit else 'sampled_not_detected')
                    records.append(dict(seed=int(seed),model=model,scenario=scenario,
                        method=method,game=key[0],play=key[1],player=key[2],
                        event_ordinal=int(ev.event_ordinal),duration_s=float(ev.duration_s),
                        peak_reference_mps=float(ev.peak_reference_mps),
                        touches_track_edge=bool(ev.touches_track_edge),
                        sampled=sampled,accepted=kept,detected=hit,outcome=outcome))
    events_out=pd.DataFrame(records)
    sampled_out=pd.DataFrame(epo)
    assert len(events_out)==len(events)*df[['seed','model','scenario']].drop_duplicates().shape[0]*4
    assert bool(((events_out.outcome=='detected')==events_out.detected).all())
    assert bool(((events_out.outcome=='not_sampled')==~events_out.sampled).all())
    assert bool((events_out.detected<=events_out.accepted).all() and
                (events_out.accepted<=events_out.sampled).all())
    for keys,x in events_out.groupby(['seed','model','scenario']):
        pivot=x.pivot(index=['game','play','player','event_ordinal'],columns='method',values='sampled')
        assert (pivot.nunique(axis=1)==1).all(),'The algorithms must have the same sampling phases'

    seed_summary,result=summarize_events(events_out,sampled_out)
    return events,events_out,sampled_out,seed_summary,result


def summarize_events(events_out,sampled_out):
    seed_summary=[]
    for (seed,model,scenario,method),g in events_out.groupby(['seed','model','scenario','method']):
        r=sampled_out[(sampled_out.seed==seed)&(sampled_out.model==model)&
                      (sampled_out.scenario==scenario)&(sampled_out.method==method)]
        n=len(g);samples=int(r.sampled_epochs.sum());accepted=int(r.accepted_epochs.sum())
        se=float(r.accepted_sq_error_sum.sum()); neg=int(r.reference_negative_epochs.sum())
        seed_summary.append(dict(seed=int(seed),model=model,scenario=scenario,method=method,
          reference_events=n,sampled_events=int(g.sampled.sum()),detected_events=int(g.detected.sum()),
          not_sampled=int((g.outcome=='not_sampled').sum()),
          no_valid_solution=int((g.outcome=='sampled_no_valid_solution').sum()),
          sampled_not_detected=int((g.outcome=='sampled_not_detected').sum()),
          sample_ceiling_pct=100*float(g.sampled.mean()),
          event_recall_pct=100*float(g.detected.mean()),
          recall_given_sampled_pct=100*float(g.detected.sum())/max(1,int(g.sampled.sum())),
          accepted_epoch_pct=100*accepted/max(1,samples),
          false_alerts_per_100_negative_samples=100*int(r.false_high_speed_epochs.sum())/max(1,neg),
          false_high_speed_epochs=int(r.false_high_speed_epochs.sum()),
          speed_rmse_on_accepted_mps=np.sqrt(se/max(1,accepted))))
    seed_summary=pd.DataFrame(seed_summary)
    pooled=[]
    for keys,g in seed_summary.groupby(['model','scenario','method']):
        pool=dict(zip(['model','scenario','method'],keys))
        n=int(g.reference_events.sum());sm=int(g.sampled_events.sum());hits=int(g.detected_events.sum())
        pool.update(runs=len(g),reference_event_trials=n,sampled_event_trials=sm,
            detected_event_trials=hits,event_recall_pct=100*hits/n,
            sampling_ceiling_pct=100*sm/n,
            sampling_loss_pct=100*int(g.not_sampled.sum())/n,
            no_valid_solution_pct=100*int(g.no_valid_solution.sum())/n,
            sampled_not_detected_pct=100*int(g.sampled_not_detected.sum())/n,
            event_recall_seed_min_pct=float(g.event_recall_pct.min()),
            event_recall_seed_max_pct=float(g.event_recall_pct.max()),
            mean_epoch_availability_pct=float(g.accepted_epoch_pct.mean()),
            mean_speed_rmse_on_accepted_mps=float(g.speed_rmse_on_accepted_mps.mean()),
            mean_false_alerts_per_100_negative_samples=float(g.false_alerts_per_100_negative_samples.mean()),
            total_false_high_speed_epochs=int(g.false_high_speed_epochs.sum()))
        pooled.append(pool)
    result=pd.DataFrame(pooled).sort_values(['model','scenario','method']).reset_index(drop=True)
    return seed_summary,result



def plot_results(summary,out,model):
    d=summary[summary.model==model].copy()
    scens=['observed_like','injected_faults','combined_degradation']
    meths=['wls','fde','robust','wls_epoch_gate']
    x=np.arange(len(scens));width=.18
    fig,ax=plt.subplots(figsize=(9.5,5.1))
    for i,method in enumerate(meths):
        y=[d[(d.scenario==s)&(d.method==method)].event_recall_pct.iloc[0] for s in scens]
        ax.bar(x+(i-1.5)*width,y,width=width,label={'wls':'WLS','fde':'FDE','robust':'Robust','wls_epoch_gate':'WLS + epoch rejection'}[method])
    ceiling=[d[(d.scenario==s)&(d.method=='wls')].sampling_ceiling_pct.iloc[0] for s in scens]
    ax.plot(x,ceiling,marker='D',ls='--',label='Ideal 1 Hz sampling ceiling')
    ax.set_xticks(x,[s.replace('_','\n') for s in scens]);ax.set_ylim(0,104)
    ax.set_ylabel('Reference high-speed episodes detected (%)')
    ax.set_title(f'End-to-end NFL event detection: {model} simulated GNSS errors')
    ax.legend(fontsize=9,ncol=2);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/f'phase5_event_detection_{model}.png',dpi=180);plt.close(fig)


def plot_failure_breakdown(summary,out,model='persistent',scenario='combined_degradation'):
    d=summary[(summary.model==model)&(summary.scenario==scenario)].set_index('method')
    meths=['wls','fde','robust','wls_epoch_gate'];fig,ax=plt.subplots(figsize=(10.2,5.1))
    bottom=np.zeros(len(meths))
    for col,label in [('sampling_loss_pct','Never sampled'),('no_valid_solution_pct','Sampled; no accepted solution'),
                      ('sampled_not_detected_pct','Accepted; below threshold'),('event_recall_pct','Detected')]:
        vals=d.loc[meths,col].to_numpy(float)
        ax.bar(meths,vals,bottom=bottom,label=label)
        bottom+=vals
    assert np.max(np.abs(bottom-100))<1e-8
    ax.set(ylim=(0,105),ylabel='Share of reference high-speed episodes (%)',
           title=f'Why 1 Hz tracking misses events: {model} / {scenario.replace("_"," ")}')
    ax.set_xticks(range(4),['WLS','FDE','Robust','WLS + epoch gate'])
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=9)
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'phase5_event_failure_breakdown.png',dpi=180,bbox_inches='tight');plt.close(fig)


def run(root=None,threshold=7.0,min_frames=1):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/phase5';out.mkdir(parents=True,exist_ok=True)
    ref,event_out,epochs,seed_summary,pooled=analyze(root,threshold,min_frames)
    ref.to_csv(out/'reference_events_10hz.csv',index=False)
    event_out.to_csv(out/'matched_event_outcomes.csv',index=False)
    epochs.to_csv(out/'matched_epoch_error_counts.csv',index=False)
    seed_summary.to_csv(out/'seed_level_event_summary.csv',index=False)
    pooled.to_csv(out/'end_to_end_event_summary.csv',index=False)
    for model in sorted(pooled.model.unique()):plot_results(pooled,out,model)
    plot_failure_breakdown(pooled,out)
    # Sensitivity to what constitutes an event. Analysis reuses the exact same
    # GNSS epoch outputs, avoiding fresh random draws or post-hoc retuning.
    sensitivity=[]
    ev_low,_,_,_,low=analyze(root,5.5,1)
    for r in low.to_dict('records'):
        r.update(speed_threshold_mps=5.5,min_reference_frames=1)
        sensitivity.append(r)
    for minimum in (3,5):
        filtered=event_out[event_out.duration_s>=(minimum/10.-1e-8)]
        seed_sub, p=summarize_events(filtered,epochs)
        for r in p.to_dict('records'):
            r.update(speed_threshold_mps=7.0,min_reference_frames=minimum)
            sensitivity.append(r)
    pd.DataFrame(sensitivity).to_csv(out/'event_threshold_duration_sensitivity.csv',index=False)
    # Reference data provenance and limits.
    md={'primary_threshold_mps':threshold,'minimum_event_frames':min_frames,
       'real_reference_data':'NFL Big Data Bowl 2026 Analytics, 2023 Week 1 pre-pass player tracking (10 Hz)',
       'gnss_measurements':'All NFL Doppler observations, faults, outages, satellite geometry, noise are simulated. Real wearable residual proxies inform errors.',
       'samples':'Exactly one simulated GNSS measurement per second; phase randomized by seed/model/scenario/track, shared across QC methods.',
       'analysis_reference':'Held-out NFL games only; six Monte Carlo seeds, repeated simulation realizations on the same tracks.',
       'methods':'WLS, residual FDE (z_cut=5), Huber robust weighting (k=3.5) held fixed for held-out evaluation; plus pre-specified WLS epoch gate (>=6 satellites, baseline formal 3D sigma <=2 m/s and chi-square <=99th percentile). No test tuning.',
       'event_definition':'Contiguous 10 Hz NFL reference speed >= threshold. Hit requires >=1 accepted, threshold-crossing simulated GNSS speed at a sample inside that event.',
       'failure_categories':'not_sampled, sampled_no_valid_solution, sampled_not_detected, detected; mutually exclusive and exhaustive',
       'false_alert_definition':'accepted sampled GNSS speed >= threshold while NFL reference speed at same sample is below threshold; reported per 100 reference-negative samples',
       'sampled_event_ceiling':'An ideal noiseless 1Hz velocity sensor detects a high-speed event only when sampling intersects its reference interval.',
       'prohibitions':['Do not treat repeats as independent NFL games or this dataset as true GNSS.',
                       'Do not transfer SnapperGPS positioning errors in metres directly to Doppler errors in metres/second.',
                       'Do not interpret 10-Hz NFL player tracking as perfectly accurate or as a real wearable GNSS system.',
                       'Do not claim real Pixel Watch Doppler velocity RMSE without appropriate independent velocity reference.']}
    (out/'phase5_methodology.json').write_text(json.dumps(md,indent=2))
    print('Matched reference events',len(ref))
    print(pooled[['model','scenario','method','event_recall_pct','sampling_ceiling_pct','mean_epoch_availability_pct','mean_false_alerts_per_100_negative_samples']].to_string(index=False))
    return pooled


if __name__=='__main__':run()
