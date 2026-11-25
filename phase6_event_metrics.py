"""Vectorized event detection and error/availability accounting.

High-speed events come exclusively from real NFL 10 Hz reference speeds.
NFL GNSS observations and estimated velocities remain simulation outputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from phase5_event_integration import read_reference, reference_events


def vectorized_events(root:Path, epoch:pd.DataFrame, threshold:float=7.0, minimum_frames:int=1):
    df=epoch.copy()
    gate=df[df.method.eq('wls')].copy()
    gate_ok=(gate.accepted.astype(bool) & gate.n_satellites.ge(6) & gate.baseline_formal_sigma.le(2.) &
       gate.baseline_dof.gt(0) & gate.baseline_chi2.le(chi2.ppf(.99,gate.baseline_dof.clip(lower=1))))
    gate['accepted']=gate_ok
    gate.loc[~gate_ok,'est_speed_mps']=np.nan
    gate['method']='wls_epoch_gate'
    df=pd.concat([df,gate],ignore_index=True)
    df['accepted']=df.accepted.astype(bool)
    idcols=['seed','model','scenario','game','play','player','frame']
    assert df.groupby(idcols).method.nunique().eq(4).all()
    assert df.groupby(idcols).size().eq(4).all()
    assert df.loc[~df.accepted,'est_speed_mps'].isna().all()
    ref=read_reference(root,df)
    ev=reference_events(ref,threshold,minimum_frames)
    if not len(ev):raise ValueError('No events')
    # One row per NFL 10-Hz reference frame belonging to a high-speed event.
    frame_rows=[]
    for e in ev.itertuples(index=False):
        frame_rows.extend((e.game,e.play,e.player,int(f),e.event_ordinal) for f in range(e.start_frame,e.end_frame+1))
    frames=pd.DataFrame(frame_rows,columns=['game','play','player','frame','event_ordinal'])
    tagged=df.merge(frames,on=['game','play','player','frame'],how='inner',validate='many_to_one')
    tagged['detected']=tagged.accepted & tagged.est_speed_mps.ge(threshold)
    event_id=['seed','model','scenario','method','game','play','player','event_ordinal']
    stats=tagged.groupby(event_id,sort=False).agg(
        sampled=('frame','size'),accepted=('accepted','any'),detected=('detected','any')).reset_index()
    # Cross-product ensures missing event samples are counted as not sampled.
    combos=df[['seed','model','scenario','method']].drop_duplicates().copy()
    ev_keys=ev[['game','play','player','event_ordinal','duration_s','peak_reference_mps','touches_track_edge']]
    full=combos.merge(ev_keys,how='cross').merge(stats,on=event_id,how='left',validate='one_to_one')
    full['sampled']=full.sampled.fillna(0).gt(0)
    full['accepted']=full.accepted.astype('boolean').fillna(False).astype(bool)
    full['detected']=full.detected.astype('boolean').fillna(False).astype(bool)
    full['outcome']=np.select(
       [~full.sampled,full.sampled & ~full.accepted,full.detected],
       ['not_sampled','sampled_no_valid_solution','detected'],default='sampled_not_detected')
    assert (full.detected<=full.accepted).all() and (full.accepted<=full.sampled).all()
    # Every method uses the same sampling phases within a seed/model/scenario/track.
    pivot=full.pivot(index=['seed','model','scenario','game','play','player','event_ordinal'],
                     columns='method',values='sampled')
    assert pivot.nunique(axis=1).eq(1).all()
    full=full.drop(columns=[])
    # Consistent epoch-denominator definitions and false alerts from all sampled epochs.
    df['truth_positive']=df.true_speed_mps.ge(threshold)
    df['alert']=df.accepted & df.est_speed_mps.ge(threshold)
    df['accepted_sq_error']=np.where(df.accepted,(df.est_speed_mps-df.true_speed_mps)**2,0.)
    df['reference_negative']=(~df.truth_positive).astype(int)
    df['false_alert']=(df.alert & ~df.truth_positive).astype(int)
    epstats=df.groupby(['seed','model','scenario','method','game','play','player'],sort=False).agg(
         sampled_epochs=('frame','size'),accepted_epochs=('accepted','sum'),
         reference_negative_epochs=('reference_negative','sum'),
         false_high_speed_epochs=('false_alert','sum'),
         positive_predictions=('alert','sum'),
         accepted_sq_error_sum=('accepted_sq_error','sum')).reset_index()
    seeds,pooled=summarize_fast(full,epstats)
    return ev,full,epstats,seeds,pooled


def summarize_fast(events_out:pd.DataFrame, sampled_out:pd.DataFrame):
    keys=['seed','model','scenario','method']
    e=events_out.assign(not_sampled=events_out.outcome.eq('not_sampled').astype(int),
         no_valid_solution=events_out.outcome.eq('sampled_no_valid_solution').astype(int),
         sampled_not_detected=events_out.outcome.eq('sampled_not_detected').astype(int))
    g=e.groupby(keys,sort=True).agg(reference_events=('detected','size'),
       sampled_events=('sampled','sum'),detected_events=('detected','sum'),
       not_sampled=('not_sampled','sum'),no_valid_solution=('no_valid_solution','sum'),
       sampled_not_detected=('sampled_not_detected','sum')).reset_index()
    r=sampled_out.groupby(keys,sort=True)[['sampled_epochs','accepted_epochs',
       'reference_negative_epochs','false_high_speed_epochs','accepted_sq_error_sum']].sum().reset_index()
    seeds=g.merge(r,on=keys,validate='one_to_one')
    seeds['sample_ceiling_pct']=100*seeds.sampled_events/seeds.reference_events
    seeds['event_recall_pct']=100*seeds.detected_events/seeds.reference_events
    seeds['recall_given_sampled_pct']=100*seeds.detected_events/seeds.sampled_events.clip(lower=1)
    seeds['accepted_epoch_pct']=100*seeds.accepted_epochs/seeds.sampled_epochs.clip(lower=1)
    seeds['false_alerts_per_100_negative_samples']=100*seeds.false_high_speed_epochs/seeds.reference_negative_epochs.clip(lower=1)
    seeds['speed_rmse_on_accepted_mps']=np.sqrt(seeds.accepted_sq_error_sum/seeds.accepted_epochs.clip(lower=1))
    group=['model','scenario','method']
    grouped=seeds.groupby(group,sort=True)
    pooled=grouped.agg(runs=('seed','size'),reference_event_trials=('reference_events','sum'),
       sampled_event_trials=('sampled_events','sum'),detected_event_trials=('detected_events','sum'),
       total_not_sampled=('not_sampled','sum'),total_no_valid=('no_valid_solution','sum'),
       total_sampled_not_detected=('sampled_not_detected','sum'),
       event_recall_seed_min_pct=('event_recall_pct','min'),
       event_recall_seed_max_pct=('event_recall_pct','max'),
       mean_epoch_availability_pct=('accepted_epoch_pct','mean'),
       mean_speed_rmse_on_accepted_mps=('speed_rmse_on_accepted_mps','mean'),
       mean_false_alerts_per_100_negative_samples=('false_alerts_per_100_negative_samples','mean'),
       total_false_high_speed_epochs=('false_high_speed_epochs','sum')).reset_index()
    denom=pooled.reference_event_trials
    pooled['event_recall_pct']=100*pooled.detected_event_trials/denom
    pooled['sampling_ceiling_pct']=100*pooled.sampled_event_trials/denom
    pooled['sampling_loss_pct']=100*pooled.total_not_sampled/denom
    pooled['no_valid_solution_pct']=100*pooled.total_no_valid/denom
    pooled['sampled_not_detected_pct']=100*pooled.total_sampled_not_detected/denom
    pooled=pooled.drop(columns=['total_not_sampled','total_no_valid','total_sampled_not_detected'])
    return seeds,pooled
