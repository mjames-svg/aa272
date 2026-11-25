"""Temporal-correlation and threshold sensitivity experiment (simulated NFL GNSS).

Uses only observed NFL player kinematics and Pixel Watch-derived *proxy* error
statistics. Does not claim to model historical NFL-stadium radio conditions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from phase2_validation import prepare, sample_error
from phase2_nfl_simulation import select_trajectories, matrix_rotation, solve, SPEED_THRESHOLD

MAIN_SEED=20261110


def estimated_autocorrelation(path, out):
    df=pd.read_csv(path).sort_values(['sv','epoch']).copy()
    df['demeaned_by_sv']=(df.loo_prediction_residual_mps-
                          df.groupby('sv').loo_prediction_residual_mps.transform('mean'))
    lag_rows=[]
    for lag in range(1,11):
        v=df.groupby('sv').demeaned_by_sv.shift(lag)
        prev=df.groupby('sv').epoch.shift(lag)
        x=df.loc[df.epoch-prev==lag, 'demeaned_by_sv']
        y=v.loc[df.epoch-prev==lag]
        lag_rows.append(dict(lag_seconds=lag,pairs=len(x),within_satellite_centered_corr=float(x.corr(y))))
    lag_data=pd.DataFrame(lag_rows)
    lag_data.to_csv(out/'real_satellite_residual_autocorrelation.csv',index=False)
    lag1=float(lag_data.iloc[0].within_satellite_centered_corr)
    fig,ax=plt.subplots(figsize=(8.0,4.0))
    ax.plot(lag_data.lag_seconds,lag_data.within_satellite_centered_corr,'o-')
    ax.axhline(0,color='gray',lw=.9)
    ax.set(xlabel='Consecutive 1-second lag (s)',ylabel='Within-satellite residual autocorrelation',
           title='Pixel Watch: time correlation of Doppler leave-one-out residual proxies')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'real_residual_autocorrelation.png',dpi=160);plt.close(fig)
    return lag1


def fde_adjustable(H,y,sigma,z_cut=4.0,p_cut=.999):
    ix=np.arange(len(y),dtype=int)
    removed=[]
    while len(ix)>=6:
        sol=solve(H[ix],y[ix],sigma[ix])
        if sol is None: return None,removed
        j=int(np.argmax(np.abs(sol['z'])))
        if sol['chi2']<=chi2.ppf(p_cut,sol['dof']) or abs(sol['z'][j])<=z_cut:
            return sol,removed
        removed.append(int(ix[j]));ix=np.delete(ix,j)
    return solve(H[ix],y[ix],sigma[ix]) if len(ix)>=4 else None, removed


def robust_adjustable(H,y,sigma,huber_k):
    sol=solve(H,y,sigma)
    if sol is None:return None
    for _ in range(8):
        scaled=np.abs(sol['r'])/np.maximum(sigma,.15)
        factor=np.minimum(1.,huber_k/np.maximum(scaled,1e-9))
        candidate=solve(H,y,sigma,factor)
        if candidate is None:break
        diff=np.linalg.norm(candidate['x']-sol['x']);sol=candidate
        if diff<1e-6:break
    return sol


def step_markov(rng,blocked,active_fraction=.3,stay_blocked=.9):
    # Two-state Markov: target stationary probability is active_fraction.
    start_p=active_fraction*(1-stay_blocked)/(1-active_fraction)
    return np.where(blocked,rng.random(len(blocked))<stay_blocked,
                    rng.random(len(blocked))<start_p)


def simulate_track(rng,track,template,rot,bank,scenario,model,phase):
    H=template['H'];sig=template['sigma'];cn0=template['cn0']
    step=10; idx=np.arange(phase,len(track['speed']),step)
    phi=.5 if model=='persistent' else 0.
    # Same template retained along a short player trajectory instead of
    # independently sampling unrelated satellite geometries each second.
    last=np.zeros(len(sig)); blocked=rng.random(len(sig))<scenario['block']
    fault_active=rng.random(len(sig))<scenario['fault_rate']
    fault_sign=rng.choice([-1.,1.],size=len(sig))
    fault_amplitude=rng.uniform(5.,10.,size=len(sig))
    for frameidx in idx:
        if model=='persistent':
            if scenario['block']>0:
                blocked=step_markov(rng,blocked,scenario['block'],.85)
            if scenario['fault_rate']>0:
                fault_active=step_markov(rng,fault_active,scenario['fault_rate'],.7)
                start_new=fault_active&(rng.random(len(sig))<.2)
                fault_sign[start_new]=rng.choice([-1.,1.],size=int(start_new.sum()))
                fault_amplitude[start_new]=rng.uniform(5.,10.,size=int(start_new.sum()))
        else:
            blocked=rng.random(len(sig))<scenario['block']
            fault_active=rng.random(len(sig))<scenario['fault_rate']
            fault_sign=rng.choice([-1.,1.],size=len(sig))
            fault_amplitude=rng.uniform(5.,10.,size=len(sig))
        innovation=sample_error(rng,cn0,bank)
        last=phi*last+np.sqrt(1-phi*phi)*innovation
        if scenario['noise']!=1:noise=scenario['noise']*last
        else:noise=last.copy()
        noise+=np.where(fault_active,fault_sign*fault_amplitude,0)
        keep=~blocked
        truevel=track['vel_enu'][frameidx];state=np.r_[rot.T@truevel,0.]
        yield frameidx,H[keep],H[keep]@state+noise[keep],sig[keep],fault_active[keep]


def build_evaluation(root,out,real,tracks,reference_seed=MAIN_SEED):
    out.mkdir(parents=True,exist_ok=True)
    # Different GAMES—not just different frames of the same play—are held out.
    games=sorted(set(int(t['key'][0]) for t in tracks))
    rngsplit=np.random.default_rng(reference_seed)
    shuffled=np.asarray(games)[rngsplit.permutation(len(games))]
    ntest=max(3,round(len(games)*.30));test_games=set(int(g) for g in shuffled[:ntest])
    train_games=set(games)-test_games
    rot=matrix_rotation(real.epoch_info)
    scenarios={'observed_like':dict(noise=1.,fault_rate=0.,block=0.),
               'injected_faults':dict(noise=1.,fault_rate=.05,block=0.),
               'combined_degradation':dict(noise=2.,fault_rate=.05,block=.30)}
    settings=[('wls',None)]+[('fde',x) for x in (3.,4.,5.)]+[('robust',k) for k in (1.2,2.2,3.5)]
    records=[]
    run_id=0
    for simkind in ('independent','persistent'):
        for name,scenario in scenarios.items():
            for trial in range(2):
                # Split RNG by scenario and trial (repeatability, no parameter look-ahead).
                rng=np.random.default_rng(reference_seed+run_id*1009);run_id+=1
                for t in tracks:
                    phase=int(rng.integers(10))
                    template=real.templates[int(rng.integers(len(real.templates)))]
                    split='test' if t['key'][0] in test_games else 'train'
                    group_id=f'{trial}_{t["key"][0]}_{t["key"][1]}_{t["key"][2]}'
                    for idx,H,y,sigma,known_faults in simulate_track(rng,t,template,rot,real.bank,scenario,simkind,phase):
                        base=solve(H,y,sigma) if len(y)>=4 else None
                        common=(len(y)>=6 and base is not None and base['formal_sigma']<=4)
                        for method,parameter in settings:
                            if method=='wls':sol=base;excluded=[]
                            elif not common:sol=None;excluded=[]
                            elif method=='fde':sol,excluded=fde_adjustable(H,y,sigma,z_cut=parameter)
                            else:sol=robust_adjustable(H,y,sigma,parameter);excluded=[]
                            accepted=False;estimate=np.nan
                            if common and sol is not None:
                                v=rot@sol['x'][:3]
                                estimate=float(np.linalg.norm(v[:2]))
                                accepted=(sol['dof']>0 and sol['formal_sigma']<=4 and
                                          np.isfinite(estimate) and estimate<=15.)
                            records.append(dict(model=simkind,scenario=name,split=split,trial=trial,
                               group_id=group_id,game=t['key'][0],play=t['key'][1],player=t['key'][2],
                               frame=int(t['frames'][idx]),method=method,parameter=parameter if parameter is not None else 0.,
                               n_satellites=len(y),true_speed_mps=float(t['speed'][idx]),
                               accepted=bool(accepted),estimated_speed_mps=estimate if accepted else np.nan,
                               n_deliberate_faults=int(known_faults.sum()),
                               fde_exclusions=len(excluded),fde_true_faults_removed=int(np.sum(known_faults[excluded])) if len(excluded) else 0))
    a=pd.DataFrame(records)
    a.to_csv(out/'nfl_temporal_threshold_epoch_results.csv',index=False)
    summ=[]
    for group,seg in a.groupby(['model','scenario','split','method','parameter'],sort=True):
        good=seg[seg.accepted]
        err=good.estimated_speed_mps-good.true_speed_mps
        # unavailable high-speed epochs are treated as missed (not as false negatives on accepted-only)
        true_positive=int(((good.true_speed_mps>=SPEED_THRESHOLD)&(good.estimated_speed_mps>=SPEED_THRESHOLD)).sum())
        truth_pos=int((seg.true_speed_mps>=SPEED_THRESHOLD).sum())
        false_positive=int(((good.true_speed_mps<SPEED_THRESHOLD)&(good.estimated_speed_mps>=SPEED_THRESHOLD)).sum())
        summ.append(dict(zip(['model','scenario','split','method','parameter'],group),
                         total_epochs=len(seg),accepted_epochs=len(good),availability_pct=100*len(good)/len(seg),
                         speed_rmse_accepted_mps=float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
                         high_speed_recall_including_unavailable=true_positive/truth_pos if truth_pos else np.nan,
                         false_high_speed_epochs=false_positive,
                         fde_exclusions=int(seg.fde_exclusions.sum()),
                         fde_known_faults_removed=int(seg.fde_true_faults_removed.sum()),
                         injected_faults=int(seg.n_deliberate_faults.sum())))
    summary=pd.DataFrame(summ)
    summary.to_csv(out/'nfl_temporal_threshold_method_summary.csv',index=False)
    (out/'phase3_split.json').write_text(json.dumps(dict(train_games=sorted(train_games),test_games=sorted(test_games),
                                  samples='1Hz downsampling of 10Hz NFL trajectories; two independent trials',
                                  time_correlation_phi=.5,
                                  p_stay_faulted=.7,p_stay_blocked=.85,
                                  note='Independent vs persistent are sensitivity models, not field-validated NFL conditions; parameter grids chosen in advance.'),indent=2))
    return a,summary


def evaluate_tuning(df,summary,out):
    # Select one threshold per method on training GAMES across all scenarios,
    # requiring >=95% of aggregate WLS availability when possible.
    # Held-out games have no role in threshold selection.
    chosen=[]
    global_params={}
    for model in sorted(summary.model.unique()):
        training=df[(df.split=='train') & (df.model==model)]
        wls_training=training[training.method=='wls']
        baseline_coverage=float(wls_training.accepted.mean())
        for method in ('fde','robust'):
            scores=[]
            for param,group in training[training.method==method].groupby('parameter'):
                valid=group[group.accepted]
                err=valid.estimated_speed_mps-valid.true_speed_mps
                rmse=float(np.sqrt(np.mean(err**2))) if len(valid) else np.inf
                scores.append(dict(parameter=float(param),coverage=float(group.accepted.mean()),rmse=rmse))
            candidates=pd.DataFrame(scores)
            feasible=candidates[candidates.coverage>=.95*baseline_coverage]
            if len(feasible):
                best=feasible.sort_values('rmse').iloc[0]
            else:
                best=candidates.sort_values(['coverage','rmse'],ascending=[False,True]).iloc[0]
            global_params[(model,method)]=float(best.parameter)
    for model in sorted(summary.model.unique()):
      for scen in sorted(summary.scenario.unique()):
        train=summary[(summary.model==model)&(summary.scenario==scen)&(summary.split=='train')]
        base=train[train.method=='wls'].iloc[0]
        for method in ('fde','robust'):
            best=train[(train.method==method)&(train.parameter==global_params[(model,method)])].iloc[0]
            test=summary[(summary.model==model)&(summary.scenario==scen)&(summary.split=='test')&
                (summary.method==method)&(summary.parameter==best.parameter)].iloc[0]
            base_test=summary[(summary.model==model)&(summary.scenario==scen)&(summary.split=='test')&
                (summary.method=='wls')].iloc[0]
            chosen.append(dict(model=model,scenario=scen,method=method,selected_parameter=float(best.parameter),
                training_rmse_mps=float(best.speed_rmse_accepted_mps),
                training_availability_pct=float(best.availability_pct),
                test_speed_rmse_mps=float(test.speed_rmse_accepted_mps),
                test_availability_pct=float(test.availability_pct),
                test_wls_rmse_mps=float(base_test.speed_rmse_accepted_mps),
                test_wls_availability_pct=float(base_test.availability_pct)))
    selection=pd.DataFrame(chosen)
    selection.to_csv(out/'train_selected_parameters_heldout_results.csv',index=False)
    # Paired RMSE differences on same accepted epochs, cluster-bootstrap by player trajectory.
    rng=np.random.default_rng(MAIN_SEED+9)
    comparison=[]
    for p in selection.itertuples(index=False):
        block=df[(df.split=='test')&(df.model==p.model)&(df.scenario==p.scenario)]
        base=block[block.method=='wls'].set_index(['group_id','frame'])
        alt=block[(block.method==p.method)&(block.parameter==p.selected_parameter)].set_index(['group_id','frame'])
        pair=base.join(alt,how='inner',lsuffix='_wls',rsuffix='_alt')
        pair=pair[pair.accepted_wls & pair.accepted_alt].copy()
        if len(pair)==0:continue
        pair['sq_wls']=(pair.estimated_speed_mps_wls-pair.true_speed_mps_wls)**2
        pair['sq_alt']=(pair.estimated_speed_mps_alt-pair.true_speed_mps_alt)**2
        clusters=pair.reset_index().groupby('game_wls').agg(n=('sq_wls','size'),sw=('sq_wls','sum'),sa=('sq_alt','sum')).to_numpy()
        rmse_diff=np.sqrt(clusters[:,2].sum()/clusters[:,0].sum())-np.sqrt(clusters[:,1].sum()/clusters[:,0].sum())
        picks=rng.integers(len(clusters),size=(600,len(clusters)))
        sums=clusters[picks].sum(axis=1)
        diffs=np.sqrt(sums[:,2]/sums[:,0])-np.sqrt(sums[:,1]/sums[:,0])
        low,high=np.quantile(diffs,[.025,.975])
        comparison.append(dict(model=p.model,scenario=p.scenario,method=p.method,parameter=p.selected_parameter,
              paired_epochs=len(pair),clusters=len(clusters),
              delta_rmse_mps=float(rmse_diff),bootstrap95_low=float(low),bootstrap95_high=float(high),
              note='Positive means worse than WLS. Resampled held-out NFL games (few clusters; descriptive intervals).'))
    paired=pd.DataFrame(comparison)
    paired.to_csv(out/'heldout_paired_bootstrap.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,5.4))
    scenarios=['observed_like','injected_faults','combined_degradation']
    x=np.arange(len(scenarios)); offsets={'wls':-.26,'fde':0,'robust':.26}
    for model,fill in [('independent','none'),('persistent','full')]:
        for method in ('wls','fde','robust'):
            values=[]
            for sc in scenarios:
                if method=='wls':
                    a=summary[(summary.model==model)&(summary.scenario==sc)&(summary.split=='test')&(summary.method=='wls')]
                else:
                    q=selection[(selection.model==model)&(selection.scenario==sc)&(selection.method==method)].iloc[0]
                    a=summary[(summary.model==model)&(summary.scenario==sc)&(summary.split=='test')&
                        (summary.method==method)&(summary.parameter==q.selected_parameter)]
                values.append(float(a.speed_rmse_accepted_mps.iloc[0]))
            dx=offsets[method]+(-.08 if model=='independent' else .08)
            ax.bar(x+dx,values,width=.155,label=f'{method} ({model})',alpha=.65 if model=='independent' else .95,
                   hatch='//' if model=='independent' else '')
    ax.set_xticks(x,[s.replace('_','\n') for s in scenarios])
    ax.set(ylabel='Speed RMSE on accepted epochs (m/s)',
           title='Held-out NFL games: sensitivity to temporal persistence')
    ax.grid(alpha=.2,axis='y');ax.legend(ncol=3,fontsize=8);fig.tight_layout()
    fig.savefig(out/'heldout_temporal_model_comparison.png',dpi=180);plt.close(fig)
    return selection,paired


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/phase3';out.mkdir(parents=True,exist_ok=True)
    lag1=estimated_autocorrelation(root/'results/phase2/empirical_leave_one_out_residuals.csv',out)
    real=prepare(root,root/'results/phase2')
    tracks=select_trajectories(root/'data/input_2023_w01.csv',high_n=60,other_n=30)
    results,summary=build_evaluation(root,out,real,tracks)
    sel,pairs=evaluate_tuning(results,summary,out)
    metadata=dict(real_loo_within_sat_lag1_corr=lag1,tracks=len(tracks),
      row_count=len(results),scenarios=sorted(results.scenario.unique()),
      caution='All NFL satellite observations and faults are SIMULATED. Correlation in LOO residual proxies can reflect unmodeled biases and dynamics. No external NFL GNSS validation.',
      reference='data/Raw.csv, precise SP3 orbit, NFL 2023 Week 1 tracking; see Phase 2 README')
    (out/'phase3_temporal_summary.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
    print('TRAIN-SELECTED THRESHOLDS / HELD-OUT TEST:')
    print(sel.to_string(index=False))
    print('PAIRED HELD-OUT:')
    print(pairs.to_string(index=False))
    return results,summary,sel,pairs

if __name__=='__main__':run()
