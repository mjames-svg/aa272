"""Controlled GNSS satellite-level error simulations using observed NFL player motion.

All reported RMSEs in this script are *simulation* metrics with tracking
trajectories as reference. The estimated GNSS errors are not real NFL GNSS.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from real_gnss_analysis import ecef_to_enu
from phase2_validation import prepare, sample_error

YARD_M = 0.9144
SPEED_THRESHOLD = 7.0  # m/s, illustrative research threshold
SEED = 272


def select_trajectories(path, seed=SEED, high_n=80, other_n=40):
    frame=pd.read_csv(path,usecols=['game_id','play_id','nfl_id','frame_id','x','y','s','dir'])
    frame=frame.sort_values(['game_id','play_id','nfl_id','frame_id'])
    g=frame.groupby(['game_id','play_id','nfl_id'],sort=False)
    stats=g.agg(frames=('s','size'),max_speed=('s','max'),mean_speed=('s','mean')).reset_index()
    stats=stats[stats.frames>=23]
    hi=stats[stats.max_speed*YARD_M>=SPEED_THRESHOLD+.2]
    lo=stats[stats.max_speed*YARD_M<SPEED_THRESHOLD+.2]
    rng=np.random.default_rng(seed)
    pick_hi=hi.iloc[rng.choice(len(hi),size=min(high_n,len(hi)),replace=False)]
    pick_lo=lo.iloc[rng.choice(len(lo),size=min(other_n,len(lo)),replace=False)]
    keys=pd.concat([pick_hi,pick_lo])[['game_id','play_id','nfl_id']]
    selected=frame.merge(keys,on=['game_id','play_id','nfl_id'])
    trajectories=[]
    for key,group in selected.groupby(['game_id','play_id','nfl_id'],sort=True):
        group=group.sort_values('frame_id')
        if group.frame_id.diff().iloc[1:].ne(1).any():
            continue
        theta=np.deg2rad(group['dir'].to_numpy())
        s=group.s.to_numpy()*YARD_M
        # NFL direction: 0 degrees points along +y, 90 degrees along +x;
        # confirmed from finite differences of provided player x/y trajectories.
        vxy=np.column_stack((s*np.sin(theta),s*np.cos(theta),np.zeros(len(s))))
        trajectories.append({'key':key,'vel_enu':vxy,'speed':s,'frames':group.frame_id.to_numpy()})
    return trajectories


def solve(H,y,sigma,relative=None):
    if len(y)<4 or np.linalg.matrix_rank(H)<4:
        return None
    w=1/np.maximum(sigma,0.15)**2
    if relative is not None: w=w*relative
    A=H*np.sqrt(w[:,None]); b=y*np.sqrt(w)
    x, *_=np.linalg.lstsq(A,b,rcond=None)
    residual=y-H@x
    inv=np.linalg.pinv(H.T@(w[:,None]*H))
    hatdiag=w*np.einsum('ij,jk,ik->i',H,inv,H)
    z=residual/np.maximum(sigma*np.sqrt(np.maximum(1-hatdiag,.04)),.1)
    out=dict(x=x,r=residual,z=z,chi2=float(np.sum((residual/np.maximum(sigma,.15))**2)),
             dof=len(y)-4,formal_sigma=float(np.sqrt(np.trace(inv[:3,:3]))))
    return out


def solve_fde(H,y,sigma,return_removed=False):
    ix=np.arange(len(y)); removed=0;removed_indices=[]
    for _ in range(len(y)):
        sol=solve(H[ix],y[ix],sigma[ix])
        if sol is None: return None,(removed_indices if return_removed else removed)
        if len(ix)<6: break
        score=np.abs(sol['z']);bad=int(np.argmax(score))
        if sol['chi2']<=chi2.ppf(.999,sol['dof']) or score[bad]<=4.0:
            break
        removed_indices.append(int(ix[bad]))
        ix=np.delete(ix,bad);removed+=1
    return sol,(removed_indices if return_removed else removed)


def solve_robust(H,y,sigma):
    sol=solve(H,y,sigma)
    if sol is None:return None
    relative=np.ones(len(y))
    for _ in range(6):
        residual_z=np.abs(sol['r'])/np.maximum(sigma,.15)
        new=np.minimum(1.,2.2/np.maximum(residual_z,1e-10))
        candidate=solve(H,y,sigma,new)
        if candidate is None:break
        if np.linalg.norm(candidate['x']-sol['x'])<1e-5:
            sol=candidate
            break
        sol=candidate;relative=new
    return sol


def synthetic_measurements(rng, template, true_vel_enu, bank, scenario, rotation):
    H=template['H'].copy()
    sigma=template['sigma'].copy();cn0=template['cn0'].copy()
    if scenario['block']:
        keep=rng.random(len(sigma))>=scenario['block']
        H=H[keep];sigma=sigma[keep];cn0=cn0[keep]
    if len(sigma)<4:return {'insufficient_satellites':int(len(sigma))}
    # Real satellite LOS geometry but hypothetical athlete motion in ENU;
    # simulated velocity has the same local coordinate frame as the watch location.
    v_ecef=rotation.T@true_vel_enu
    true_state=np.r_[v_ecef,0.0]  # clock drift absorbed in estimated nuisance state
    error=sample_error(rng,cn0,bank)*scenario['noise_scale']
    is_fault=rng.random(len(error))<scenario['fault_probability']
    signs=rng.choice([-1,1],size=len(error))
    error+=is_fault*signs*rng.uniform(5.,10.,size=len(error))
    return H, H@true_state+error, sigma, is_fault


def matrix_rotation(info):
    r=info[['lat_deg','lon_deg']].dropna().median()
    lat,lon=np.deg2rad([r.lat_deg,r.lon_deg]);sl,cl=np.sin(lat),np.cos(lat);so,co=np.sin(lon),np.cos(lon)
    return np.array([[-so,co,0],[-sl*co,-sl*so,cl],[cl*co,cl*so,sl]])


def event_capture(speeds,frame_ids,selected_indices,threshold=SPEED_THRESHOLD):
    truth=speeds>=threshold
    starts=np.flatnonzero(truth & ~np.r_[False,truth[:-1]])
    ends=np.flatnonzero(truth & ~np.r_[truth[1:],False])
    total=len(starts)
    found=sum(np.any((selected_indices>=a)&(selected_indices<=b)) for a,b in zip(starts,ends))
    return total,found


def run_simulation(root, out, real, repeats=2, high_n=80, other_n=40):
    out.mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(SEED)
    trajectories=select_trajectories(root/'data/input_2023_w01.csv',SEED,high_n,other_n)
    rot=matrix_rotation(real.epoch_info)
    scenarios={
      'observed_like':dict(noise_scale=1.,fault_probability=0.,block=0.),
      'increased_noise':dict(noise_scale=2.,fault_probability=0.,block=0.),
      'injected_faults':dict(noise_scale=1.,fault_probability=.05,block=0.),
      'reduced_satellites':dict(noise_scale=1.,fault_probability=0.,block=.30),
      'combined_degradation':dict(noise_scale=2.,fault_probability=.05,block=.30),
    }
    methods=['wls','fde','robust','wls_epoch_gate']
    rows=[]
    sampling=[]
    template_bank=real.templates
    total_cases=0
    for rate in [1,5]:
        # 5Hz is hypothetical and deliberately NOT a measured watch rate.
        available = ['observed_like','combined_degradation'] if rate==5 else list(scenarios)
        increment=int(10/rate)
        for repetition in range(repeats if rate==1 else 1):
            for t in trajectories:
                phase=int(rng.integers(increment))
                idx=np.arange(phase,len(t['speed']),increment)
                if not len(idx):continue
                n_events,caught=event_capture(t['speed'],t['frames'],idx)
                sampling.append(dict(rate_hz=rate,trial=repetition,game=t['key'][0],play=t['key'][1],
                                     player=t['key'][2],reference_sprint_events=n_events,
                                     sampled_sprint_events=caught))
                for scenario_name in available:
                    scenario=scenarios[scenario_name]
                    for index in idx:
                        true_vel=t['vel_enu'][index]
                        # Shared satellite geometry & realized errors for ALL methods.
                        template=template_bank[rng.integers(len(template_bank))]
                        mock=synthetic_measurements(rng,template,true_vel,real.bank,scenario,rot)
                        total_cases+=1
                        if isinstance(mock,dict):
                            for meth in methods:
                                rows.append(dict(rate_hz=rate,trial=repetition,scenario=scenario_name,
                                    game=t['key'][0],play=t['key'][1],player=t['key'][2],
                                    frame=int(t['frames'][index]),method=meth,
                                    true_speed_mps=float(t['speed'][index]),est_speed_mps=np.nan,
                                    velocity_vector_error_mps=np.nan, retained=False,
                                    n_satellites=mock['insufficient_satellites'],known_faults=0,fde_exclusions=0,
                                    fde_injected_faults_removed=0))
                            continue
                        H,y,sigma,fault_mask=mock
                        nfault=int(fault_mask.sum())
                        estimates={'wls':solve(H,y,sigma),
                                   'fde':None,'robust':solve_robust(H,y,sigma)}
                        fde,removed_indices=solve_fde(H,y,sigma,return_removed=True)
                        excluded=len(removed_indices)
                        injected_faults_removed=int(np.sum(fault_mask[removed_indices])) if excluded else 0
                        estimates['fde']=fde
                        baseline=estimates['wls']
                        # Conservative gate based only on estimated uncertainty and
                        # measurement-consistency test; never accesses true velocity.
                        gate_good=(baseline is not None and len(y)>=6 and
                                   baseline['formal_sigma']<=2.0 and baseline['dof']>0 and
                                   baseline['chi2']<=chi2.ppf(.99,baseline['dof']))
                        estimates['wls_epoch_gate']=baseline if gate_good else None
                        # Shared conditioning policy applied BEFORE considering which
                        # method won: >=6 observations and <=4 m/s formal 3D sigma.
                        common_ok=bool(len(y)>=6 and baseline is not None and
                                       baseline['formal_sigma']<=4.0)
                        for meth,est in estimates.items():
                            if est is None:
                                speed=np.nan;verr=np.nan;raw_speed=np.nan
                                method_ok=False
                            else:
                                velocity_enu=rot@est['x'][:3]
                                raw_speed=float(np.linalg.norm(velocity_enu[:2]))
                                # Application-specific sanity threshold independent
                                # of the actual NFL reference velocity (15 m/s).
                                method_ok=bool(common_ok and est['dof']>0 and
                                   est['formal_sigma']<=4.0 and np.isfinite(raw_speed)
                                   and raw_speed<=15.0)
                                speed=raw_speed if method_ok else np.nan
                                verr=float(np.linalg.norm(velocity_enu-true_vel)) if method_ok else np.nan
                            rows.append(dict(rate_hz=rate,trial=repetition,scenario=scenario_name,
                              game=t['key'][0],play=t['key'][1],player=t['key'][2],
                              frame=int(t['frames'][index]),method=meth,
                              true_speed_mps=float(t['speed'][index]),est_speed_mps=speed,
                              raw_estimated_speed_mps=raw_speed,
                              velocity_vector_error_mps=verr,
                              retained=bool(method_ok),common_precondition_passed=common_ok,
                              n_satellites=len(y),known_faults=nfault,
                              fde_exclusions=int(excluded if meth=='fde' else 0),
                              fde_injected_faults_removed=injected_faults_removed if meth=='fde' else 0))
    df=pd.DataFrame(rows)
    df.to_csv(out/'nfl_simulated_epoch_results.csv',index=False)
    sam=pd.DataFrame(sampling)
    sam.to_csv(out/'nfl_sampling_event_coverage.csv',index=False)
    summary=[]
    for (rate,sc,meth),a in df.groupby(['rate_hz','scenario','method'],sort=True):
        good=a[a.retained];n=len(a)
        sp_err=good.est_speed_mps.to_numpy()-good.true_speed_mps.to_numpy()
        thr=good.true_speed_mps.to_numpy()>=SPEED_THRESHOLD
        pred=good.est_speed_mps.to_numpy()>=SPEED_THRESHOLD
        truths=a.true_speed_mps>=SPEED_THRESHOLD
        # Rejected truth-positive epochs count as undetected for the overall recall.
        tp=int(np.sum(thr&pred));fp=int(np.sum(~thr&pred));fn=int(np.sum(thr&~pred))
        total_truth=int(truths.sum())
        summary.append(dict(rate_hz=rate, scenario=sc, method=meth, epochs=n,
            retained_epochs=len(good),retained_pct=100*len(good)/n,
            rms_speed_error_mps=float(np.sqrt(np.mean(sp_err**2))) if len(sp_err) else np.nan,
            rms_vector_error_mps=float(np.sqrt(np.mean(good.velocity_vector_error_mps.to_numpy()**2))) if len(good) else np.nan,
            false_high_speed_epochs=fp,
            missed_high_speed_epochs_among_retained=fn,
            truth_high_speed_epochs=total_truth,
            recall_including_rejected_as_missed=tp/total_truth if total_truth else np.nan,
            false_high_speed_rate_among_retained_negative=fp/max(1,int(np.sum(~thr))),
            fde_exclusions=int(a.fde_exclusions.sum()),
            fde_injected_faults_removed=int(a.fde_injected_faults_removed.sum()),
            injected_fault_observations=int(a.known_faults.sum())))
    metrics=pd.DataFrame(summary)
    metrics.to_csv(out/'nfl_simulation_method_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5.5))
    one=metrics[metrics.rate_hz==1].copy()
    for meth,sub in one.groupby('method'):
        sub=sub.set_index('scenario').reindex(scenarios)
        ax.plot(np.arange(len(sub)),sub.rms_speed_error_mps,marker='o',label=meth)
    ax.set_xticks(range(len(scenarios)),[s.replace('_','\n') for s in scenarios])
    ax.set(ylabel='RMSE of simulated GNSS speed (m/s)',title='Controlled NFL simulation: retained-epoch speed error (1 Hz)')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'nfl_speed_rmse_by_scenario.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5))
    for meth,sub in one.groupby('method'):
        sub=sub.set_index('scenario').reindex(scenarios)
        ax.plot(np.arange(len(sub)),sub.retained_pct,marker='o',label=meth)
    ax.set_xticks(range(len(scenarios)),[s.replace('_','\n') for s in scenarios])
    ax.set(ylabel='Accepted velocity epochs (%)',ylim=(0,105),title='Controlled NFL simulation: tracking availability (1 Hz)')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'nfl_availability_by_scenario.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for meth,sub in one.groupby('method'):
        ax.scatter(sub.retained_pct,sub.rms_speed_error_mps,s=55,label=meth)
    ax.set(xlabel='Accepted velocity epochs (%)',ylabel='RMSE on accepted epochs (m/s)',
           title='Accuracy-versus-availability tradeoff (multiple degradation scenarios)')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'nfl_accuracy_availability_tradeoff.png',dpi=180);plt.close(fig)
    sm=sam.groupby('rate_hz')[['reference_sprint_events','sampled_sprint_events']].sum()
    sampling_recall={str(int(rate)):float(row.sampled_sprint_events/row.reference_sprint_events) if row.reference_sprint_events else None for rate,row in sm.iterrows()}
    metadata=dict(seed=SEED,selected_tracks=len(trajectories),repeats=repeats,scenario_definitions=scenarios,
                  admissibility_policy='Shared: at least 6 satellites and baseline formal 3D velocity sigma <=4 m/s. Each final estimate must have a residual DOF, formal sigma <=4 m/s, and horizontal speed <=15 m/s (athlete-specific plausibility). Extra WLS gate: formal sigma <=2 m/s and chi2 <=99th percentile. Rejected estimates count against tracking availability.',
                  primary_measurement_rate_hz=1,hypothetical_comparison_rate_hz=5,
                  sprint_threshold_mps=SPEED_THRESHOLD,
                  sampled_truth_event_detection_rate=sampling_recall,
                  total_simulated_gnss_cases=total_cases,
                  model='NFL speed/direction as reference motion; real wearables LOS geometry; empirical leave-one-out Doppler residual proxies; injected controlled faults and blockage',
                  critical_limitations=[
                      'Real GNSS measurement errors are not known exactly without an external reference; leave-one-out residuals only proxy errors.',
                      'Satellite geometry is sampled from a wearable recording in France, NOT actual NFL stadium satellite constellations.',
                      'Temporal correlations, signal tracking/lock dynamics, body orientation effects and stadium obstruction are not explicitly reproduced.',
                      'NFL player tracking is reference motion for simulation, not independent truth.',
                      'RMSE is conditioned on accepted epochs; compare availability in parallel.',
                      '5Hz is a hypothetical sampling rate, not the observed Pixel Watch measurement rate.',
                      'The speed threshold of 7m/s is an illustrative analysis choice, not a universal NFL sprint definition.',
                  ])
    (out/'nfl_simulation_metadata.json').write_text(json.dumps(metadata,indent=2))
    return metrics, metadata


if __name__=='__main__':
    root=Path(__file__).resolve().parent;out=root/'results/phase2'
    real=prepare(root,out)
    summary,info=run_simulation(root,out,real)
    print(summary.to_string(index=False))
    print(json.dumps(info,indent=2))
