"""Phase 11: strictly causal Kalman GNSS-Doppler study on the paired Phase-10 simulation.

The player velocities are measured NFL RFID tracking references. ALL NFL satellite
Doppler measurements are synthetic. This study evaluates only the observed 1-Hz
GNSS sample times; prediction at missing epochs is never equivalent to a GNSS fix.
The process noise is fixed before evaluating held-out NFL trajectories.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase2_validation import prepare, sample_error
from phase2_nfl_simulation import matrix_rotation, solve
from phase3_temporal_sensitivity import step_markov
from phase10_geometry_stadium import SKIES, SIGNALS, sky_geometry
from run_final_analysis import SEEDS, choose_tracks
from phase5_event_integration import read_reference, reference_events
from phase6_event_metrics import summarize_fast

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'phase11'
SEED_MODEL='kalman_paired'
DEFAULT_Q=2.5     # std of a 1-s CV velocity increment, m/s
LOW_Q=0.75
HIGH_Q=5.0
CA_Q=3.0          # std change in acceleration over 1 second, m/s^2
PREDICTION_LIMIT_S=2.0
SIGMA_LIMIT=4.0
PREDICTION_SIGMA_LIMIT=8.0
SPEED_LIMIT=15.0
INNOVATION_GATE_Z=5.0
METHODS=('wls','wls4','kf_cv6','kf_cv4','kf_ca4','kf_cv4_lowq','kf_cv4_highq')
DYNAMIC_CASES=(('partial_canopy','entry'),('deep_canopy','entry'),
               ('open','outage1'),('open','outage2'),
               ('partial_canopy','outage1'),('partial_canopy','outage2'))
LABELS={'wls':'WLS (6 sat)','wls4':'WLS (4 sat)','kf_cv6':'CV-KF (6 sat)',
        'kf_cv4':'CV-KF (4 sat)','kf_ca4':'CA-KF (4 sat)',
        'kf_cv4_lowq':'CV-KF (low Q)','kf_cv4_highq':'CV-KF (high Q)'}
ROOF_LABELS={'open':'Open sky','partial_canopy':'Partial canopy',
             'deep_canopy':'Deep canopy','sealed_roof':'All blocked'}

@dataclass
class FilterConfig:
    method:str
    min_sats:int
    q_scale:float
    model:str='cv'

CONFIGS=(FilterConfig('kf_cv6',6,DEFAULT_Q),
         FilterConfig('kf_cv4',4,DEFAULT_Q),
         FilterConfig('kf_ca4',4,CA_Q,'ca'),
         FilterConfig('kf_cv4_lowq',4,LOW_Q),
         FilterConfig('kf_cv4_highq',4,HIGH_Q))


class DopplerKF:
    """Velocity/clock-drift filter, with optional acceleration states.

    CV state [vX,vY,vZ,clock_drift]; CA state [vX,vY,vZ,aX,aY,aZ,clock_drift].
    Initial state comes ONLY from a qualifying single-epoch Doppler WLS solution.
    No known NFL reference speed, athlete direction, or future samples enter filter.
    """
    def __init__(self,config:FilterConfig,rotation:np.ndarray):
        self.config=config
        self.rotation=rotation
        self.n=7 if config.model=='ca' else 4
        self.x=None
        self.P=None
        self.previous_frame=None
        self.last_update_frame=None

    def _design(self,H):
        if self.n==4:return H
        extended=np.zeros((len(H),7))
        extended[:,:3]=H[:,:3]
        extended[:,6]=H[:,3]
        return extended

    def _predict(self,dt):
        F=np.eye(self.n)
        Q=np.zeros((self.n,self.n))
        if self.n==4:
            # Random-walk velocity, equivalent to white unresolved acceleration.
            Q[:3,:3]=np.eye(3)*self.config.q_scale**2*dt
            Q[3,3]=0.25*dt
        else:
            F[:3,3:6]=np.eye(3)*dt
            # White jerk: exact constant-acceleration transition covariance.
            q=self.config.q_scale**2
            Q[:3,:3]=np.eye(3)*q*dt**3/3
            Q[:3,3:6]=np.eye(3)*q*dt**2/2
            Q[3:6,:3]=np.eye(3)*q*dt**2/2
            Q[3:6,3:6]=np.eye(3)*q*dt
            Q[6,6]=0.25*dt
        self.x=F@self.x
        self.P=F@self.P@F.T+Q
        self.P=(self.P+self.P.T)/2

    def _init(self,frame,H,y,sigma):
        if len(y)<6:return False
        est=solve(H,y,sigma)
        if est is None or est['dof']<1 or est['formal_sigma']>SIGMA_LIMIT:
            return False
        if np.linalg.norm((self.rotation@est['x'][:3])[:2])>SPEED_LIMIT:
            return False
        w=1/np.maximum(sigma,0.15)**2
        cov=np.linalg.pinv(H.T@(w[:,None]*H))
        if not np.all(np.isfinite(cov)):return False
        self.x=np.zeros(self.n)
        self.P=np.zeros((self.n,self.n))
        if self.n==4:
            self.x[:]=est['x'];self.P[:]=cov
        else:
            self.x[:3]=est['x'][:3]
            self.x[6]=est['x'][3]
            self.P[:3,:3]=cov[:3,:3]
            self.P[:3,6]=cov[:3,3]
            self.P[6,:3]=cov[3,:3]
            self.P[6,6]=cov[3,3]
            self.P[3:6,3:6]=np.eye(3)*3.0**2
        self.P=(self.P+self.P.T)/2
        self.previous_frame=frame
        self.last_update_frame=frame
        return True

    def _update(self,H,y,sigma):
        if len(y)<self.config.min_sats:return False,0,0
        # Measurements may have fewer than 6 satellite directions. Require the
        # four observed velocity/clock states to be geometrically identifiable.
        if np.linalg.matrix_rank(H,tol=1e-9)<4:return False,0,0
        B=self._design(H)
        Rdiag=np.maximum(sigma,0.15)**2
        innovation=y-B@self.x
        prior_var=np.einsum('ij,jk,ik->i',B,self.P,B)+Rdiag
        z=np.abs(innovation)/np.sqrt(np.maximum(prior_var,1e-10))
        good=np.isfinite(z)&(z<=INNOVATION_GATE_Z)
        rejected=int((~good).sum())
        if np.sum(good)<self.config.min_sats or np.linalg.matrix_rank(H[good],tol=1e-9)<4:
            return False,0,rejected
        B=B[good];y=y[good];Rdiag=Rdiag[good]
        R=np.diag(Rdiag)
        S=B@self.P@B.T+R
        try:
            K=np.linalg.solve(S,B@self.P).T
        except np.linalg.LinAlgError:
            return False,0,rejected
        self.x=self.x+K@(y-B@self.x)
        I=np.eye(self.n)
        J=I-K@B
        self.P=J@self.P@J.T+K@R@K.T
        self.P=(self.P+self.P.T)/2
        return True,len(y),rejected

    def step(self,frame:int,H:np.ndarray,y:np.ndarray,sigma:np.ndarray):
        if self.x is None:
            initialized=self._init(frame,H,y,sigma)
            if not initialized:
                return (False,np.nan,'unavailable',np.nan,np.nan,0,0)
            update='initialized';used=len(y);rejected=0
        else:
            dt=(frame-self.previous_frame)/10.0
            if not (0<dt<=5):raise ValueError(f'nonconsecutive track timing {dt}')
            self._predict(dt)
            self.previous_frame=frame
            ok,used,rejected=self._update(H,y,sigma)
            if ok:
                self.last_update_frame=frame
                update='gnss_update'
            else:
                update='prediction'
        age=(frame-self.last_update_frame)/10.0
        covsigma=float(np.sqrt(max(0.,np.trace(self.P[:3,:3]))))
        speed=float(np.linalg.norm((self.rotation@self.x[:3])[:2]))
        accepted=(np.isfinite(speed) and speed<=SPEED_LIMIT and np.isfinite(covsigma) and
                  covsigma <= (PREDICTION_SIGMA_LIMIT if update=='prediction' else SIGMA_LIMIT) and
                  (update!='prediction' or age<=PREDICTION_LIMIT_S+1e-8))
        if update=='prediction' and not accepted:
            update='prediction_withheld'
        elif not accepted:
            update='update_withheld'
        return (bool(accepted),speed if accepted else np.nan,update,covsigma,age,used,rejected)


def wls_matched_phase10(h,y,s,rot,min_sats=6):
    if len(y)<min_sats:
        return False,np.nan,np.nan
    base=solve(h,y,s)
    if base is None or base['formal_sigma']>SIGMA_LIMIT:
        return False,np.nan,np.nan
    speed=float(np.linalg.norm((rot@base['x'][:3])[:2]))
    ok=bool((base['dof']>0 or min_sats==4) and np.isfinite(speed) and speed<=SPEED_LIMIT)
    return ok,speed if ok else np.nan,float(base['formal_sigma'])


def run_simulation(seeds=SEEDS):
    OUT.mkdir(parents=True,exist_ok=True)
    real=prepare(ROOT,ROOT/'results'/'phase2')
    rot=matrix_rotation(real.epoch_info)
    tracks,cohort=choose_tracks(ROOT)
    rows=[]
    for seed in seeds:
        for ti,t in enumerate(tracks):
            game,play,player=t['key']
            rng=np.random.default_rng(np.random.SeedSequence([seed,game,play,player,1081]))
            phase=int(rng.integers(10))
            template=real.templates[int(rng.integers(len(real.templates)))]
            orientation=float(rng.uniform(0,180))
            H=template['H'];sig=template['sigma'];cn0=template['cn0']
            _,_,masks=sky_geometry(H,rot,orientation)
            sampled=np.arange(phase,len(t['speed']),10)
            last=np.zeros(len(sig))
            faults_active=rng.random(len(sig))<.05
            fsign=rng.choice([-1.,1.],len(sig))
            famp=rng.uniform(5.,10.,len(sig))
            scenarios=[(signal,sky,'static') for signal in SIGNALS for sky in SKIES]
            scenarios += [('measured_proxy',sky,mode) for sky,mode in DYNAMIC_CASES]
            filters={(signal,sky,mode,cfg.method):DopplerKF(cfg,rot)
                     for signal,sky,mode in scenarios for cfg in CONFIGS}
            for ordinal,idx in enumerate(sampled):
                innovation=sample_error(rng,cn0,real.bank)
                last=.5*last+np.sqrt(.75)*innovation
                faults_active=step_markov(rng,faults_active,.05,.7)
                newly=faults_active&(rng.random(len(sig))<.2)
                fsign[newly]=rng.choice([-1.,1.],int(newly.sum()))
                famp[newly]=rng.uniform(5.,10.,int(newly.sum()))
                faults=np.where(faults_active,fsign*famp,0.)
                truevel=t['vel_enu'][idx]
                state=np.r_[rot.T@truevel,0.]
                base=H@state
                frame=int(t['frames'][idx])
                for signal,sky,mode in scenarios:
                    disturbance=last if signal=='measured_proxy' else 2*last+faults
                    if mode=='entry':
                        # First sampled epoch is open, then the athlete passes
                        # beneath an assumed canopy for the remaining epochs.
                        mask=masks['open'] if ordinal==0 else masks[sky]
                    elif mode in ('outage1','outage2') and ordinal%4 in (
                            (1,) if mode=='outage1' else (1,2)):
                        # Deliberate complete satellite blackout of 1 or 2 s,
                        # within a short player track, beginning AFTER initialization.
                        mask=np.zeros(len(sig),dtype=bool)
                    else:
                        mask=masks[sky]
                    h=H[mask]; y=(base+disturbance)[mask];s=sig[mask]
                    scenario=f'{sky}__{signal}' if mode=='static' else f'{sky}__{signal}__{mode}'
                    accepted,speed,formal=wls_matched_phase10(h,y,s,rot)
                    analysis_acceleration=float(t['speed'][idx]-t['speed'][max(0,idx-10)]) /max((idx-max(0,idx-10))/10,1e-8)
                    rows.append((seed,SEED_MODEL,scenario,game,play,player,frame,'wls',
                                 bool(accepted),speed,float(t['speed'][idx]),len(y),
                                 'gnss_update' if accepted else 'unavailable',
                                 float(formal) if np.isfinite(formal) else np.nan,0.,
                                 len(y) if accepted else 0,0,sky,signal,analysis_acceleration))
                    accepted4,speed4,formal4=wls_matched_phase10(h,y,s,rot,min_sats=4)
                    rows.append((seed,SEED_MODEL,scenario,game,play,player,frame,'wls4',
                                 bool(accepted4),speed4,float(t['speed'][idx]),len(y),
                                 'gnss_update' if accepted4 else 'unavailable',
                                 float(formal4) if np.isfinite(formal4) else np.nan,0.,
                                 len(y) if accepted4 else 0,0,sky,signal,analysis_acceleration))
                    for cfg in CONFIGS:
                        f=filters[(signal,sky,mode,cfg.method)]
                        ok,v,status,cov,age,used,rejected=f.step(frame,h,y,s)
                        rows.append((seed,SEED_MODEL,scenario,game,play,player,frame,cfg.method,
                                     ok,v,float(t['speed'][idx]),len(y),status,cov,age,used,rejected,sky,signal,
                                     analysis_acceleration))
        print(f'Phase 11 reconstructed seed {seed}: {len(rows):,} method-epoch records',flush=True)
    cols=['seed','model','scenario','game','play','player','frame','method','accepted','est_speed_mps',
          'true_speed_mps','n_satellites','solution_type','filter_formal_sigma_mps','seconds_since_update',
          'satellites_used_in_update','innovation_rejections','sky','signal','reference_speed_change_mps2']
    df=pd.DataFrame.from_records(rows,columns=cols)
    df.to_csv(OUT/'kalman_paired_epoch_results.csv',index=False)
    meta={'seeds':list(seeds),'cohort':cohort,'model':'All NFL Doppler measurements simulated; real wearable GPS geometry and LOO residual proxy',
          'sampling':'Only actual simulated 1-Hz sampled epochs; no interpolation; original 10-Hz RFID provides reference only',
          'cv_model':'state [ECEF 3D velocity, receiver clock drift], F=I; Qv=q^2*dt',
          'ca_model':'state [ECEF velocity, ECEF acceleration, receiver clock drift]; Fv,a=dt and white-jerk Q',
          'process_std_1s_cv_mps':{'low':LOW_Q,'default':DEFAULT_Q,'high':HIGH_Q},
          'process_std_1s_ca_acceleration_change_mps2':CA_Q,
          'clock_drift_random_walk_variance_mps2_per_s':.25,
          'R':'raw per-satellite Doppler sigma from Pixel Watch (minimum 0.15 m/s); not independently calibrated',
          'initialization':'requires >=6 GPS satellites, full-rank independent Doppler WLS, formal sigma<=4 m/s, speed<=15 m/s; no ground truth',
          'innovation_outlier_gate_z':INNOVATION_GATE_Z,
          'wls4_note':'Four-satellite WLS is a separate fair control with zero residual redundancy at exactly four satellites; initialization of all Kalman filters still requires six',
          'cv6_min_update_satellites':6,'cv4_min_update_satellites':4,'ca4_min_update_satellites':4,
          'max_prediction_gap_seconds':PREDICTION_LIMIT_S,'acceptance_max_sigma_mps':SIGMA_LIMIT,
          'acceptance_max_speed_mps':SPEED_LIMIT,
          'prediction_max_sigma_mps':PREDICTION_SIGMA_LIMIT,
          'prediction_policy_note':'Prediction-only estimates have a deliberately looser uncertainty threshold than satellite updates; they are not equivalent-quality GNSS fixes',
          'sealed_roof':'no available Doppler observation for the entire player track (no possible initialization)',
          'chosen_settings':'All parameters specified before held-out analysis, not selected by test-set ground-truth scores',
          'additional_dynamics':'entry first open-sky 1-Hz epoch then fixed partial/deep canopy; outage1/outage2 deliberately mask ALL satellites at epoch index 1 or indices 1,2 of each 4-sample cycle; these are hypothetical transient outages and not Phase-10 roof results',
          'study_limit':'Do not interpret model-based predictions as satellite fixes or as evidence of NFL receiver performance.'}
    (OUT/'methods_and_assumptions.json').write_text(json.dumps(meta,indent=2))
    return df


def validate_paired_wls(df,strict=True):
    p10=pd.read_csv(ROOT/'results/phase10/paired_epoch_results.csv',
        usecols=['seed','scenario','game','play','player','frame','method','accepted','est_speed_mps'])
    ours=df[df.method.eq('wls') & df.scenario.str.match(r'^(open|partial_canopy|deep_canopy|sealed_roof)__(measured_proxy|degraded_sensitivity)$')][['seed','scenario','game','play','player','frame','accepted','est_speed_mps']].copy()
    p10=p10[p10.method.eq('wls')].drop(columns='method')
    keys=['seed','scenario','game','play','player','frame']
    joined=ours.merge(p10,on=keys,suffixes=('_p11','_p10'),validate='one_to_one',how='outer',indicator=True)
    mismatching=(joined['_merge'].ne('both')|
                  joined.accepted_p11.ne(joined.accepted_p10)|
                 ~np.isclose(joined.est_speed_mps_p11,joined.est_speed_mps_p10,rtol=1e-9,atol=1e-8,equal_nan=True))
    n=int(mismatching.sum())
    result={'matched_rows':int(len(joined)),'unmatched_or_mismatching':n,
            'baseline_exactly_reconstructed':bool(n==0)}
    (OUT/'baseline_reconstruction_check.json').write_text(json.dumps(result,indent=2))
    if strict and n:raise AssertionError(f'Paired experiment not identical to Phase 10: {result}')
    return result


def analyze(df):
    required=['seed','model','scenario','game','play','player','frame','method']
    assert not df.duplicated(required).any()
    assert df.groupby(required[:-1]).method.nunique().eq(len(METHODS)).all()
    assert df.loc[~df.accepted,'est_speed_mps'].isna().all()
    refs=read_reference(ROOT,df)
    events=reference_events(refs,7.)
    assert len(events)==287, f'Unexpected reference event count {len(events)}'
    event_keys=['game','play','player','event_ordinal']
    pairs=[]
    for ev in events.itertuples(index=False):
        pairs.extend((ev.game,ev.play,ev.player,f,ev.event_ordinal)
                     for f in range(ev.start_frame,ev.end_frame+1))
    tag=pd.DataFrame(pairs,columns=['game','play','player','frame','event_ordinal'])
    tagged=df.merge(tag,on=['game','play','player','frame'],how='inner',validate='many_to_one')
    tagged['hit']=tagged.accepted & tagged.est_speed_mps.ge(7.)
    tagged['gnss_hit']=tagged.hit & tagged.solution_type.isin(['gnss_update','initialized'])
    group=['seed','model','scenario','method','game','play','player','event_ordinal']
    tagged_events=tagged.groupby(group,sort=False).agg(sampled=('frame','size'),
        accepted=('accepted','any'),detected=('hit','any'),gnss_detected=('gnss_hit','any')).reset_index()
    combos=df[['seed','model','scenario','method']].drop_duplicates()
    evcols=events[event_keys+['duration_s','peak_reference_mps','touches_track_edge']]
    full=combos.merge(evcols,how='cross').merge(tagged_events,on=group,how='left',validate='one_to_one')
    full['sampled']=full.sampled.fillna(0).gt(0)
    for col in ('accepted','detected','gnss_detected'):
        full[col]=full[col].astype('boolean').fillna(False).astype(bool)
    full['pred_only_added_event']=full.detected & ~full.gnss_detected
    full['outcome']=np.select([~full.sampled,full.sampled&~full.accepted,full.detected],
                  ['not_sampled','sampled_no_valid_solution','detected'],default='sampled_not_detected')
    chk=full.pivot(index=['seed','model','scenario']+event_keys,columns='method',values='sampled')
    assert chk.nunique(axis=1).eq(1).all()
    df=df.copy()
    df['positive_truth']=df.true_speed_mps.ge(7.)
    df['alert']=df.accepted & df.est_speed_mps.ge(7.)
    df['neg_sample']=(~df.positive_truth).astype(int)
    df['false_alert']=(df.alert & ~df.positive_truth).astype(int)
    df['squared_error']=np.where(df.accepted,(df.est_speed_mps-df.true_speed_mps)**2,0.)
    df['update_accepted']=(df.accepted & df.solution_type.isin(['initialized','gnss_update'])).astype(int)
    df['prediction_accepted']=(df.accepted & df.solution_type.eq('prediction')).astype(int)
    df['update_sq_error']=np.where(df.update_accepted>0,df.squared_error,0.)
    df['prediction_sq_error']=np.where(df.prediction_accepted>0,df.squared_error,0.)
    epochkeys=['seed','model','scenario','method','game','play','player']
    epoch=df.groupby(epochkeys,sort=False).agg(sampled_epochs=('frame','size'),
        accepted_epochs=('accepted','sum'),reference_negative_epochs=('neg_sample','sum'),
        false_high_speed_epochs=('false_alert','sum'),accepted_sq_error_sum=('squared_error','sum'),
        gnss_updates=('update_accepted','sum'),prediction_only=('prediction_accepted','sum'),
        gnss_update_sq_err=('update_sq_error','sum'),prediction_only_sq_err=('prediction_sq_error','sum')).reset_index()
    seedstats,summary=summarize_fast(full,epoch)
    addon=epoch.groupby(['seed','model','scenario','method'],sort=False)[
        ['sampled_epochs','accepted_epochs','gnss_updates','prediction_only','gnss_update_sq_err','prediction_only_sq_err']].sum().reset_index()
    addon['gnss_update_pct']=100*addon.gnss_updates/addon.sampled_epochs
    addon['prediction_only_pct']=100*addon.prediction_only/addon.sampled_epochs
    addon['gnss_update_rmse_mps']=np.where(addon.gnss_updates>0,np.sqrt(addon.gnss_update_sq_err/addon.gnss_updates.clip(lower=1)),np.nan)
    addon['prediction_only_rmse_mps']=np.where(addon.prediction_only>0,np.sqrt(addon.prediction_only_sq_err/addon.prediction_only.clip(lower=1)),np.nan)
    added=full.groupby(['seed','model','scenario','method'],sort=False).agg(
        pred_only_added_events=('pred_only_added_event','sum'),
        gnss_detected_events=('gnss_detected','sum')).reset_index()
    seedstats=seedstats.merge(addon.drop(columns=['sampled_epochs','accepted_epochs','gnss_update_sq_err','prediction_only_sq_err']),
                               on=['seed','model','scenario','method'],validate='one_to_one')
    seedstats=seedstats.merge(added,on=['seed','model','scenario','method'],validate='one_to_one')
    overall=seedstats.groupby(['model','scenario','method'],sort=True).agg(
        gnss_update_pct=('gnss_update_pct','mean'),
        prediction_only_pct=('prediction_only_pct','mean'),
        pred_only_added_events=('pred_only_added_events','sum'),
        gnss_detected_events=('gnss_detected_events','sum'),
        gnss_update_rmse_mps=('gnss_update_rmse_mps','mean'),
        prediction_only_rmse_mps=('prediction_only_rmse_mps','mean')).reset_index()
    summary=summary.merge(overall,on=['model','scenario','method'],validate='one_to_one')
    summary.loc[summary.mean_epoch_availability_pct.eq(0),'mean_speed_rmse_on_accepted_mps']=np.nan
    events.to_csv(OUT/'reference_events.csv',index=False)
    full.to_csv(OUT/'event_outcomes.csv',index=False)
    seedstats.to_csv(OUT/'seed_level_results.csv',index=False)
    summary.to_csv(OUT/'kalman_method_comparison.csv',index=False)
    return df,full,summary,seedstats


def paired_bootstrap(full,n_boot=1000):
    selected=full[full.method.isin(['wls','kf_cv6','kf_cv4','kf_ca4'])]
    all_games=np.array(sorted(selected.game.unique()))
    rng=np.random.default_rng(2722026)
    rows=[]
    for signal in SIGNALS:
        for sky in SKIES[:3]:
            sub=selected[selected.scenario.eq(f'{sky}__{signal}')].copy()
            piv=sub.pivot(index=['seed','game','play','player','event_ordinal'],columns='method',values='detected')
            base=piv['wls'].astype(float)
            for method in ('kf_cv6','kf_cv4','kf_ca4'):
                delta=(piv[method].astype(float)-base).reset_index(name='difference')
                gameagg=delta.groupby('game').difference.agg(['sum','count']).reindex(all_games)
                point=100*delta.difference.mean()
                bs=np.empty(n_boot)
                for b in range(n_boot):
                    pick=rng.integers(0,len(all_games),len(all_games))
                    bs[b]=100*gameagg['sum'].to_numpy()[pick].sum()/gameagg['count'].to_numpy()[pick].sum()
                rows.append(dict(sky=sky,signal=signal,method=method,delta_recall_pp=point,
                            lower_95_pp=float(np.percentile(bs,2.5)),upper_95_pp=float(np.percentile(bs,97.5)),
                            resampling_unit='five held-out games; repeated seeds are NOT independent games'))
    for scenario in ['partial_canopy__measured_proxy__entry',
                     'deep_canopy__measured_proxy__entry',
                     'open__measured_proxy__outage1','open__measured_proxy__outage2']:
        sub=full[(full.scenario==scenario)&full.method.isin(['wls','wls4','kf_cv6','kf_cv4','kf_ca4'])]
        piv=sub.pivot(index=['seed','game','play','player','event_ordinal'],columns='method',values='detected')
        base=piv['wls'].astype(float)
        for method in ['wls4','kf_cv6','kf_cv4','kf_ca4']:
            diff=(piv[method].astype(float)-base).reset_index(name='difference')
            gameagg=diff.groupby('game').difference.agg(['sum','count']).reindex(all_games)
            delta=100*diff.difference.mean()
            bs=np.empty(n_boot)
            for b in range(n_boot):
                ids=rng.integers(0,len(all_games),len(all_games))
                bs[b]=100*gameagg['sum'].to_numpy()[ids].sum()/gameagg['count'].to_numpy()[ids].sum()
            rows.append({'sky':scenario,'signal':'dynamic','method':method,
                         'delta_recall_pp':delta,
                         'lower_95_pp':float(np.percentile(bs,2.5)),
                         'upper_95_pp':float(np.percentile(bs,97.5)),
                         'resampling_unit':'Five held-out games; paired six seeds'})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'paired_game_bootstrap.csv',index=False)
    return result


def compare_against_wls4(full,n_boot=1200):
    rng=np.random.default_rng(27244)
    games=np.array(sorted(full.game.unique()))
    scenarios=['open__measured_proxy','partial_canopy__measured_proxy','deep_canopy__measured_proxy',
        'partial_canopy__measured_proxy__entry','deep_canopy__measured_proxy__entry',
        'open__measured_proxy__outage1','open__measured_proxy__outage2']
    rec=[]
    for scenario in scenarios:
        sub=full[(full.scenario==scenario)&full.method.isin(['wls4','kf_cv4','kf_ca4'])]
        pivot=sub.pivot(index=['seed','game','play','player','event_ordinal'],columns='method',values='detected')
        for method in ['kf_cv4','kf_ca4']:
            dif=(pivot[method].astype(float)-pivot['wls4'].astype(float)).reset_index(name='delta')
            totals=dif.groupby('game').delta.agg(['sum','count']).reindex(games)
            v=totals.to_numpy()
            samples=rng.integers(len(games),size=(n_boot,len(games)))
            numerator=v[samples,0].sum(axis=1)
            denominator=v[samples,1].sum(axis=1)
            boot=100*numerator/denominator
            rec.append(dict(scenario=scenario,method=method,reference='wls4',
                            delta_event_recall_pp=100*dif.delta.mean(),
                            lower_95_pp=float(np.percentile(boot,2.5)),
                            upper_95_pp=float(np.percentile(boot,97.5)),
                            caveat='Game-cluster descriptive bootstrap, five held-out games; six paired seeds are not independent NFL games'))
    out=pd.DataFrame(rec)
    out.to_csv(OUT/'four_satellite_matched_control.csv',index=False)
    return out


def figures(df,summary,boot,control):
    # Four estimators on fixed, identical synthetic Doppler observations.
    color={'wls':'#206679','wls4':'#73948F','kf_cv6':'#B07D34','kf_cv4':'#A44F51','kf_ca4':'#546B74'}
    main=summary[(summary.scenario.str.endswith('measured_proxy'))&summary.method.isin(color)]
    x=np.arange(4);width=.155
    fig,ax=plt.subplots(figsize=(9,4.2))
    for i,method in enumerate(color):
        vals=[float(main[(main.scenario.eq(f'{s}__measured_proxy'))&(main.method.eq(method))].event_recall_pct.iloc[0]) for s in SKIES]
        ax.bar(x+(i-2)*width,vals,width,color=color[method],label=LABELS[method])
    ax.axhline(main.loc[(main.scenario.eq('open__measured_proxy'))&(main.method.eq('wls')),'sampling_ceiling_pct'].iloc[0],ls='--',lw=1,color='#666',label='1-Hz perfect-measurement ceiling')
    ax.set_xticks(x,[ROOF_LABELS[s] for s in SKIES]);ax.set_ylim(0,82)
    ax.set_ylabel('Observed-interval recall (%)');ax.grid(axis='y',alpha=.18)
    ax.legend(ncol=3,fontsize=7.2,frameon=False,loc='upper right');fig.tight_layout()
    fig.savefig(OUT/'11_kalman_event_recall.pdf',bbox_inches='tight');fig.savefig(OUT/'11_kalman_event_recall.png',dpi=180,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(9.7,3.4))
    for j,metric in enumerate(['mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps']):
        for method in color:
            vals=[float(main[(main.scenario.eq(f'{s}__measured_proxy'))&(main.method.eq(method))][metric].iloc[0]) for s in SKIES[:3]]
            ax[j].plot(np.arange(3),vals,'o-',label=LABELS[method],color=color[method])
        ax[j].set_xticks(range(3),[ROOF_LABELS[s] for s in SKIES[:3]])
        ax[j].set_ylabel('Accepted sampled epochs (%)' if j==0 else 'Speed RMSE on accepted epochs (m/s)')
        ax[j].grid(alpha=.2)
    ax[0].legend(fontsize=7,frameon=False,loc='best');fig.tight_layout()
    fig.savefig(OUT/'11_kalman_rmse_availability.png',dpi=190,bbox_inches='tight');plt.close(fig)

    # Dynamic scenarios are essential: in the static roof masks there are no
    # mid-track satellite changes for temporal prediction to bridge.
    dyn_scenarios=['partial_canopy__measured_proxy__entry',
                   'deep_canopy__measured_proxy__entry',
                   'open__measured_proxy__outage1',
                   'open__measured_proxy__outage2']
    dyn_names=['Enter partial\ncanopy','Enter deep\ncanopy','1-s complete\noutage','2-s complete\noutage']
    fig,ax=plt.subplots(figsize=(9.5,4.2))
    xx=np.arange(len(dyn_scenarios));w=.24
    for i,method in enumerate(['kf_cv6','kf_cv4','kf_ca4']):
        sub=summary[summary.method.eq(method)].set_index('scenario')
        a=np.array([sub.loc[sc,'gnss_update_pct'] for sc in dyn_scenarios])
        b=np.array([sub.loc[sc,'prediction_only_pct'] for sc in dyn_scenarios])
        ax.bar(xx+(i-1)*w,a,w,color=color[method],label=LABELS[method])
        ax.bar(xx+(i-1)*w,b,w,bottom=a,facecolor='white',edgecolor=color[method],hatch='///',linewidth=.8)
    ax.set_xticks(xx,dyn_names)
    ax.set_ylabel('Accepted / all sampled epochs (%)')
    ax.set_ylim(0,112);ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02))
    fig.text(.5,.01,'Solid: GNSS update     Hatched: prediction only',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'11_gnss_update_vs_prediction.png',dpi=190,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,4))
    meth=['wls','wls4','kf_cv6','kf_cv4','kf_ca4']
    xx=np.arange(len(dyn_scenarios));w=.16
    for i,method in enumerate(meth):
        sub=summary[summary.method.eq(method)].set_index('scenario')
        vals=np.array([sub.loc[sc,'event_recall_pct'] for sc in dyn_scenarios])
        ax.bar(xx+(i-2)*w,vals,w,label=LABELS[method],color=color[method])
    ax.set_xticks(xx,dyn_names)
    ax.set_ylabel('High-speed interval recall (%)')
    ax.set_ylim(0,80);ax.grid(axis='y',alpha=.18)
    ax.legend(ncol=3,frameon=False,fontsize=7.6)
    fig.tight_layout();fig.savefig(OUT/'11_intermit_event_recall.pdf',bbox_inches='tight')
    fig.savefig(OUT/'11_intermit_event_recall.png',dpi=190,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(9.6,3.6))
    opts=[('kf_cv4_lowq','Lower process noise'),('kf_cv4','Default process noise'),('kf_cv4_highq','Higher process noise')]
    for j,metric in enumerate(['event_recall_pct','mean_false_alerts_per_100_negative_samples']):
        for i,(m,label) in enumerate(opts):
            vals=[float(main.loc[(main.scenario.eq(f'{s}__measured_proxy'))&(main.method.eq(m)),metric].iloc[0]) if m=='kf_cv4' else
                  float(summary.loc[(summary.scenario.eq(f'{s}__measured_proxy'))&(summary.method.eq(m)),metric].iloc[0])
                  for s in SKIES[:3]]
            ax[j].plot(range(3),vals,'o-',lw=1.8,label=label)
        ax[j].set_xticks(range(3),[ROOF_LABELS[s] for s in SKIES[:3]])
        ax[j].set_ylabel('Event recall (%)' if j==0 else 'False alerts per 100 negative samples')
        ax[j].grid(alpha=.2)
    ax[0].legend(fontsize=8,frameon=False);fig.tight_layout()
    fig.savefig(OUT/'11_process_noise_sensitivity.png',dpi=190,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,3.8))
    choose=boot[boot.signal.eq('measured_proxy')&boot.method.isin(['kf_cv6','kf_cv4','kf_ca4'])]
    for i,method in enumerate(['kf_cv6','kf_cv4','kf_ca4']):
        b=choose[choose.method.eq(method)].set_index('sky').loc[list(SKIES[:3])]
        mean=b.delta_recall_pp.to_numpy()
        err=np.vstack((mean-b.lower_95_pp.to_numpy(),b.upper_95_pp.to_numpy()-mean))
        ax.errorbar(np.arange(3)+(i-1)*.11,mean,yerr=err,fmt='o-',capsize=3,
                    color=color[method],label=LABELS[method])
    ax.axhline(0,color='#666',lw=1)
    ax.set_xticks(range(3),[ROOF_LABELS[s] for s in SKIES[:3]])
    ax.set_ylabel('Change in event recall vs paired WLS (percentage points)')
    ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(OUT/'11_paired_bootstrap.png',dpi=190,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.7,3.7))
    for method in ['kf_cv6','kf_cv4','kf_ca4']:
        subset=df[(df.method.eq(method))&df.accepted&df.solution_type.eq('prediction')&df.scenario.str.contains('__measured_proxy')]
        if not len(subset):continue
        points=subset.groupby('seconds_since_update').apply(
            lambda g:pd.Series({'rmse':np.sqrt(np.mean((g.est_speed_mps-g.true_speed_mps)**2)),
                               'n':len(g)}),include_groups=False)
        ax.plot(points.index,points.rmse,'o-',label=LABELS[method],color=color[method])
    ax.set_xlabel('Time since last satellite update (s)')
    ax.set_ylabel('Prediction-only speed RMSE (m/s)')
    ax.set_xticks([1,2]);ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(OUT/'11_outage_prediction_error.png',dpi=190,bbox_inches='tight');plt.close(fig)

    # Acceleration bins are derived from reference motion only for post-hoc
    # error analysis; they are never fed to the filter itself.
    bins=[-1,1,3,1e9]
    labels=['<1','1–3','≥3']
    ref=df[(df.scenario=='open__measured_proxy')&df.accepted&df.method.isin(['wls','wls4','kf_cv4','kf_ca4'])].copy()
    ref['bin']=pd.cut(np.abs(ref.reference_speed_change_mps2),bins=bins,labels=labels)
    fig,ax=plt.subplots(figsize=(7.4,3.7))
    for method in ['wls','wls4','kf_cv4','kf_ca4']:
        vec=[]
        for label in labels:
            g=ref[(ref.method==method)&(ref['bin']==label)]
            vec.append(float(np.sqrt(np.mean((g.est_speed_mps-g.true_speed_mps)**2))) if len(g) else np.nan)
        ax.plot(range(len(labels)),vec,'o-',label=LABELS[method],color=color[method])
    ax.set_xticks(range(len(labels)),labels)
    ax.set(xlabel='Reference speed change, |Δspeed/Δt| (m/s²)',ylabel='Speed RMSE (m/s)')
    ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(OUT/'11_maneuver_sensitivity.png',dpi=190,bbox_inches='tight');plt.close(fig)

    study=control[control.scenario.isin(['partial_canopy__measured_proxy__entry',
                                     'deep_canopy__measured_proxy__entry',
                                     'open__measured_proxy__outage1',
                                     'open__measured_proxy__outage2'])]
    scenarios=['partial_canopy__measured_proxy__entry','deep_canopy__measured_proxy__entry',
               'open__measured_proxy__outage1','open__measured_proxy__outage2']
    labels=['Partial entry','Deep entry','1-s outage','2-s outage']
    fig,ax=plt.subplots(figsize=(8.7,4))
    for i,method in enumerate(['kf_cv4','kf_ca4']):
        d=study[study.method.eq(method)].set_index('scenario').loc[scenarios]
        mean=d.delta_event_recall_pp.to_numpy()
        err=np.vstack([mean-d.lower_95_pp.to_numpy(),d.upper_95_pp.to_numpy()-mean])
        ax.errorbar(np.arange(4)+(i-.5)*.10,mean,yerr=err,fmt='o-',capsize=3,
                    label=LABELS[method],lw=1.8)
    ax.axhline(0,c='gray',lw=.8)
    ax.set_xticks(range(4),labels)
    ax.set_ylabel('Change in event recall vs matched 4-satellite WLS (pp)')
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'11_temporal_benefit_vs_wls4.png',dpi=180,bbox_inches='tight');plt.close(fig)

    # Time-domain case selected by a deterministic rule (not best-looking example).
    example=df[(df.seed.eq(int(df.seed.min())))&(df.scenario.eq('open__measured_proxy__outage1'))&
               df.method.isin(['wls','kf_cv4','kf_ca4'])]
    gr=example.groupby(['game','play','player'])
    selected_key=next((k for k,g in gr if g.frame.nunique()>=5 and
                    g[(g.method=='kf_cv4')&g.solution_type.eq('prediction')].shape[0]>=1 and
                    g.true_speed_mps.max()>=7.),None)
    if selected_key is not None:
        eg=gr.get_group(selected_key)
        fig,ax=plt.subplots(figsize=(8,3.8))
        first=eg[eg.method.eq('wls')].sort_values('frame')
        t=(first.frame-first.frame.min()).to_numpy()/10.
        ax.plot(t,first.true_speed_mps,'k-o',label='NFL reference (10-Hz source)',ms=4)
        for method in ['wls','kf_cv4','kf_ca4']:
            zz=eg[eg.method.eq(method)].sort_values('frame')
            ax.plot(t,zz.est_speed_mps,'o--',ms=4,label=LABELS[method],color=color[method])
            if method=='kf_cv4':
                withheld=zz.solution_type.eq('prediction')
                ax.scatter(t[withheld],zz.loc[withheld,'est_speed_mps'],s=95,facecolors='none',
                           edgecolors=color[method],linewidths=1.3,zorder=4)
        ax.axhline(7,color='gray',ls=':',label='7 m/s threshold')
        ax.set(xlabel='Time since first sampled epoch (s)',ylabel='Horizontal speed (m/s)')
        ax.grid(alpha=.2);ax.legend(fontsize=7.2,ncol=2,frameon=False)
        fig.tight_layout();fig.savefig(OUT/'11_example_track.png',dpi=190,bbox_inches='tight');plt.close(fig)
        (OUT/'example_track.json').write_text(json.dumps({'seed':int(df.seed.min()),'game':int(selected_key[0]),
            'play':int(selected_key[1]),'player':int(selected_key[2]),
            'note':'Deterministically first track with >=5 sampled epochs, a reference sampled speed >=7 m/s, and >=1 CV-KF prediction epoch under explicit 1-second outages; not selected by accuracy'},indent=2))


def run(seeds=SEEDS):
    df=run_simulation(seeds)
    reconstruction=validate_paired_wls(df,strict=len(seeds)==len(SEEDS))
    df,full,summary,seedstats=analyze(df)
    b=paired_bootstrap(full)
    control=compare_against_wls4(full)
    figures(df,summary,b,control)
    print('WLS matched reference:',reconstruction)
    print(summary[summary.scenario.str.endswith('measured_proxy')][['scenario','method','event_recall_pct',
           'mean_epoch_availability_pct','gnss_update_pct','prediction_only_pct','mean_speed_rmse_on_accepted_mps',
           'mean_false_alerts_per_100_negative_samples']].to_string(index=False,float_format=lambda v:f'{v:.2f}'))
    return summary

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--seeds',type=int,default=6,help='Number of paired preset seeds (1 through 6)')
    a.add_argument('--analyze-only',action='store_true',help='Analyze existing Phase 11 per-epoch CSV without rerunning')
    args=a.parse_args()
    if args.analyze_only:
        frame=pd.read_csv(OUT/'kalman_paired_epoch_results.csv')
        _,full,summary,_=analyze(frame)
        b=paired_bootstrap(full)
        control=compare_against_wls4(full)
        figures(frame,summary,b,control)
    else:
        run(SEEDS[:args.seeds])
