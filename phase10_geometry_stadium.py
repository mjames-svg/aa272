"""Paired, geometry-aware hypothetical stadium-obstruction study.

NFL trajectories are real RFID-derived motion; satellite observations are synthetic.
The roof masks are stylized engineering assumptions, not surveyed stadiums.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from phase2_validation import prepare, sample_error
from phase2_nfl_simulation import matrix_rotation, solve
from phase3_temporal_sensitivity import fde_adjustable, robust_adjustable, step_markov
from run_final_analysis import choose_tracks, SEEDS
from phase6_event_metrics import vectorized_events

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'phase10'
SKIES=('open','partial_canopy','deep_canopy','sealed_roof')
SIGNALS=('measured_proxy','degraded_sensitivity')
METHODS=('wls','fde','robust')
COLORS={'open':'#257C78','partial_canopy':'#D3A04A','deep_canopy':'#B95E4E','sealed_roof':'#686A78'}
LABELS={'open':'Open sky','partial_canopy':'Partial canopy','deep_canopy':'Deep canopy','sealed_roof':'Sealed roof'}


def sky_geometry(H, rot, orientation_deg):
    local=-(rot@H[:,:3].T).T
    az=(np.degrees(np.arctan2(local[:,0],local[:,1]))+360)%360
    el=np.degrees(np.arcsin(np.clip(local[:,2],-1,1)))
    sep=lambda center: np.abs((az-(center+orientation_deg)+180)%360-180)
    side=lambda half_width: (sep(90)<=half_width)|(sep(270)<=half_width)
    open_mask=el>=10
    partial=open_mask & ~(side(25)&(el<40))
    deep=open_mask & ~(side(45)&(el<50))
    closed=np.zeros(len(el),dtype=bool)
    assert np.all(deep<=partial) and np.all(partial<=open_mask)
    return az,el,dict(zip(SKIES,(open_mask,partial,deep,closed)))


def solve_methods(H,y,sigma,rot):
    blank=(False,np.nan,np.nan,np.nan,0,np.nan,np.nan)
    if len(y)<6:
        return {m:blank for m in METHODS}, blank
    base=solve(H,y,sigma)
    if base is None:
        return {m:blank for m in METHODS},blank
    formally_ok=base['formal_sigma']<=4
    if not formally_ok:
        return {m:blank for m in METHODS},blank
    fde,removed=fde_adjustable(H,y,sigma,z_cut=5.)
    robust=robust_adjustable(H,y,sigma,huber_k=3.5)
    estimates={'wls':base,'fde':fde,'robust':robust}
    result={}
    for meth,sol in estimates.items():
        if sol is None:
            result[meth]=blank;continue
        vel=rot@sol['x'][:3]
        speed=float(np.linalg.norm(vel[:2]))
        ok=bool(sol['dof']>0 and sol['formal_sigma']<=4 and np.isfinite(speed) and speed<=15)
        result[meth]=(ok, speed if ok else np.nan, float(sol['formal_sigma']),
                      float(sol['chi2']),len(removed) if meth=='fde' else 0,
                      float(sol['dof']),float(base['chi2']))
    g=bool(result['wls'][0] and len(y)>=6 and base['formal_sigma']<=2 and base['dof']>0
           and base['chi2']<=chi2.ppf(.99,base['dof']))
    gate=(g,result['wls'][1] if g else np.nan,float(base['formal_sigma']),float(base['chi2']),0,float(base['dof']),float(base['chi2']))
    return result,gate


def generate(seeds=SEEDS):
    OUT.mkdir(exist_ok=True,parents=True)
    real=prepare(ROOT,ROOT/'results'/'phase2')
    rot=matrix_rotation(real.epoch_info)
    tracks,cohort=choose_tracks(ROOT)
    records=[]; geometry=[]
    for seed in seeds:
        for t in tracks:
            game,play,player=t['key']
            draw=np.random.default_rng(np.random.SeedSequence([seed,game,play,player,1081]))
            phase=int(draw.integers(10))
            template=real.templates[int(draw.integers(len(real.templates)))]
            orientation=float(draw.uniform(0,180))
            H=template['H']; sig=template['sigma']; cn0=template['cn0']
            az,el,masks=sky_geometry(H,rot,orientation)
            geo_key=(seed,game,play,player)
            for sky in SKIES:
                geometry.append((*geo_key,sky,orientation,len(sig),int(masks[sky].sum()),
                                 int((el>=10).sum()),float(np.mean(el[masks[sky]])) if np.any(masks[sky]) else np.nan))
            serial=np.arange(phase,len(t['speed']),10)
            last=np.zeros(len(sig))
            fault_active=draw.random(len(sig))<.05
            fsign=draw.choice([-1.,1.],len(sig))
            famp=draw.uniform(5.,10.,len(sig))
            for idx in serial:
                innovation=sample_error(draw,cn0,real.bank)
                last=.5*last+np.sqrt(.75)*innovation
                fault_active=step_markov(draw,fault_active,.05,.7)
                newly=fault_active & (draw.random(len(sig))<.2)
                fsign[newly]=draw.choice([-1.,1.],int(newly.sum()))
                famp[newly]=draw.uniform(5.,10.,int(newly.sum()))
                fault= np.where(fault_active,fsign*famp,0.)
                truth=t['vel_enu'][idx]; state=np.r_[rot.T@truth,0.]
                mean=H@state
                for signal in SIGNALS:
                    disturbance=last if signal=='measured_proxy' else 2*last+fault
                    for sky in SKIES:
                        mask=masks[sky]
                        h=H[mask]; y=(mean+disturbance)[mask]; s=sig[mask]
                        solved,gate=solve_methods(h,y,s,rot)
                        n=int(mask.sum()); nf=int(fault_active[mask].sum()) if signal!='measured_proxy' else 0
                        for method in METHODS:
                            accepted,speed,formal,chi,excl,dof,basechi=solved[method]
                            records.append((seed,'geometry',f'{sky}__{signal}',game,play,player,int(t['frames'][idx]),
                              method,accepted,speed,float(t['speed'][idx]),n,basechi,dof,formal,nf,excl,
                              sky,signal,orientation))
        print(f'Completed paired roof experiment seed {seed}: {len(records):,} method-epoch rows',flush=True)
    cols=['seed','model','scenario','game','play','player','frame','method','accepted','est_speed_mps',
          'true_speed_mps','n_satellites','baseline_chi2','baseline_dof','baseline_formal_sigma',
          'deliberately_faulted_satellites','fde_removed_satellites','sky','signal','roof_orientation_deg']
    df=pd.DataFrame.from_records(records,columns=cols)
    gdf=pd.DataFrame.from_records(geometry,columns=['seed','game','play','player','sky','roof_orientation_deg',
                  'template_n_satellites','retained_n_satellites','open_n_satellites','mean_visible_elevation_deg'])
    df.to_csv(OUT/'paired_epoch_results.csv',index=False)
    gdf.to_csv(OUT/'paired_roof_geometry.csv',index=False)
    metadata={'seeds':list(seeds),'cohort':cohort,'sky_masks':{
        'open':'Satellite elevation >=10 degrees',
        'partial_canopy':'Open mask, excluding two opposite azimuth sectors centered at 90 and 270 deg relative to a randomized roof orientation, half-width 25 deg, below 40 deg elevation',
        'deep_canopy':'Open mask, excluding sectors with half-width 45 deg, below 50 deg elevation',
        'sealed_roof':'All direct GPS line-of-sight signals blocked; 0 satellites'},
        'roof_orientation':'Uniform in [0,180) degrees, fixed within each player track; shared across all scenarios for given seed/track',
        'signal_modes':{'measured_proxy':'Leave-one-out Pixel Watch residual proxy bootstrap with AR(1) phi=0.5; no faults',
                        'degraded_sensitivity':'Same random proxy draws x2 plus common persistent 5% satellite faults of 5-10 m/s amplitude'},
        'receiver':'GPS-only 7-9 satellite observation templates from real Pixel Watch in an unrelated location and time; hypothetical stadium geometries',
        'sampling':'1 Hz randomly phased with 10-Hz NFL reference, held-out five NFL games',
        'processing':'>=6 satellites, <=4 m/s formal 3D velocity sigma, <=15 m/s solution speed; FDE z=5, robust Huber=3.5',
        'science_limit':'No roof geometry, RF propagation, live NFL GPS receivers, real stadium GNSS measurements, or measured sprinting GNSS data.'}
    (OUT/'methods_and_assumptions.json').write_text(json.dumps(metadata,indent=2))
    return df,gdf,real,rot


def analyze(df,gdf):
    ev,outcomes,epochs,seedstats,summary=vectorized_events(ROOT,df,threshold=7.)
    ev.to_csv(OUT/'reference_events.csv',index=False)
    outcomes.to_csv(OUT/'event_outcomes.csv',index=False)
    seedstats.to_csv(OUT/'seed_level_results.csv',index=False)
    summary.loc[summary.mean_epoch_availability_pct.eq(0),'mean_speed_rmse_on_accepted_mps']=np.nan
    summary.to_csv(OUT/'stadium_method_comparison.csv',index=False)
    paired=[]
    for signal in SIGNALS:
        for method in ('wls','fde','robust','wls_epoch_gate'):
            rows=outcomes[(outcomes.method==method)&(outcomes.scenario.str.endswith(signal))]
            pivot=rows.pivot(index=['seed','game','play','player','event_ordinal'],columns='scenario',values='detected')
            main=pivot[f'open__{signal}'].astype(int)
            for sky in SKIES[1:]:
                other=pivot[f'{sky}__{signal}'].astype(int)
                for seed in sorted(rows.seed.unique()):
                    ss=main.index.get_level_values('seed')==seed
                    paired.append({'signal':signal,'method':method,'sky':sky,'seed':int(seed),
                      'baseline_detected':int(main.loc[ss].sum()),'condition_detected':int(other.loc[ss].sum()),
                      'reference_events':int(ss.sum()),'delta_recall_pp':100*float((other.loc[ss]-main.loc[ss]).mean())})
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_event_recall_differences.csv',index=False)
    bootstrap=[]
    for signal in SIGNALS:
        for method in ('wls','fde','robust','wls_epoch_gate'):
            sub=outcomes[(outcomes.method==method)&(outcomes.scenario.str.endswith(signal))]
            p=sub.pivot(index=['seed','game','play','player','event_ordinal'],columns='scenario',values='detected')
            games=np.unique(p.index.get_level_values('game'))
            base=p[f'open__{signal}'].to_numpy(int)
            gameids=p.index.get_level_values('game').to_numpy()
            rng=np.random.default_rng(np.random.SeedSequence([654, len(method),len(signal)]))
            indices={int(g):np.flatnonzero(gameids==g) for g in games}
            draw=rng.integers(0,len(games),size=(2500,len(games)))
            for sky in SKIES[1:]:
                delta=(p[f'{sky}__{signal}'].to_numpy(int)-base)*100
                cluster_sums=np.asarray([delta[indices[int(g)]].sum() for g in games])
                cluster_ns=np.asarray([len(indices[int(g)]) for g in games])
                boots=cluster_sums[draw].sum(axis=1)/cluster_ns[draw].sum(axis=1)
                bootstrap.append(dict(signal=signal,method=method,sky=sky,
                    point_delta_recall_pp=float(delta.mean()),lower_95_pp=float(np.percentile(boots,2.5)),
                    upper_95_pp=float(np.percentile(boots,97.5)),n_games=int(len(games)),
                    note='Paired game-cluster bootstrap across five held-out games; limited-cluster descriptive uncertainty'))
    bdf=pd.DataFrame(bootstrap)
    bdf.to_csv(OUT/'paired_game_cluster_bootstrap.csv',index=False)
    print('\nSIMULATED ROOF RESULT\n',summary[(summary.method=='wls')][['scenario','event_recall_pct',
          'mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps','mean_false_alerts_per_100_negative_samples']].to_string(index=False))
    return summary,seedstats,ev,outcomes,bdf


def plot_masks(real,rot):
    template=real.templates[int(np.argmax([len(t['sigma']) for t in real.templates]))]
    h=template['H']; az,el,_=sky_geometry(h,rot,0)
    fig,axs=plt.subplots(1,3,subplot_kw={'projection':'polar'},figsize=(13,4.7))
    angles=np.linspace(0,2*np.pi,361); rad=np.linspace(0,90,91)
    ang,rr=np.meshgrid(np.rad2deg(angles),rad)
    for ax,sky in zip(axs,SKIES[:3]):
        half={'open':0,'partial_canopy':25,'deep_canopy':45}[sky]
        cap={'open':0,'partial_canopy':40,'deep_canopy':50}[sky]
        sep=lambda cen: abs((ang-cen+180)%360-180)
        roof=((sep(90)<=half)|(sep(270)<=half))&(90-rr<cap)
        blocked=(90-rr<10)|roof
        ax.pcolormesh(angles,rad,blocked,cmap=matplotlib.colors.ListedColormap(['#DAECE8','#D8D9DC']),shading='auto',vmin=0,vmax=1)
        ax.scatter(np.deg2rad(az),90-el,c='#1C5464',s=46,edgecolor='white',linewidth=.6,zorder=3)
        ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
        ax.set_rticks([10,30,50,70,90]);ax.set_yticklabels(['80°','60°','40°','20°','0°'],fontsize=7)
        ax.set_title(LABELS[sky],fontweight='bold',fontsize=12)
    fig.suptitle('Idealized satellite sky masks — illustrative receiver sky; not a surveyed stadium',fontsize=13)
    fig.text(.5,.012,'Teal: direct LOS permitted   |   Gray: blocked   |   Dots: one real Pixel Watch GPS satellite geometry snapshot',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.93))
    fig.savefig(OUT/'10_synthetic_roof_sky_masks.png',dpi=175,bbox_inches='tight');plt.close(fig)


def plot_summary(summary,gdf,bdf):
    look=summary[(summary.scenario.str.endswith('measured_proxy'))].copy()
    fig,ax=plt.subplots(figsize=(10.5,5.5))
    x=np.arange(4);w=.19
    for j,method in enumerate(('wls','fde','robust','wls_epoch_gate')):
        vals=[float(look[(look.scenario==f'{sky}__measured_proxy')&(look.method==method)].event_recall_pct.iloc[0]) for sky in SKIES]
        ax.bar(x+(j-1.5)*w,vals,width=w,label={'wls':'WLS','fde':'FDE','robust':'Robust WLS','wls_epoch_gate':'Strict epoch gate'}[method])
    ceiling=float(look[look.scenario=='open__measured_proxy'].sampling_ceiling_pct.iloc[0])
    ax.axhline(ceiling,ls='--',c='#4F5F70',label=f'1-Hz sampling ceiling ({ceiling:.1f}%)')
    ax.set(ylim=(0,80),ylabel='High-speed interval recall (%)',title='Roof geometry versus NFL sprint detection (same epochs and noise draws)')
    ax.set_xticks(x,[LABELS[sky] for sky in SKIES]);ax.grid(axis='y',alpha=.2);ax.legend(ncol=3,fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'10_event_recall_by_roof.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14.3,4.6));x=np.arange(4);w=.36
    for i,(col,ylab) in enumerate((('mean_epoch_availability_pct','Valid estimates (%)'),
          ('mean_speed_rmse_on_accepted_mps','Speed RMSE on valid epochs (m/s)'),
          ('mean_false_alerts_per_100_negative_samples','False alerts per 100 negative samples'))):
        ax=axes[i]
        for j,signal in enumerate(SIGNALS):
            val=[]
            for sky in SKIES:
                row=summary[(summary.scenario==f'{sky}__{signal}')&(summary.method=='wls')].iloc[0]
                val.append(float(row[col]) if np.isfinite(row[col]) else 0.0)
            ax.bar(x+(j-.5)*w,val,width=w,label='Measured-residual proxy' if j==0 else 'Degraded noise/faults',color=['#287E7C','#D09A4B'][j])
        ax.set_xticks(x,[LABELS[s] for s in SKIES],rotation=19,ha='right')
        ax.set(ylabel=ylab);ax.grid(axis='y',alpha=.18)
        if i==0:ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Geometry obstruction and signal degradation are separate sensitivity dimensions (WLS)')
    fig.tight_layout();fig.savefig(OUT/'10_accuracy_availability_false_alerts.png',dpi=180);plt.close(fig)

    geom=gdf.groupby('sky',sort=False).retained_n_satellites.agg(['mean','median',lambda v:100*(v>=6).mean()]).reset_index()
    geom.columns=['sky','mean_satellites','median_satellites','at_least_6_pct']
    geom.to_csv(OUT/'roof_visibility_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.9))
    for sky in SKIES:
        subset=gdf[gdf.sky==sky].retained_n_satellites.to_numpy()
        ax.hist(subset,bins=np.arange(-.5,10.6,1),histtype='step',density=True,lw=2.1,
                label=f'{LABELS[sky]} (≥6: {(subset>=6).mean()*100:.1f}%)',color=COLORS[sky])
    ax.set(xlabel='Unobstructed GPS satellites in sky mask',ylabel='Fraction of paired track trials',
           title='Sky geometry: reduction in GPS-only satellite availability')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'10_satellite_visibility.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.8,4.8));x=np.arange(3)
    subset=bdf[(bdf.signal=='measured_proxy')&(bdf.method.isin(['wls','fde','robust']))]
    for j,meth in enumerate(('wls','fde','robust')):
        rows=subset[subset.method==meth].set_index('sky').loc[list(SKIES[1:])]
        y=rows.point_delta_recall_pp.to_numpy(float)
        low=y-rows.lower_95_pp.to_numpy(float);up=rows.upper_95_pp.to_numpy(float)-y
        ax.errorbar(x+(j-1)*.12,y,yerr=np.vstack((low,up)),fmt='o-',capsize=4,label=meth.upper(),lw=1.8)
    ax.axhline(0,c='#777',lw=.7)
    ax.set_xticks(x,[LABELS[s] for s in SKIES[1:]])
    ax.set(ylabel='Change in event recall vs paired open-sky case (percentage points)',
           title='Paired event-detection effect of assumed roof geometry')
    ax.grid(axis='y',alpha=.18);ax.legend()
    fig.text(.5,.012,'Whiskers: 95% game-cluster bootstrap (5 held-out games); descriptive, not stadium-population confidence',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'10_paired_game_bootstrap.png',dpi=180);plt.close(fig)
    return geom


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=6)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.plots_only:
        real=prepare(ROOT,ROOT/'results'/'phase2');rot=matrix_rotation(real.epoch_info)
        summary=pd.read_csv(OUT/'stadium_method_comparison.csv')
        bdf=pd.read_csv(OUT/'paired_game_cluster_bootstrap.csv')
        gdf=pd.read_csv(OUT/'paired_roof_geometry.csv')
        ev=pd.read_csv(OUT/'reference_events.csv')
    else:
        df,gdf,real,rot=generate(SEEDS[:args.seeds])
        summary,seedstats,ev,outcomes,bdf=analyze(df,gdf)
    plot_masks(real,rot);geom=plot_summary(summary,gdf,bdf)
    assert len(ev)==287, f'Expected 287 events from fixed heldout reference; observed {len(ev)}'
    assert (summary[summary.scenario.str.startswith('sealed_roof')].event_recall_pct==0).all()
    assert (summary[summary.scenario.str.startswith('sealed_roof')].mean_epoch_availability_pct==0).all()
    print('Phase 10 complete:',OUT,'reference events',len(ev),'paired geometry',len(gdf))

if __name__=='__main__':main()
