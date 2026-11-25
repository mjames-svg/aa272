"""Supplementary real-field GNSS and simulated obstruction sensitivity study.

Real w02/w03/w06 position errors and synthetic satellite Doppler tests are
separate evidence streams. No recording represents a roofed NFL stadium.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'phase9'
OUT.mkdir(parents=True, exist_ok=True)
SNAPPER = ROOT / 'results' / 'phase4' / 'snapper' / 'snapper_matched_position_errors.csv'
SIM = ROOT / 'results' / 'phase2' / 'nfl_simulation_method_comparison.csv'
FINAL = ROOT / 'results' / 'final' / 'main_method_comparison.csv'


def read_inputs():
    for path in (SNAPPER, SIM, FINAL):
        if not path.exists():
            raise FileNotFoundError(f'Missing existing project result: {path}')
    measured = pd.read_csv(SNAPPER)
    simulated = pd.read_csv(SIM)
    final = pd.read_csv(FINAL)
    return measured, simulated, final


def make_w02_figure(measured):
    d = measured[measured.track == 'w02'].sort_values('utc').copy()
    if len(d) < 400:
        raise AssertionError('Expected approximately 480 matched w02 fixes')
    re = d.rtk_east_m.to_numpy(); rn = d.rtk_north_m.to_numpy()
    fe = d.fix_east_m.to_numpy(); fn = d.fix_north_m.to_numpy()
    re, fe = re - np.median(re), fe - np.median(re)
    rn, fn = rn - np.median(rn), fn - np.median(rn)
    tt = (pd.to_datetime(d.utc, utc=True) - pd.to_datetime(d.utc.iloc[0], utc=True)).dt.total_seconds().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw={'width_ratios':[1.12, 1]})
    ax = axes[0]
    ax.plot(re,rn,lw=2.1,color='#0C6A62',label='RTK reference trajectory',zorder=3)
    ax.scatter(fe,fn,c=tt,cmap='plasma',s=10,alpha=.65,label='SnapperGPS fixes',zorder=2)
    for i in range(0,len(d),25):
        ax.plot([re[i],fe[i]],[rn[i],fn[i]],color='gray',lw=.6,alpha=.33,zorder=1)
    ax.set(aspect='equal',xlabel='Local east displacement (m)',ylabel='Local north displacement (m)',
           title='American football pitch (w02)\nMeasured GNSS vs. RTK')
    ax.legend(frameon=False,loc='best',fontsize=8)
    ax.grid(alpha=.18)
    err=d.horizontal_error_m.to_numpy()
    ax=axes[1]
    ax.plot(tt/60,err,lw=1.2,color='#954E8B')
    median=float(np.median(err)); p95=float(np.percentile(err,95))
    ax.axhline(median,color='#0C6A62',lw=1.4,ls='--',label=f'Median: {median:.2f} m')
    ax.axhline(p95,color='#B85D27',lw=1.4,ls=':',label=f'95th percentile: {p95:.2f} m')
    ax.set(xlabel='Elapsed time (min)',ylabel='Horizontal position disagreement (m)',
           title='Position disagreement over the recording')
    ax.legend(frameon=False,fontsize=9)
    ax.grid(alpha=.2)
    fig.suptitle('Real GNSS field recording — walking, not NFL-speed motion', fontsize=13,y=1.02)
    fig.tight_layout()
    fig.savefig(OUT/'football_field_w02_real_rtk.png',dpi=190,bbox_inches='tight')
    plt.close(fig)
    return d


def make_environment_figure(measured):
    samples = [('w02','American football pitch'),('w03','Park'),('w06','Urban')]
    rows=[]
    for track,environment in samples:
        e=measured.loc[measured.track==track,'horizontal_error_m'].dropna().to_numpy()
        if len(e)<100: raise AssertionError(f'Insufficient track data {track}')
        rows.append({'track':track,'recording_environment':environment,'matched_fixes':len(e),
                     'median_error_m':float(np.median(e)),'p95_error_m':float(np.percentile(e,95)),
                     'fraction_over_20m':float(np.mean(e>20))})
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'real_environment_track_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(9.1,5.2))
    x=np.arange(len(df)); w=.34
    bars1=ax.bar(x-w/2,df.median_error_m,width=w,color='#286E77',label='Median')
    bars2=ax.bar(x+w/2,df.p95_error_m,width=w,color='#C77B44',label='95th percentile')
    ax.bar_label(bars1,fmt='%.1f',padding=3,fontsize=9)
    ax.bar_label(bars2,fmt='%.1f',padding=3,fontsize=9)
    ax.set_xticks(x,[f'{r.track} · {r.recording_environment}\n{r.matched_fixes} matched fixes' for r in df.itertuples()])
    ax.set(ylabel='Horizontal GNSS–RTK position disagreement (m)',ylim=(0,max(df.p95_error_m)*1.23),
           title='Independent walking recordings — descriptive comparison only')
    ax.legend(frameon=False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.text(.5,.02,'These are different sites/recordings. No controlled roof effect is identifiable.',ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=(0,.065,1,1))
    fig.savefig(OUT/'real_field_vs_urban_position_errors.png',dpi=190,bbox_inches='tight')
    plt.close(fig)
    return df


def make_stress_figure(simulated, final):
    labels=[('observed_like','Open-like signal baseline'),('reduced_satellites','30% random satellite loss'),
            ('combined_degradation','30% loss + 2× noise + 5% faults')]
    df=simulated[(simulated.rate_hz==1)&(simulated.method=='wls')].set_index('scenario').loc[[r[0] for r in labels]].reset_index()
    if len(df)!=3: raise AssertionError('Expected three Phase 2 simulation cases')
    df['scenario_description']=[x[1] for x in labels]
    df['high_speed_epoch_recall_pct']=df.recall_including_rejected_as_missed*100
    df[['scenario','scenario_description','epochs','retained_pct','rms_speed_error_mps','high_speed_epoch_recall_pct']].to_csv(OUT/'exploratory_obstruction_stress_test.csv',index=False)
    main=final[(final.model=='persistent')&(final.method=='wls')&(final.scenario.isin(['observed_like','combined_degradation']))].copy()
    main[['model','scenario','method','runs','event_recall_pct','mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps']].to_csv(OUT/'main_six_seed_event_reference.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(14,4.8))
    xs=np.arange(3)
    vals=[('retained_pct','Valid velocity epochs (%)'),('rms_speed_error_mps','Speed RMSE on valid epochs (m/s)'),
          ('high_speed_epoch_recall_pct','High-speed epoch recall (%)')]
    for ax,(col,name) in zip(axs,vals):
        bars=ax.bar(xs,df[col],color=['#327E77','#BB8D4B','#B9584A'],width=.63)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks(xs,['Baseline','Satellite loss','Loss + noise/faults'],rotation=14,ha='right')
        ax.set_ylabel(name)
        ax.set_ylim(0,max(float(df[col].max())*1.21, 1))
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('NFL motion + simulated Doppler — 1 Hz WLS exploratory stress test',fontsize=13)
    fig.text(.5,.012,'827 simulated epochs per scenario · not actual stadium roof measurements · epoch recall ≠ interval/event detection',ha='center',fontsize=8.9,color='#555')
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(OUT/'nfl_obstruction_analogue_stress_test.png',dpi=190,bbox_inches='tight')
    plt.close(fig)
    return df,main


def write_report(env,sim,main):
    w02=env[env.track=='w02'].iloc[0]
    rows='\n'.join(f'| {r.track} | {r.recording_environment} | {r.matched_fixes} | {r.median_error_m:.2f} | {r.p95_error_m:.2f} |' for r in env.itertuples())
    scen='\n'.join(f'| {r.scenario_description} | {r.retained_pct:.1f}% | {r.rms_speed_error_mps:.2f} | {r.high_speed_epoch_recall_pct:.1f}% |' for r in sim.itertuples())
    base=main.set_index('scenario')
    main_lines='\n'.join(f'| {scenario} | {base.loc[scenario,"mean_epoch_availability_pct"]:.2f}% | {base.loc[scenario,"event_recall_pct"]:.2f}% |' for scenario in ('observed_like','combined_degradation'))
    content=f'''# Phase 9 — Football-field GNSS and stadium-obstruction research extension

## Research question

How might satellite-signal obstruction associated with roofed stadiums affect GNSS velocity accuracy, monitoring availability, and detection of fast NFL player movements?

This addition separates three evidence streams: **measured position errors on an outdoor-type American football pitch**, **measured urban walking position errors in an independent recording**, and **simulated NFL satellite-observation degradation**. None of our measured GNSS recordings is from inside a covered NFL stadium. Do not label the stress-test scenarios as validated stadium conditions.

## Measured data (position, not Doppler)

SnapperGPS `w02` is documented as a walking recording on an *American football pitch*; `w06` is classified urban and `w03` park. Their GNSS fixes have RTK reference positions. Local displacement coordinates in the w02 figure are **not** registered to any stadium outline or actual field markings.

| Track | Documented setting | Matched fixes | Median horizontal error (m) | 95th percentile (m) |
|---|---|---:|---:|---:|
{rows}

For the w02 walking case, median error is {w02.median_error_m:.2f} m (n={int(w02.matched_fixes)}). Distinct recordings and routes have different conditions, motion and antenna behavior; their observed differences are **not a causal roof-obstruction estimate**. No per-satellite Doppler observables are provided in these processed fix files.

![Measured football pitch GNSS and RTK trajectories](results/phase9/football_field_w02_real_rtk.png)

![Real tracks, environmental context](results/phase9/real_field_vs_urban_position_errors.png)

## Exploratory 1-Hz simulated Doppler stress test (Phase 2 data reused)

This is a *supplementary, single-run, 827-epoch simulation*. It uses NFL player motion and GNSS signal templates from real wearable observations, but synthetic per-satellite errors. WLS is shown for interpretability. Scenario settings are **assumptions**, not calibrated roof measurements:

- Baseline: observed-like error distribution without added faults or satellite masking.
- Satellite-loss case: independent 30% random masking per satellite.
- Compound degradation: independent 30% masking, 2× error scale and 5% injected faults per remaining satellite.

| Hypothetical signal case | Valid GNSS velocity epochs | RMSE on valid epochs (m/s) | High-speed *epoch* recall |
|---|---:|---:|---:|
{scen}

![Simulated football satellite-obstruction cases](results/phase9/nfl_obstruction_analogue_stress_test.png)

This stress test is separate from the project's more rigorous six-seed, held-out game *event-level* results. Do not confuse high-speed epoch recall in the table above with complete sprint-interval detection.

## Existing primary six-seed event-level result (WLS, persistent error model)

| Scenario | Mean measurement availability | Sprint-interval detection |
|---|---:|---:|
{main_lines}

This establishes the event-level consequences of a *synthetic* combined degradation stressor, not the measured effect of a physical stadium roof.

## Interpretation and validity limits

1. **Measured:** processed SnapperGPS positions on a football pitch differ from independent RTK positions by several metres. This is not a direct GNSS speed-error or Doppler WLS test.
2. **Modeled:** satellite masking sharply reduces tracking availability; combined error/masking impacts high-speed classification and estimation reliability. Differences among the exploratory stress-test cases include random effects and should not be interpreted as calibrated causal roof effects.
3. **Unmeasured:** how many satellites or how much C/N0 a specific roof eliminates. A physically grounded stadium model would require measured satellite azimuth/elevation and sky visibility/roof geometry, or an external matched dataset taken inside and outside the stadium.
4. **No sports-system claim:** NFL Next Gen Stats data used here are player-tracking trajectories, not NFL GNSS observations. NFL tracking does not become GNSS merely because the motion is on a football field.
5. **Scientific next improvement:** construct an elevation/azimuth-dependent stadium sky mask, link loss to measured C/N0 and path delay, run paired six-seed NFL comparisons under identical trajectories and random draws, and show confidence intervals for event detection, false alarms, availability and RMSE.

## Sources

- Jonas Beuchert (2023), *SnapperGPS: Collection of GNSS Signal Snapshots 2*, University of Oxford, https://doi.org/10.5287/ora-xq5b8xva7 ; recording classifications in https://github.com/JonasBchrt/snapshot-gnss-data-2
- Kim et al. (2025), *A quantitative method for evaluating GNSS signal disruption in sports stadiums*, *Sports Engineering*, https://doi.org/10.1007/s12283-025-00530-7 . Reports stadium-roof effects but does **not** supply a paired raw stadium dataset for calibrating our exact simulations.

## Reproduction (offline, from project root)

```powershell
python phase9_stadium_context.py
```

Inputs: existing `results/phase4/snapper/snapper_matched_position_errors.csv`, `results/phase2/nfl_simulation_method_comparison.csv`, and `results/final/main_method_comparison.csv`. All generated Phase 9 outputs are under `results/phase9`.
'''
    (ROOT/'PHASE9_STADIUM_CONTEXT.md').write_text(content,encoding='utf-8')


def run():
    measured, simulated, final = read_inputs()
    make_w02_figure(measured)
    environment=make_environment_figure(measured)
    stress,main=make_stress_figure(simulated,final)
    write_report(environment,stress,main)
    print('REAL POSITION COMPARISON\n',environment.to_string(index=False))
    print('\nSIMULATED WLS STRESS TEST\n',stress[['scenario','epochs','retained_pct','rms_speed_error_mps','high_speed_epoch_recall_pct']].to_string(index=False))
    print('\nPHASE 9 COMPLETE:',OUT)


if __name__ == '__main__':
    run()
