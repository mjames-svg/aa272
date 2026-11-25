"""Controlled Doppler fault injection into the real Pixel Watch recording.

This study measures *incremental estimator sensitivity* to known artificial faults.
The unmodified measurement solution is a within-recording comparison, NOT
independent physical velocity ground truth. Do not report these as velocity RMSE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from real_gnss_analysis import (
    load_watch_data, load_sp3, prepare_measurements,
    solve_wls, solve_fde, solve_robust, ecef_to_enu,
)

MAGNITUDES_MPS = (1.0, 3.0, 5.0, 10.0, 20.0)
SEED = 120272


def method_solutions(group: pd.DataFrame) -> dict:
    baseline = solve_wls(group)
    fde, fde_removed = solve_fde(group, threshold=5.0)
    robust, weights = solve_robust(group, huber_k=3.5)
    return {'WLS': (baseline, set()), 'FDE': (fde, set(fde_removed)),
            'Huber WLS': (robust, set()), 'robust_weights': weights}


def sensitivity_record(epoch: int, sat: str, amplitude: float, sign: int,
                       n_sat: int, mean_cn0: float, clean: dict, corrupt: dict,
                       sample: pd.DataFrame, reference_speed: float | None, latitude: float,
                       longitude: float) -> list[dict]:
    outputs = []
    new_fde_exclusions = corrupt['FDE'][1] - clean['FDE'][1]
    for method in ('WLS', 'FDE', 'Huber WLS'):
        current = corrupt[method][0]
        original = clean[method][0]
        if current is None or original is None:
            continue
        clean_v_enu = ecef_to_enu(original['est'][:3], latitude, longitude)
        corrupt_v_enu = ecef_to_enu(current['est'][:3], latitude, longitude)
        clean_speed = float(np.linalg.norm(clean_v_enu[:2]))
        corrupt_speed = float(np.linalg.norm(corrupt_v_enu[:2]))
        # This is the change caused by injection, not error against a ground-truth velocity.
        delta_horiz = float(np.linalg.norm(corrupt_v_enu[:2] - clean_v_enu[:2]))
        delta_speed = abs(corrupt_speed - clean_speed)
        post_count = n_sat - (len(corrupt['FDE'][1]) if method == 'FDE' else 0)
        added = {'epoch': int(epoch), 'fault_satellite': sat,
                 'fault_magnitude_mps': amplitude, 'fault_sign': sign,
                 'method': method, 'n_satellites': n_sat, 'mean_cn0_dbhz': mean_cn0,
                 'n_satellites_after_exclusion': post_count,
                 'horizontal_velocity_perturbation_mps': delta_horiz,
                 'absolute_horizontal_speed_perturbation_mps': delta_speed,
                 'injected_fault_excluded': int(method == 'FDE' and sat in corrupt['FDE'][1]),
                 'injected_fault_newly_excluded': int(method == 'FDE' and sat in new_fde_exclusions),
                 'other_new_exclusions': int(method == 'FDE' and len(new_fde_exclusions - {sat})),
                 'total_exclusions': int(method == 'FDE' and len(corrupt['FDE'][1])),
                 'enough_after_exclusion': bool(post_count >= 6),
                 'watch_speed_with_injection_mps': corrupt_speed,
                 'clean_watch_speed_mps': clean_speed,
                 'novatel_head_speed_mps': reference_speed}
        if method == 'Huber WLS':
            idx = np.flatnonzero(sample.sv.eq(sat))
            added['injected_observation_huber_factor'] = float(corrupt['robust_weights'][int(idx[0])]) if len(idx) else np.nan
        outputs.append(added)
    return outputs


def run(root: Path, out: Path, max_epochs: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw, fix = load_watch_data(root/'data/Raw.csv', root/'data/Fix.csv')
    groups, info, rejected_orbit = prepare_measurements(raw, fix, load_sp3(root/'data/Sta23062.sp3'))
    matched = pd.read_csv(root/'results/phase7/matched_watch_rg_velocity_epochs.csv')
    ref = dict(zip(matched['epoch'].astype(int), matched['rg_inferred_horizontal_speed_mps']))
    candidates = sorted((ep, g) for ep, g in groups.items() if len(g) >= 7)
    if max_epochs is not None:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, min(max_epochs,len(candidates)), dtype=int)]
    rng = np.random.default_rng(SEED)
    rows = []
    baseline = []
    for epoch, g in candidates:
        clean = method_solutions(g)
        if any(clean[name][0] is None for name in ('WLS', 'FDE', 'Huber WLS')):
            continue
        satidx = int(rng.integers(0,len(g)))
        sat = str(g.iloc[satidx]['sv'])
        sign = int(rng.choice([-1,1]))
        lat = float(info.loc[epoch, 'lat_deg']); lon = float(info.loc[epoch, 'lon_deg'])
        base_removed = clean['FDE'][1]
        baseline.append({'epoch': int(epoch), 'n_satellites': len(g),
                         'clean_fde_exclusions': len(base_removed),
                         'clean_fde_excluded_selected_satellite': int(sat in base_removed),
                         'selected_satellite': sat})
        for amp in MAGNITUDES_MPS:
            faulted = g.copy()
            faulted.loc[faulted.index[satidx],'y'] += sign * amp
            injected = method_solutions(faulted)
            rows.extend(sensitivity_record(epoch,sat,amp,sign,len(g),float(g.cn0.mean()),
                clean,injected,g,float(ref[epoch]) if epoch in ref else None, lat,lon))
    data = pd.DataFrame(rows)
    base = pd.DataFrame(baseline)
    data.to_csv(out/'measured_watch_injected_fault_epoch_results.csv',index=False)
    base.to_csv(out/'unmodified_watch_fde_exclusion_baseline.csv',index=False)
    summary = data.groupby(['fault_magnitude_mps','method'],sort=True).agg(
       evaluated_epochs=('epoch','size'),
       median_horizontal_velocity_perturbation_mps=('horizontal_velocity_perturbation_mps','median'),
       rms_horizontal_velocity_perturbation_mps=('horizontal_velocity_perturbation_mps',lambda x:float(np.sqrt(np.mean(np.square(x))))),
       median_abs_speed_perturbation_mps=('absolute_horizontal_speed_perturbation_mps','median'),
       p95_abs_speed_perturbation_mps=('absolute_horizontal_speed_perturbation_mps',lambda x:float(np.quantile(x,.95))),
       fault_exclusion_fraction=('injected_fault_excluded','mean'),
       newly_excluded_fault_fraction=('injected_fault_newly_excluded','mean'),
       other_new_exclusion_fraction=('other_new_exclusions',lambda x:float(np.mean(x>0))),
       viable_after_exclusion_fraction=('enough_after_exclusion','mean'),
    ).reset_index()
    summary.to_csv(out/'fault_injection_method_summary.csv',index=False)
    # Paired bootstrap at epoch level: resample epochs, not individual satellites.
    boot_rng = np.random.default_rng(SEED+1)
    bootstrap=[]
    for amp in MAGNITUDES_MPS:
        pair=data[data.fault_magnitude_mps.eq(amp)].pivot(index='epoch',columns='method',values='horizontal_velocity_perturbation_mps').dropna()
        if len(pair)<10: continue
        arr=pair[['WLS','FDE','Huber WLS']].to_numpy()
        idx=boot_rng.integers(0,len(arr),size=(800,len(arr)))
        b=np.sqrt(np.mean(arr[idx]**2,axis=1))
        for j,method in enumerate(('FDE','Huber WLS')):
            diff=b[:,j+1]-b[:,0]
            bootstrap.append({'fault_magnitude_mps':amp,'comparison':method+' - WLS',
                 'rms_horizontal_perturbation_delta_mps':float(np.sqrt(np.mean(arr[:,j+1]**2))-np.sqrt(np.mean(arr[:,0]**2))),
                 'resample_percentile_low_mps':float(np.quantile(diff,.025)),
                 'resample_percentile_high_mps':float(np.quantile(diff,.975)),
                 'note':'Descriptive epoch bootstrap from one recording, not independent-device confidence interval'})
    pd.DataFrame(bootstrap).to_csv(out/'paired_epoch_resampling_sensitivity.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.5),layout='constrained')
    colors={'WLS':'#2F6985','FDE':'#B17D3E','Huber WLS':'#76975F'}
    for method in colors:
        t=summary[summary.method.eq(method)]
        axes[0].plot(t.fault_magnitude_mps,t.rms_horizontal_velocity_perturbation_mps,
                    marker='o',label=method,color=colors[method],linewidth=1.8)
    axes[0].set(xlabel='Injected single-satellite error (m/s)',ylabel='RMS change in horizontal velocity (m/s)',title='Sensitivity to known artificial errors')
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False)
    s=summary[summary.method.eq('FDE')]
    axes[1].plot(s.fault_magnitude_mps,100*s.fault_exclusion_fraction,marker='o',color=colors['FDE'],label='Faulty satellite removed')
    axes[1].plot(s.fault_magnitude_mps,100*s.other_new_exclusion_fraction,marker='s',color='#85556A',label='Other satellite newly excluded')
    axes[1].set(xlabel='Injected single-satellite error (m/s)',ylabel='Fraction of trials (%)',title='FDE exclusions on measured GNSS geometry',ylim=(0,108))
    axes[1].grid(alpha=.25)
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(out/'measured_watch_fault_injection.pdf',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    metadata={'study':'Controlled one-satellite additions to measured Pixel Watch GPS pseudorange rates',
       'measured_source':'GPS-only S1-A7 smartwatch, March 19 2024',
       'n_unmodified_epochs':len(base),
       'n_injected_trial_methods':len(data),
       'unique_epoch_injection_pairs':len(base)*len(MAGNITUDES_MPS),
       'known_fault_magnitudes_mps':list(MAGNITUDES_MPS),
       'fault_selection':'one randomly chosen satellite and one random sign per epoch, held fixed across magnitudes',
       'random_seed':SEED,
       'method_settings':'real_gnss_analysis.py solve_wls, solve_fde(threshold=5), solve_robust(huber_k=3.5)',
       'clean_baseline_epochs_with_any_fde_exclusion':int((base.clean_fde_exclusions>0).sum()),
       'orbit_rejected_measurement_rows':int(rejected_orbit),
       'bootstrap_interpretation':'Descriptive within-one-walk epoch bootstrap, not population or independently verified accuracy',
       'strict_limitation':'Differences are due to artificial injection relative to each method on the unmodified same-epoch data; no certified reference velocity, no true intrinsic measurement-error classification',
       'separate_real_head_reference':'NovAtel measurements are head mounted and decoding unofficial; comparison here not used as ground truth.'}
    (out/'validation_scope.json').write_text(json.dumps(metadata,indent=2))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(json.dumps(metadata,indent=2))
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',type=Path)
    p.add_argument('--max-epochs',type=int)
    args=p.parse_args()
    run(args.root,args.out or args.root/'results/phase12',args.max_epochs)
