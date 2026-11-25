"""Paired descriptive head-reference/wrist-GNSS comparison and temporal sensitivity.

No claim of co-located watch-velocity truth is made. Bootstrap intervals quantify
variation within this single walk, not population or certified sensor accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {
    'Android reported speed': 'gps_fix_speed_mps',
    'Doppler WLS': 'wls_horizontal_speed_mps',
    'Doppler FDE': 'fde_horizontal_speed_mps',
    'Doppler robust WLS': 'robust_horizontal_speed_mps',
}
SEED = 272


def filtered_common(data: pd.DataFrame, include_android: bool = False) -> pd.DataFrame:
    columns = ['rg_inferred_horizontal_speed_mps', 'wls_horizontal_speed_mps',
               'fde_horizontal_speed_mps', 'robust_horizontal_speed_mps']
    if include_android:
        columns.append('gps_fix_speed_mps')
    out = data.loc[data.gps_sats >= 7].dropna(subset=columns).copy()
    out['utc'] = pd.to_datetime(out.utc, utc=True)
    return out.sort_values('utc').reset_index(drop=True)


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(error, dtype=float) ** 2)))


def mae(error: np.ndarray) -> float:
    return float(np.mean(np.abs(error)))


def basic_metrics(a: np.ndarray, ref: np.ndarray) -> dict:
    err = np.asarray(a) - np.asarray(ref)
    return {'n': len(err), 'rmse_mps': rmse(err), 'mae_mps': mae(err),
            'median_absolute_disagreement_mps': float(np.median(np.abs(err))),
            'mean_signed_difference_mps': float(err.mean()),
            'correlation': float(np.corrcoef(a, ref)[0, 1])}


def moving_block_indices(n: int, block: int, n_reps: int, rng: np.random.Generator):
    """Circular within-recording moving-block resampling for paired epochs."""
    if n < 2 or block < 1 or n_reps < 1:
        raise ValueError('Bootstrap needs at least two observations, positive blocks and replicates.')
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_reps, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_reps, -1)[:, :n]


def paired_block_interval(df: pd.DataFrame, alternative: str, block: int = 20,
                          n_reps: int = 4000, seed: int = SEED) -> dict:
    reference = df.rg_inferred_horizontal_speed_mps.to_numpy()
    base = df.wls_horizontal_speed_mps.to_numpy() - reference
    alt = df[alternative].to_numpy() - reference
    idx = moving_block_indices(len(df), block, n_reps, np.random.default_rng(seed))
    base_b, alt_b = base[idx], alt[idx]
    rmse_delta = np.sqrt((alt_b ** 2).mean(axis=1)) - np.sqrt((base_b ** 2).mean(axis=1))
    mae_delta = np.abs(alt_b).mean(axis=1) - np.abs(base_b).mean(axis=1)
    return {'alternative_column': alternative, 'block_size_epochs': block,
            'n_bootstrap': n_reps, 'n_paired_epochs': len(df),
            'observed_rmse_change_vs_wls_mps': rmse(alt) - rmse(base),
            'rmse_change_95pct_descriptive_interval_mps': list(map(float, np.quantile(rmse_delta, [.025, .975]))),
            'observed_mae_change_vs_wls_mps': mae(alt) - mae(base),
            'mae_change_95pct_descriptive_interval_mps': list(map(float, np.quantile(mae_delta, [.025, .975])))}


def smoothed_comparison(data: pd.DataFrame) -> pd.DataFrame:
    df = filtered_common(data, include_android=True).set_index('utc')
    outputs = []
    for window_seconds, min_points in [(1, 1), (3, 3), (5, 3), (10, 7)]:
        if window_seconds == 1:
            smoothed = df
        else:
            cols = ['rg_inferred_horizontal_speed_mps', *METHODS.values()]
            smoothed = df[cols].rolling(f'{window_seconds}s', center=True,
                                        min_periods=min_points).mean()
        for label, column in METHODS.items():
            pair = smoothed[[column, 'rg_inferred_horizontal_speed_mps']].dropna()
            if len(pair) < 2:
                continue
            outputs.append({'window_s': window_seconds, 'method': label,
                            'matched_epochs': len(pair),
                            **{k: v for k,v in basic_metrics(pair[column].to_numpy(),
                                       pair.rg_inferred_horizontal_speed_mps.to_numpy()).items() if k != 'n'}})
    return pd.DataFrame(outputs)


def satellite_strata(data: pd.DataFrame) -> pd.DataFrame:
    df = filtered_common(data)
    out = []
    for n_sats, group in df.groupby('gps_sats'):
        if len(group) < 15:
            continue
        ref = group.rg_inferred_horizontal_speed_mps.to_numpy()
        for label, col in METHODS.items():
            q = group[[col, 'rg_inferred_horizontal_speed_mps']].dropna()
            if len(q) < 15:
                continue
            out.append({'satellites': int(n_sats), 'method': label,
                        'n': len(q), 'rmse_mps': rmse(q[col].to_numpy()-q.rg_inferred_horizontal_speed_mps.to_numpy())})
    return pd.DataFrame(out)


def make_plot(df: pd.DataFrame, smooth: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    df = df.sort_values('utc')
    t = (df.utc-df.utc.iloc[0]).dt.total_seconds()/60
    ax[0].plot(t,df.rg_inferred_horizontal_speed_mps, color='#222222',label='NovAtel head (inferred)',linewidth=1.4)
    ax[0].plot(t,df.gps_fix_speed_mps, color='#2074a5',alpha=.8,linewidth=.95,label='Android reported')
    ax[0].scatter(t,df.wls_horizontal_speed_mps, s=6,color='#de8840',alpha=.55,label='Watch Doppler WLS')
    ax[0].set(xlabel='Minutes from first common epoch',ylabel='Horizontal speed (m/s)',title='Simultaneous real-device speed recordings')
    ax[0].legend(fontsize=8)
    sub=smooth.pivot(index='window_s',columns='method',values='rmse_mps')
    for label,color in [('Android reported speed','#2074a5'),('Doppler WLS','#de8840'),
                        ('Doppler FDE','#943b62'),('Doppler robust WLS','#358765')]:
        ax[1].plot(sub.index,sub[label],'-o',label=label,markersize=4,color=color)
    ax[1].set(xlabel='Centered smoothing window (s)',ylabel='RMS speed disagreement (m/s)',
              title='Temporal smoothing sensitivity (common epochs)')
    ax[1].legend(fontsize=7)
    for a in ax: a.grid(alpha=.22)
    fig.suptitle('Exploratory head vs wrist comparison — not co-located velocity accuracy',fontsize=11)
    fig.savefig(out, dpi=175)
    plt.close(fig)


def run(root: Path | None = None, n_reps: int = 4000) -> dict:
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/phase8'
    out.mkdir(exist_ok=True,parents=True)
    data=pd.read_csv(root/'results/phase7/matched_watch_rg_velocity_epochs.csv')
    df=filtered_common(data)
    common=filtered_common(data, include_android=True)
    ref=df.rg_inferred_horizontal_speed_mps.to_numpy()
    tables=[]
    for label,column in METHODS.items():
        sample=df.dropna(subset=[column]); r=sample.rg_inferred_horizontal_speed_mps.to_numpy()
        tables.append({'method': label,**basic_metrics(sample[column].to_numpy(),r)})
    all_comparisons=pd.DataFrame(tables)
    all_comparisons.to_csv(out/'real_device_speed_summary.csv',index=False)
    common_comparisons=pd.DataFrame([{'method':label, **basic_metrics(common[column].to_numpy(),
             common.rg_inferred_horizontal_speed_mps.to_numpy())} for label,column in METHODS.items()])
    common_comparisons.to_csv(out/'common_665_epoch_comparison.csv',index=False)
    intervals=[paired_block_interval(df,column,block=block,n_reps=n_reps)
          for column in ['fde_horizontal_speed_mps','robust_horizontal_speed_mps']
          for block in [10,20,30]]
    pd.DataFrame([{'alternative':z['alternative_column'],'block_epochs':z['block_size_epochs'],
                   'n':z['n_paired_epochs'],'rmse_delta_mps':z['observed_rmse_change_vs_wls_mps'],
                   'rmse_ci_low_mps':z['rmse_change_95pct_descriptive_interval_mps'][0],
                   'rmse_ci_high_mps':z['rmse_change_95pct_descriptive_interval_mps'][1],
                   'mae_delta_mps':z['observed_mae_change_vs_wls_mps'],
                   'mae_ci_low_mps':z['mae_change_95pct_descriptive_interval_mps'][0],
                   'mae_ci_high_mps':z['mae_change_95pct_descriptive_interval_mps'][1]}
                  for z in intervals]).to_csv(out/'paired_block_bootstrap.csv',index=False)
    smooth=smoothed_comparison(data)
    smooth.to_csv(out/'temporal_smoothing_sensitivity.csv',index=False)
    strata=satellite_strata(data)
    strata.to_csv(out/'satellite_count_sensitivity.csv',index=False)
    make_plot(common,smooth,out/'real_reference_robustness.png')
    vector_differences={}
    for name in ['wls','fde','robust']:
        east=df[f'{name}_ve_mps']-df.rg_inferred_ve_mps
        north=df[f'{name}_vn_mps']-df.rg_inferred_vn_mps
        vector_differences[name]={'rms_horizontal_vector_difference_mps':rmse(np.hypot(east,north)),
           'mean_east_component_difference_mps':float(east.mean()),
           'mean_north_component_difference_mps':float(north.mean())}
    timing=data['absolute_match_offset_ms'] if 'absolute_match_offset_ms' in data else pd.Series(dtype=float)
    summary={'scope':'Single S1-A7 walking recording; exploratory head-vs-wrist GNSS comparison',
             'matched_epochs_any_satellites':int(len(data)),
             'matched_epochs_7plus_satellites':int(len(df)),
             'matched_common_epochs_with_android':int(len(common)),
             'match_abs_time_offset_ms_median':float(timing.median()) if len(timing) else None,
             'match_abs_time_offset_ms_max':float(timing.max()) if len(timing) else None,
             'fde_modified_epoch_count':int(np.sum(np.abs(df.fde_horizontal_speed_mps-df.wls_horizontal_speed_mps)>1e-7)),
             'robust_modified_epoch_count':int(np.sum(np.abs(df.robust_horizontal_speed_mps-df.wls_horizontal_speed_mps)>1e-7)),
             'block_bootstrap_seed':SEED,'n_bootstrap':n_reps,
             'rms_component_vector_differences_mps':vector_differences,
             'methods':tables,'paired_intervals':intervals,
             'caveats':['NovAtel reverse trajectory binary offsets inferred, no official field or quality definitions',
                        'Head-mounted GNSS and forearm-mounted watch do not have identical instantaneous velocities',
                        'Android processed speeds are not independent truth and have different smoothing/processing',
                        'All windows and bootstrap resamples come from one recording, not independent subject samples',
                        'Centered smoothing uses past and future data and is not a real-time algorithm',
                        'Bootstrap 95 percent intervals are descriptive sensitivity, not calibrated sensor performance or population confidence intervals',
                        'No GNSS satellite observations from NFL games are included in this measured experiment']}
    (out/'phase8_robustness_audit.json').write_text(json.dumps(summary,indent=2))
    print(all_comparisons.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print(pd.read_csv(out/'paired_block_bootstrap.csv').to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    return summary


if __name__=='__main__':
    run()
