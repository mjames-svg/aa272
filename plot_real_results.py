"""Create publication-ready diagnostic plots from Phase 1 CSV outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(out):
    out = Path(out)
    res = pd.read_csv(out/'real_velocity_epochs.csv')
    obs = pd.read_csv(out/'real_satellite_diagnostics.csv')
    t = (res['utc_unix_ms'] - res['utc_unix_ms'].iloc[0])/1000

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(t, res['gps_fix_speed_mps'], label='Android GPS Fix (sanity check)', lw=2, alpha=.85)
    ax.plot(t, res['wls_horizontal_speed_mps'], label='Independent GPS Doppler WLS', lw=.9, alpha=.7)
    ax.plot(t, res['fde_horizontal_speed_mps'], label='Doppler WLS + FDE', lw=.9, alpha=.65)
    ax.set(xlabel='Seconds since start of recording',ylabel='Horizontal speed (m/s)',
           title='Pixel Watch A7: measured Doppler velocity and device-reported speed')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'real_speed_comparison.png', dpi=190)
    plt.close(fig)

    o = obs.copy()
    o['abs_residual'] = o['wls_postfit_residual_mps'].abs()
    fig, ax = plt.subplots(figsize=(9.5,4.8))
    ax.scatter(o['cn0_dbhz'],o['abs_residual'],s=7,alpha=.12,label='Individual satellite observations')
    b = o.groupby(pd.cut(o.cn0_dbhz, np.arange(18,47,3)),observed=True).agg(
        median_abs_residual=('abs_residual','median'),mean_cn0=('cn0_dbhz','mean'),n=('cn0_dbhz','size')).reset_index(drop=True)
    ax.plot(b.mean_cn0,b.median_abs_residual,'o-',lw=2,label='Median absolute residual by C/N₀ bin')
    ax.set(xlabel='Carrier-to-noise density C/N₀ (dB-Hz)',ylabel='Absolute post-fit Doppler residual (m/s)',
           title='Real GNSS data: Doppler consistency versus signal quality')
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'real_residual_vs_cn0.png',dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11,3.9))
    ax.plot(t,res['gps_sats'],lw=1.5,label='GPS satellites after initial screening')
    flagged=res['n_fde_excluded'].gt(0)
    ax.scatter(t[flagged],res.loc[flagged,'gps_sats'],marker='x',s=20,
               label='Epoch with ≥1 FDE exclusion')
    ax.axhline(4,ls='--',lw=.9,label='Minimum GPS-only velocity-state measurement count')
    ax.set(xlabel='Seconds since start of recording',ylabel='GPS satellites',
           title='Usable satellite count and preliminary FDE flags',ylim=(2,10))
    ax.legend(fontsize=9,ncol=2,loc='lower right')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'real_satellite_coverage.png',dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t,abs(res['fde_horizontal_speed_mps'] - res['wls_horizontal_speed_mps']),
            label='|FDE − WLS|',lw=1)
    ax.plot(t,abs(res['robust_horizontal_speed_mps'] - res['wls_horizontal_speed_mps']),
            label='|Robust − WLS|',lw=1)
    ax.set(xlabel='Seconds since start of recording',ylabel='Absolute difference in horizontal speed (m/s)',
           title='How quality-control algorithms change the real-data velocity estimates')
    ax.grid(alpha=.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out/'real_method_disagreement.png',dpi=190)
    plt.close(fig)
    print('Saved real-data plots in',out)

if __name__=='__main__':
    run(Path(__file__).resolve().parent/'results')
