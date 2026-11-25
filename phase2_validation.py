"""Audit Phase 1 satellite estimates and derive an empirical GNSS error sampler.

Error samples are leave-one-out prediction residual proxies, not independent
measurement-error truth. No real-data velocity RMSE is claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from real_gnss_analysis import (load_watch_data, load_sp3, prepare_measurements,
                                solve_wls)

BINS = np.array([0., 26., 30., 34., 100.])


@dataclass
class RealGNSS:
    bank: dict
    templates: list
    validation: dict
    epoch_info: pd.DataFrame


def prepare(root: Path, out: Path) -> RealGNSS:
    out.mkdir(parents=True, exist_ok=True)
    raw, fix = load_watch_data(root/'data/Raw.csv', root/'data/Fix.csv')
    groups, info, _ = prepare_measurements(raw, fix, load_sp3(root/'data/Sta23062.sp3'))
    records, samples, templates = [], [], []
    for ep, g in groups.items():
        if len(g) < 4:
            continue
        sol = solve_wls(g)
        if sol is None:
            continue
        record = dict(epoch=ep, satellites=len(g), dof=len(g)-4,
                      formal_velocity_sigma_mps=sol['formal_vel_sigma'],
                      rms_residual_mps=float(np.sqrt(np.mean(sol['residual']**2))),
                      wls_speed_mps=float(np.linalg.norm(sol['est'][:3])),
                      gnss_fix_speed_mps=float(info.loc[ep,'speed_mps']),
                      mean_cn0_dbhz=float(g.cn0.mean()))
        records.append(record)
        if len(g) >= 7:
            H = np.column_stack((g[['hx','hy','hz']].to_numpy(),np.ones(len(g))))
            template = {'H':H, 'sigma':g.sigma.to_numpy(), 'cn0':g.cn0.to_numpy(),
                        'sats':tuple(g.sv), 'epoch':int(ep)}
            templates.append(template)
            for i in range(len(g)):
                leverage = float(sol['leverage'][i])
                if leverage > .80:
                    continue
                loo = float(sol['residual'][i]/(1-leverage))
                samples.append(dict(epoch=ep,sv=g.sv.iloc[i],cn0_dbhz=g.cn0.iloc[i],
                                    sigma_mps=g.sigma.iloc[i],leverage=leverage,
                                    loo_prediction_residual_mps=loo))
    audit = pd.DataFrame(records)
    bank_frame = pd.DataFrame(samples)
    bank_frame['bin']=pd.cut(bank_frame.cn0_dbhz,BINS,labels=False,include_lowest=True)
    bank = {}
    binsummary=[]
    for ib in range(4):
        sub=bank_frame[bank_frame['bin']==ib]
        arr=sub.loo_prediction_residual_mps.to_numpy()
        median=float(np.median(arr))
        bank[ib]=arr-median
        binsummary.append({'cn0_bin_dbhz':f'{BINS[ib]:g}-{BINS[ib+1]:g}',
                          'observations':len(arr),
                          'median_abs_leave_one_out_residual_mps':float(np.median(abs(arr))),
                          'p95_abs_leave_one_out_residual_mps':float(np.percentile(abs(arr),95))})
    bank_frame.to_csv(out/'empirical_leave_one_out_residuals.csv',index=False)
    audit.to_csv(out/'real_quality_audit.csv',index=False)
    by_sat = (audit.groupby('satellites').agg(
        epochs=('epoch','size'), median_formal_sigma_mps=('formal_velocity_sigma_mps','median'),
        median_wls_speed_mps=('wls_speed_mps','median')).reset_index())
    by_sat.to_csv(out/'real_quality_by_satellite_count.csv',index=False)
    rho,p=spearmanr(bank_frame.cn0_dbhz,abs(bank_frame.loo_prediction_residual_mps))
    first= (audit.satellites==4).sum()
    good=int((audit.satellites>=7).sum())
    validation=dict(
        method='WLS GPS-only satellite pseudorange rate with precise SP3 geometry',
        interpretation='observed residuals are proxies; no external truth for absolute velocity RMSE',
        watch_epoch_count=len(audit), four_satellite_epochs=int(first),
        epochs_at_least_seven_satellites=good,
        empirical_residual_count=len(bank_frame),
        cn0_vs_absolute_LOO_spearman_rho=float(rho),
        cn0_vs_absolute_LOO_spearman_p=float(p),
        bin_statistics=binsummary,
        ppk_reference='Proprietary Waypoint/NovAtel PPK_expe_AG.rg is not decoded: no independent velocity ground truth',
        gps_fix_comparison='GPS Fix speed is same receiver processing, not independent validation',
        caveat='LOO residuals can contain model mismatch, biases and motion effects; not i.i.d. measurement-error ground truth'
    )
    (out/'real_validation_summary.json').write_text(json.dumps(validation,indent=2))

    fig,ax=plt.subplots(figsize=(8.5,4.8))
    data=[abs(bank_frame.loc[bank_frame.bin==i,'loo_prediction_residual_mps'].to_numpy()) for i in range(4)]
    ax.boxplot(data,showfliers=False,labels=[f'{BINS[i]:.0f}–{BINS[i+1]:.0f}' if i<3 else '≥34' for i in range(4)])
    ax.set(xlabel='C/N₀ bin (dB-Hz)',ylabel='Absolute leave-one-out residual (m/s)',
           title='Measured GPS Doppler prediction residuals by signal quality')
    ax.grid(alpha=.2,axis='y');fig.tight_layout();fig.savefig(out/'real_LOO_residuals_by_CNo.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.scatter(audit.satellites, audit.formal_velocity_sigma_mps,s=12,alpha=.24)
    ax.set(xlabel='Number of usable GPS satellites',ylabel='Formal 3D velocity uncertainty (m/s)',
           title='Satellite redundancy and formal WLS uncertainty')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'real_satellite_count_uncertainty.png',dpi=170);plt.close(fig)
    return RealGNSS(bank,templates,validation,info)


def sample_error(rng, cn0, bank):
    labels=np.clip(np.searchsorted(BINS[1:],cn0,side='right'),0,3)
    result=np.empty(len(cn0))
    for i,binid in enumerate(labels):
        arr=bank[int(binid)]
        result[i]=float(arr[rng.integers(len(arr))])
    return result


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    report=prepare(root,root/'results/phase2')
    print(json.dumps(report.validation,indent=2))
