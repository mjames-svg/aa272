"""Audit first-order receive-time vs transmit-time/Sagnac GPS Doppler model.

Finite-difference geometric range at a fixed receiver approximates satellite
range rate and includes signal travel time and Earth rotation. It is a model
sensitivity audit, not a reference-validated velocity solution.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from real_gnss_analysis import (C, GPS_UTC_LEAP_SECONDS, geodetic_to_ecef, ecef_to_enu,
    load_watch_data, load_sp3, prepare_measurements, satellite_state, solve_wls)

OMEGA_E=7.2921151467e-5

def rotate_to_reception(r_tx, tau):
    th=-OMEGA_E*tau
    ct,st=np.cos(th),np.sin(th)
    x,y,z=r_tx
    return np.array([ct*x-st*y,st*x+ct*y,z])


def corrected_pseudorange(orbits, sv, receive_gps_sec, rx):
    nominal=satellite_state(orbits,sv,receive_gps_sec)
    if nominal is None:return None
    tau=np.linalg.norm(nominal[0]-rx)/C
    tx=None
    for _ in range(4):
        tx=satellite_state(orbits,sv,receive_gps_sec-tau)
        if tx is None:return None
        sat_rotated=rotate_to_reception(tx[0],tau)
        rho=np.linalg.norm(sat_rotated-rx)
        tau=rho/C
    return rho,sat_rotated,receive_gps_sec-tau


def corrected_observation(orbits,sv,receive_gps_sec,rx,rate):
    central=corrected_pseudorange(orbits,sv,receive_gps_sec,rx)
    plus=corrected_pseudorange(orbits,sv,receive_gps_sec+.35,rx)
    minus=corrected_pseudorange(orbits,sv,receive_gps_sec-.35,rx)
    if any(v is None for v in (central,plus,minus)):
        return None
    rho,rsat,transmit_gps_sec=central
    station_geo_rate=(plus[0]-minus[0])/.70
    clock=satellite_state(orbits,sv,transmit_gps_sec)
    if clock is None:return None
    u=(rsat-rx)/rho
    y=rate-station_geo_rate+C*clock[2]
    return y,-u,station_geo_rate


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/phase3';out.mkdir(parents=True,exist_ok=True)
    raw,fix=load_watch_data(root/'data/Raw.csv',root/'data/Fix.csv')
    orbits=load_sp3(root/'data/Sta23062.sp3')
    groups,info,orphan=prepare_measurements(raw,fix,orbits)
    merged=raw[['epoch','sv','unix_ms']].drop_duplicates(['epoch','sv'])
    rows=[]; obs_deltas=[];reject=0
    for ep,frame in groups.items():
        if len(frame)<4:continue
        orig=solve_wls(frame)
        if orig is None:continue
        rx=geodetic_to_ecef(info.loc[ep,'lat_deg'],info.loc[ep,'lon_deg'],info.loc[ep,'alt_m'])
        d=frame.merge(merged[merged.epoch==ep][['sv','unix_ms']],on='sv',how='left')
        updated=[]
        for item in d.itertuples(index=False):
            change=corrected_observation(orbits,item.sv,item.unix_ms/1000.+GPS_UTC_LEAP_SECONDS,rx,item.pr_rate)
            if change is None:
                reject+=1;continue
            y,u,stationgeo=change
            updated.append(dict(sv=item.sv,y=y,sigma=item.sigma,cn0=item.cn0,
                        hx=u[0],hy=u[1],hz=u[2]))
            obs_deltas.append(dict(epoch=ep,sv=item.sv,correction_in_observation_mps=float(y-item.y),
                                   corrected_stationary_range_rate_mps=float(stationgeo)))
        newer=pd.DataFrame(updated)
        if len(newer)!=len(frame):continue
        sol=solve_wls(newer)
        if sol is None:continue
        pos=info.loc[ep]
        uncorrected_v=ecef_to_enu(orig['est'][:3],pos['lat_deg'],pos['lon_deg'])
        corrected_v=ecef_to_enu(sol['est'][:3],pos['lat_deg'],pos['lon_deg'])
        rows.append(dict(epoch=ep,unix_ms=int(pos['unix_ms']),gps_satellites=len(frame),
            initial_horizontal_speed_mps=float(np.linalg.norm(uncorrected_v[:2])),
            corrected_horizontal_speed_mps=float(np.linalg.norm(corrected_v[:2])),
            initial_vertical_velocity_mps=float(uncorrected_v[2]),
            corrected_vertical_velocity_mps=float(corrected_v[2]),
            horizontal_velocity_change_norm_mps=float(np.linalg.norm((corrected_v-uncorrected_v)[:2])),
            three_d_velocity_change_norm_mps=float(np.linalg.norm(corrected_v-uncorrected_v)),
            watch_fix_speed_mps=float(pos['speed_mps']) if np.isfinite(pos['speed_mps']) else np.nan))
    summary_frame=pd.DataFrame(rows)
    observation_frame=pd.DataFrame(obs_deltas)
    summary_frame.to_csv(out/'real_orbit_range_rate_correction_epochs.csv',index=False)
    observation_frame.to_csv(out/'real_orbit_range_rate_correction_observations.csv',index=False)
    summary=dict(epochs_analyzed=len(summary_frame),orbit_rejected_obs=int(reject),
        median_absolute_observation_correction_mps=float(observation_frame.correction_in_observation_mps.abs().median()),
        p95_absolute_observation_correction_mps=float(observation_frame.correction_in_observation_mps.abs().quantile(.95)),
        median_horizontal_velocity_solution_change_mps=float(summary_frame.horizontal_velocity_change_norm_mps.median()),
        p95_horizontal_velocity_solution_change_mps=float(summary_frame.horizontal_velocity_change_norm_mps.quantile(.95)),
        median_3d_velocity_solution_change_mps=float(summary_frame.three_d_velocity_change_norm_mps.median()),
        initial_median_horizontal_speed_mps=float(summary_frame.initial_horizontal_speed_mps.median()),
        corrected_median_horizontal_speed_mps=float(summary_frame.corrected_horizontal_speed_mps.median()),
        caveats=['Transmit-time and Sagnac numerical range-rate correction included.',
                 'No full receiver clock modeling or proprietary PPK reference velocity validation.',
                 'Finite-difference satellite-induced geometric range rate assumes a fixed receiver; receiver velocity solved linearly.',
                 'Corrections on the order of hundredths of m/s would not resolve residual outliers at the meter/s level.'])
    (out/'real_orbit_model_audit.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(8,4.6))
    ax.hist(summary_frame.horizontal_velocity_change_norm_mps,bins=35)
    ax.set(xlabel='Change in estimated horizontal velocity vector (m/s)',
        ylabel='Number of watch epochs',
        title='First-order vs transmit-time/Sagnac-corrected Doppler reconstruction')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'real_orbit_model_correction_histogram.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':run()
