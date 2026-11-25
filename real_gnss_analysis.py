"""Offline Doppler velocity reconstruction from Pixel Watch GNSS raw measurements.

GPS-only first-stage analysis with precise SP3 orbits. Results are preliminary
until independently validated against a synchronized external velocity reference.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import chi2

C = 299792458.0
GPS_UTC_LEAP_SECONDS = 18
EARTH_A = 6378137.0
EARTH_E2 = 6.69437999014e-3

RAW_COLS = {
    1: 'unix_ms', 11: 'svid', 16: 'cn0_dbhz', 17: 'pr_rate_mps',
    18: 'pr_rate_sigma_mps', 22: 'carrier_hz', 28: 'constellation',
}
FIX_COLS = {1: 'provider', 2: 'lat_deg', 3: 'lon_deg', 4: 'alt_m',
            5: 'speed_mps', 8: 'unix_ms'}


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = np.deg2rad([lat_deg, lon_deg])
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = EARTH_A / np.sqrt(1 - EARTH_E2 * sin_lat**2)
    return np.array([(n + alt_m) * cos_lat * np.cos(lon),
                     (n + alt_m) * cos_lat * np.sin(lon),
                     (n * (1 - EARTH_E2) + alt_m) * sin_lat])


def ecef_to_enu(v, lat_deg, lon_deg):
    lat, lon = np.deg2rad([lat_deg, lon_deg])
    a = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                  [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return a @ v


def load_sp3(path):
    sat_records = {}
    epoch = None
    with open(path, 'r', encoding='ascii', errors='replace') as f:
        for line in f:
            if line.startswith('*'):
                fields = line[1:].split()
                if len(fields) >= 6:
                    yy, mo, dd, h, mi = map(int, fields[:5])
                    sec = float(fields[5])
                    epoch = datetime(yy, mo, dd, h, mi, tzinfo=timezone.utc).timestamp() + sec
            elif line.startswith('PG') and epoch is not None:
                satellite = line[1:4]
                fields = line[4:].split()
                if len(fields) < 4:
                    continue
                nums = [float(s) for s in fields[:4]]
                if max(abs(v) for v in nums[:3]) >= 99999:
                    continue
                sat_records.setdefault(satellite, []).append((epoch, *nums))
    if not sat_records:
        raise ValueError('No GPS satellite records found in SP3 file.')
    result = {}
    for satellite, rows in sat_records.items():
        a = np.asarray(rows, dtype=float)
        order = np.argsort(a[:, 0])
        a = a[order]
        x = a[:, 1:4] * 1000.0
        t = a[:, 0]
        splines = [CubicSpline(t - t[0], x[:, i], extrapolate=False) for i in range(3)]
        clk = None
        if np.all(np.isfinite(a[:, 4])) and np.max(np.abs(a[:, 4])) < 999999:
            clk = CubicSpline(t - t[0], a[:, 4] * 1e-6, extrapolate=False)
        result[satellite] = (t[0], t[-1], splines, clk)
    return result


def satellite_state(orbits, sv, gps_epoch):
    if sv not in orbits:
        return None
    t0, t1, splines, clk = orbits[sv]
    if not t0 <= gps_epoch <= t1:
        return None
    t = gps_epoch - t0
    pos = np.array([s(t) for s in splines])
    vel = np.array([s(t, 1) for s in splines])
    clk_drift = float(clk(t, 1)) if clk is not None else 0.0
    return pos, vel, clk_drift


def load_watch_data(raw_path, fix_path):
    raw = pd.read_csv(raw_path, header=None, usecols=list(RAW_COLS), low_memory=False)
    raw = raw.rename(columns=RAW_COLS)
    raw = raw[raw['constellation'].eq(1)].copy()
    raw['sv'] = 'G' + raw['svid'].astype(int).astype(str).str.zfill(2)
    raw['epoch'] = np.rint((raw['unix_ms'] - raw['unix_ms'].min()) / 1000).astype(int)
    raw = raw[
        raw['pr_rate_mps'].between(-1500, 1500)
        & raw['pr_rate_sigma_mps'].between(0.05, 10)
        & raw['cn0_dbhz'].ge(18)
    ].copy()
    raw = raw.sort_values(['epoch', 'sv', 'pr_rate_sigma_mps'])
    raw = raw.drop_duplicates(['epoch', 'sv'], keep='first')

    fix = pd.read_csv(fix_path, header=None, usecols=list(FIX_COLS), low_memory=False)
    fix = fix.rename(columns=FIX_COLS)
    fix = fix[fix['provider'].eq('GPS')].copy().sort_values('unix_ms')
    fix = fix[fix['lat_deg'].between(-90, 90) & fix['lon_deg'].between(-180, 180)]
    return raw, fix


def prepare_measurements(raw, fix, orbits):
    t = raw.groupby('epoch', as_index=False)['unix_ms'].median()
    t = t.sort_values('unix_ms')
    t['unix_ms'] = t['unix_ms'].round().astype('int64')
    gpsfix = fix[['unix_ms', 'lat_deg', 'lon_deg', 'alt_m', 'speed_mps']]
    merged = pd.merge_asof(t, gpsfix, on='unix_ms', direction='nearest', tolerance=2200)
    fallback = gpsfix[['lat_deg', 'lon_deg', 'alt_m']].median()
    for f in ['lat_deg', 'lon_deg', 'alt_m']:
        merged[f] = merged[f].fillna(fallback[f])
    epoch_info = merged.set_index('epoch')

    measurement_groups = {}
    orbit_rejected = 0
    for epoch, group in raw.groupby('epoch'):
        info = epoch_info.loc[epoch]
        rx = geodetic_to_ecef(info['lat_deg'], info['lon_deg'], info['alt_m'])
        obs = []
        for a in group.itertuples(index=False):
            gps_sec = a.unix_ms / 1000.0 + GPS_UTC_LEAP_SECONDS
            state = satellite_state(orbits, a.sv, gps_sec)
            if state is None:
                orbit_rejected += 1
                continue
            satp, satv, satclk = state
            los = satp - rx
            rho = np.linalg.norm(los)
            if not (1e7 < rho < 5e7):
                orbit_rejected += 1
                continue
            unit = los / rho
            elevation = np.degrees(np.arcsin(np.clip(np.dot(unit,rx/np.linalg.norm(rx)), -1, 1)))
            y = a.pr_rate_mps - np.dot(unit, satv) + C * satclk
            obs.append((a.sv, y, a.pr_rate_sigma_mps, a.cn0_dbhz,
                        elevation, *(-unit), a.pr_rate_mps))
        if obs:
            measurement_groups[int(epoch)] = pd.DataFrame(obs, columns=[
                'sv', 'y', 'sigma', 'cn0', 'elevation', 'hx', 'hy', 'hz', 'pr_rate'])
    return measurement_groups, epoch_info, orbit_rejected


def solve_wls(data, weight_factor=None):
    h = np.column_stack((data[['hx', 'hy', 'hz']].to_numpy(dtype=float), np.ones(len(data))))
    y = data['y'].to_numpy(dtype=float)
    sigma = data['sigma'].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(sigma, 0.15)**2
    if weight_factor is not None:
        weights *= weight_factor
    a = h * np.sqrt(weights[:, None])
    b = y * np.sqrt(weights)
    if np.linalg.matrix_rank(a) < 4:
        return None
    est, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual = y - h @ est
    invnormal = np.linalg.pinv(h.T @ (h * weights[:, None]))
    leverage = np.einsum('ij,jk,ik->i', h, invnormal, h) * weights
    z = residual / np.maximum(sigma * np.sqrt(np.maximum(1-leverage, 0.04)), 0.1)
    chi = float(np.sum((residual / sigma)**2))
    dof = len(data) - 4
    sigma_velocity = float(np.sqrt(np.trace(invnormal[:3, :3])))
    return dict(est=est, residual=residual, stdresid=z, chi2=chi,
                dof=dof, formal_vel_sigma=sigma_velocity, leverage=leverage)


def solve_fde(data, threshold=4.0):
    working = data.copy()
    excluded = []
    for _ in range(max(0, len(data)-5)+1):
        sol = solve_wls(working)
        if sol is None:
            return None, excluded
        if len(working) < 6:
            break
        maxi = int(np.argmax(np.abs(sol['stdresid'])))
        global_problem = sol['chi2'] > chi2.ppf(.999, max(sol['dof'],1))
        if not(global_problem and abs(sol['stdresid'][maxi]) > threshold):
            break
        excluded.append(working.iloc[maxi]['sv'])
        working = working.drop(working.index[maxi]).reset_index(drop=True)
    return sol, excluded


def solve_robust(data, huber_k=2.2):
    factor = np.ones(len(data))
    sol = solve_wls(data)
    if sol is None:
        return None, factor
    for _ in range(15):
        z = np.abs(sol['residual']) / np.maximum(data['sigma'].to_numpy(dtype=float), 0.15)
        new_factor = np.minimum(1.0, huber_k / np.maximum(z, 1e-12))
        candidate = solve_wls(data, new_factor)
        if candidate is None:
            break
        if np.linalg.norm(candidate['est'] - sol['est']) < 1e-5:
            sol = candidate
            factor = new_factor
            break
        factor, sol = new_factor, candidate
    return sol, factor


def run_analysis(raw_path, fix_path, sp3_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, fix = load_watch_data(raw_path, fix_path)
    orbits = load_sp3(sp3_path)
    groups, info, orphan = prepare_measurements(raw, fix, orbits)
    records, obs_records = [], []
    for epoch, a in groups.items():
        if len(a) < 4:
            continue
        baseline = solve_wls(a)
        if baseline is None:
            continue
        fde, excluded = solve_fde(a)
        robust, factors = solve_robust(a)
        if fde is None or robust is None:
            continue
        row = info.loc[epoch]
        output = dict(epoch=epoch, utc_unix_ms=row['unix_ms'],
                      iso_utc=pd.to_datetime(row['unix_ms'], unit='ms', utc=True).isoformat(),
                      gps_fix_speed_mps=row['speed_mps'],
                      gps_sats=len(a), n_fde_excluded=len(excluded),
                      fde_excluded_ids=';'.join(excluded),
                      mean_cn0_dbhz=a['cn0'].mean(),
                      gps_mean_doppler_sigma_mps=a['sigma'].mean(),
                      gps_max_postfit_stdresidual=float(np.max(np.abs(baseline['stdresid']))))
        for name, sol in [('wls', baseline), ('fde', fde), ('robust',robust)]:
            v = sol['est'][:3]
            enu = ecef_to_enu(v, row['lat_deg'], row['lon_deg'])
            output.update({f'{name}_ve_mps':enu[0],f'{name}_vn_mps':enu[1],
                           f'{name}_vu_mps':enu[2],f'{name}_speed_mps':np.linalg.norm(v),
                           f'{name}_horizontal_speed_mps':np.linalg.norm(enu[:2]),
                           f'{name}_clock_drift_mps':sol['est'][3],
                           f'{name}_residual_rms_mps':np.sqrt(np.mean(sol['residual']**2)),
                           f'{name}_formal_vel_sigma_mps':sol['formal_vel_sigma'],
                           f'{name}_chi2':sol['chi2'],f'{name}_dof':sol['dof']})
        for i, obs in enumerate(a.itertuples(index=False)):
            obs_records.append(dict(epoch=epoch, iso_utc=output['iso_utc'],
                                    sv=obs.sv, cn0_dbhz=obs.cn0,
                                    pr_rate_sigma_mps=obs.sigma,
                                    elevation_deg=obs.elevation,
                                    wls_postfit_residual_mps=baseline['residual'][i],
                                    wls_standardized_residual=baseline['stdresid'][i],
                                    robust_relative_weight=factors[i],
                                    excluded_by_fde=obs.sv in excluded))
        records.append(output)
    res = pd.DataFrame.from_records(records).sort_values('epoch')
    obs = pd.DataFrame.from_records(obs_records)
    res.to_csv(output_dir / 'real_velocity_epochs.csv',index=False)
    obs.to_csv(output_dir / 'real_satellite_diagnostics.csv',index=False)
    summary = {
        'status':'PRELIMINARY (not independent ground-truth accuracy validation)',
        'measurement_model':'GPS L1 pseudorange rate; IAC precise SP3 orbit cubic-spline interpolation; WLS clock drift',
        'raw_satellite_measurement_count': int(pd.read_csv(raw_path, header=None, usecols=[0]).shape[0]),
        'gps_after_screening':len(raw),
        'usable_gps_epochs':len(res),
        'gps_epochs_total_after_screening':int(raw['epoch'].nunique()),
        'gps_sats_median':float(res['gps_sats'].median()),
        'wls_median_speed_mps':float(res['wls_speed_mps'].median()),
        'robust_median_speed_mps':float(res['robust_speed_mps'].median()),
        'fde_median_speed_mps':float(res['fde_speed_mps'].median()),
        'fde_epochs_with_exclusions':int((res['n_fde_excluded']>0).sum()),
        'fde_satellite_exclusions':int(res['n_fde_excluded'].sum()),
        'gps_fix_speed_median_mps':float(res['gps_fix_speed_mps'].median()),
        'wls_horizontal_speed_minus_fix_median_mps':float((res['wls_horizontal_speed_mps']-res['gps_fix_speed_mps']).median()),
        'wls_horizontal_speed_minus_fix_mae_mps':float(np.nanmean(abs(res['wls_horizontal_speed_mps']-res['gps_fix_speed_mps']))),
        'raw_gps_rows_missing_sp3':orphan,
        'note':'Comparisons to Android GPS Fix speed are nonindependent sanity checks, NOT velocity RMSE.',
    }
    (output_dir/'phase1_summary.json').write_text(json.dumps(summary,indent=2))
    return res,obs,summary


def main():
    p=argparse.ArgumentParser()
    root=Path(__file__).resolve().parent
    p.add_argument('--raw',type=Path,default=root/'data'/'Raw.csv')
    p.add_argument('--fix',type=Path,default=root/'data'/'Fix.csv')
    p.add_argument('--sp3',type=Path,default=root/'data'/'Sta23062.sp3')
    p.add_argument('--out',type=Path,default=root/'results')
    a=p.parse_args()
    _,_,s=run_analysis(a.raw,a.fix,a.sp3,a.out)
    print(json.dumps(s,indent=2))

if __name__=='__main__':
    main()
