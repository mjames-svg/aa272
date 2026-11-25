from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase3_reference_audit import parse_rss

GPS_UTC_OFFSET_S = 18
GPS_EPOCH = pd.Timestamp('1980-01-06', tz='UTC')
EARTH_A = 6378137.0
EARTH_E2 = 6.69437999014e-3
RECORD_HEADER = b'\xdc\x52'
RECORD_LENGTH = 224
START = 512


def read_rg(path: Path) -> pd.DataFrame:
    data = path.read_bytes()
    if not data.startswith(b'$WPGTRJ\x00'):
        raise ValueError('Unexpected .rg file signature.')
    footer = data.find(b'$WPGFTR\x00', START)
    if footer < 0 or (footer - START) % RECORD_LENGTH:
        raise ValueError('Unrecognized .rg record layout or footer.')
    n = (footer - START) // RECORD_LENGTH
    if any(data[START + i * RECORD_LENGTH: START + i * RECORD_LENGTH + 2] != RECORD_HEADER for i in range(n)):
        raise ValueError('Inconsistent .rg record header.')

    def field(offset: int, dtype: str) -> np.ndarray:
        return np.ndarray((n,), dtype=dtype, buffer=data, offset=START + offset, strides=(RECORD_LENGTH,)).copy()

    week = field(10, '<u2').astype(np.int64)
    tow = field(2, '<f8')
    lat = field(47, '<f8')
    lon = field(55, '<f8')
    height = field(63, '<f8')
    ve = field(122, '<f4').astype(float)
    vn = field(126, '<f4').astype(float)
    vu = field(130, '<f4').astype(float)
    if not ((week == 2306).all() and np.isfinite(tow).all() and np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError('File differs from the verified March 2024 sample; review decoding before use.')
    utc = GPS_EPOCH + pd.to_timedelta(week * 604800 + tow - GPS_UTC_OFFSET_S, unit='s')
    return pd.DataFrame({'utc': utc, 'gps_week': week, 'gps_tow_s': tow,
                         'rg_latitude_deg': np.rad2deg(lat), 'rg_longitude_deg': np.rad2deg(lon),
                         'rg_ellipsoidal_height_m': height,
                         'rg_inferred_ve_mps': ve, 'rg_inferred_vn_mps': vn, 'rg_inferred_vu_mps': vu,
                         'rg_inferred_horizontal_speed_mps': np.hypot(ve, vn)}).sort_values('utc').reset_index(drop=True)


def verify_positions(rg: pd.DataFrame, rss_path: Path) -> tuple[pd.DataFrame, dict]:
    rss = parse_rss(rss_path)
    matches = pd.merge_asof(rss, rg, on='utc', direction='nearest', tolerance=pd.Timedelta(milliseconds=110))
    matches = matches.dropna(subset=['rg_latitude_deg']).copy()
    lat0 = np.deg2rad(matches['rg_latitude_deg'].to_numpy())
    north = (matches['latitude_deg'] - matches['rg_latitude_deg']) * 111132.9
    east = (matches['longitude_deg'] - matches['rg_longitude_deg']) * 111320 * np.cos(lat0)
    matches['rss_rg_horizontal_difference_m'] = np.hypot(east, north)
    stats = {'rss_log_positions': int(len(rss)), 'matched_positions': int(len(matches)),
             'median_rss_rg_horizontal_difference_m': float(matches.rss_rg_horizontal_difference_m.median()),
             'p95_rss_rg_horizontal_difference_m': float(matches.rss_rg_horizontal_difference_m.quantile(.95)),
             'max_rss_rg_horizontal_difference_m': float(matches.rss_rg_horizontal_difference_m.max())}
    return matches, stats


def verify_velocities(rg: pd.DataFrame) -> dict:
    lat = np.deg2rad(rg.rg_latitude_deg.to_numpy())
    lon = np.deg2rad(rg.rg_longitude_deg.to_numpy())
    alt = rg.rg_ellipsoidal_height_m.to_numpy()
    t = rg.gps_week.to_numpy() * 604800 + rg.gps_tow_s.to_numpy()
    dt = t[2:] - t[:-2]
    good = np.isclose(dt, 0.4, atol=1e-4)
    phi = lat[1:-1]
    w = 1 - EARTH_E2 * np.sin(phi) ** 2
    m = EARTH_A * (1-EARTH_E2) / w ** 1.5
    nn = EARTH_A / np.sqrt(w)
    east = (lon[2:] - lon[:-2]) * (nn+alt[1:-1]) * np.cos(phi) / dt
    north = (lat[2:] - lat[:-2]) * (m+alt[1:-1]) / dt
    ve = rg.rg_inferred_ve_mps.to_numpy()[1:-1]
    vn = rg.rg_inferred_vn_mps.to_numpy()[1:-1]
    return {'finite_difference_n': int(good.sum()),
            'east_velocity_vs_position_difference_correlation': float(np.corrcoef(ve[good],east[good])[0,1]),
            'north_velocity_vs_position_difference_correlation': float(np.corrcoef(vn[good],north[good])[0,1]),
            'east_velocity_vs_position_difference_rmse_mps': float(np.sqrt(np.mean((ve[good]-east[good])**2))),
            'north_velocity_vs_position_difference_rmse_mps': float(np.sqrt(np.mean((vn[good]-north[good])**2)))}


def compare_watch(rg: pd.DataFrame, epochs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    watch = pd.read_csv(epochs_path)
    watch['utc'] = pd.to_datetime(watch.utc_unix_ms, unit='ms', utc=True)
    watch = watch.sort_values('utc')
    rg = rg.sort_values('utc').copy()
    rg['rg_reference_utc'] = rg['utc']
    matched = pd.merge_asof(watch, rg, on='utc', direction='nearest', tolerance=pd.Timedelta(milliseconds=110))
    matched = matched.dropna(subset=['rg_inferred_horizontal_speed_mps']).copy()
    matched['absolute_match_offset_ms'] = (matched['utc'] - matched['rg_reference_utc']).dt.total_seconds().abs() * 1000
    methods = [('Android GPS reported', 'gps_fix_speed_mps'),
               ('Doppler WLS', 'wls_horizontal_speed_mps'),
               ('Doppler FDE', 'fde_horizontal_speed_mps'),
               ('Doppler robust WLS', 'robust_horizontal_speed_mps')]
    summaries = []
    for subset, frame in [('All matched epochs', matched), ('At least 7 satellites', matched[matched.gps_sats >= 7])]:
        for label, col in methods:
            sample = frame.dropna(subset=[col])
            error = sample[col].to_numpy() - sample.rg_inferred_horizontal_speed_mps.to_numpy()
            summaries.append({'subset': subset, 'method': label, 'matched_epochs': len(sample),
                              'speed_disagreement_mae_mps': float(np.mean(np.abs(error))),
                              'speed_disagreement_rmse_mps': float(np.sqrt(np.mean(error**2))),
                              'speed_disagreement_median_abs_mps': float(np.median(np.abs(error))),
                              'signed_speed_disagreement_bias_mps': float(np.mean(error)),
                              'speed_correlation': float(sample[col].corr(sample.rg_inferred_horizontal_speed_mps))})
    return matched, pd.DataFrame(summaries)


def make_figure(paired: pd.DataFrame, path: Path) -> None:
    df = paired[paired.gps_sats >= 7].copy()
    t = (df.utc - df.utc.min()).dt.total_seconds()/60
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, df.rg_inferred_horizontal_speed_mps, linewidth=1.3, label='NovAtel reverse-processed head reference (inferred)')
    axes[0].scatter(t, df.gps_fix_speed_mps, s=6, alpha=.7, label='Android reported watch speed')
    axes[0].set_ylabel('Horizontal speed (m/s)')
    axes[0].set_title('Real recordings: approximate head vs wrist speed comparison')
    axes[0].legend(fontsize=8)
    axes[1].plot(t, df.rg_inferred_horizontal_speed_mps, linewidth=1.3, label='NovAtel head reference (inferred)')
    axes[1].scatter(t, df.wls_horizontal_speed_mps, s=6, alpha=.7, label='Watch Doppler WLS')
    axes[1].set_ylabel('Horizontal speed (m/s)')
    axes[1].set_xlabel('Time since first matched watch epoch (min)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rg = read_rg(root/'data/PPK_expe_AG.rg')
    matches, poscheck = verify_positions(rg, root/'data/PPK_expe_AG.rss')
    velcheck = verify_velocities(rg)
    paired, comparison = compare_watch(rg, root/'results/real_velocity_epochs.csv')
    rg.to_csv(output/'novatel_rg_inferred_trajectory.csv', index=False)
    matches.to_csv(output/'rss_position_verification.csv', index=False)
    paired.to_csv(output/'matched_watch_rg_velocity_epochs.csv', index=False)
    comparison.to_csv(output/'speed_disagreement_comparison.csv', index=False)
    make_figure(paired, output/'rg_watch_speed_comparison.png')
    dt = rg.gps_tow_s.diff()
    summary = {'status': 'EXPLORATORY inferred binary format; not validated manufacturer export or certified velocity truth',
               'n_rg_records': len(rg), 'record_rate_hz_nominal': 5,
               'first_rg_utc': rg.utc.iloc[0].isoformat(), 'last_rg_utc': rg.utc.iloc[-1].isoformat(),
               'n_gaps_greater_than_0_2_seconds': int((dt > .21).sum()),
               'n_watch_wls_epochs': len(pd.read_csv(root/'results/real_velocity_epochs.csv')),
               'n_watch_epochs_matched_within_110ms': len(paired),
               'n_watch_epochs_matched_with_7plus_satellites': int((paired.gps_sats >= 7).sum()),
               'position_verification': poscheck, 'inferred_velocity_verification': velcheck,
               'caveats': ['Reverse GNSS solution (RG), not necessarily optimal combined CG solution',
                           'Position verification against RSS uses same NovAtel processing output',
                           'Velocity components inferred from binary offsets, not manufacturer-certified field schema',
                           'NovAtel head-mounted and Pixel Watch forearm-mounted: speed disagreement is not watch velocity RMSE',
                           'No independent binary quality/fixed-float filter was decoded',
                           'Watch Android speed is a processed speed estimate and is not independent physical ground truth',
                           'Dataset notes mention March 2023 but measurement UTC and NovAtel summary establish March 2024']}
    (output/'rg_reference_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return summary


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    run(root, root/'results/phase7')
