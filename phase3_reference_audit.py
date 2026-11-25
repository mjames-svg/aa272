"""Extract sparse ambiguity-resolution position log and assess alignment with Pixel Watch.

The .rss Kar RestorePos records are NOT continuous validated ground-truth
velocity. NovAtel reference is on the head; watch is on a swinging forearm.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from real_gnss_analysis import geodetic_to_ecef, ecef_to_enu

GPS_UTC_OFFSET_S = 18


def dms_to_decimal(deg, minutes, seconds):
    deg = float(deg)
    return np.sign(deg) * (abs(deg) + float(minutes)/60 + float(seconds)/3600)


def parse_rss(path):
    text = Path(path).read_text(encoding='latin1')
    pattern = (r'EngageTime:\s+[^\n]*?(\d\d:\d\d:\d\d(?:\.\d+)?)\s+'
               r'(\d\d/\d\d/\d{4}).*?RestorePos:\s+([^\n]+)')
    records = []
    for clock, day, coordinates in re.findall(pattern, text, flags=re.DOTALL):
        parts = coordinates.split()
        if len(parts) != 7:
            continue
        time_gps = pd.to_datetime(f'{day} {clock}', utc=True, format='%m/%d/%Y %H:%M:%S.%f')
        records.append(dict(utc=time_gps-pd.Timedelta(seconds=GPS_UTC_OFFSET_S),
                            latitude_deg=dms_to_decimal(*parts[:3]),
                            longitude_deg=dms_to_decimal(*parts[3:6]),
                            height_m=float(parts[6]), record_type='ARTK RestorePos'))
    if not records:
        raise ValueError('No time-stamped ARTK RestorePos records parsed.')
    result=pd.DataFrame(records).sort_values('utc').reset_index(drop=True)
    assert result.utc.is_monotonic_increasing
    assert result.latitude_deg.between(40,55).all()
    assert result.longitude_deg.between(-10,10).all()
    return result


def run(root, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ppk=parse_rss(root/'data/PPK_expe_AG.rss')
    watch=pd.read_csv(root/'data/Fix.csv',header=None,usecols=[1,2,3,4,5,8]);
    watch.columns=['provider','latitude_deg','longitude_deg','height_m','device_speed_mps','unix_ms']
    watch=watch[watch.provider.eq('GPS')].copy()
    watch['utc']=pd.to_datetime(watch.unix_ms, unit='ms',utc=True)
    watch=watch.sort_values('utc')
    # Full satellite WLS analysis, grouped at 1 Hz, is external to this audit.
    epochs=pd.read_csv(root/'results/real_velocity_epochs.csv')
    epochs['utc']=pd.to_datetime(epochs.utc_unix_ms,unit='ms',utc=True)
    epochs=epochs.sort_values('utc')
    # Sparse positions: no interpolation, find only nearly simultaneous samples.
    paired=pd.merge_asof(ppk,watch[['utc','latitude_deg','longitude_deg','height_m','device_speed_mps']],
                         on='utc',direction='nearest',tolerance=pd.Timedelta(seconds=2),
                         suffixes=('_reference','_watch'))
    paired=paired.dropna(subset=['latitude_deg_watch']).copy()
    paired['north_separation_m']=(paired.latitude_deg_watch-paired.latitude_deg_reference)*111132.9
    paired['east_separation_m']=(paired.longitude_deg_watch-paired.longitude_deg_reference)*111320*np.cos(np.deg2rad(paired.latitude_deg_reference))
    paired['horizontal_separation_m']=np.hypot(paired.north_separation_m,paired.east_separation_m)
    paired.to_csv(out/'novatel_sparse_matched_position_checks.csv',index=False)
    # Slow-motion velocity approximations from sparse positions; NOT velocity ground truth.
    ppk['t_s']=(ppk.utc-ppk.utc.iloc[0]).dt.total_seconds()
    xyz=np.array([geodetic_to_ecef(t.latitude_deg,t.longitude_deg,t.height_m) for t in ppk.itertuples(index=False)])
    vel=[]
    for i in range(1,len(ppk)-1):
        dt=ppk.t_s.iloc[i+1]-ppk.t_s.iloc[i-1]
        if not (3 <= dt <= 15):
            continue
        v_ecef=(xyz[i+1]-xyz[i-1])/dt
        ve=ecef_to_enu(v_ecef,ppk.latitude_deg.iloc[i],ppk.longitude_deg.iloc[i]);
        vel.append(dict(utc=ppk.utc.iloc[i],centered_difference_window_s=float(dt),
                        approximate_reference_ve_mps=float(ve[0]),approximate_reference_vn_mps=float(ve[1]),
                        approximate_reference_horizontal_speed_mps=float(np.linalg.norm(ve[:2]))))
    vel=pd.DataFrame(vel)
    joined=pd.merge_asof(vel,epochs[['utc','wls_ve_mps','wls_vn_mps','wls_horizontal_speed_mps','gps_fix_speed_mps','gps_sats']],
                          on='utc',direction='nearest',tolerance=pd.Timedelta(seconds=1.2)).dropna(subset=['wls_horizontal_speed_mps'])
    joined['absolute_speed_disagreement_mps']=(joined.wls_horizontal_speed_mps-joined.approximate_reference_horizontal_speed_mps).abs()
    joined.to_csv(out/'novatel_sparse_speed_consistency_checks.csv',index=False)
    summary=dict(source='Waypoint/NovAtel .rss Reverse Summary, ARTK RestorePos diagnostic entries',
                 interpretation='Sparse processing-log diagnostic positions, NOT an independently certified continuous trajectory or instantaneous wrist/watch velocity reference.',
                 n_log_positions=int(len(ppk)),
                 first_utc=ppk.utc.min().isoformat(), last_utc=ppk.utc.max().isoformat(),
                 watched_gps_fix_epochs=int(len(watch)),
                 nearly_simultaneous_pairs_within_2s=int(len(paired)),
                 median_gps_fix_to_sparse_position_separation_m=float(paired.horizontal_separation_m.median()),
                 p90_separation_m=float(paired.horizontal_separation_m.quantile(.9)),
                 sparse_centered_difference_velocity_pairs_matched=int(len(joined)),
                 median_approx_reference_horizontal_speed_mps=float(joined.approximate_reference_horizontal_speed_mps.median()),
                 median_absolute_watch_wls_vs_approx_ref_speed_disagreement_mps=float(joined.absolute_speed_disagreement_mps.median()),
                 constraints=['GPS week/time conversion uses +18 GPS-UTC seconds for March 2024.',
                              'EngageTime/RestorePos event-time meaning should be confirmed with NovAtel documentation.',
                              'Reference antenna on head versus Pixel Watch on swinging left forearm: instantaneous velocities differ.',
                              'Reference positions are irregular and sparse, making derivatives sensitive to event selection.',
                              'Cannot call these differences true WLS velocity error or velocity RMSE.',
                              'Full .rg trajectory export still needed for quantitative independent validation.'])
    (out/'novatel_reference_audit.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(9,4.7))
    if len(joined):
        t=(joined.utc-joined.utc.min()).dt.total_seconds()/60
        ax.scatter(t,joined.approximate_reference_horizontal_speed_mps,label='Sparse head-reference finite difference',s=30,alpha=.85)
        ax.scatter(t,joined.wls_horizontal_speed_mps,label='Watch Doppler WLS at nearby epoch',s=27,marker='x')
    ax.set(xlabel='Time since first comparison (min)',ylabel='Horizontal speed (m/s)',
           title='Exploratory cross-device speed consistency (NOT ground-truth RMSE)')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'novatel_sparse_speed_check.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    run(root,root/'results/phase3')
