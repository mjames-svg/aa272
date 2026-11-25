"""SnapperGPS RTK-position audit (distinct from Android Doppler validation).

Reads public snapshot-gnss-data-2 `fixes` and `ground-truth` JSON files.
No use of raw 12 ms snapshots: they contain baseband bits, not ready Doppler.

Source: https://github.com/JonasBchrt/snapshot-gnss-data-2
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

A = 6378137.0
E2 = 6.6943799901413165e-3


def ecef_from_latlon(lat, lon):
    lat, lon = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    sinlat, coslat = np.sin(lat), np.cos(lat)
    N = A / np.sqrt(1 - E2 * sinlat**2)
    return np.column_stack((N*coslat*np.cos(lon), N*coslat*np.sin(lon), N*(1-E2)*sinlat))


def local_xy(lat, lon, origin_lat, origin_lon):
    p = ecef_from_latlon(lat, lon)
    origin = ecef_from_latlon([origin_lat], [origin_lon])[0]
    lat0, lon0 = np.deg2rad([origin_lat, origin_lon])
    sl,cl = np.sin(lat0),np.cos(lat0)
    so,co = np.sin(lon0),np.cos(lon0)
    R = np.array([[-so, co, 0],[-sl*co, -sl*so,cl]])
    return (p-origin)@R.T


def _records(value,kind):
    required = {'latitude','longitude'}
    if isinstance(value, list):
        if not value or not isinstance(value[0], dict): return []
        if required.issubset(value[0]): return value
    if isinstance(value,dict):
        for key in ([kind, 'fixes', 'ground-truth', 'ground_truth', 'data', 'records', 'points']):
            if key in value:
                result = _records(value[key], kind)
                if result: return result
        for nested in value.values():
            if isinstance(nested,list):
                result = _records(nested,kind)
                if result: return result
    return []


def read_points(path, kind, reference_date=None):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    records = _records(payload, kind)
    if not records:
        raise ValueError(f'No {kind} lat/lon records recognized in {path}')
    df = pd.DataFrame(records)
    tcol = 'datetime' if kind == 'fixes' else 'time'
    if tcol not in df:
        raise ValueError(f'Missing {tcol} UTC timestamp in {path}: {list(df.columns)}')

    if kind == 'ground-truth':
        if reference_date is None:
            raise ValueError('Ground-truth records contain only UTC time of day; pass the date from their matching fixes file')
        anchor = pd.Timestamp(reference_date)
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize('UTC')
        else:
            anchor = anchor.tz_convert('UTC')
        base = anchor.normalize()
        raw = df[tcol].astype(str)
        if raw.str.fullmatch(r'\d{1,2}:\d{2}:\d{2}(?:\.\d+)?').all():
            tod = pd.to_timedelta(raw)
            # Unwrap a recording crossing midnight, then select the absolute
            # date nearest the matching fix recording.
            jumps = tod.diff().dt.total_seconds().fillna(0)
            day_offsets = (jumps < -12 * 3600).cumsum() - (jumps > 12 * 3600).cumsum()
            ref_times = base + tod + pd.to_timedelta(day_offsets, unit='D')
            midpoint = ref_times.iloc[len(ref_times)//2]
            days = round((anchor - midpoint).total_seconds() / 86400)
            df['utc'] = ref_times + pd.Timedelta(days=days)
        else:
            df['utc'] = pd.to_datetime(df[tcol], utc=True, errors='coerce')
    else:
        df['utc'] = pd.to_datetime(df[tcol], utc=True, errors='coerce')

    for c in ['latitude', 'longitude']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['utc', 'latitude', 'longitude'])
    df = df[df.latitude.between(-90, 90) & df.longitude.between(-180, 180)]
    return df.sort_values('utc').drop_duplicates('utc').reset_index(drop=True)


def interpolate_reference(fixes,truth,max_bracket_s=3.0):
    """Linearly interpolate RTK east/north positions at fix times; reject gaps."""
    if len(truth)<2 or len(fixes)<1:return pd.DataFrame()
    olat,olon=float(truth.latitude.median()),float(truth.longitude.median())
    gtxy=local_xy(truth.latitude,truth.longitude,olat,olon)
    fixxy=local_xy(fixes.latitude,fixes.longitude,olat,olon)
    gt_t=truth.utc.astype('int64').to_numpy()/1e9
    fx_t=fixes.utc.astype('int64').to_numpy()/1e9
    right=np.searchsorted(gt_t,fx_t,side='left')
    left=np.clip(right-1,0,len(gt_t)-1)
    right=np.clip(right,0,len(gt_t)-1)
    exact=(gt_t[right]==fx_t)
    left=np.where(exact,right,left)
    gap=gt_t[right]-gt_t[left]
    valid=exact|((right>left)&(gap<=max_bracket_s)&(gap>0))
    alpha=np.zeros(len(fx_t))
    alpha[~exact]=np.divide(fx_t[~exact]-gt_t[left[~exact]],gap[~exact],
                              out=np.full(np.count_nonzero(~exact),np.nan),where=(gap[~exact]>0))
    valid&=(alpha>=0)&(alpha<=1)
    ref=np.full((len(fx_t),2),np.nan)
    ref[valid]=gtxy[left[valid]]+(gtxy[right[valid]]-gtxy[left[valid]])*alpha[valid,None]
    out=pd.DataFrame({'utc':fixes.utc,'fix_east_m':fixxy[:,0],'fix_north_m':fixxy[:,1],
                      'rtk_east_m':ref[:,0],'rtk_north_m':ref[:,1],
                      'reference_bracket_s':np.where(exact,0.,gap)})
    out=out.loc[valid].copy()
    out['horizontal_error_m']=np.hypot(out.fix_east_m-out.rtk_east_m,out.fix_north_m-out.rtk_north_m)
    if 'confidence' in fixes:
        out['reported_confidence50_m']=pd.to_numeric(fixes.loc[out.index,'confidence'],errors='coerce').to_numpy()
    return out.reset_index(drop=True)


def rtk_reference_speed(truth,smooth_window=9,max_gap_s=2.5):
    """Finite-difference RTK reference motion, not guaranteed independent speed truth."""
    if len(truth)<3:return pd.DataFrame()
    olat,olon=float(truth.latitude.median()),float(truth.longitude.median())
    xy=local_xy(truth.latitude,truth.longitude,olat,olon)
    sec=truth.utc.astype('int64').to_numpy()/1e9
    groups=np.cumsum(np.r_[0, np.diff(sec)>max_gap_s])
    vx=np.full(len(truth),np.nan);vy=np.full(len(truth),np.nan)
    for k in np.unique(groups):
        idx=np.flatnonzero(groups==k)
        if len(idx)<4:continue
        tt=sec[idx];xx=xy[idx]
        if np.any(np.diff(tt)<=0):continue
        window=min(smooth_window,len(idx) if len(idx)%2 else len(idx)-1)
        if window>=5:
            xx=savgol_filter(xx,window,min(2,window-2),axis=0)
        vx[idx]=np.gradient(xx[:,0],tt)
        vy[idx]=np.gradient(xx[:,1],tt)
    return pd.DataFrame({'utc':truth.utc,'reference_speed_mps':np.hypot(vx,vy)})


def inspect_root(root):
    root=Path(root)
    possible=[root, root/'snapshot-gnss-data-2-main',root/'snapshot-gnss-data-2']
    for p in possible:
        if (p/'fixes').is_dir() and (p/'ground-truth').is_dir(): return p
    raise FileNotFoundError('Expected fixes/ and ground-truth/ JSON folders under '+str(root))


def run(root,out):
    repo=inspect_root(root)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fixdir=repo/'fixes';truthdir=repo/'ground-truth'
    # File stems should match (e.g. w02, c00), independent of file ordering.
    fixes={p.stem:p for p in fixdir.rglob('*.json')}
    truth={p.stem:p for p in truthdir.rglob('*.json')}
    common=sorted(set(fixes)&set(truth))
    if not common:raise RuntimeError(f'No same-named JSON tracks in {fixdir} and {truthdir}; found {list(fixes)[:4]} / {list(truth)[:4]}')
    summary=[]; matched=[];motion=[]
    for track in common:
        try:
            ff=read_points(fixes[track],'fixes')
            tt=read_points(truth[track],'ground-truth',reference_date=ff.utc.iloc[len(ff)//2])
            pairs=interpolate_reference(ff,tt)
            speed=rtk_reference_speed(tt)
            if not pairs.empty:
                pairs.insert(0,'track',track)
                matched.append(pairs)
            speed.insert(0,'track',track)
            motion.append(speed)
            err=pairs.horizontal_error_m if len(pairs) else pd.Series(dtype=float)
            summary.append({'track':track,'activity':'walking' if track.lower().startswith('w') else 'cycling' if track.lower().startswith('c') else 'unknown',
                'n_fixes':len(ff),'n_rtk_positions':len(tt),'matched_fixes':len(pairs),
                'matched_fraction':len(pairs)/len(ff) if len(ff) else np.nan,
                'median_horizontal_error_m':float(err.median()) if len(err) else np.nan,
                'p95_horizontal_error_m':float(err.quantile(.95)) if len(err) else np.nan,
                'horizontal_position_rmse_m':float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
                'median_rtk_derived_speed_mps':float(speed.reference_speed_mps.median())})
        except Exception as exc:
            summary.append({'track':track,'processing_error':str(exc)})
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'snapper_track_summary.csv',index=False)
    if matched:
        pd.concat(matched,ignore_index=True).to_csv(out/'snapper_matched_position_errors.csv',index=False)
    if motion:
        pd.concat(motion,ignore_index=True).to_csv(out/'snapper_rtk_derived_speed.csv',index=False)
    good=summary.dropna(subset=['median_horizontal_error_m'])
    if len(good):
        fig,ax=plt.subplots(figsize=(9,4.5))
        colors=['tab:blue' if a=='walking' else 'tab:orange' for a in good.activity]
        ax.bar(good.track,good.median_horizontal_error_m,color=colors)
        ax.set(ylabel='Median horizontal position error (m)',xlabel='SnapperGPS track',title='SnapperGPS raw fixes vs interpolated RTK positions')
        ax.tick_params(axis='x',labelrotation=90)
        fig.tight_layout();fig.savefig(out/'snapper_position_error_by_track.png',dpi=180);plt.close(fig)
    report={'source_repo':'https://github.com/JonasBchrt/snapshot-gnss-data-2',
         'tracks_processed':int(len(summary)), 'tracks_with_matches':int(len(good)),
         'total_fixes_matched':int(summary.matched_fixes.fillna(0).sum()),
         'caution':'Position comparisons on separate SnapperGPS recordings. Not an independent Android-watch Doppler velocity RMSE. RTK derived speeds are finite differences of reference positions.'}
    (out/'snapper_audit_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='external/snapshot-gnss-data-2-main')
    parser.add_argument('--out',default='results/phase4/snapper')
    args=parser.parse_args()
    run(args.repo,args.out)
