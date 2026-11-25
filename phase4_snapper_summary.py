"""Summarize measured SnapperGPS/RTK position accuracy, including rare outliers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def run(out='results/phase4/snapper'):
    out = Path(out)
    matched = pd.read_csv(out / 'snapper_matched_position_errors.csv')
    tracks = pd.read_csv(out / 'snapper_track_summary.csv')
    groups = [('all', matched), ('walking', matched[matched.track.str.startswith('w')]),
              ('cycling', matched[matched.track.str.startswith('c')])]
    rows = []
    for activity, df in groups:
        e = df.horizontal_error_m.to_numpy(float)
        rows.append({
            'activity': activity,
            'n_matched_fixes': len(e),
            'median_horizontal_error_m': float(np.median(e)),
            'p95_horizontal_error_m': float(np.quantile(e,.95)),
            'p99_horizontal_error_m': float(np.quantile(e,.99)),
            'horizontal_rmse_all_m': float(np.sqrt(np.mean(e**2))),
            'n_errors_over_100m': int(np.sum(e>100)),
            'n_errors_over_1000m': int(np.sum(e>1000)),
            'fraction_within_reported_50pct_radius': float(np.mean(df.horizontal_error_m<=df.reported_confidence50_m)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'snapper_descriptive_statistics.csv',index=False)
    report = {
        'source': 'User-uploaded snapshot-gnss-data-2-main.zip; fixes and RTK ground-truth only',
        'tracks': int(len(tracks)),
        'total_fix_records': int(tracks.n_fixes.sum()),
        'total_time_aligned_fixes': int(tracks.matched_fixes.sum()),
        'matched_fraction': float(tracks.matched_fixes.sum()/tracks.n_fixes.sum()),
        'median_track_reference_speed_mps': {a:float(g.median_rtk_derived_speed_mps.median()) for a,g in tracks.groupby('activity')},
        'note':'This measures SnapperGPS position error on separate recordings, not Android Pixel Watch Doppler velocity RMSE. RTK-derived speeds come from numerical differentiation and smoothing. No extreme position error is silently removed.'
    }
    (out / 'snapper_extended_summary.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False))
    return summary

if __name__=='__main__':
    run()
