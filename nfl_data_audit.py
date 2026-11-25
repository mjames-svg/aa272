"""Load and audit the NFL reference motion needed for Phase B."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

YARD = 0.9144
GAME, PLAY, PLAYER = 2023091010, 3640, 46279


def run(csv_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    source = pd.read_csv(csv_path)
    a = source[(source.game_id == GAME)&(source.play_id == PLAY)&(source.nfl_id == PLAYER)].copy()
    a = a.sort_values('frame_id')
    if len(a) < 10:
        raise RuntimeError('Selected NFL trajectory missing; inspect game/play/player identifiers.')
    a['time_s'] = (a.frame_id-a.frame_id.iloc[0])/10
    a['x_m'] = a.x * YARD
    a['y_m'] = a.y * YARD
    a['nfl_reported_speed_mps'] = a.s * YARD
    x,y = a[['x_m','y_m']].to_numpy().T
    wl = min(9, len(a)//2*2-1)
    x_s = savgol_filter(x,wl,3)
    y_s = savgol_filter(y,wl,3)
    a['x_smooth_m'] = x_s
    a['y_smooth_m'] = y_s
    a['vx_from_smooth_position_mps'] = np.gradient(x_s,a.time_s.to_numpy())
    a['vy_from_smooth_position_mps'] = np.gradient(y_s,a.time_s.to_numpy())
    a['speed_from_smooth_position_mps'] = np.hypot(a.vx_from_smooth_position_mps,a.vy_from_smooth_position_mps)
    a[['game_id','play_id','nfl_id','frame_id','time_s','x_m','y_m','x_smooth_m','y_smooth_m',
       'nfl_reported_speed_mps','vx_from_smooth_position_mps','vy_from_smooth_position_mps',
       'speed_from_smooth_position_mps']].to_csv(out_dir/'nfl_reference_example.csv',index=False)

    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(a.time_s,a.nfl_reported_speed_mps,label='NFL-reported tracking speed')
    ax.plot(a.time_s,a.speed_from_smooth_position_mps,label='Speed from smoothed positions',alpha=.8)
    ax.set(xlabel='Play-relative time (s)',ylabel='Player speed (m/s)',
           title='NFL Big Data Bowl: real player motion used as simulated-GNSS reference')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir/'nfl_reference_speed.png',dpi=190)
    plt.close(fig)
    summary={
        'source':'NFL Big Data Bowl 2026 Analytics, Week 1 2023 input tracking',
        'source_rows':len(source),
        'example_game_id':GAME,'example_play_id':PLAY,'example_nfl_id':PLAYER,
        'selected_frames':len(a),'duration_seconds':float(a.time_s.iloc[-1]),
        'frame_rate_hz':10,
        'max_nfl_reported_speed_mps':float(a.nfl_reported_speed_mps.max()),
        'position_derived_vs_nfl_reported_speed_rmse_mps':float(np.sqrt(np.mean((a.speed_from_smooth_position_mps-a.nfl_reported_speed_mps)**2))),
        'note':'Comparing two quantities from the same NFL tracking system is an internal consistency check, NOT independent ground-truth validation.'
    }
    (out_dir/'nfl_reference_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    run(root/'data'/'input_2023_w01.csv',root/'results')
