"""Exact phase-enumerated loss of NFL high-speed episodes under subsampling.

Uses observed NFL Big Data Bowl 2026 input tracking at 10 Hz, not GNSS data.
This quantifies a pure temporal sampling limitation before GNSS errors arise.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

YARD_M=0.9144
RATES=(10,5,2,1)
THRESHOLDS=(5.5,7.0)
MIN_FRAMES=(1,3,5)


def contiguous_intervals(mask):
    x=np.asarray(mask,dtype=bool)
    if not len(x): return []
    starts=np.flatnonzero(x & ~np.r_[False,x[:-1]])
    ends=np.flatnonzero(x & ~np.r_[x[1:],False])
    return [(int(a),int(b)) for a,b in zip(starts,ends)]


def capture_fraction(a,b,step):
    """Exact fraction of possible clock phases that sample >=1 event frame."""
    return sum(np.any(np.arange(phase,b+1,step)>=a) and (phase<=b) for phase in range(step))/step


def audit(csv_path,out):
    df=pd.read_csv(csv_path,usecols=['game_id','play_id','nfl_id','frame_id','s'])
    df=df.dropna(subset=['game_id','play_id','nfl_id','frame_id','s'])
    df=df.sort_values(['game_id','play_id','nfl_id','frame_id'])
    events=[]
    ntracks=0;nsegments=0
    for key,group in df.groupby(['game_id','play_id','nfl_id'],sort=False):
        if len(group)<10:continue
        frame=group.frame_id.to_numpy()
        speed=group.s.to_numpy(float)*YARD_M
        boundaries=np.r_[0,np.flatnonzero(np.diff(frame)!=1)+1,len(frame)]
        ntracks+=1
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            if right-left<10:continue
            nsegments+=1
            ss=speed[left:right]
            for threshold in THRESHOLDS:
                for a,b in contiguous_intervals(ss>=threshold):
                    count=b-a+1
                    for nmin in MIN_FRAMES:
                        if count<nmin:continue
                        row={'game_id':int(key[0]),'play_id':int(key[1]),'nfl_id':int(key[2]),
                             'threshold_mps':threshold,'minimum_event_duration_s':nmin/10,
                             'event_duration_s':count/10,'start_frame':int(frame[left+a]),'end_frame':int(frame[left+b]),
                             'max_event_speed_mps':float(ss[a:b+1].max())}
                        for hz in RATES:
                            row[f'capture_probability_{hz}hz']=capture_fraction(a,b,int(10/hz))
                        events.append(row)
    events=pd.DataFrame(events)
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    events.to_csv(out/'nfl_high_speed_event_sampling_detail.csv',index=False)
    rows=[]
    for threshold in THRESHOLDS:
      for nmin in MIN_FRAMES:
        e=events[(events.threshold_mps==threshold)&(events.minimum_event_duration_s==nmin/10)]
        for hz in RATES:
            if len(e):
                grp=e.groupby('game_id')[f'capture_probability_{hz}hz'].mean()
                rows.append({'threshold_mps':threshold,'minimum_event_duration_s':nmin/10,'rate_hz':hz,
                             'event_count':len(e),'games_with_events':int(e.game_id.nunique()),
                             'mean_event_capture_probability':float(e[f'capture_probability_{hz}hz'].mean()),
                             'median_duration_s':float(e.event_duration_s.median()),
                             'min_game_mean_capture':float(grp.min()),'max_game_mean_capture':float(grp.max())})
            else:
                rows.append({'threshold_mps':threshold,'minimum_event_duration_s':nmin/10,'rate_hz':hz,
                             'event_count':0,'games_with_events':0,'mean_event_capture_probability':np.nan})
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'nfl_sampling_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    for threshold in THRESHOLDS:
        ss=summary[(summary.threshold_mps==threshold)&(summary.minimum_event_duration_s==0.1)].sort_values('rate_hz')
        ax.plot(ss.rate_hz,ss.mean_event_capture_probability*100,marker='o',label=f'{threshold:g} m/s high-speed threshold')
    ax.set(xlabel='Simulated GNSS sampling rate (Hz)',ylabel='Fraction of true NFL high-speed events sampled (%)',
           title='Intrinsic event loss from downsampling 10-Hz NFL tracking',ylim=(0,105))
    ax.set_xticks(RATES);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'nfl_sprint_capture_vs_sampling_rate.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    aa=summary[summary.threshold_mps==7.0]
    for md in sorted(aa.minimum_event_duration_s.unique()):
        ss=aa[aa.minimum_event_duration_s==md].sort_values('rate_hz')
        ax.plot(ss.rate_hz,ss.mean_event_capture_probability*100,marker='o',label=f'Event duration >= {md:.1f} s')
    ax.set(xlabel='Sampling rate (Hz)',ylabel='Fraction of true events sampled (%)',ylim=(0,105),
           title='Sampling effects depend on the definition of a sprint event')
    ax.set_xticks(RATES);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'nfl_event_duration_sensitivity.png',dpi=200);plt.close(fig)
    report={'source':'NFL Big Data Bowl 2026 Analytics, 2023 Week 1 pre-pass tracking',
            'track_count':ntracks,'continuous_track_segments_1s_or_more':nsegments,
            'rates_hz':list(RATES),'thresholds_mps':list(THRESHOLDS),
            'note':'Exact enumeration of sampling phases, no GNSS measurement errors or NFL RF data. 10Hz source treated as reference motion.'}
    (out/'nfl_sampling_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print(summary[(summary.threshold_mps==7.0)&(summary.minimum_event_duration_s==.1)][['rate_hz','event_count','mean_event_capture_probability']].to_string(index=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input',default='data/input_2023_w01.csv')
    parser.add_argument('--out',default='results/phase4')
    args=parser.parse_args();audit(args.input,args.out)
