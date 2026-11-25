"""Small independent checks of time matching, spatial units and sample-phase enumeration."""
import numpy as np
import pandas as pd
from phase4_snapper_reference import local_xy,interpolate_reference,rtk_reference_speed,read_points,run as snapper_run
from phase4_nfl_sampling_audit import capture_fraction,contiguous_intervals
from pathlib import Path
import tempfile,json


def test():
    d=local_xy([0.,0.],[0.,0.00001],0.,0.)
    assert abs(d[1,0]-1.1132)<.02 and abs(d[0,0])<1e-9
    assert abs(d[1,1])<1e-6
    assert abs(capture_fraction(0,0,10)-.1)<1e-10
    assert abs(capture_fraction(0,9,10)-1.)<1e-10
    assert abs(capture_fraction(4,8,10)-.5)<1e-10
    assert contiguous_intervals([False,True,True,False,True])==[(1,2),(4,4)]
    t0=pd.Timestamp('2023-01-01',tz='UTC')
    truth=pd.DataFrame({'utc':[t0+pd.Timedelta(seconds=k) for k in range(7)],
        'latitude':[0.]*7,'longitude':[k*0.00001 for k in range(7)]})
    fixes=pd.DataFrame({'utc':[t0+pd.Timedelta(seconds=i+.5) for i in (0,1,2,3)],
        'latitude':[0.]*4,'longitude':[(i+.5)*0.00001 for i in (0,1,2,3)]})
    paired=interpolate_reference(fixes,truth)
    assert len(paired)==4 and paired.horizontal_error_m.max()<1e-5
    speed=rtk_reference_speed(truth)
    assert 1.0<speed.reference_speed_mps.median()<1.2
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'w02.json'
        path.write_text(json.dumps({'fixes':[{'datetime':'2023-01-01T00:00:00Z','latitude':0,'longitude':0,'confidence':10}]}))
        assert len(read_points(path,'fixes'))==1
        # Public RTK records contain HH:MM:SS only; align to fix UTC date.
        tod=Path(folder)/'time_only.json'
        tod.write_text(json.dumps([{'time':'10:00:00','latitude':0,'longitude':0},
                                   {'time':'10:00:01','latitude':0,'longitude':0.00001}]))
        parsed=read_points(tod,'ground-truth',reference_date='2022-10-18T10:00:00Z')
        assert len(parsed)==2
        assert parsed.utc.iloc[0]==pd.Timestamp('2022-10-18T10:00:00Z')
        repo=Path(folder)/'repo'
        (repo/'fixes').mkdir(parents=True)
        (repo/'ground-truth').mkdir(parents=True)
        fixrows=[{'datetime':(t0+pd.Timedelta(seconds=i)).isoformat(),'latitude':0.0,
                  'longitude':i*0.00001+0.000002,'confidence':10} for i in range(7)]
        truerows=[{'time':(t0+pd.Timedelta(seconds=i)).isoformat(),'latitude':0.0,
                   'longitude':i*0.00001} for i in range(7)]
        (repo/'fixes/w02.json').write_text(json.dumps(fixrows))
        (repo/'ground-truth/w02.json').write_text(json.dumps(truerows))
        summary=snapper_run(repo,Path(folder)/'out')
        assert summary['tracks_with_matches']==1 and summary['total_fixes_matched']==7
        med=pd.read_csv(Path(folder)/'out/snapper_track_summary.csv').median_horizontal_error_m.iloc[0]
        assert .1<med<.5
    print('Phase 4 unit tests passed.')

if __name__=='__main__':test()
