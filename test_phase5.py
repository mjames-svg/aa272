from pathlib import Path
import numpy as np
import pandas as pd
from phase5_event_integration import reference_events
from phase6_event_metrics import vectorized_events
from phase5_paired_audit import run as paired_audit


def test_reference_segment_boundaries():
    # A reference event cannot silently bridge a missing 10-Hz frame.
    f=pd.DataFrame(dict(game_id=[1]*7,play_id=[2]*7,nfl_id=[3]*7,
                        frame_id=[10,11,12,14,15,16,17],s=[0,8,8,8,8,0,8]))
    ev=reference_events(f,7.0,1)
    assert len(ev)==3,ev
    assert list(zip(ev.start_frame,ev.end_frame))==[(11,12),(14,15),(17,17)]
    ev3=reference_events(f,7.0,3)
    assert len(ev3)==0
    # Verify yards/second -> metres/second conversion.
    f2=f.copy(); f2['s']=7.1
    assert len(reference_events(f2,7.,1))==0


def test_existing_data(root):
    simulation=pd.read_csv(root/'results/phase3/heldout_monte_carlo_epoch_results.csv')
    ref,events,epochs,seeds,summary=vectorized_events(root,simulation)
    assert len(ref)==22
    assert len(seeds)==36*4, len(seeds)
    assert len(summary)==24
    assert events.groupby(['seed','model','scenario','method']).size().eq(len(ref)).all()
    assert ((summary.sampling_loss_pct+summary.no_valid_solution_pct+
            summary.sampled_not_detected_pct+summary.event_recall_pct)-100).abs().max()<1e-8
    assert (summary.event_recall_pct<=summary.sampling_ceiling_pct+1e-8).all()
    assert ((summary.mean_epoch_availability_pct>=0)&(summary.mean_epoch_availability_pct<=100)).all()
    # All source methods use identical scheduled sampling epochs.
    k=['seed','model','scenario','game','play','player']
    frame_check=epochs.groupby(k).sampled_epochs.nunique()
    assert frame_check.eq(1).all()
    assert not summary[['event_recall_pct','mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps']].isna().any().any()
    pairs,spread=paired_audit(root)
    assert len(pairs)==18 and len(spread)==18
    assert (pairs.net_delta_events==pairs.recovered_by_alternative-pairs.lost_by_alternative).all()
    print('Phase 5 tests passed; 22 held-out NFL high-speed episodes; 6 simulation seeds')


if __name__=='__main__':
    test_reference_segment_boundaries()
    test_existing_data(Path(__file__).resolve().parent)
