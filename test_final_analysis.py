"""Fast reproducibility, provenance, pairing and denominator checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from phase6_event_metrics import vectorized_events


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/final'
    epochs=pd.read_csv(out/'heldout_expanded_epoch_results.csv')
    events=pd.read_csv(out/'reference_events.csv')
    outcomes=pd.read_csv(out/'event_outcomes.csv')
    summary=pd.read_csv(out/'main_method_comparison.csv')
    boot=pd.read_csv(out/'paired_bootstrap_uncertainty.csv')
    cohorts=json.loads((out/'expanded_cohort.json').read_text())
    test_games=set(cohorts['heldout_games'])
    assert set(epochs.game)==test_games
    assert set(epochs.method)=={'wls','fde','robust'}
    assert set(epochs.model)=={'independent','persistent'}
    assert set(epochs.scenario)=={'observed_like','injected_faults','combined_degradation'}
    assert set(epochs.seed)=={272,847,1927,3117,9872,11234}
    assert len(events)==287
    assert len(events.query('touches_track_edge == False'))==65
    assert cohorts['included_high_speed_tracks']==281
    assert cohorts['included_tracks']==556
    assert epochs[['seed','model','scenario','game','play','player','frame','method']].duplicated().sum()==0
    n=epochs.groupby(['seed','model','scenario','game','play','player','frame']).method.nunique()
    assert n.eq(3).all()
    assert epochs.loc[~epochs.accepted,'est_speed_mps'].isna().all()
    assert outcomes.groupby(['seed','model','scenario','method']).size().eq(287).all()
    assert outcomes.groupby(['seed','model','scenario','game','play','player','event_ordinal']).sampled.nunique().eq(1).all()
    assert ((summary.sampling_loss_pct+summary.no_valid_solution_pct+
             summary.sampled_not_detected_pct+summary.event_recall_pct)-100).abs().max()<1e-8
    assert summary.event_recall_pct.le(summary.sampling_ceiling_pct+1e-9).all()
    assert summary.mean_epoch_availability_pct.between(0,100).all()
    assert summary.loc[summary.method=='wls','sampling_ceiling_pct'].max()-summary.loc[summary.method=='wls','sampling_ceiling_pct'].min()<1e-8
    assert boot.loc[boot.method=='wls','observed_delta_recall_pp'].abs().max()<1e-10
    assert (boot.hierarchical_bootstrap_95ci_upper_pp>=boot.hierarchical_bootstrap_95ci_lower_pp).all()
    assert not summary[['event_recall_pct','mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps']].isna().any().any()
    fast_ref,fast_events,fast_stats,fast_seeds,fast_summary=vectorized_events(root,epochs,7.0)
    comp=summary.merge(fast_summary,on=['model','scenario','method'],suffixes=('_file','_regen'))
    for metric in ['event_recall_pct','sampling_ceiling_pct','mean_epoch_availability_pct',
                   'mean_speed_rmse_on_accepted_mps','mean_false_alerts_per_100_negative_samples']:
        assert np.allclose(comp[metric+'_file'],comp[metric+'_regen'],rtol=0,atol=1e-9)
    assert len(fast_ref)==len(events)
    older=pd.read_csv(root/'results/phase5/end_to_end_event_summary.csv')
    assert len(older)==24 and len(summary)==24
    print('Final analysis tests passed: 287 reference intervals, six seeds, five held-out games, shared observations, four methods.')

if __name__=='__main__':run()
