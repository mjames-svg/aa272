from pathlib import Path
import pandas as pd
from phase6_event_metrics import vectorized_events
from phase6_event_metrics import summarize_fast


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    out=root/'results/final'
    df=pd.read_csv(out/'heldout_expanded_epoch_results.csv')
    rows=[]
    for threshold in (5.5,7.0):
        ev,matched,eps,_,_=vectorized_events(root,df,threshold=threshold)
        for minimum in (1,3,5):
            filtered=matched[matched.duration_s>=minimum/10-1e-8]
            _,tab=summarize_fast(filtered,eps)
            tab['speed_threshold_mps']=threshold
            tab['minimum_duration_s']=minimum/10
            tab['reference_unique_intervals']=len(ev[ev.duration_s>=minimum/10-1e-8])
            rows.append(tab)
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(out/'event_definition_sensitivity.csv',index=False)
    print('Threshold / min duration / unique intervals')
    print(result[['speed_threshold_mps','minimum_duration_s','reference_unique_intervals']].drop_duplicates().to_string(index=False))
    return result

if __name__=='__main__':run()
