import numpy as np
import pandas as pd
from pathlib import Path
from phase8_real_reference_robustness import filtered_common, moving_block_indices, paired_block_interval, run

root=Path(__file__).resolve().parent
sample=pd.read_csv(root/'results/phase7/matched_watch_rg_velocity_epochs.csv')
a=filtered_common(sample)
b=filtered_common(sample,include_android=True)
assert len(a)==674 and len(b)==665
assert a.utc.is_monotonic_increasing
assert (a.gps_sats>=7).all()
idx=moving_block_indices(25,7,5,np.random.default_rng(1))
assert idx.shape==(5,25) and idx.max()<25 and idx.min()>=0
s=paired_block_interval(a,'fde_horizontal_speed_mps',20,100,272)
assert s['n_paired_epochs']==674
assert len(s['rmse_change_95pct_descriptive_interval_mps'])==2
summary=run(root,n_reps=4000)
assert summary['matched_epochs_7plus_satellites']==674
assert summary['match_abs_time_offset_ms_max']<=110
assert summary['fde_modified_epoch_count']>0
print('Phase 8 paired temporal-bootstrap, complete-case, and robustness tests passed.')
