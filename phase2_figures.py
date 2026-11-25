"""Create a compact paired bootstrap visualization from Phase 2 results."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent/'results/phase2'
a=pd.read_csv(out/'paired_common_epoch_bootstrap.csv')
a=a[a.other_method.isin(['fde','robust'])]
order=['observed_like','increased_noise','injected_faults','reduced_satellites','combined_degradation']
fig,ax=plt.subplots(figsize=(9,6.1))
for j,meth in enumerate(['fde','robust']):
    sub=a[a.other_method==meth].set_index('scenario').reindex(order)
    y=np.arange(len(order))*2.0+(j-.5)*.6
    x=sub.rmse_difference_other_minus_wls_mps.to_numpy()
    low=sub.bootstrap_95ci_low_mps.to_numpy()
    high=sub.bootstrap_95ci_high_mps.to_numpy()
    ax.errorbar(x,y,xerr=np.vstack((x-low,high-x)),fmt='o',capsize=3,label=meth)
ax.axvline(0,ls='--',lw=1,alpha=.75)
ax.set_yticks(np.arange(len(order))*2.0,[x.replace('_',' ') for x in order])
ax.invert_yaxis(); ax.set(xlabel='Speed RMSE difference vs WLS (m/s): negative favors alternative',
    title='Matched-epoch NFL simulation: paired trajectory-cluster bootstrap')
ax.grid(axis='x',alpha=.2);ax.legend();fig.tight_layout()
fig.savefig(out/'paired_rmse_differences_95ci.png',dpi=180)
plt.close(fig)
