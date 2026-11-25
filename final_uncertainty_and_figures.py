"""Paired, game/seed-resampled intervals and final summary figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase6_event_metrics import summarize_fast

METHODS=['wls','fde','robust','wls_epoch_gate']


def paired_bootstrap(events:pd.DataFrame,bootstrap=2500):
    rnd=np.random.default_rng(20261111)
    output=[]
    for population,subset in [('all_observed_windows',events),('fully_observed_only',events[~events.touches_track_edge])]:
      for (model,scenario),block in subset.groupby(['model','scenario']):
        seeds=sorted(block.seed.unique().tolist());games=sorted(block.game.unique().tolist())
        n=np.zeros(len(games),dtype=int)
        hits=np.zeros((len(seeds),len(games),len(METHODS)),dtype=int)
        for gi,game in enumerate(games):
            e=block[(block.game==game)&(block.seed==seeds[0])&block.method.eq('wls')]
            n[gi]=len(e)
        for si,seed in enumerate(seeds):
          for gi,game in enumerate(games):
            src=block[(block.seed==seed)&(block.game==game)]
            for mi,method in enumerate(METHODS):
              hits[si,gi,mi]=int(src.loc[src.method==method,'detected'].sum())
        assert n.sum()>0
        base_ind=METHODS.index('wls')
        for mi,meth in enumerate(METHODS):
            numerator=hits[:,:,mi].sum()-hits[:,:,base_ind].sum()
            observed=100*numerator/(n.sum()*len(seeds))
            estimates=np.empty(bootstrap)
            for ri in range(bootstrap):
                draw_games=rnd.integers(0,len(games),size=len(games))
                draw_seeds=rnd.integers(0,len(seeds),size=len(seeds))
                total=float(n[draw_games].sum()*len(draw_seeds))
                diff=hits[np.ix_(draw_seeds,draw_games,[mi])][:,:,0]-hits[np.ix_(draw_seeds,draw_games,[base_ind])][:,:,0]
                estimates[ri]=100*diff.sum()/total
            low,high=np.percentile(estimates,[2.5,97.5])
            output.append(dict(population=population,model=model,scenario=scenario,
                 method=meth,observed_delta_recall_pp=observed,
                 hierarchical_bootstrap_95ci_lower_pp=float(low),
                 hierarchical_bootstrap_95ci_upper_pp=float(high),
                 reference_events_per_seed=int(n.sum()),games=len(games),seeds=len(seeds)))
    return pd.DataFrame(output)


def run(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    d=root/'results/final'
    e=pd.read_csv(d/'event_outcomes.csv')
    epo=pd.read_csv(d/'epoch_classification_counts.csv')
    paired=paired_bootstrap(e)
    paired.to_csv(d/'paired_bootstrap_uncertainty.csv',index=False)
    interior,interior_pool=summarize_fast(e[~e.touches_track_edge],epo)
    interior_pool.to_csv(d/'fully_observed_event_sensitivity.csv',index=False)
    interior.to_csv(d/'fully_observed_seed_level.csv',index=False)
    main=pd.read_csv(d/'main_method_comparison.csv')
    cols=['event_recall_pct','mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps','mean_false_alerts_per_100_negative_samples']
    table=main[(main.model=='persistent')&(main.scenario=='combined_degradation')].set_index('method').loc[METHODS].copy()
    fig,ax=plt.subplots(figsize=(9.4,5.2))
    x=np.arange(len(METHODS))
    ax.bar(x,table.event_recall_pct.to_numpy(float),label='Events detected')
    ax.scatter(x,table.sampling_ceiling_pct.to_numpy(float),marker='D',s=65,label='Ideal 1-Hz sampling ceiling')
    ax.set_xticks(x,['WLS','FDE','Robust','WLS + epoch gate'])
    ax.set(ylabel='Reference 10-Hz high-speed intervals detected (%)',ylim=(0,105),
           title='Final held-out NFL simulation: persistent combined degradation')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(d/'final_main_event_detection.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.4,5.2))
    for method in METHODS:
        series=main[(main.method==method)&main.model.eq('persistent')]
        ax.scatter(series.mean_epoch_availability_pct,series.mean_speed_rmse_on_accepted_mps,
                   s=75,label=method)
    ax.set(xlabel='Accepted scheduled GNSS epochs (%)',ylabel='Speed RMSE among accepted epochs (m/s)',
           title='NFL simulation: velocity accuracy vs tracking availability')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(d/'final_accuracy_availability.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.5,5.2))
    x=np.arange(len(METHODS));width=.3
    allrec=main[(main.model=='persistent')&(main.scenario=='combined_degradation')].set_index('method').loc[METHODS]
    inrec=interior_pool[(interior_pool.model=='persistent')&(interior_pool.scenario=='combined_degradation')].set_index('method').loc[METHODS]
    ax.bar(x-width/2,allrec.event_recall_pct,width,label='All observed intervals')
    ax.bar(x+width/2,inrec.event_recall_pct,width,label='Fully observed intervals only')
    ax.set_xticks(x,['WLS','FDE','Robust','WLS + epoch gate'])
    ax.set(ylabel='Detected observed high-speed intervals (%)',ylim=(0,100),
           title='Sensitivity to intervals truncated by NFL tracking window')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(d/'final_boundary_event_sensitivity.png',dpi=200);plt.close(fig)
    report=table[cols].copy()
    print('Final persistent combined degradation\n',report.to_string(),flush=True)
    print('Paired bootstrap key rows\n',paired[(paired.population=='all_observed_windows')&(paired.model=='persistent')&(paired.scenario=='combined_degradation')].to_string(index=False),flush=True)
    print('Interior event count',len(e[(e.seed==e.seed.min())&(e.model=='persistent')&(e.scenario=='combined_degradation')&(e.method=='wls')&~e.touches_track_edge]),flush=True)
    return paired

if __name__=='__main__':run()
