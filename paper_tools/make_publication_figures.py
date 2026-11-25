from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT=Path(__file__).resolve().parent.parent/'project_data'
OUT=Path(__file__).resolve().parent.parent/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'savefig.facecolor':'white','pdf.fonttype':42})
C=['#206679','#AB763C','#A24F54','#60676B']

def save(name):
 plt.savefig(OUT/name,bbox_inches='tight',pad_inches=.09);plt.close()

# Real C/N0 leave-one-out residuals from measured watch data
r=pd.read_csv(ROOT/'results/phase2/empirical_leave_one_out_residuals.csv')
r=r[np.isfinite(r['cn0_dbhz']) & np.isfinite(r['loo_prediction_residual_mps'])].copy()
r['absres']=r.loo_prediction_residual_mps.abs()
bins=[0,26,30,34,100];lab=['<26','26–30','30–34','≥34']
r['quality']=pd.cut(r.cn0_dbhz,bins=bins,labels=lab,include_lowest=True,right=False)
med=[r.loc[r.quality==b,'absres'].median() for b in lab]
n=[r.quality.eq(b).sum() for b in lab]
fig,ax=plt.subplots(figsize=(6.8,2.6))
bars=ax.bar(np.arange(4),med,color=C[0],width=.6)
ax.set_xticks(np.arange(4),lab)
ax.set_xlabel('Carrier-to-noise-density ratio, C/N₀ (dB-Hz)')
ax.set_ylabel('Median absolute LOO residual (m/s)')
ax.set_ylim(0,max(med)*1.36);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
for i,(b,v) in enumerate(zip(bars,med)):
 ax.text(b.get_x()+b.get_width()/2,v+.035,f'{v:.2f}\nn={n[i]}',ha='center',va='bottom',fontsize=8)
fig.tight_layout();save('fig2_cn0_residuals.pdf')
print('real data medians:', dict(zip(lab, med)))

# Sampling frequency
s=pd.read_csv(ROOT/'results/phase4/nfl_sampling_summary.csv');s=s[(s.threshold_mps==7)&(s.minimum_event_duration_s==.1)].sort_values('rate_hz')
fig,ax=plt.subplots(figsize=(7.0,2.7))
ax.plot(s.rate_hz,s.mean_event_capture_probability*100,marker='o',ms=7.5,lw=2.2,color=C[0])
ax.set(xlabel='Hypothetical sampling frequency (Hz)',ylabel='Ideal interval capture (%)',ylim=(55,108),xlim=(.7,10.45))
ax.set_xticks([1,2,5,10]);ax.grid(alpha=.27)
for _,z in s.iterrows():
 ax.annotate(f'{100*z.mean_event_capture_probability:.1f}%', (z.rate_hz,z.mean_event_capture_probability*100),xytext=((-3,16) if z.rate_hz==1 else ((-12,12) if z.rate_hz==10 else (0,9))),textcoords='offset points',ha='center',va='bottom',fontsize=9)
fig.tight_layout(pad=.8);save('fig3_sampling.pdf')

# Difficult-case algorithm comparison
p=pd.read_csv(ROOT/'results/final/main_method_comparison.csv');p=p[(p.model=='persistent')&(p.scenario=='combined_degradation')]
names={'wls':'WLS','fde':'FDE','robust':'Robust WLS','wls_epoch_gate':'Strict gate'}
p=p.set_index('method').loc[list(names)]
fig,ax=plt.subplots(1,2,figsize=(7.3,3),gridspec_kw={'wspace':.30})
x=np.arange(4)
for j,col in enumerate(['event_recall_pct','mean_epoch_availability_pct']):
 vals=p[col].values
 ax[j].bar(x,vals,color=[C[0],C[1],C[2],C[3]],width=.64)
 ax[j].set_xticks(x,['WLS','FDE','Robust','Strict'],fontsize=9)
 ax[j].set_ylim(0,61 if j==0 else 66)
 ax[j].set_ylabel('Event recall (%)' if j==0 else 'Accepted epochs (%)')
 ax[j].grid(axis='y',alpha=.25);ax[j].set_axisbelow(True)
 for i,v in enumerate(vals):ax[j].text(i,v+1.3,f'{v:.1f}%',ha='center',fontsize=8)
fig.tight_layout();save('fig4_method_comparison.pdf')

# Paired roof comparison by methods
p10_all=pd.read_csv(ROOT/'results/phase10/stadium_method_comparison.csv');p10=p10_all[p10_all.scenario.str.endswith('__measured_proxy')].copy()
skies=['open','partial_canopy','deep_canopy','sealed_roof'];methods=['wls','fde','robust']
fig,ax=plt.subplots(figsize=(7.25,2.95))
x=np.arange(4);width=.24
for i,m in enumerate(methods):
 vals=[float(p10[(p10.method==m)&(p10.scenario==sky+'__measured_proxy')].event_recall_pct.iloc[0]) for sky in skies]
 ax.bar(x+(i-1)*width,vals,width,label=names[m],color=C[i],edgecolor='white',linewidth=.45)
ax.set_xticks(x,['Open Sky','Partial Canopy','Deep Canopy','Signal-Blocking\nRoof'])
ax.set_ylabel('High-speed interval recall (%)');ax.set_ylim(0,60);ax.grid(axis='y',alpha=.24);ax.set_axisbelow(True)
ax.legend(ncol=3,frameon=False,loc='upper right')
fig.tight_layout();save('fig6_roof_recall.pdf')

# Availability versus accepted-only RMSE for sky scenarios, WLS and two noise models
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8),gridspec_kw={'wspace':.3})
for j,term in enumerate(['mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps']):
 for k,model in enumerate(['measured_proxy','degraded_sensitivity']):
  vals=[p10_all[(p10_all.method=='wls')&(p10_all.scenario==sky+'__'+model)][term].iloc[0] for sky in skies[:3]]
  ax[j].bar(np.arange(3)+(k-.5)*.34,vals,width=.34,label='Baseline' if k==0 else 'Noise + faults',color=C[0] if k==0 else C[2])
 ax[j].set_xticks(range(3),['Open','Partial','Deep']);ax[j].grid(axis='y',alpha=.25);ax[j].set_axisbelow(True)
 ax[j].set_ylabel('Valid epochs (%)' if j==0 else 'Speed RMSE on valid epochs (m/s)')
 ax[j].set_ylim(0,112 if j==0 else 3.85)
ax[0].legend(frameon=False,fontsize=8,loc='upper right')
fig.tight_layout();save('fig7_roof_accuracy_availability.pdf')

# Copy PNGs from results
from shutil import copyfile
for source,out in [
 ('01_football_field_w02_real_rtk.png','fig1_football_field.png'),
 ('09_10_synthetic_roof_sky_masks.png','fig5_roof_masks_original.png'),
 ('13_10_paired_game_bootstrap.png','fig8_roof_bootstrap.png'),
 ('07_real_reference_robustness.png','fig9_real_reference.png'),
 ('08_real_field_vs_urban_position_errors.png','fig10_field_urban.png'),
 ('14_final_boundary_event_sensitivity.png','fig11_boundary.png')
]:copyfile(ROOT/'FINAL_DELIVERABLES/CURATED_FIGURES'/source,OUT/out)
print('finished',len(list(OUT.iterdir())))

# Apply the final layout pass after rebuilding the original numeric figures.
import subprocess, sys
subprocess.run([sys.executable, str(Path(__file__).parent/'rebuild_corrected_figures.py')], check=True)
