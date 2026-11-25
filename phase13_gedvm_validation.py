"""AA272 GEDVM: aligned Android GPS-provider speed vs phone-specific GNSS/INS reference.
Standalone Python script. Does not imply raw-Doppler WLS/independent certified truth.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GPS_UNIX_OFFSET=315964800
LEAP_2024=18
REF_FREQ_MAX_GAP=1.01
COLOR=['#23637d','#b27b40','#9c5158']


def parse_fix_and_cno(phone_zip: Path):
    fixes=[]
    sat={}
    with zipfile.ZipFile(phone_zip) as z:
        with z.open(z.namelist()[0]) as src:
            ix=None
            for ln in src:
                if ln.startswith(b'# Raw,'):
                    h=ln.decode('utf8',errors='replace').strip()[2:].split(',')
                    ix={k:i for i,k in enumerate(h)}
                    continue
                if ln.startswith(b'Fix,'):
                    a=ln.decode('utf8',errors='replace').strip().split(',')
                    if len(a)<8 or a[1].lower()!='gps':continue
                    try:
                        v=float(a[5]);acc=float(a[6]);t=float(a[7]);
                        if math.isfinite(v) and 0<=v<=60 and math.isfinite(t): fixes.append((t,v,acc))
                    except ValueError:pass
                if ix is not None and ln.startswith(b'Raw,'):
                    a=ln.decode('utf8',errors='replace').strip().split(',')
                    try:
                        if int(a[ix['ConstellationType']])!=1:continue
                        f=float(a[ix['CarrierFrequencyHz']]);
                        if abs(f-1575420000)>2500000:continue
                        cn=float(a[ix['Cn0DbHz']]);prr=float(a[ix['PseudorangeRateMetersPerSecond']]);
                        si=float(a[ix['PseudorangeRateUncertaintyMetersPerSecond']]);
                        if not all(map(math.isfinite,(cn,prr,si))) or si<=0:continue
                        t=a[ix['TimeNanos']]
                        if not t:continue
                        k=(int(a[ix['HardwareClockDiscontinuityCount']]),int(t))
                        v=sat.setdefault(k,[0,0.,0,0.])
                        v[0]+=1;v[1]+=cn;v[2]+=min(si,100);v[3]+=cn<30
                    except (ValueError,IndexError,OverflowError):pass
    fixes=pd.DataFrame(fixes,columns=['utc_ms','gps_speed_mps','gps_position_accuracy_m'])
    n_original=len(fixes)
    fixes=fixes.sort_values('utc_ms').drop_duplicates('utc_ms',keep='first').reset_index(drop=True)
    return fixes, n_original, sat


def align_reference(fix:pd.DataFrame,ref:pd.DataFrame)->pd.DataFrame:
    gpsweek=ref['gps_week'].to_numpy(float)
    sow=ref['gps_sow_s'].to_numpy(float)
    tr=(GPS_UNIX_OFFSET+gpsweek*604800+sow-LEAP_2024)*1000
    re=ref[['vel_e_mps','vel_n_mps','vel_u_mps','acc_e_mps2','acc_n_mps2','solution_quality_code']].to_numpy(float)
    idx=np.argsort(tr);tr=tr[idx];re=re[idx]
    tf=fix.utc_ms.to_numpy(float)
    hi=np.searchsorted(tr,tf,side='left');lo=np.maximum(hi-1,0);hi=np.minimum(hi,len(tr)-1)
    dt=(tr[hi]-tr[lo])/1000
    ok=(dt>=0)&(dt<=REF_FREQ_MAX_GAP)&(tf>=tr[lo]-10)&(tf<=tr[hi]+10)
    frac=np.divide(tf-tr[lo],tr[hi]-tr[lo],out=np.zeros_like(tf),where=(tr[hi]-tr[lo])>0)
    frac=np.clip(frac,0,1)
    refint=re[lo]*(1-frac[:,None])+re[hi]*frac[:,None]
    result=fix.copy()
    result['ref_speed_mps']=np.hypot(refint[:,0],refint[:,1]);
    result['ref_up_mps']=refint[:,2]
    result['ref_accel_horizontal_mps2']=np.hypot(refint[:,3],refint[:,4])
    result['ref_quality_code_nearest']=np.where(frac<0.5,re[lo,5],re[hi,5]);
    result['time_to_nearest_reference_s']=np.minimum(np.abs(tf-tr[lo]),np.abs(tf-tr[hi]))/1000
    result=result[ok & np.isfinite(result['ref_speed_mps']) & np.isfinite(result['gps_speed_mps'])].copy()
    result['speed_error_mps']=result.gps_speed_mps-result.ref_speed_mps
    result['abs_speed_error_mps']=np.abs(result.speed_error_mps)
    return result


def summarize(df,device,session):
    e=df.speed_error_mps.to_numpy(float)
    return dict(session=session,device=device,n=int(len(e)),rmse_mps=float(np.sqrt(np.mean(e*e))) if len(e) else None,
      mae_mps=float(np.mean(abs(e))) if len(e) else None,
      median_abs_error_mps=float(np.median(abs(e))) if len(e) else None,
      p95_abs_error_mps=float(np.quantile(abs(e),.95)) if len(e) else None,
      bias_mps=float(np.mean(e)) if len(e) else None,
      ref_speed_median_mps=float(df.ref_speed_mps.median()) if len(e) else None,
      phone_speed_median_mps=float(df.gps_speed_mps.median()) if len(e) else None,
      ref_quality_codes=json.dumps({str(k):int(v) for k,v in df.ref_quality_code_nearest.value_counts().items()}))


def process(root:Path,out:Path):
    rawroot=root/'data'/'raw';refs=root/'data'/'reference'
    out.mkdir(parents=True,exist_ok=True)
    aligned=[];metadata=[]
    for zip_path in sorted(rawroot.glob('*.zip')):
        stem=zip_path.stem
        ref=pd.read_csv(refs/(stem+'.csv'))
        fixes,nraw,sat=parse_fix_and_cno(zip_path)
        d=align_reference(fixes,ref)
        d['session']=stem
        d['drive']=stem.split('_')[0]
        d['device']=stem.split('_')[1]
        aligned.append(d)
        metadata.append(dict(session=stem,reference_rows=len(ref),phone_unique_gps_fixes=len(fixes),duplicate_removed=nraw-len(fixes),matched=len(d),gps_l1_clock_epochs=len(sat),median_l1_satellites_per_epoch=float(np.median([v[0] for v in sat.values()])) if sat else None,median_l1_cno_dbhz=float(np.median([v[1]/v[0] for v in sat.values()])) if sat else None))
        print(stem,'ref',len(ref),'gps fix unique',len(fixes),'matched',len(d),flush=True)
    all_df=pd.concat(aligned,ignore_index=True)
    all_df.to_csv(out/'gedvm_aligned_fix_speed.csv',index=False,float_format='%.8f')
    pd.DataFrame(metadata).to_csv(out/'gedvm_data_quality.csv',index=False)
    sess=[]
    for stem,d in all_df.groupby('session'):
        sess.append(summarize(d,stem.split('_')[1],stem))
    pd.DataFrame(sess).to_csv(out/'gedvm_session_summary.csv',index=False)
    devices=[]
    for name,d in all_df.groupby('device'):
        devices.append(summarize(d,name,'ALL'))
    devices.append(summarize(all_df,'ALL','ALL'))
    pd.DataFrame(devices).to_csv(out/'gedvm_device_summary.csv',index=False)
    bins=[0,2,7,15,60];labs=['<2','2–7','7–15','≥15']
    all_df['ref_speed_bin']=pd.cut(all_df.ref_speed_mps,bins=bins,labels=labs,include_lowest=True,right=False)
    summaries=[]
    for (s,b),df in all_df.groupby(['session','ref_speed_bin'],observed=True):
        x=summarize(df,s.split('_')[1],s); x['speed_bin']=str(b);summaries.append(x)
    pd.DataFrame(summaries).to_csv(out/'gedvm_speed_bin_summary.csv',index=False)
    for name,mask in [('low_accel', all_df.ref_accel_horizontal_mps2<0.5),('high_accel', all_df.ref_accel_horizontal_mps2>=0.5)]:
        sub=all_df[mask]
        print(name,'n',len(sub),'rmse',np.sqrt((sub.speed_error_mps**2).mean()),flush=True)
    # paired comparison at nearest common UTC reference second, no pseudoreplication
    perdrive=[]
    for drive,sub in all_df.groupby('drive'):
        devices_here=sorted(sub.device.unique())
        if len(devices_here)<2:continue
        table=[]
        for dev in devices_here:
            df=sub[sub.device==dev].copy();df['timebin']=np.rint(df.utc_ms/1000).astype('int64')
            df=df.drop_duplicates('timebin').set_index('timebin')
            table.append(df[['abs_speed_error_mps']].rename(columns={'abs_speed_error_mps':dev}))
        merged=table[0].join(table[1],how='inner').dropna()
        if len(merged):perdrive.append(dict(drive=drive,n_paired=len(merged),device1=devices_here[0],device2=devices_here[1],mean_diff_abs_error_mps=float(np.mean(merged[devices_here[0]]-merged[devices_here[1]]))))
    pd.DataFrame(perdrive).to_csv(out/'gedvm_paired_device.csv',index=False)
    plot_figures(all_df,pd.DataFrame(sess),pd.DataFrame(metadata),out)
    report={'dataset':'GEDVM dataset main ZIP, six phone GNSS logs and six matching GNSS/INS references',
      'method':'Android gps-provider Fix.Speed compared with linear interpolation of phone-specific GNSS/INS ENU speed; no time lag fitted',
      'is_raw_doppler_wls':False,'is_independent_certified_reference':False,
      'notable_exclusions':'fused/network Fix locations excluded, duplicate GPS Fix timestamps removed, invalid reference gaps >1.01s not bridged',
      'gps_minus_utc_seconds':18,'phone_fix_speed_not_direct_implementation':'The Android GPS provider speed is an opaque device processing output. This does not validate this project’s own satellite-level WLS/FDE/robust or Kalman estimators.',
      'missing_ephemerides':'No broadcast/precise satellite ephemeris is included in GEDVM release; raw pseudorange rates cannot be converted to an independent reference-relative velocity WLS result with this archive alone.',
      'matched_total':len(all_df),'sessions':len(sess),
      'reference_caution':'Smoothed phone-lever-arm-specific NovAtel GNSS/INS outputs; no per-epoch velocity uncertainty and no guaranteed independence in GNSS error sources',
      'gps_fix_caution':'Fix timestamps are UTC and asynchronous to GNSS raw epochs; not mixed with receiver clock epochs',
      'grouping':'Two phones on each drive share a vehicle and reference movement; sample counts are epochs, not independent subjects',
      'metadata':metadata,'overall':summarize(all_df,'ALL','ALL')}
    (out/'gedvm_methodology.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return pd.DataFrame(sess),pd.DataFrame(devices),all_df


def plot_figures(all_df,sessions,meta,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'savefig.facecolor':'white'})
    g=all_df.copy()
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5),gridspec_kw={'width_ratios':[1.25,1]})
    for j,(name,df) in enumerate(g.groupby('session')):
        if j>=6:break
        axs[0].scatter(df.ref_speed_mps.iloc[::max(1,len(df)//450)],df.gps_speed_mps.iloc[::max(1,len(df)//450)],s=5,alpha=.25,label=name)
    axs[0].plot([0,35],[0,35],color='black',ls='--',lw=1);axs[0].set_xlim(0,35);axs[0].set_ylim(0,35)
    axs[0].set(xlabel='GNSS/INS reference horizontal speed (m/s)',ylabel='Android GPS-provider speed (m/s)')
    axs[0].legend(fontsize=6,frameon=False,ncol=2,loc='upper left');axs[0].grid(alpha=.2)
    labs=sessions.session.tolist();x=np.arange(len(labs));axs[1].bar(x,sessions.rmse_mps,color=['#23637d','#b27b40']*3)
    axs[1].set_xticks(x,[s.replace('drive','D').replace('_','\n') for s in labs],fontsize=7)
    axs[1].set_ylabel('Speed disagreement RMSE (m/s)');axs[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'gedvm_reference_velocity_validation.pdf',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.9,3.3))
    for j,(s,df) in enumerate(g.groupby('session')):
        axs[0].scatter(df.ref_speed_mps.iloc[::max(1,len(df)//500)],df.abs_speed_error_mps.iloc[::max(1,len(df)//500)],alpha=.23,s=4,label=s)
    axs[0].set(xlim=(0,35),ylim=(0,np.quantile(g.abs_speed_error_mps,.99)*1.05),xlabel='Reference speed (m/s)',ylabel='Absolute speed difference (m/s)')
    axs[0].grid(alpha=.18)
    for i,(name,sub) in enumerate([('Low acceleration',g[g.ref_accel_horizontal_mps2<0.5]),('High acceleration',g[g.ref_accel_horizontal_mps2>=0.5])]):
        err=sub.speed_error_mps.clip(-7,7).to_numpy()
        axs[1].hist(err,bins=np.linspace(-7,7,75),density=True,histtype='step',lw=1.5,label=f'{name} (n={len(sub):,})',color=COLOR[i])
    axs[1].set(xlabel='GPS-provider minus GNSS/INS speed (m/s)',ylabel='Density');axs[1].legend(fontsize=8,frameon=False);axs[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'gedvm_speed_error_sensitivity.pdf',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(8.6,4.5),sharex=True,gridspec_kw={'height_ratios':[1.2,1]})
    s=g[g.session==sorted(g.session.unique())[0]].sort_values('utc_ms').copy()
    m=(s.utc_ms>s.utc_ms.min()+600000)&(s.utc_ms<s.utc_ms.min()+870000)
    if m.sum()<30:m=(s.utc_ms<s.utc_ms.min()+270000)
    s=s[m]
    t=(s.utc_ms-s.utc_ms.iloc[0])/1000
    axs[0].plot(t,s.ref_speed_mps,lw=1.5,label='Phone-specific GNSS/INS reference')
    axs[0].plot(t,s.gps_speed_mps,lw=.85,alpha=.8,label='Android GPS-provider speed')
    axs[0].set_ylabel('Horizontal speed (m/s)');axs[0].legend(fontsize=8,frameon=False);axs[0].grid(alpha=.2)
    axs[1].plot(t,s.speed_error_mps,lw=.8,color=COLOR[2]);axs[1].axhline(0,c='black',lw=.6)
    axs[1].set(xlabel='Elapsed time in selected segment (s)',ylabel='Speed difference (m/s)');axs[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'gedvm_example_time_series.pdf',bbox_inches='tight',pad_inches=.12);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default=str(Path(__file__).resolve().parent/'gedvm_dataset-main'));p.add_argument('--out',default=str(Path(__file__).resolve().parent/'results'/'phase13'));args=p.parse_args()
    sess,dev,df=process(Path(args.dataset),Path(args.out))
    print(sess[['session','n','rmse_mps','mae_mps','p95_abs_error_mps','bias_mps']].to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('OVERALL',dev.tail(1).to_string(index=False,float_format=lambda v:f'{v:.3f}'))
if __name__=='__main__':main()
