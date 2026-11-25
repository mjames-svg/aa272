"""Build the integrated AA272 GNSS research narrative, figure atlas, and deck."""
from __future__ import annotations
import csv
import html
import json
import math
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from PIL import Image
from weasyprint import HTML
import fitz
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'; T=RES/'phase10'; DEST=ROOT/'FINAL_DELIVERABLES'
DEST.mkdir(exist_ok=True)
PLOT_DIR=DEST/'CURATED_FIGURES';PLOT_DIR.mkdir(exist_ok=True)
s=pd.read_csv(T/'stadium_method_comparison.csv')
b=pd.read_csv(T/'paired_game_cluster_bootstrap.csv')
g=pd.read_csv(T/'roof_visibility_summary.csv')
old=pd.read_csv(RES/'final'/'main_method_comparison.csv')
real=pd.read_csv(RES/'phase4/snapper/snapper_track_summary.csv')
samp=pd.read_csv(RES/'phase4/nfl_sampling_summary.csv')
val=json.loads((RES/'phase2/real_validation_summary.json').read_text())

INFO=[
('01','Measured football-field trajectory','phase9/football_field_w02_real_rtk.png','REAL position data','Real GNSS and RTK positions on walking recording w02; field markings are not georeferenced'),
('02','Measured Doppler quality dependence','phase2/real_LOO_residuals_by_CNo.png','REAL Doppler','Leave-one-out pseudorange-rate prediction residuals versus measured C/N0'),
('03','GNSS satellite geometry and uncertainty','phase2/real_satellite_count_uncertainty.png','REAL Doppler','Satellite-count and formal velocity uncertainty from the GPS-only Pixel Watch solution'),
('04','Sampling-frequency limitation','phase4/nfl_sprint_capture_vs_sampling_rate.png','NFL reference / ideal sampling','10-Hz NFL motion and purely mathematical sampling-rate sensitivity'),
('05','Original algorithm comparison','final/final_main_event_detection.png','SIMULATED NFL GNSS','Held-out persistent-error comparison, with four filtering approaches'),
('06','Accuracy and availability tradeoff','final/final_accuracy_availability.png','SIMULATED NFL GNSS','Speed RMSE only over accepted epochs versus percentage of accepted epochs'),
('07','Matched external receiver comparison','phase8/real_reference_robustness.png','REAL cross-device comparison','Head-mounted NovAtel versus forearm-mounted Pixel Watch; not co-located truth'),
('08','Real football field versus urban','phase9/real_field_vs_urban_position_errors.png','REAL position data','Different walking recordings; not a controlled roof comparison'),
('09','Geometric roof-model visualization','phase10/10_synthetic_roof_sky_masks.png','HYPOTHETICAL geometry','Azimuth/elevation masking in assumed roofs; no surveyed venue'),
('10','Geometry and GPS satellite visibility','phase10/10_satellite_visibility.png','HYPOTHETICAL geometry','GPS-only template counts after nested masking'),
('11','NFL detection by hypothetical roof','phase10/10_event_recall_by_roof.png','SIMULATED NFL GNSS','1-Hz high-speed interval recall under matched stadium masks'),
('12','Accuracy availability and false alarms','phase10/10_accuracy_availability_false_alerts.png','SIMULATED NFL GNSS','Separately vary satellite geometry and noise/fault conditions'),
('13','Paired roof-effect uncertainty','phase10/10_paired_game_bootstrap.png','SIMULATED NFL GNSS','Paired game-cluster bootstrapped differences versus open sky; only five games'),
('14','Boundary sensitivity','final/final_boundary_event_sensitivity.png','SIMULATED NFL GNSS','Many observed high-speed intervals are truncated by the tracking window'),
('15','Exploratory random-mask stress test','phase9/nfl_obstruction_analogue_stress_test.png','SIMULATED NFL GNSS','Preliminary single-run sensitivity, superseded by Phase 10'),
]

for num,title,src,kind,desc in INFO:
    p=RES/src
    assert p.exists(),p
    shutil.copy2(p,PLOT_DIR/(num+'_'+Path(src).name))
rows=[]
for p in sorted(RES.rglob('*.png')):
    category='Simulated or mixed — consult source script'
    for num,title,src,kind,desc in INFO:
        if p == RES/src:
            category=kind;break
    rows.append({'relative_path':str(p.relative_to(ROOT)),'category':category,
                 'is_curated':'yes' if any(p==(RES/entry[2]) for entry in INFO) else 'no'})
with (DEST/'FIGURE_INVENTORY.csv').open('w',newline='') as f:
    dw=csv.DictWriter(f,fieldnames=['relative_path','category','is_curated']);dw.writeheader();dw.writerows(rows)


def cell(sky,signal='measured_proxy',method='wls'):
    return s[(s.scenario==f'{sky}__{signal}')&(s.method==method)].iloc[0]

def fmt(value,nd=1,unit=''):
    return f'{value:.{nd}f}{unit}' if pd.notna(value) else 'N/A'

def phase10_table(signal):
    out=[]
    for sky,label in [('open','Open sky'),('partial_canopy','Partial canopy'),('deep_canopy','Deep canopy'),('sealed_roof','Fully signal-blocking roof')]:
        a=cell(sky,signal)
        out.append([label,fmt(a.mean_epoch_availability_pct,1,'%'),fmt(a.mean_speed_rmse_on_accepted_mps,2),
                    fmt(a.event_recall_pct,2,'%'),fmt(a.mean_false_alerts_per_100_negative_samples,2)])
    return out

main=old[(old.model=='persistent')&(old.scenario=='combined_degradation')]
original={m:main[main.method==m].iloc[0] for m in ['wls','fde','robust','wls_epoch_gate']}
w02=real[real.track=='w02'].iloc[0]
low=val['bin_statistics'][0]['median_abs_leave_one_out_residual_mps']
high=val['bin_statistics'][-1]['median_abs_leave_one_out_residual_mps']

# Human-readable story, with references to exact figure and data files.
story=f'''# WHEN GPS GOES MISSING
## GNSS Doppler estimation, high-speed NFL events, and hypothetical stadium obstruction

Stanford AA 272 final project — integrated November 2026 research narrative.

### The research question
How do satellite signal quality, velocity filtering, sampling rate, and obstruction affect our ability to identify short high-speed athlete movements when satellite GNSS is used as a hypothetical tracking modality?

### The answer in one paragraph
Measured smartphone/wearable GNSS Doppler residual proxies depend on received signal quality. Our 10-Hz NFL tracking data contain high-speed segments that can be missed simply by sampling at 1 Hz. When NFL motion is fed through synthesized GPS pseudorange-rate observations, conventional WLS, FDE, and robust WLS have similar interval detection performance under the studied assumptions; rejecting entire questionable epochs reduces accepted-only velocity error while reducing detection. New *paired* geometric sky-mask experiments show that, under our hypothetical partial and deep stadium overhangs, event recall falls from {cell('open').event_recall_pct:.2f}% (open sky) to {cell('partial_canopy').event_recall_pct:.2f}% and {cell('deep_canopy').event_recall_pct:.2f}%, primarily through lower measurement availability. A completely opaque, signal-blocking enclosure provides no direct GNSS solution by construction. None of these results is a measured NFL stadium GPS test.

### Evidence ladder and datasets
1. **Measured, RTK-referenced position:** SnapperGPS w02 is an eight-minute *walking* recording labelled American football pitch. {int(w02.matched_fixes)} paired RTK positions give {w02.median_horizontal_error_m:.2f} m median and {w02.p95_horizontal_error_m:.2f} m 95th-percentile horizontal disagreement. This is not velocity RMSE, and the outdoor field is not an NFL stadium.
2. **Measured wearable Doppler:** Pixel Watch GPS-only data provide {val['watch_epoch_count']} satellite-measurement epochs ({val['epochs_at_least_seven_satellites']} with at least seven satellites). Median absolute leave-one-out Doppler prediction residual is {low:.3f} m/s below C/N0 26 dB-Hz, versus {high:.3f} m/s at C/N0 at least 34 dB-Hz. These residuals are error *proxies*, not independent white-noise truth.
3. **Measured cross-device check:** The recovered NovAtel .rg trajectory aligns with the logged .rss positions and provides head-mounted speed comparisons to the forearm watch. They are not co-located, and binary velocity semantics remain inferred.
4. **NFL reference movement:** The NFL file contains 10-Hz RFID-based player tracking, not NFL GPS. The full 16-game sample contains 847 observed >=7 m/s intervals; ideal random-phase 1-Hz sampling intersects 64.51% of them (2 Hz: 83.73%; 5 Hz: 95.51%; 10 Hz: 100%).
5. **Simulation:** Five held-out games, 556 selected player trajectories, 287 observed >=7 m/s intervals and six Monte Carlo seeds support our comparison of WLS, FDE, robust WLS, and a deliberately strict epoch gate.

### Primary algorithm study: persistent combined degradation
Method | Event recall | Accepted epochs | Speed RMSE on accepted epochs | False alerts/100 negatives
---|---:|---:|---:|---:
WLS | {original['wls'].event_recall_pct:.2f}% | {original['wls'].mean_epoch_availability_pct:.2f}% | {original['wls'].mean_speed_rmse_on_accepted_mps:.2f} m/s | {original['wls'].mean_false_alerts_per_100_negative_samples:.2f}
FDE | {original['fde'].event_recall_pct:.2f}% | {original['fde'].mean_epoch_availability_pct:.2f}% | {original['fde'].mean_speed_rmse_on_accepted_mps:.2f} m/s | {original['fde'].mean_false_alerts_per_100_negative_samples:.2f}
Robust WLS | {original['robust'].event_recall_pct:.2f}% | {original['robust'].mean_epoch_availability_pct:.2f}% | {original['robust'].mean_speed_rmse_on_accepted_mps:.2f} m/s | {original['robust'].mean_false_alerts_per_100_negative_samples:.2f}
Strict epoch gate | {original['wls_epoch_gate'].event_recall_pct:.2f}% | {original['wls_epoch_gate'].mean_epoch_availability_pct:.2f}% | {original['wls_epoch_gate'].mean_speed_rmse_on_accepted_mps:.2f} m/s | {original['wls_epoch_gate'].mean_false_alerts_per_100_negative_samples:.2f}

### New physically motivated — but *hypothetical* — stadium experiment
We use satellite azimuth/elevation from 674 real-watch GPS geometry templates. The three nested geometric masks are: open sky (elevation >=10 degrees); a partial canopy (two opposing 25-degree azimuth half-width overhangs blocking satellites below 40 degrees); and a deeper canopy (45-degree azimuth half-width blocking satellites below 50 degrees). Roof orientation changes across paired player trials but stays fixed within a trial. A fourth limiting case excludes all direct GPS signals. Each case uses identical player motion, 1-Hz sample epochs, satellite templates, unmasked noise draws, and Monte Carlo seeds. Six seeds; all five held-out games; measured-residual-proxy noise plus a separate noise-doubling/5%-fault sensitivity. A mask is an engineering illustration, **not a surveyed or RF-calibrated stadium**.

#### GPS-only WLS — measured-residual-proxy condition
Roof model | Valid epochs | Accepted-only RMSE | Interval recall | False alerts/100 negatives
---|---:|---:|---:|---:
'''
for line in phase10_table('measured_proxy'):
    story+=' | '.join(line)+'\n'
story+='''
#### GPS-only WLS — degraded-noise/fault sensitivity
Roof model | Valid epochs | Accepted-only RMSE | Interval recall | False alerts/100 negatives
---|---:|---:|---:|---:
'''
for line in phase10_table('degraded_sensitivity'):story+=' | '.join(line)+'\n'
boot=b[(b.signal=='measured_proxy')&(b.method=='wls')].set_index('sky')
story+=f'''
### Paired effect sizes
For WLS under the baseline measured-residual proxy, the modeled partial canopy changes recall relative to paired open sky by {boot.loc['partial_canopy','point_delta_recall_pp']:.2f} percentage points (descriptive five-game bootstrap interval {boot.loc['partial_canopy','lower_95_pp']:.2f} to {boot.loc['partial_canopy','upper_95_pp']:.2f}); the deeper canopy changes it by {boot.loc['deep_canopy','point_delta_recall_pp']:.2f} percentage points. Bootstrap intervals resample only five held-out NFL games and do **not** account for uncertainty in roof physics or the lack of a stadium measurements dataset.

### Method
At each epoch, the synthetic GPS Doppler observation vector is y = H[v_ECEF; c-dot] + e, where H is a measured satellite line-of-sight design matrix, and e is sampled from measured receiver leave-one-out residual proxies with a modeled temporal correlation. The weighted least squares estimate minimizes (y-Hx)^T W (y-Hx), with W diagonal and inverse estimated variance. FDE tests residual consistency and may exclude satellites; Huber WLS reweights residual outliers. Accepted estimates require >=6 satellites, formal 3D velocity uncertainty <=4 m/s and horizontal speed <=15 m/s. A reference interval is counted as detected if any *accepted, sampled* GNSS speed within that interval meets >=7 m/s. All unsampled or rejected intervals count as missed. Speed RMSE is computed **only** on accepted epochs and cannot be compared without their availability fractions.

### Main scientific conclusions
- Satellite visibility and temporal coverage can be more limiting than the particular WLS variant.
- A method can improve reported error on selected epochs but become worse at continuous event monitoring.
- Increasing the sampling rate raises the ideal chance of observing short high-speed intervals; a 1-Hz receiver alone cannot reconstruct unobserved events without prediction or other sensors.
- A more enclosed geometric sky mask reduces GPS-only solution availability. When all direct signals are blocked, an unaided GPS-only Doppler solution is not available.
- No tested method is universally best. FDE and robust WLS do not consistently improve event detection over WLS in these simulations.

### Crucial limits
The NFL player velocities were observed using RFID, not satellite GNSS; the entire NFL satellite Doppler stream is simulated. The smartphone and watch data were recorded away from NFL stadiums, and the SnapperGPS football-field data concern position rather than per-satellite velocity estimation. The NovAtel and watch are on different body sites, and the .rg fields are not officially documented. Each hypothetical stadium model uses azimuth/elevation blockage but lacks roof geometry surveys, reflection/refraction models, calibrated attenuation, multi-constellation tracking, or stadium field measurements. A real enclosed stadium can sometimes receive attenuated/reflected signals; the sealed-roof case is only a limit where all direct GPS signals are absent. Five held-out games limit population generalization; 222 of 287 observed NFL intervals touch the pre-pass tracking-window edge.

### Selected figure sequence
'''
for num,title,src,kind,desc in INFO:
    story+=f'{num}. **{title}** — {kind}. `{src}`. {desc}.\n'
story+='''
### Reproduce in VS Code
`python -m pip install -r requirements.txt` then `python phase10_geometry_stadium.py --plots-only` to reproduce the phase-10 figures from included CSV results, or `python phase10_geometry_stadium.py --seeds 6` to rebuild the full paired simulation. Run `python test_phase10.py` for the new checks. For original phases, see `run_complete.py` (or `python run_complete.py --full-simulation` for a complete expensive rebuild). Run `python build_final_story.py` to regenerate the integrated report/deck/figure atlas.

### References and external context
- University of Oxford, SnapperGPS signal snapshots 2: https://ora.ox.ac.uk/objects/uuid:cf48364c-c837-4dbd-b615-982eb6eb7aff
- Beuchert dataset source and recording descriptions: https://github.com/JonasBchrt/snapshot-gnss-data-2
- NFL Football Operations, Next Gen Stats RFID receiver system and 10-Hz tracking: https://edge-operations.nfl.com/game-operations-logistics/technology/performance-tracking-data-next-gen-stats
- Kim et al. (2025), *A quantitative method for evaluating GNSS signal disruption in sports stadiums*, Sports Engineering, doi:10.1007/s12283-025-00530-7. This is external evidence that roof coverage can degrade GNSS, not a calibration of our hypothetical masks.
- Zenodo Pixel Watch GNSS data: https://zenodo.org/records/12566912
'''
(ROOT/'FINAL_INTEGRATED_STORY.md').write_text(story,encoding='utf-8')

# HTML/PDF authored from the same numeric source files.
CSS='''
@page {size: Letter; margin: 0.65in 0.66in 0.62in; @bottom-right {content:"AA 272 / " counter(page); color:#60717a; font-size:8pt;} }
body{font-family:Arial,Helvetica,sans-serif;color:#192A3B;font-size:10pt;line-height:1.38;}
h1{font-size:27pt;line-height:1.05;color:#14283B;margin:0 0 9px} h2{font-size:15.5pt;color:#185E6A;margin:23px 0 7px}
h3{font-size:11.5pt;color:#26465E;margin:13px 0 6px} p{margin:7px 0 10px}
.eyebrow{font-size:9pt;color:#5A787D;font-weight:bold;letter-spacing:1.1px;margin:0 0 12px}
.subtitle{font-size:14pt;color:#4F6671;margin-bottom:20px}.lead{font-size:12pt;line-height:1.5;color:#283F4C}
.meta{background:#EFF4F4;padding:12px 14px;border-left:4px solid #2C797D;margin:14px 0 16px}
.meta strong{color:#142D3D}.note{font-size:9pt;color:#61727D;line-height:1.38}
.page{break-before:page}.avoid{break-inside:avoid}
img{display:block;max-width:100%;max-height:4.7in;margin:auto} figure{break-inside:avoid;margin:15px 0 22px}
figcaption{font-size:8.4pt;color:#647785;text-align:center;margin:6px 2px 0}
table{border-collapse:collapse;width:100%;font-size:8.4pt;margin:11px 0 15px;break-inside:avoid}
th{background:#204F5E;color:white;text-align:left;padding:7px 6px}
td{padding:7px 6px;border-bottom:1px solid #D4DFE3} tr:nth-child(even) td{background:#F5F8F9}
.cites{font-size:8pt;word-break:break-word;color:#5A6A73}
.small{font-size:8.7pt;color:#5C6972}.kpirow{display:flex;gap:10px;margin:18px 0}
.kpi{flex:1;border:1px solid #CEDDDC;border-radius:5px;padding:11px;text-align:center}
.kpi b{display:block;font-size:20pt;color:#17686B}.kpi span{font-size:8pt;color:#5C747B}
.bar{height:5px;background:#2E777A;margin-bottom:23px}.two{display:flex;gap:12px}.two>*{flex:1}
ul{padding-left:22px}li{margin:6px 0}
'''

def table_html(headers,values):
    head=''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)
    cells=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in r)+'</tr>' for r in values)
    return '<table><thead><tr>'+head+'</tr></thead><tbody>'+cells+'</tbody></table>'

def figure(src,title,caption):
    uri=(RES/src).resolve().as_uri()
    return f'<figure><img src="{uri}"/><figcaption><b>{html.escape(title)}.</b> {html.escape(caption)}</figcaption></figure>'

orig_table=[]
for meth,label in [('wls','WLS'),('fde','FDE'),('robust','Robust WLS'),('wls_epoch_gate','Strict epoch gate')]:
    z=original[meth]
    orig_table.append([label,fmt(z.event_recall_pct,2,'%'),fmt(z.mean_epoch_availability_pct,2,'%'),
                       fmt(z.mean_speed_rmse_on_accepted_mps,2),fmt(z.mean_false_alerts_per_100_negative_samples,2)])
bootp=b[(b.signal=='measured_proxy')&(b.method=='wls')].set_index('sky')
H=[]
H.append('''<h1>WHEN GPS GOES MISSING</h1><div class="bar"></div>
<div class="subtitle">GNSS Doppler, NFL sprint detection, and hypothetical stadium obstruction</div>
<p class="eyebrow">STANFORD AA 272 • INTEGRATED FINAL RESEARCH REPORT • NOVEMBER 2026</p>
<p class="lead">Our result: GNSS tracking reliability is a balance of <b>velocity error, available satellite measurements, and event sampling</b>. Even a precise estimator is of limited use if it cannot return a solution when an athlete sprints.</p>''')
H.append('''<div class="meta"><b>Scope and provenance.</b> Real Pixel Watch GPS Doppler, real SnapperGPS/RTK walking positions, and real NFL player motion inform the analysis. All satellite observations attached to NFL motion and all stadium roofs are simulated. The NFL production system uses RFID, not GNSS.</div>''')
H.append(f'''<div class="kpirow"><div class="kpi"><b>{int(val['watch_epoch_count'])}</b><span>Measured watch GPS epochs</span></div><div class="kpi"><b>287</b><span>Held-out observed fast intervals</span></div><div class="kpi"><b>6</b><span>Paired random seeds</span></div><div class="kpi"><b>{cell('partial_canopy').event_recall_pct:.1f}%</b><span>Partial-canopy WLS recall</span></div></div>''')
H.append('''<h2>Research question</h2><p>How do signal quality, GNSS velocity filtering, receiver sampling, and satellite obstruction affect our ability to detect short, high-speed athletic movements?</p>
<h2>Executive conclusion</h2><p>In our tested simulations, FDE and robust WLS give little consistent improvement over ordinary WLS. Stricter whole-epoch rejection gives lower accepted-only RMSE, but loses many high-speed events. Modeling more obstructed satellite geometry strongly reduces the fraction of valid speed estimates and therefore sprint-event recall.</p>
<h2>Evidence strategy</h2><p><b>Measured position</b> (SnapperGPS w02 football pitch) → <b>measured Doppler residual proxies</b> (Pixel Watch) → <b>NFL reference motion</b> (10-Hz RFID trajectories) → <b>simulated GNSS</b> (WLS/FDE/robust) → <b>paired hypothetical roof masks</b>.</p>''')
H.append('<div class="page"></div><h2>1. Real football-field positions: a grounding case</h2>')
H.append(f'<p>SnapperGPS <b>w02</b> is labeled an American football pitch, recorded while walking. {int(w02.matched_fixes)} fixes were matched to independent RTK positions. The median horizontal disagreement was <b>{w02.median_horizontal_error_m:.2f} m</b>; the 95th percentile was <b>{w02.p95_horizontal_error_m:.2f} m</b>. These are position errors, not GNSS Doppler speed errors, and the recording is not a covered NFL stadium.</p>')
H.append(figure('phase9/football_field_w02_real_rtk.png','Figure 1','Real measured positions against RTK; each dot is a GNSS fix and the green line denotes the reference walking route.'))
H.append('<div class="page"></div><h2>2. Measured Doppler quality affects our error model</h2>')
H.append(f'<p>We analyzed {val["watch_epoch_count"]} GPS epochs, including {val["epochs_at_least_seven_satellites"]} epochs with at least seven usable GPS satellites. The median absolute leave-one-out residual decreased from <b>{low:.3f} m/s</b> at C/N0 below 26 dB-Hz to <b>{high:.3f} m/s</b> at C/N0 of at least 34 dB-Hz. This motivates an empirical, quality-conditioned noise proxy; it is not a direct measurement-error ground truth.</p>')
H.append(figure('phase2/real_LOO_residuals_by_CNo.png','Figure 2','Observed leave-one-out prediction residual distributions; lower C/N0 bins show greater dispersion.'))
H.append('<p class="note">Independent-device check: the inferred NovAtel head trajectory enables comparison with forearm Pixel Watch speed, but different mounting locations and undocumented .rg field layout preclude certified device-accuracy claims.</p>')
H.append('<div class="page"></div><h2>3. Short NFL events can be missed before any GNSS noise</h2>')
H.append('''<p>The 10-Hz NFL pre-pass reference sample contains 847 observed high-speed intervals over 16 games (illustrative threshold: 7 m/s). Random-phase direct sampling at 1 Hz intersects only 64.51% of them; 2 Hz reaches 83.73%, 5 Hz reaches 95.51%, and 10 Hz reaches 100%. This is a sampling constraint, not estimator bias.</p>''')
H.append(figure('phase4/nfl_sprint_capture_vs_sampling_rate.png','Figure 3','Ideal intersection probabilities from measured NFL reference intervals, without GNSS errors or interpolation.'))
H.append('<div class="page"></div><h2>4. Original estimator comparison: tracking availability matters</h2>')
H.append('<p>We tested six seeds over five held-out games and 287 observed fast intervals. All NFL satellite Doppler measurements were synthesized from real-watch geometry and residual proxies. Under a severe persistent-error condition with 30% random satellite loss, doubled error proxies and 5% faults, four policies yield:</p>')
H.append(table_html(['Algorithm','Event recall','Valid epochs','Accepted-only RMSE (m/s)','False alerts / 100 negatives'],orig_table))
H.append(figure('final/final_main_event_detection.png','Figure 4','High-speed interval detection under the original difficult simulated conditions.'))
H.append('<p class="note">The strict gate trades lower reported RMSE and fewer false alerts for very low monitoring availability. Accepted-only errors are conditional on which epochs survived filtering.</p>')
H.append('<div class="page"></div><h2>5. The controlled stadium extension: sky masks, not stadium measurements</h2>')
H.append('''<p>We reconstructed satellite elevation/azimuth from real GPS-only watch geometry and imposed four nested visibility cases. Open sky keeps satellites above 10° elevation; partial canopies block two opposed 25°-halfwidth roof sectors below 40°; deep canopies block 45°-halfwidth sectors below 50°; a fully signal-blocking enclosure excludes every direct GPS signal. Each roof orientation is randomized across trials but kept fixed within each paired player trial.</p>
<p>All roof conditions use the same reference player velocities, satellite template, sample phases, synthetic error innovations and six seeds. The comparison therefore isolates <b>assumed geometry</b>, while a second matched sensitivity separately doubles residual-proxy noise and adds persistent 5% satellite faults. No physical stadium roof was surveyed.</p>''')
H.append(figure('phase10/10_synthetic_roof_sky_masks.png','Figure 5','Polar sky plots of stylized visibility masks. The dots are one real watch satellite-geometry snapshot; gray is blocked.'))
H.append('<div class="page"></div><h2>6. New primary result: fewer satellites, fewer detected sprints</h2>')
H.append(table_html(['Roof geometry','Valid epochs','Accepted-only RMSE (m/s)','Event recall','False alerts / 100 negatives'],phase10_table('measured_proxy')))
H.append(f'<p>Relative to the paired open-sky WLS recall ({cell("open").event_recall_pct:.2f}%), the modeled partial canopy gives <b>{cell("partial_canopy").event_recall_pct:.2f}%</b>, and the deeper canopy <b>{cell("deep_canopy").event_recall_pct:.2f}%</b>. For the fully signal-blocking limit there is no GNSS measurement and thus no standalone solution. Across those cases, the speed RMSE of surviving epochs changes only from {cell("open").mean_speed_rmse_on_accepted_mps:.2f} to {cell("deep_canopy").mean_speed_rmse_on_accepted_mps:.2f} m/s—demonstrating why accuracy should not be reported independently of availability.</p>')
H.append(figure('phase10/10_event_recall_by_roof.png','Figure 6','Matched, simulated interval-detection comparison for WLS, FDE, robust weighting, and strict epoch rejection.'))
H.append('<div class="page"></div><h2>7. Satellite geometry explains the dominant mechanism</h2>')
H.append(figure('phase10/10_satellite_visibility.png','Figure 7','Distribution of GPS satellite counts after each geometric mask is applied.'))
H.append(f'''<p>Our GPS-only templates usually contain seven to nine satellites. The fractions of paired track trials with at least six surviving satellites are: <b>{g.loc[g.sky.eq('open'),'at_least_6_pct'].iloc[0]:.1f}%</b> open; <b>{g.loc[g.sky.eq('partial_canopy'),'at_least_6_pct'].iloc[0]:.1f}%</b> partial; and <b>{g.loc[g.sky.eq('deep_canopy'),'at_least_6_pct'].iloc[0]:.1f}%</b> deep. The six-satellite processing criterion makes availability particularly sensitive to geometric loss in this single-constellation setup.</p>''')
H.append('<div class="page"></div><h2>8. Separate signal-noise effects from geometric obstruction</h2>')
H.append(table_html(['Roof geometry','Valid epochs','Accepted-only RMSE (m/s)','Event recall','False alerts / 100 negatives'],phase10_table('degraded_sensitivity')))
H.append(figure('phase10/10_accuracy_availability_false_alerts.png','Figure 8','Two signal-noise assumptions evaluated on the same nested roof masks using WLS.'))
H.append('<p class="note">Zero observations under the fully blocked model imply zero availability and no defined RMSE. Zero false alerts under total outage are not evidence of good detection; the method cannot produce any alert.</p>')
H.append('<div class="page"></div><h2>9. Paired uncertainty and interpretation</h2>')
H.append(figure('phase10/10_paired_game_bootstrap.png','Figure 9','Change in detection relative to the paired open-sky experiment; game-cluster bootstrap intervals resample only five games.'))
H.append(f'<p>For WLS, the paired partial-canopy recall difference is <b>{bootp.loc["partial_canopy","point_delta_recall_pp"]:.2f} percentage points</b> (descriptive 95% game-cluster interval {bootp.loc["partial_canopy","lower_95_pp"]:.2f} to {bootp.loc["partial_canopy","upper_95_pp"]:.2f}). The deeper canopy gives <b>{bootp.loc["deep_canopy","point_delta_recall_pp"]:.2f} percentage points</b>. These intervals describe finite-game resampling conditional on the simulation, not uncertainty over all real stadium roofs.</p>')
H.append('<h2>What the results do and do not establish</h2><ul><li>Established within the model: the nested roof masks reduce GPS-only satellite redundancy, accepted velocity epochs, and short-event detection.</li><li>Observed in measured data: receiver Doppler prediction residuals depend on C/N0; football-field position fixes differ from RTK positions.</li><li>Not established: numerical GNSS accuracy for actual NFL players inside any named open or covered NFL stadium, or a universal algorithm ranking.</li></ul>')
H.append('<div class="page"></div><h2>10. Conclusions, engineering implication, and limitations</h2>')
H.append('''<h3>Three defensible takeaways</h3><ol><li><b>Sampling:</b> 1-Hz receiver data cannot directly observe every short high-speed event, even with zero measurement error.</li><li><b>Estimation:</b> WLS is a credible baseline here; FDE/robust weighting do not consistently dominate; whole-epoch rejection can be misleading when reported only as RMSE.</li><li><b>Reception:</b> in a GPS-only receiver with seven-to-nine observed satellites, partial sky obstruction can remove the redundancy necessary to provide continuous velocity estimates.</li></ol>
<h3>Limits to highlight in any presentation</h3><p>The NFL Next Gen Stats dataset is RFID-derived, not GNSS. Satellite Doppler observations for NFL motion are entirely simulated. Real-wearable error proxies were measured elsewhere; temporal noise correlation, fault occurrence, and the stadium roof angles are modeled assumptions. The closed-roof case is deliberately fully signal-blocking, not an empirical description of every indoor stadium. Many reference intervals touch the pre-pass window boundary, so we measure observed high-speed segments rather than complete real-world sprints. The NovAtel .rg binary decoding is inferred and the watch is mounted on a different body segment.</p>
<h3>Practical engineering extension</h3><p>To generalize beyond this sensitivity study, use actual stadium roof geometry or matched indoor/outdoor receiver logs; include multi-constellation GNSS, inertial fusion and precise C/N0/azimuth/elevation. For actual NFL games, the existing RFID/UWB venue system addresses the indoor-coverage problem without direct GNSS visibility.</p>''')
H.append('''<h2>Sources</h2><div class="cites">
<p>Oxford SnapperGPS: https://ora.ox.ac.uk/objects/uuid:cf48364c-c837-4dbd-b615-982eb6eb7aff</p>
<p>NFL Next Gen Stats RFID: https://edge-operations.nfl.com/game-operations-logistics/technology/performance-tracking-data-next-gen-stats</p>
<p>Kim et al., Sports Engineering (2025): https://doi.org/10.1007/s12283-025-00530-7</p>
<p>Pixel Watch raw data: https://zenodo.org/records/12566912</p>
<p>Exact machine-readable tables and scripts: results/phase10/ and phase10_geometry_stadium.py in the packaged project.</p>
</div>''')
page='<html><head><meta charset="utf-8"/><style>'+CSS+'</style></head><body>\n'+''.join(H)+'</body></html>'
(DEST/'INTEGRATED_RESEARCH_REPORT.html').write_text(page)
HTML(string=page,base_url=str(ROOT)).write_pdf(str(DEST/'INTEGRATED_RESEARCH_REPORT.pdf'))

# A complete, searchable figure atlas: one image per page, with classification and provenance.
allplots=sorted(RES.rglob('*.png'))
doc=fitz.open()
for i,p in enumerate(allplots):
    im=Image.open(p); width,height=im.size
    r=width/height
    pg=doc.new_page(width=900,height=660)
    pg.draw_rect(fitz.Rect(0,0,900,8),color=(.13,.42,.47),fill=(.13,.42,.47))
    name=str(p.relative_to(ROOT))
    pg.insert_text((35,45),f'AA272 FIGURE ATLAS  |  {i+1} / {len(allplots)}',fontsize=11,color=(.12,.38,.43))
    pg.insert_textbox(fitz.Rect(35,60,865,103),name,fontsize=17,fontname='helv',color=(.12,.18,.23))
    box=fitz.Rect(32,116,868,574)
    if r>box.width/box.height:
        newh=box.width/r; y=(box.y0+box.y1-newh)/2;target=fitz.Rect(box.x0,y,box.x1,y+newh)
    else:
        neww=box.height*r;x=(box.x0+box.x1-neww)/2;target=fitz.Rect(x,box.y0,x+neww,box.y1)
    pg.insert_image(target,filename=str(p),keep_proportion=True)
    tag=next((kind for num,title,src,kind,desc in INFO if p==RES/src),'See script for data provenance')
    pg.insert_textbox(fitz.Rect(35,594,865,648),f'Provenance: {tag}. See figure inventory and generating script for exact analysis.',fontsize=10,color=(.42,.48,.53))
doc.set_metadata({'title':'AA272 - Complete Figure Atlas','subject':'All existing and newly generated GNSS project figures'})
doc.save(str(DEST/'COMPLETE_FIGURE_ATLAS.pdf'),garbage=4,deflate=True)
doc.close()

# Editable widescreen slide deck with original data-backed charts.
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
BLU=RGBColor(19,40,57); TEAL=RGBColor(32,111,116); GRAY=RGBColor(79,100,109)
WHITE=RGBColor(255,255,255); LIGHT=RGBColor(237,243,244); ORANGE=RGBColor(195,126,54)

def text(shapes,x,y,w,h,txt,size=18,color=BLU,bold=False,align=None):
    tb=shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=tb.text_frame;tf.clear();tf.word_wrap=True;tf.margin_top=Pt(0);tf.margin_bottom=Pt(0)
    for k,line in enumerate(str(txt).split('\n')):
        p=tf.paragraphs[0] if k==0 else tf.add_paragraph();p.text=line
        p.font.size=Pt(size);p.font.name='Aptos';p.font.bold=bold;p.font.color.rgb=color
        p.space_after=Pt(6)
        if align is not None:p.alignment=align
    return tb

def slide(title,subtitle=None):
    sld=prs.slides.add_slide(prs.slide_layouts[6]); sh=sld.shapes
    bg=sh.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.13));bg.fill.solid();bg.fill.fore_color.rgb=TEAL;bg.line.fill.background()
    text(sh,.55,.38,12.2,.58,title,29,BLU,True)
    if subtitle:text(sh,.57,1.02,12.1,.4,subtitle,13,GRAY)
    line=sh.add_shape(MSO_SHAPE.RECTANGLE,Inches(.56),Inches(6.98),Inches(12.2),Pt(1))
    line.fill.solid();line.fill.fore_color.rgb=RGBColor(201,216,221);line.line.fill.background()
    text(sh,.56,7.04,10,.22,'STANFORD AA 272  •  REAL GNSS + NFL-TRAJECTORY SIMULATION',9,GRAY)
    text(sh,12.11,7.03,.5,.24,str(len(prs.slides)),9,GRAY,align=PP_ALIGN.RIGHT)
    return sld

def add_pic(shapes,src,x,y,w,h):
    img=Image.open(RES/src);ratio=img.width/img.height;boxratio=w/h
    if ratio>boxratio: width=w;height=w/ratio
    else: height=h;width=h*ratio
    shapes.add_picture(str(RES/src),Inches(x+(w-width)/2),Inches(y+(h-height)/2),width=Inches(width),height=Inches(height))

def bullet_sidebar(shapes,lines,x=9.32,y=1.7,w=3.35,size=17):
    y0=y
    for title,body in lines:
        dot=shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y0+.04),Inches(.14),Inches(.14))
        dot.fill.solid();dot.fill.fore_color.rgb=TEAL;dot.line.fill.background()
        text(shapes,x+.23,y0,w-.28,.38,title,size,BLU,True)
        text(shapes,x+.23,y0+.46,w-.3,1.05,body,size-3,GRAY)
        y0+=1.55

st=slide('WHEN GPS GOES MISSING','GNSS athlete-velocity estimation • NFL high-speed events • Hypothetical stadium roofs')
bg_rect=st.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.65),Inches(1.72),Inches(12.05),Inches(4.48))
bg_rect.fill.solid();bg_rect.fill.fore_color.rgb=LIGHT;bg_rect.line.fill.background()
text(st.shapes,1,2.15,11.3,1.5,'The most accurate estimate cannot detect a sprint if there is no usable GNSS observation.',31,BLU,True)
text(st.shapes,1,4.25,10.7,1.0,'A measured-data-informed, six-seed controlled analysis using 10-Hz NFL player motion and synthetic GPS Doppler.',19,TEAL)

st=slide('The question and the evidence','Measured GNSS informs the model; NFL GNSS itself remains simulated')
for j,(a,bod) in enumerate([
 ('01 • REAL FIELD','SnapperGPS w02: football-field walking GNSS positions vs RTK'),
 ('02 • REAL DOPPLER','Pixel Watch per-satellite rates, residuals and C/N0'),
 ('03 • NFL REFERENCE','10-Hz RFID player speed and event intervals'),
 ('04 • SYNTHETIC GNSS','WLS/FDE/robust tracking under noise, outages and sky masks')]):
  x=.8+(j%2)*6.25;y=1.65+(j//2)*2.1
  q=st.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(5.6),Inches(1.7))
  q.fill.solid();q.fill.fore_color.rgb=LIGHT;q.line.fill.background()
  text(st.shapes,x+.25,y+.20,5.1,.45,a,18,TEAL,True)
  text(st.shapes,x+.25,y+.78,5.03,.7,bod,15,BLU)

spec=[
('Field data are real — but they measure position','phase9/football_field_w02_real_rtk.png',[
  ('9.76 m','Median horizontal GNSS–RTK disagreement'),('Walking case','Not NFL sprinting; no indoor comparator')]),
('Signal strength changes Doppler residual behavior','phase2/real_LOO_residuals_by_CNo.png',[
  ('1.014 m/s','Median |LOO| below 26 dB-Hz'),('0.492 m/s','Median |LOO| above 34 dB-Hz')]),
('Sampling imposes its own detection ceiling','phase4/nfl_sprint_capture_vs_sampling_rate.png',[
  ('64.5%','Ideal 1-Hz capture of >=7 m/s intervals'),('95.5%','Ideal capture at 5 Hz')]),
('Baseline filtering offers no universal winner','final/final_main_event_detection.png',[
  ('24.45%','WLS recall, difficult persistent scenario'),('24.80%','Robust WLS, same simulation')]),
('Accepted-only error can hide missing estimates','final/final_accuracy_availability.png',[
  ('2.18 m/s','Strict gate RMSE on accepted epochs'),('11.16%','Only this fraction of epochs remains')]),
('Model the stadium roof using satellite directions','phase10/10_synthetic_roof_sky_masks.png',[
  ('Geometric','Nested azimuth/elevation sky masks'),('Not field data','No calibrated roof or NFL RF measurement')]),
('Obstruction reduces receiver redundancy','phase10/10_satellite_visibility.png',[
  ('71.8%','Partial-canopy trials with ≥6 satellites'),('26.6%','Deep-canopy trials with ≥6 satellites')]),
('The headline result: fewer detected high-speed intervals','phase10/10_event_recall_by_roof.png',[
  ('48.0% → 36.0%','WLS open to partial-canopy recall'),('→ 13.4%','WLS recall with deep canopy')]),
('Noise and geometry are distinct stressors','phase10/10_accuracy_availability_false_alerts.png',[
  ('Separate effects','Compare same sky mask under two error models'),('RMSE is conditional','Undefined when 0% of epochs survive')]),
('Paired estimates quantify the modeled effect','phase10/10_paired_game_bootstrap.png',[
  (f'{bootp.loc["partial_canopy","point_delta_recall_pp"]:.1f} pp','Partial versus paired open-sky WLS'),('Five games','Bootstrap does not capture stadium-model uncertainty')]),
]
for title,fig_src,bullets in spec:
  st=slide(title,'All stadium roof measurements in this analysis are hypothetical')
  add_pic(st.shapes,fig_src,.48,1.52,8.55,5.25)
  bullet_sidebar(st.shapes,bullets,x=9.32,y=1.8)

st=slide('Four performance metrics tell the whole story','No algorithm should be ranked on accepted-only RMSE')
for j,(title,body) in enumerate([
 ('Speed error','RMSE over valid estimates only; requires a coverage denominator'),
 ('Availability','Percent of scheduled receiver epochs that produce an accepted solution'),
 ('Event recall','Share of reference >=7 m/s intervals detected at least once'),
 ('False alarms','High-speed alerts during sampled non-high-speed reference epochs')]):
  x=.75+(j%2)*6.25;y=1.55+(j//2)*2.2
  q=st.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(5.8),Inches(1.73))
  q.fill.solid();q.fill.fore_color.rgb=LIGHT;q.line.fill.background()
  text(st.shapes,x+.3,y+.2,5.2,.5,title,22,TEAL,True)
  text(st.shapes,x+.3,y+.88,5.15,.7,body,15,BLU)

st=slide('Conclusions','What we learned — and what we cannot claim')
for y,(headline,detail) in enumerate([
 ('Visibility and sampling control coverage','A 1-Hz receiver and GPS-only 7–9-satellite templates are vulnerable to missed observations.'),
 ('Filtering cannot replace missing satellites','WLS, FDE and robust WLS behave similarly in the paired roof experiments.'),
 ('No actual NFL-stadium GNSS is measured','The NFL uses RFID; roof dimensions and synthetic Doppler errors are modeling assumptions.'),
 ('Practical next validation','Survey a roof sky mask and collect actual stadium GNSS/IMU logs, ideally multi-constellation.')]):
  yy=1.52+y*1.24
  text(st.shapes,.82,yy,11.4,.40,headline,20,TEAL,True)
  text(st.shapes,.82,yy+.48,11.1,.58,detail,15,GRAY)

st=slide('References and reproducibility','Full figures, CSV results, report and scripts are included in the project ZIP')
refs=[
'University of Oxford, SnapperGPS Signal Snapshots 2 (2023)',
'NFL Football Operations: Next Gen Stats venue-based RFID/UWB system',
'Kim et al. (2025), Sports Engineering, DOI: 10.1007/s12283-025-00530-7',
'Zenodo 12566912: Pixel Watch GNSS pedestrian measurements',
'Project scripts: phase10_geometry_stadium.py, phase6_event_metrics.py, phase2_validation.py',
'Output data: results/phase10/stadium_method_comparison.csv + all paired epoch records'
]
for j,r in enumerate(refs):
  text(st.shapes,.9,1.6+j*.79,11.6,.62,r,17,BLU)

talking_points=[
    "Our motivating question is not just how accurate GNSS is when it works, but whether the receiver produces measurements at the time a high-speed event occurs. We combine measured wearable GNSS characteristics with actual NFL player motion, not real NFL satellite observations.",
    "This four-part evidence chain prevents us from overclaiming. SnapperGPS offers measured football-field positions and RTK reference; Pixel Watch offers real satellite Doppler; the NFL tracks are 10-Hz RFID. All NFL GPS measurements and all roof masks are constructed in the simulation.",
    "This is recording w02: a person walking on an American football pitch, not sprinting inside an NFL stadium. The median GNSS-versus-RTK horizontal position disagreement was 9.76 meters over 480 matched observations. This grounds the study in field data, not in device-velocity ground truth.",
    "The Pixel Watch raw pseudorange-rate data show that lower C/N0 is associated with larger leave-one-out Doppler prediction residuals. We see about 1.014 m/s median absolute residual below 26 dB-Hz and 0.492 m/s above 34 dB-Hz. These residuals are useful proxies, but they include motion and modeling effects.",
    "The 10-Hz NFL trajectories contain 847 observed fast intervals across 16 games. At the 7 m/s illustrative threshold, direct 1-Hz sampling intersects only 64.51% of these intervals, even with perfect accuracy. This is a fundamental timing limitation before any GNSS noise is added.",
    "Our original primary experiment uses five held-out games and 287 observed fast intervals. Under persistent, severe synthetic degradation, WLS detected 24.45%, FDE 24.33%, and robust WLS 24.80%. The differences are small, so there is no defensible algorithm winner here.",
    "The strict epoch rejection rule looks better if we only quote accepted-only RMSE: 2.18 m/s instead of about 2.86. But it accepts only 11.16% of scheduled epochs, causing event recall to drop to 5.11%. Error without an availability denominator can mislead.",
    "This new experiment replaces independent random satellite drops with direction-dependent sky masks. We use real watch satellite azimuth/elevation, and model opposing partial and deep roof overhangs. The exact geometric angles are idealized assumptions, not measurements of a named stadium.",
    "Because the watch GPS-only templates contain only about seven to nine satellites, restricting sky visibility rapidly removes solution redundancy. We require six observations for our standard acceptance policy. About 71.8% of partial-canopy trials and 26.6% of deep-canopy trials still have at least six satellites.",
    "The headline numerical finding: using paired seeds and identical reference motion, WLS high-speed interval recall falls from 47.97% open sky to 36.00% with partial canopy and 13.41% with deep canopy. The fully signal-blocked theoretical limit produces no satellite-only result. FDE and robust WLS follow very similar patterns.",
    "We separately doubled modeled noise and added persistent 5% satellite faults. This increases accepted-only error substantially; changing the sky mask reduces the fraction of epochs that even produce an estimate. These are two different failure mechanisms and must not be confused.",
    "Paired data let us estimate the effect of one roof mask versus another while holding measurements fixed. The WLS partial-canopy change is -11.96 percentage points relative to open sky. Uncertainty whiskers come from resampling just five held-out games: they do not represent all real stadiums or roof-physics uncertainty.",
    "Please evaluate speed RMSE, valid-epoch availability, complete reference-interval recall, and false alarms together. Reporting only accuracy on retained estimates would favor methods that discard the difficult measurements precisely when athletes move fast.",
    "Our defensible conclusion: for this GPS-only single-constellation simulation, visibility and sampling control event detection more than fine-tuning WLS. NFL production tracking uses RFID, not GNSS; measured stadium satellite data would be required for a real venue comparison.",
    "The exact machine-readable results are in the phase10 CSV files and Python scripts. The original phases, raw inputs, figure collection, integrated report, and references are all preserved in the final ZIP. External roof-coverage literature supports the motivation, not calibration of our chosen masks."
]
assert len(talking_points)==len(prs.slides)
for slideobj,point in zip(prs.slides,talking_points):
    slideobj.notes_slide.notes_text_frame.text=point
pptx=DEST/'FINAL_RESEARCH_STORY.pptx';prs.save(pptx)
if shutil.which('libreoffice'):
    subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/aa272_lo_story_'+str(Path(__file__).stat().st_size),
                   '--headless','--convert-to','pdf','--outdir',str(DEST),str(pptx)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
notes='# Presentation speaker notes — 15 slides\n\n'
for i,(slideobj,point) in enumerate(zip(prs.slides,talking_points),1):
    title=slideobj.shapes[1].text if hasattr(slideobj.shapes[1],'text') else f'Slide {i}'
    notes+=f'## Slide {i}: {title}\n\n{point}\n\n'
notes+='Final line: In these tested models, an accurate GPS estimate is not useful if the receiver cannot return a speed during the movement we want to detect.\n'
(DEST/'PRESENTATION_TALK_TRACK.md').write_text(notes,encoding='utf-8')
print('FINAL STORY READY',DEST)
print('PDF pages',len(fitz.open(str(DEST/'INTEGRATED_RESEARCH_REPORT.pdf'))),'deck',len(prs.slides),'atlas figures',len(allplots))
