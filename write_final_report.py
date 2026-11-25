from pathlib import Path
import json
import pandas as pd

root=Path(__file__).resolve().parent
out=root/'results/final'
m=pd.read_csv(out/'main_method_comparison.csv')
b=pd.read_csv(out/'paired_bootstrap_uncertainty.csv')
e=pd.read_csv(out/'reference_events.csv')
s=pd.read_csv(out/'event_definition_sensitivity.csv')
i=pd.read_csv(out/'fully_observed_event_sensitivity.csv')
meta=json.loads((out/'expanded_cohort.json').read_text())
valid=json.loads((root/'results/phase2/real_validation_summary.json').read_text())
def tbl(df, cols,rename=None, digits=2):
 d=df[cols].copy()
 if rename:d=d.rename(columns=rename)
 for c in d.select_dtypes(include='number'):
  if d[c].dtype.kind=='f':d[c]=d[c].round(digits)
 return d.to_markdown(index=False)

core=m[(m.model=='persistent')&(m.scenario=='combined_degradation')].set_index('method').loc[['wls','fde','robust','wls_epoch_gate']].reset_index()
core['method']=core.method.replace({'wls':'WLS','fde':'FDE','robust':'Robust weighting','wls_epoch_gate':'WLS + strict epoch rejection'})
all_scenarios=m.pivot_table(index=['model','scenario'],columns='method',values='event_recall_pct').reset_index().rename(columns={'wls':'WLS','fde':'FDE','robust':'Robust','wls_epoch_gate':'WLS + epoch gate'})
ci=b[(b.population=='all_observed_windows')&(b.model=='persistent')&(b.scenario=='combined_degradation')].query("method != 'wls'").copy()
ci['method']=ci.method.replace({'fde':'FDE','robust':'Robust weighting','wls_epoch_gate':'WLS + strict epoch rejection'})
interior=i[(i.model=='persistent')&(i.scenario=='combined_degradation')].set_index('method').loc[['wls','fde','robust','wls_epoch_gate']].reset_index()
interior['method']=interior.method.replace({'wls':'WLS','fde':'FDE','robust':'Robust weighting','wls_epoch_gate':'WLS + strict epoch rejection'})
sensitivity=s[(s.model=='persistent')&(s.scenario=='combined_degradation')&(s.method=='wls')]
median_bins=[v['median_abs_leave_one_out_residual_mps'] for v in valid['bin_statistics']]
report=f'''# AA 272 final data analysis — GNSS quality control for athlete velocity tracking

Analysis completed: November 2026. This is a completed, reproducible experimental analysis of the datasets in hand, not a claim of real NFL GNSS measurement or independently validated Pixel Watch Doppler-velocity accuracy.

## 1. Research question and result

We compare conventional Doppler weighted least squares (WLS), residual-based fault detection and exclusion (FDE), Huber robust weighting, and a deliberately strict whole-epoch rejection policy. The primary question is whether these methods improve speed estimation and high-speed event detection without losing too much usable tracking data.

The experimental conclusion is conditional rather than a universal algorithm ranking: the expanded held-out NFL simulations show no consistent high-speed event-detection improvement from either FDE or robust weighting over WLS. The stricter rejection policy reduces RMSE among retained velocity estimates and false alerts, but dramatically reduces accepted epochs and detected events. At 1 Hz, some short high-speed intervals are missed even with perfect velocity estimates.

## 2. Data and verification status

| Source | What was measured | Role | Independent validation status |
|---|---|---|---|
| Android Pixel Watch / Zenodo S1-A7 | Per-satellite GPS pseudorange rate, C/N0, receiver-reported uncertainty and time | Doppler WLS and empirical residual proxies | No continuous independent watch-antenna velocity truth |
| SnapperGPS / GitHub | Position fixes and RTK position references, 15 walking and 12 cycling recordings | Independent horizontal position check and movement-regime examples | RTK-backed position comparison, not Doppler WLS validation |
| NFL Big Data Bowl 2026 Analytics, 2023 Week 1 | Real player positions, speeds, directions, frame IDs (10 Hz pre-pass tracking) | Assumed reference motion for synthetic satellite Doppler | NFL motion is real tracking data; no NFL GNSS measurements |

Sources: https://zenodo.org/records/12566912 ; https://github.com/JonasBchrt/snapshot-gnss-data-2 ; https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics .

### Pixel Watch GPS data

The watch analysis covers {valid['watch_epoch_count']} GPS measurement epochs; {valid['epochs_at_least_seven_satellites']} have at least seven satellites. Leave-one-out Doppler prediction residual proxies are larger at lower C/N0: the median absolute residual is {median_bins[0]:.3f} m/s for C/N0 below 26 dB-Hz and {median_bins[-1]:.3f} m/s above 34 dB-Hz. The Spearman correlation between C/N0 and absolute leave-one-out residual is {valid['cn0_vs_absolute_LOO_spearman_rho']:.3f}. These residuals can contain modeling error, antenna motion, and shared biases; they must not be interpreted as known hardware-noise truth.

The NovAtel processing log offers sparse independent position consistency information (61 watch-reference proximity checks within two seconds, previously reported median horizontal separation about 2.08 m), but it is not a continuous synchronized velocity reference and was worn at a different anatomical location. No independent real-watch velocity RMSE is reported here.

### SnapperGPS RTK-backed position validation

For 27 tracks, 53,585 GNSS fixes have time-aligned RTK positions. The median horizontal position difference is 14.76 m and the 95th percentile is 48.28 m. A small number of extreme failures makes raw position RMSE much larger (442.55 m); outliers were not silently removed. Track-median RTK-derived movement speeds are roughly 1.31 m/s for walking and 5.49 m/s for cycling. This is not a test of Doppler WLS, FDE, or robust weighting. See `SNAPPER_VALIDATION_RESULTS.md`.

## 3. NFL sampling-only analysis, before any GNSS errors

Across all 16 Week 1 games, the 10-Hz NFL pre-pass file contains 847 observed high-speed intervals at the illustrative 7.0 m/s threshold. Exact enumeration over possible uniform sample phases gives a 64.51% probability that a 1-Hz schedule intersects an interval. Corresponding 2-Hz, 5-Hz and 10-Hz capture fractions are 83.73%, 95.51% and 100%. These are observational-window and sample-phase results, not GNSS hardware sensitivity. They assume direct threshold-crossing measurements and no interpolation/prediction.

The game-level bootstrap 95% descriptive range for the 1-Hz capture fraction is approximately 61.5–67.3%. See `results/phase4/nfl_sampling_summary.csv` and `nfl_sprint_capture_vs_sampling_rate.png`.

## 4. Expanded held-out satellite-level NFL experiment

The project preserves the original game-separated train/test partition. The fixed FDE standardized-residual cutoff (5.0) and Huber tuning constant (3.5) were chosen in the earlier training-stage simulation and are not retuned using these expanded held-out results.

- Five held-out NFL games: {', '.join(str(x) for x in meta['heldout_games'])}.
- {meta['included_tracks']} eligible pre-pass player trajectories, including all {meta['included_high_speed_tracks']} high-speed trajectories and a fixed, stratified sample of {meta['included_non_high_speed_tracks']} other trajectories to assess false alerts.
- {len(e)} observed reference high-speed intervals at 7.0 m/s. Of those, {int(e.touches_track_edge.sum())} touch a tracking-window boundary; only {int((~e.touches_track_edge).sum())} are fully observed intervals. Track fragments are not necessarily complete sprint episodes.
- Six fixed Monte Carlo seeds, two hypothetical temporal error models (independent and persistent), and three scenarios (observed-like noise proxies, deliberate outliers, or combined elevated noise/outliers/satellite loss).
- Scheduled receiver rate: 1 Hz. Each algorithm receives exactly the same sampling times, satellite geometry and injected observations within every simulated realization. Clock-phase/geometry choices also match across error scenarios and models for fair sensitivity comparison.

All satellite Doppler observations in this NFL experiment are simulated. The geometry comes from measured Pixel Watch satellite templates in France, not historical stadium constellations. The error model uses real wearable prediction-residual distributions as proxies and an assumed temporal correlation of approximately 0.5 in the persistent case. The difficult combined scenario uses twice the residual-proxy noise scale, 5% deliberate satellite fault probability and 30% modeled blockage, none of which is claimed to describe a specific NFL stadium.

The solution policy requires at least six usable satellites, formal velocity uncertainty no greater than 4 m/s and horizontal speed no greater than 15 m/s. FDE rejects residual-inconsistent satellites; robust weighting downweights them. The strict gate additionally requires baseline formal uncertainty at most 2 m/s and a 99% chi-squared consistency test. The gate is deliberately conservative, not a statement that all epoch rejection is harmful.

### Primary comparison: persistent errors plus combined degradation

{tbl(core,['method','event_recall_pct','sampling_ceiling_pct','mean_epoch_availability_pct','mean_speed_rmse_on_accepted_mps','mean_false_alerts_per_100_negative_samples'],{'method':'Method','event_recall_pct':'Events detected (%)','sampling_ceiling_pct':'Ideal sampling ceiling (%)','mean_epoch_availability_pct':'Accepted epochs (%)','mean_speed_rmse_on_accepted_mps':'Speed RMSE on accepted epochs (m/s)','mean_false_alerts_per_100_negative_samples':'False alerts / 100 negative samples'})}

Here the same ideal 1-Hz sample-intersection ceiling is {core.sampling_ceiling_pct.iloc[0]:.2f}% for every method. The remaining loss comes from rejected estimates or accepted estimates below the 7.0 m/s detection threshold. The lower RMSE from the strict epoch gate is conditional on the much smaller retained set and cannot be interpreted in isolation as superior continuous tracking.

![Final event detection comparison](results/final/final_main_event_detection.png)

![Accuracy versus tracking availability](results/final/final_accuracy_availability.png)

### Event-detection rates across all scenarios

{tbl(all_scenarios,['model','scenario','WLS','FDE','Robust','WLS + epoch gate'],{'model':'Error memory','scenario':'Simulated condition','WLS':'WLS (%)','FDE':'FDE (%)','Robust':'Robust (%)','WLS + epoch gate':'Epoch gate (%)'})}

All entries are percentages of observed 10-Hz NFL reference intervals detected by an accepted simulated 1-Hz velocity estimate. Multiple seeds reuse the same underlying NFL tracks; they are not independent NFL events.

### Paired uncertainty, not algorithm ranking

The following exploratory 95% intervals were constructed by resampling both held-out games and simulation seeds, keeping the WLS/alternative comparison paired:

{tbl(ci,['method','observed_delta_recall_pp','hierarchical_bootstrap_95ci_lower_pp','hierarchical_bootstrap_95ci_upper_pp'],{'method':'Alternative','observed_delta_recall_pp':'Detection change vs WLS (percentage points)','hierarchical_bootstrap_95ci_lower_pp':'95% lower','hierarchical_bootstrap_95ci_upper_pp':'95% upper'})}

The intervals for FDE and robust weighting include zero in this severe scenario and also across the other five simulated conditions. They do not establish one estimator as universally better. Only five independent held-out game clusters are represented, and the bootstrap does not capture uncertainty about whether the synthetic errors match real stadium RF conditions.

## 5. Reference-window and event-definition sensitivity

{int(e.touches_track_edge.sum())} of {len(e)} observed high-speed intervals touch the tracking-window edge. They may be truncated segments rather than complete efforts. Restricting the main difficult scenario to the {int((~e.touches_track_edge).sum())} fully observed intervals produces:

{tbl(interior,['method','event_recall_pct','sampling_ceiling_pct'],{'method':'Method','event_recall_pct':'Fully observed intervals detected (%)','sampling_ceiling_pct':'Sampling ceiling (%)'})}

The method-ranking conclusion remains qualitatively unchanged, but this sensitivity sample is smaller and should not be treated as more precise.

![Tracking-boundary sensitivity](results/final/final_boundary_event_sensitivity.png)

For the persistent combined-degradation scenario, changing the event threshold or minimum observed duration gives the following WLS detection fractions (these define different reference populations, not causal gains from changing thresholds):

{tbl(sensitivity,['speed_threshold_mps','minimum_duration_s','reference_unique_intervals','sampling_ceiling_pct','event_recall_pct'],{'speed_threshold_mps':'Threshold (m/s)','minimum_duration_s':'Min duration (s)','reference_unique_intervals':'Observed reference intervals','sampling_ceiling_pct':'Ideal 1 Hz intersection (%)','event_recall_pct':'WLS detection (%)'})}

## 6. Interpretation and limitations

1. The central accuracy-versus-availability tradeoff is empirically demonstrated within a controlled NFL satellite-observation simulation. It is not a field measurement of athlete-mounted GNSS.
2. Under the tested scenarios, neither FDE nor robust weighting shows a stable high-speed event-detection improvement over conventional WLS. A more aggressive rejection rule can report substantially lower accepted-only RMSE while detecting far fewer intervals.
3. At a 1-Hz receiver rate, many brief 10-Hz NFL high-speed intervals are never sampled. Estimator improvements alone cannot recover an event for which no observation exists, unless prediction, interpolation or other sensors are introduced.
4. The NFL reference speed is measured by NFL tracking, not certified as exact physical truth. The NFL file is pre-pass-only; many intervals end at a recording boundary.
5. The simulated LOS geometry and residual proxies come from a French walking-device recording, not body-worn NFL GNSS. The independence/persistence and fault/blockage assumptions are exploratory stress tests.
6. Pixel Watch real-data velocity RMSE is not established until a continuous independent velocity truth is available at a sufficiently corresponding physical sensor location. The Snapper RTK validation concerns different GNSS fixes and positions, not Doppler-velocity truth for the watch.
7. Repeating six seeds on five games changes Monte Carlo error, not the number of independent NFL games. The paired bootstrap is descriptive and insufficient to establish a real-world ranking.

## 7. Bottom-line conclusion suitable for the final course report

Using real wearable GNSS Doppler observations as an error-characterization basis and real NFL tracking as reference motion, we completed a controlled comparison of WLS, FDE, robust weighting and whole-epoch filtering. Across expanded, game-held-out simulations, no method consistently dominated WLS in high-speed event detection. A deliberately strict rejection rule lowered error among retained estimates but substantially reduced monitoring coverage. In addition, sampling analysis of 847 reference intervals showed that a 1-Hz receiver could directly sample only about 64.5% of observed 7-m/s intervals in the full Week 1 file, even without GNSS errors. These results support designing athlete-tracking diagnostics around both accuracy and temporal availability, while remaining limited by simulated stadium signals and the absence of independently validated watch-antenna velocity truth.

## 8. Reproducibility

From this folder with Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python self_check.py
python test_phase2.py
python test_phase3.py
python test_phase4.py
python test_final_analysis.py
```

The final source scripts are `run_final_analysis.py`, `phase6_event_metrics.py`, `refresh_final_sensitivity.py`, `final_uncertainty_and_figures.py` and `write_final_report.py`. `run_final_analysis.py --seeds 6` regenerates the expanded Monte Carlo outputs from the original files. `final_uncertainty_and_figures.py` regenerates bootstrap comparisons and plots; `refresh_final_sensitivity.py` regenerates event-threshold sensitivity; `write_final_report.py` rewrites this report. Phase 1–5 scripts, data and plots are retained. No web access is required to run the included project.
'''

phase7=json.loads((root/'results/phase7/rg_reference_audit.json').read_text())
phase8=json.loads((root/'results/phase8/phase8_robustness_audit.json').read_text())
real_summary=pd.read_csv(root/'results/phase8/real_device_speed_summary.csv')
real_ci=pd.read_csv(root/'results/phase8/paired_block_bootstrap.csv')
real_ci=real_ci[real_ci.block_epochs == 20].copy()
real_ci['alternative']=real_ci.alternative.replace({'fde_horizontal_speed_mps':'FDE','robust_horizontal_speed_mps':'Robust WLS'})
report=report.replace('No continuous independent watch-antenna velocity truth',
 'Continuous inferred, synchronized NovAtel head-mounted trajectory is available; no co-located watch-antenna ground truth')
report=report.replace('The NovAtel processing log offers sparse independent position consistency information (61 watch-reference proximity checks within two seconds, previously reported median horizontal separation about 2.08 m), but it is not a continuous synchronized velocity reference and was worn at a different anatomical location. No independent real-watch velocity RMSE is reported here.',
 'We recovered an exploratory high-rate NovAtel head-mounted trajectory from the supplied `.rg` file, validated its coordinate interpretation against all 271 `.rss` log positions, and paired it with watch epochs. This gives a *continuous external reference-device comparison*, but it is not a co-located, certified watch velocity truth. The median `.rss`-`.rg` horizontal coordinate difference is approximately 0.0001 m, although both products originate from the same postprocessing pipeline; one matched coordinate deviates by 1.455 m. The `.rg` velocity layout and quality flags are not officially documented in the available material. Do not label any head-versus-wrist result as watch velocity RMSE.')
report=report.replace('Pixel Watch real-data velocity RMSE is not established until a continuous independent velocity truth is available at a sufficiently corresponding physical sensor location.',
 'Pixel Watch real-data velocity RMSE remains unvalidated because the recovered continuous NovAtel reference is not co-located with the forearm watch, the binary field definitions are inferred and solution quality is unknown.')
report=report.replace('The final source scripts are `run_final_analysis.py`',
 'For the measured NovAtel reference and robustness analysis, also run `python phase7_novatel_rg.py`, `python phase8_real_reference_robustness.py`, `python test_phase7.py`, and `python test_phase8.py`. The final source scripts are `run_final_analysis.py`')
report+=f"""

## 9. New external reference-device comparison: Phases 7 and 8

The original analysis could only compare Android watch processing against its own speed fixes, or the NovAtel `.rss` log at sparse points. The supplied `.rg` was subsequently recovered as a reverse-processed 224-byte-record trajectory file and gives {phase7['n_rg_records']:,} nominal 5-Hz records. Its latitudes, longitudes and timestamps align with {phase7['position_verification']['matched_positions']} `.rss` log coordinates (median horizontal coordinate discrepancy {phase7['position_verification']['median_rss_rg_horizontal_difference_m']:.4f} m). This is confirmation of the *file interpretation*, **not** independent physical ground truth: both are NovAtel processing outputs. An inferred velocity field has east/north correlation {phase7['inferred_velocity_verification']['east_velocity_vs_position_difference_correlation']:.3f}/{phase7['inferred_velocity_verification']['north_velocity_vs_position_difference_correlation']:.3f} against velocities independently formed by differentiating adjacent decoded positions, but proprietary field definitions and quality flags are still unverified. The binary format and its reverse/combined distinction are described in the Waypoint documentation: https://docs.novatel.com/Waypoint/Content/File_Formats/FG_RG_CG_FP_RP_and_CP_files.htm .

Of {phase8['matched_epochs_any_satellites']} total matched watch epochs, {phase8['matched_epochs_7plus_satellites']} have >=7 GPS satellites and {phase8['matched_common_epochs_with_android']} also have Android-reported speeds. Match tolerance is 110 ms. The NovAtel antenna was on the **head** and the Pixel Watch was on the **swinging left forearm**. Thus these are real GNSS **between-device horizontal-speed disagreements**, not certified watch velocity errors.

{tbl(real_summary, ['method','n','rmse_mps','mae_mps','correlation'], {'method':'Measured speed estimate','n':'Matched epochs','rmse_mps':'RMS speed disagreement (m/s)','mae_mps':'Mean absolute disagreement (m/s)','correlation':'Speed correlation'},digits=3)}

![Independent device comparison and temporal-smoothing sensitivity](results/phase8/real_reference_robustness.png)

Only {phase8['fde_modified_epoch_count']} of the {phase8['matched_epochs_7plus_satellites']} common epochs had changed FDE horizontal speed relative to WLS; robust estimation changed {phase8['robust_modified_epoch_count']}. Descriptive **paired moving-block bootstrap** intervals (20 consecutive 1-Hz epochs, circular resampling within the recording, {phase8['n_bootstrap']} draws) for the *difference in RMS device disagreement* compared with WLS:

{tbl(real_ci,['alternative','rmse_delta_mps','rmse_ci_low_mps','rmse_ci_high_mps'], {'alternative':'Alternative','rmse_delta_mps':'Observed delta RMS (m/s)','rmse_ci_low_mps':'Descriptive 95% lower','rmse_ci_high_mps':'Descriptive 95% upper'},digits=4)}

These tiny changes are not evidence of a dependable superiority for FDE or robust WLS. The intervals describe uncertainty within **one** subject's recording, do not create independent test sessions, and depend on temporal block size (10- and 30-epoch sensitivities are also supplied). On a common 665-epoch subset, centered 10-second averaging reduces WLS/reference RMS scalar-speed disagreement from about 0.572 to 0.421 m/s, whereas the processed Android speed comparison moves from about 0.169 to 0.108 m/s. This smoothing is offline and reduces temporal resolution; it cannot be read as an instantaneous GNSS fix or a solution to missing subsecond athlete events.

Official combined forward/reverse `.cg` export with epoch times, ENU velocities, fixed/float quality and uncertainties remains desirable. Waypoint's Export Wizard can generate a per-epoch ASCII file: https://docs.novatel.com/Waypoint/Content/GrafNav/Use_Export_Wizard.htm . Source notes say March **2023**, but GPS week 2306 and watch/NovAtel timestamp files correspond to March **19, 2024**. The provenance discrepancy requires author confirmation.

## 10. Estimation equations and observation-model meaning

Let `u_i` be the line-of-sight unit vector from receiver to satellite, `v_s,i` the modeled satellite velocity, `v_r` the receiver velocity, `b` the receiver clock-drift contribution in m/s, and `rho_dot_i` the watch-recorded GNSS pseudorange rate. Ignoring higher-order effects (handled approximately by the orbit-model sensitivity experiments), the Doppler velocity model is:

`rho_dot_i = u_i^T (v_s,i - v_r) + b - c * clock_dot_s,i + measurement_error_i`.

Subtracting the known satellite radial velocity and adding back the satellite-clock drift gives `y_i = -u_i^T v_r + b + error_i`. The weighted least-squares state is `x = [v_x, v_y, v_z, b]^T`, with rows of the design matrix `H_i = [-u_ix, -u_iy, -u_iz, 1]`. The estimate minimizes `sum_i (y_i - H_i x)^2/sigma_i^2`, conventionally written `x_hat = (H^T W H)^(-1) H^T W y`, or computed using QR/SVD in the supplied Python implementation for numerical robustness. The fitted ECEF velocity is rotated into local east/north/up. Horizontal speed is `sqrt(v_e^2+v_n^2)`.

FDE iteratively removes a satellite when a global residual-consistency test and the largest standardized residual both suggest an outlier. Robust WLS uses iteratively reweighted Huber residuals. A strict epoch gate rejects an otherwise available solution when its formal uncertainty and residual tests fail. The accepted-only RMSE estimates a *conditional* error on the retained epochs; it is not directly comparable between methods with very different availability. For event recall, a reference interval is detected only if an accepted simulated 1-Hz speed estimate exceeds the threshold inside that interval.

## 11. Final boundary of the evidence

**Measured:** watch satellite Doppler and receiver characteristics; cross-device NovAtel/head versus watch/wrist speed disagreement; separate SnapperGPS position/RTK checks; NFL player motion and sampling intervals.

**Simulated:** NFL satellite pseudorange rates, reception loss, deliberate faults, receiver trajectories derived from NFL tracking, and inferred NFL GNSS event outcomes. No sport-field GNSS receiver was worn by NFL athletes in this study. Treat the four-method NFL results as controlled engineering comparisons with explicit assumptions, not a validation of an actual football tracking product.
"""

(root/'FINAL_ANALYSIS_REPORT.md').write_text(report)
print('Wrote',root/'FINAL_ANALYSIS_REPORT.md','bytes',len(report))
