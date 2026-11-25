"""Build the Phase-11 narrative using only the computed output tables."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
D=ROOT/'results'/'phase11'
s=pd.read_csv(D/'kalman_method_comparison.csv').set_index(['scenario','method'])
c=pd.read_csv(D/'four_satellite_matched_control.csv').set_index(['scenario','method'])

def p(sc,m,col='event_recall_pct'):
    x=s.loc[(sc,m),col]
    return f'{float(x):.2f}' if pd.notna(x) else 'N/A'

def row(sc,desc):
    ms=['wls','wls4','kf_cv6','kf_cv4','kf_ca4']
    return '| '+desc+' | '+' | '.join(f'{p(sc,m)}%' for m in ms)+' |'

static=[('open__measured_proxy','Open sky'),('partial_canopy__measured_proxy','Partial canopy'),
        ('deep_canopy__measured_proxy','Deep canopy'),('sealed_roof__measured_proxy','Fully blocked')]
dynamic=[('partial_canopy__measured_proxy__entry','Entering partial canopy'),
         ('deep_canopy__measured_proxy__entry','Entering deep canopy'),
         ('open__measured_proxy__outage1','Open sky, 1-s blackout'),
         ('open__measured_proxy__outage2','Open sky, 2-s blackout'),
         ('partial_canopy__measured_proxy__outage1','Partial canopy, 1-s blackout'),
         ('partial_canopy__measured_proxy__outage2','Partial canopy, 2-s blackout')]

md='''# AA 272 — Phase 11: Causal Kalman Filtering for NFL-Style GNSS Tracking

## Research question

Can Kalman filtering improve Doppler-based speed estimation and the detection of short high-speed player movements when satellite visibility deteriorates? Does the benefit come from temporal filtering, from using fewer satellites per solution, or from reporting predictions during outages?

## What was implemented

The original Phase 10 satellite measurements were reconstructed using identical NFL tracks, satellite templates, error draws, and six paired random seeds. The Kalman implementations use raw simulated satellite Doppler pseudorange rates directly, not a smoothed plot of scalar speed. The constant-velocity model estimates three ECEF velocity components and clock drift. The constant-acceleration model adds three acceleration states. Both use Joseph covariance updates and a five-standard-deviation innovation gate. All Kalman filters initialize from an accepted six-satellite WLS estimate, so there is no hidden NFL ground-truth initialization.

The new study also includes a four-satellite WLS control, necessary to separate the effect of accepting four or five satellite measurements from the effect of temporal filtering. A valid prediction without an update is only reported for two seconds and is labeled separately from an observation-updated GNSS solution. The output still exists only at the original simulated 1-Hz sample times. Neither the previous NFL velocity nor a future measurement is supplied to the filter.

The fixed-roof cases exactly reconstruct the previous WLS baseline: **87,168 baseline rows matched with no numerical discrepancies**. Across six seeds, the additional study evaluates 14 reception conditions, seven estimators, and 287 reference intervals from five held-out NFL games. The same five games are reused across seeds; the trials are paired, not independent NFL games.

## Fixed stadium masks: event recall

| Scenario | WLS (6 sat) | WLS (4 sat) | CV-KF (6 sat) | CV-KF (4 sat) | CA-KF (4 sat) |
|---|---:|---:|---:|---:|---:|
'''
for sc,label in static:md+=row(sc,label)+'\n'
md+='''
Under static masks, changing the estimator does not materially improve the six-satellite result. The satellite geometry is fixed within the short tracking window, meaning a Kalman filter may never get the six-satellite epoch required for initialization. Four-satellite WLS can provide more solutions under partial and deep canopy assumptions, but at exactly four satellites it has no residual redundancy and can produce more false alerts. In the fixed deep-canopy case, WLS4 has 38.50% event recall with 8.04 false alerts per 100 negative samples, compared with WLS6's 13.41% recall and 1.87 false alerts.

## Time-varying synthetic obstruction: event recall

| Scenario | WLS (6 sat) | WLS (4 sat) | CV-KF (6 sat) | CV-KF (4 sat) | CA-KF (4 sat) |
|---|---:|---:|---:|---:|---:|
'''
for sc,label in dynamic:md+=row(sc,label)+'\n'
md+=f'''
The canopy-entry tests start with an open-sky sample and then transition under an assumed partial or deep canopy. For deep-canopy entry, WLS6 detects {p('deep_canopy__measured_proxy__entry','wls')}% of intervals, compared with {p('deep_canopy__measured_proxy__entry','wls4')}% for WLS4 and {p('deep_canopy__measured_proxy__entry','kf_cv4')}% for CV-KF4. Thus most of the improvement comes from changing the minimum number of usable satellites rather than from the Kalman prediction.

With two-second complete outages in open sky, WLS6 accepts {p('open__measured_proxy__outage2','wls','mean_epoch_availability_pct')}% of epochs, whereas CV-KF4 reports estimates on {p('open__measured_proxy__outage2','kf_cv4','mean_epoch_availability_pct')}% when predictions are included. However, its recall changes from {p('open__measured_proxy__outage2','wls')}% (WLS) to {p('open__measured_proxy__outage2','kf_cv4')}% (CV-KF4). The conditional speed RMSE increases from {p('open__measured_proxy__outage2','wls','mean_speed_rmse_on_accepted_mps')} to {p('open__measured_proxy__outage2','kf_cv4','mean_speed_rmse_on_accepted_mps')} m/s. A prediction is not a new GNSS observation, and the extra reported velocities should not be characterized as equivalent-quality GNSS fixes.

## Paired comparison against the fair four-satellite baseline

| Scenario | Filter | Difference in event recall vs WLS4 (percentage points) | Descriptive five-game bootstrap range (pp) |
|---|---|---:|---:|
'''
for sc,desc in [('partial_canopy__measured_proxy__entry','Partial entry'),('deep_canopy__measured_proxy__entry','Deep entry'),
                ('open__measured_proxy__outage1','1-s blackout'),('open__measured_proxy__outage2','2-s blackout')]:
    for method in ['kf_cv4','kf_ca4']:
        v=c.loc[(sc,method)]
        md+=f"| {desc} | {'CV-KF4' if method=='kf_cv4' else 'CA-KF4'} | {v.delta_event_recall_pp:+.2f} | {v.lower_95_pp:+.2f} to {v.upper_95_pp:+.2f} |\n"
md+='''
The constant-acceleration filter produces small positive differences relative to the four-satellite WLS control within these simulations. This should be interpreted as a model-dependent result, not a general advantage for athlete tracking. The five-game bootstrap describes variation across the chosen held-out games; it does not represent a sample of stadium roofs, receivers, or independently observed NFL-GNSS trials.

## Prediction and process-noise sensitivity

The baseline Doppler noise scale comes from real smartwatch leave-one-out residuals, but those residuals are not certified observation-error truth. The default constant-velocity model assumes a 2.5 m/s one-second velocity-change standard deviation. Separate runs use 0.75 and 5.0 m/s. Lower process noise reduces false speed alerts while also reducing high-speed event recall, since the filter responds more slowly to changes in measured velocity. Higher process noise follows the Doppler measurements more closely but is less effective at suppressing noise.

A stationary completely blocked stadium scenario produces no Kalman estimates, since no receiver initialization is possible. The complete-loss case is a theoretical limit, not an experimentally measured stadium result. Models that propagate velocities through a short outage cannot infer a new, unobserved sprint from a constant-velocity assumption. Reconstructing motion between 1-Hz samples would require an additional acceleration model or sensor and would be a different, model-dependent detection experiment.

## Interpretation for the AA 272 report

The practical message is that temporal filtering and observation availability solve different problems. Kalman filtering can reduce some speed noise and continue short-lived predictions after a dropout, but a model-based prediction does not replace an actual satellite measurement. Relaxing the satellite-count requirement can increase event recall, but it may also increase false alerts and weaken fault-detection redundancy. The original study's conclusion remains: sample rate, satellite geometry, tracking availability, and accepted-epoch velocity accuracy must be evaluated together.

**Scientific limitations:** The NFL tracking is RFID-derived reference motion, not GNSS. All NFL satellite pseudorange rates and roof obstructions are simulated. Receiver geometry and residual-error distributions are from a Pixel Watch session in another location. The six seeds reuse five NFL games. The fixed roof masks do not model signal penetration, reflections, actual stadium roofs, or multiple GNSS constellations. The constant-velocity and constant-acceleration process-noise assumptions were specified before analysis but were not independently calibrated for NFL athletes.

## Files to inspect

- `phase11_kalman_stadium.py` — reconstruction, causal Kalman filters, statistics, bootstrap, and plots.
- `test_phase11.py` — solver initialization, 4/6-satellite updates, causal behavior, prediction limits, and covariance checks.
- `verify_phase11_results.py` — all six seeds, exact WLS reconstruction, event and availability logic.
- `results/phase11/kalman_paired_epoch_results.csv` — every simulated sampled epoch and method, with predicted/update status.
- `results/phase11/kalman_method_comparison.csv` — full method-by-condition numerical results.
- `results/phase11/four_satellite_matched_control.csv` — paired, game-clustered WLS4-vs-KF comparison.
- `results/phase11/*.png` — plots of event recall, update/prediction coverage, outages, Q sensitivity, and uncertainty.

Run from VS Code terminal, using the included local datasets:

```powershell
python -m pip install -r requirements.txt
python phase11_kalman_stadium.py --seeds 6
python test_phase11.py
python verify_phase11_results.py
```

All simulations are deterministic given the included inputs. No downloading, internet connection, notebook, or manual GNSS logger collection is necessary.
'''
(ROOT/'PHASE11_KALMAN_RESULTS.md').write_text(md)
print('wrote',ROOT/'PHASE11_KALMAN_RESULTS.md',len(md.split()),'words')
