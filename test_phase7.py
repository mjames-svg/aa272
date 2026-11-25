from pathlib import Path

from phase7_novatel_rg import read_rg, verify_positions, verify_velocities, compare_watch

root = Path(__file__).resolve().parent
rg = read_rg(root/'data/PPK_expe_AG.rg')
assert len(rg) == 14580
assert rg.utc.is_monotonic_increasing
assert rg.utc.iloc[0].strftime('%Y-%m-%d') == '2024-03-19'
matched_positions, position_audit = verify_positions(rg, root/'data/PPK_expe_AG.rss')
assert len(matched_positions) == 271
assert position_audit['median_rss_rg_horizontal_difference_m'] < .001
velocity_audit = verify_velocities(rg)
assert velocity_audit['east_velocity_vs_position_difference_correlation'] > .98
assert velocity_audit['north_velocity_vs_position_difference_correlation'] > .98
matched_watch, metrics = compare_watch(rg, root/'results/real_velocity_epochs.csv')
assert len(matched_watch) == 694
assert (matched_watch.gps_sats >= 7).sum() == 674
assert len(metrics) == 8
print('Phase 7 binary layout, RSS alignment, inferred velocity, and watch matching tests passed.')
