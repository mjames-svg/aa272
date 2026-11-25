from pathlib import Path
from phase3_reference_audit import run as reference
from phase3_orbit_model_audit import run as orbit
from phase3_temporal_sensitivity import run as temporal
from phase3_repeatability import run as repeats

root=Path(__file__).resolve().parent
reference(root,root/'results/phase3')
orbit(root)
temporal(root)
repeats(root)
print('Phase 3 reproducible pipeline completed. Files saved in results/phase3/.')
