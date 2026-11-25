"""Regenerate Appendix A.1 as a clean top-to-bottom scientific workflow."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = Path(__file__).resolve().parents[1] / 'figures' / 'fig12_analysis_workflow.pdf'
fig, ax = plt.subplots(figsize=(10.2, 5.1))
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 5.1)
ax.axis('off')

EDGE = '#376477'
LINE = '#526B79'

def block(x, y, w, h, txt, fill='#E9F2F2', fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle='round,pad=0.045,rounding_size=0.095',
                               facecolor=fill, edgecolor=EDGE, linewidth=1.15))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center',
            fontsize=fs, color='#1B2A35')

def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>',
                                 color=LINE, linewidth=1.2,
                                 mutation_scale=10, shrinkA=0, shrinkB=0))

def segment(a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=LINE,
            linewidth=1.2, solid_capstyle='round')

ax.text(5.1, 4.99,
        'Real measurements characterize inputs; NFL satellite Doppler remains simulated',
        ha='center', va='top', fontsize=10, fontweight='bold', color='#1B2A35')

block(0.30, 4.13, 2.70, 0.58,
      'NFL 10-Hz player trajectories\n(reference motion)', fs=8.8)
block(3.74, 4.13, 2.70, 0.58,
      'Pixel Watch raw GPS Doppler\n(geometry, C/N₀, error proxy)', fs=8.8)
block(7.17, 4.13, 2.70, 0.58,
      'GEDVM phone speed + GNSS/INS\nSnapperGPS field position + RTK', fill='#F1F3F4', fs=8.5)

block(1.60, 2.99, 5.10, 0.60,
      'Construct synthetic NFL Doppler measurements\ncontrolled noise, satellite masks, intermittent outages',
      fill='#DFEBEF', fs=9)
block(7.35, 2.99, 2.25, 0.60,
      'Independent real-data checks\nspeed and position errors', fill='#F1F3F4', fs=8.6)

# Direct vertical arrows with no floating diagonal connections.
arrow((2.36, 4.085), (2.36, 3.645))
arrow((5.05, 4.085), (5.05, 3.645))
arrow((8.50, 4.085), (8.50, 3.645))

block(0.45, 1.70, 3.05, 0.62,
      'Six-satellite WLS, FDE,\nrobust WLS baseline', fs=8.8)
block(3.68, 1.70, 2.62, 0.62,
      'Four-satellite WLS\nmeasurement-count control',
      fill='#FFF2E4', fs=8.5)
block(6.50, 1.70, 3.25, 0.62,
      'Causal Kalman: CV6, CV4, CA4\nprocess-noise sensitivity',
      fill='#F0E8EC', fs=8.7)

# A straight, orthogonal branching connector replaces three crossed diagonals.
branch_y = 2.69
segment((4.15, 2.945), (4.15, branch_y))
segment((1.975, branch_y), (8.125, branch_y))
for x in (1.975, 4.99, 8.125):
    arrow((x, branch_y), (x, 2.365))

block(1.15, 0.38, 7.92, 0.68,
      'Compare speed RMSE, accepted GNSS updates, prediction-only coverage,\ninterval recall, false alerts, and paired uncertainty',
      fill='#E7F0E7', fs=8.9)
for x in (1.975, 4.99, 8.125):
    arrow((x, 1.65), (x, 1.105))

fig.tight_layout(pad=.35)
fig.savefig(out, bbox_inches='tight', pad_inches=.07)
fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=.07)
plt.close(fig)
print(out)
