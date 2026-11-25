"""Layout-only corrections to published AA 272 figures.

Uses the saved CSVs originally used for the paper; statistical results are unchanged.
The original sky-mask image is retained as a source to preserve the satellite positions.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'figures'
DATA = ROOT / 'figure_source_data'
C = ['#206679', '#AB763C', '#A24F54', '#60676B']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})

# Figure 3: offset the first label to keep it clear of the rising line.
s = pd.read_csv(DATA / 'nfl_sampling_summary.csv')
s = s[(s.threshold_mps == 7) & (s.minimum_event_duration_s == 0.1)].sort_values('rate_hz')
fig, ax = plt.subplots(figsize=(7.0, 2.7))
ax.plot(s.rate_hz, s.mean_event_capture_probability * 100,
        marker='o', markersize=7.5, linewidth=2.2, color=C[0], zorder=3)
ax.set(xlabel='Hypothetical sampling frequency (Hz)',
       ylabel='Ideal interval capture (%)', ylim=(55, 109), xlim=(0.6, 10.55))
ax.set_xticks([1, 2, 5, 10])
ax.grid(alpha=.27, zorder=0)
for i, (_, r) in enumerate(s.iterrows()):
    x, y = float(r.rate_hz), float(r.mean_event_capture_probability) * 100
    offset = (-3, 16) if x == 1 else ((-12, 12) if x == 10 else (0, 9))
    ax.annotate(f'{y:.1f}%', xy=(x, y), xytext=offset,
                textcoords='offset points', ha='center', va='bottom',
                fontsize=9.5, zorder=5)
fig.tight_layout(pad=.8)
fig.savefig(FIG / 'fig3_sampling.pdf', bbox_inches='tight', pad_inches=.11)
fig.savefig(FIG / 'fig3_sampling.png', dpi=300, bbox_inches='tight', pad_inches=.11)
plt.close(fig)

# Figure B.3: remove the callout arrow (the equal bar heights make the point).
p = pd.read_csv(DATA / 'stadium_method_comparison.csv')
skies = ['open', 'partial_canopy', 'deep_canopy', 'sealed_roof']
methods = ['wls', 'fde', 'robust']
labels = ['WLS', 'FDE', 'Robust WLS']
fig, ax = plt.subplots(figsize=(7.25, 2.95))
x = np.arange(4)
width = .24
for i, (m, label) in enumerate(zip(methods, labels)):
    values = [float(p.loc[(p.method == m) &
                          (p.scenario == sky + '__measured_proxy'),
                          'event_recall_pct'].iloc[0]) for sky in skies]
    ax.bar(x + (i - 1) * width, values, width, label=label,
           color=C[i], edgecolor='white', linewidth=.45, zorder=3)
ax.set_xticks(x, ['Open Sky', 'Partial Canopy', 'Deep Canopy',
                  'Signal-Blocking\nRoof'])
ax.set_ylabel('High-speed interval recall (%)')
ax.set_ylim(0, 60)
ax.grid(axis='y', alpha=.24, zorder=0)
ax.legend(ncol=3, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(FIG / 'fig6_roof_recall.pdf', bbox_inches='tight', pad_inches=.12)
fig.savefig(FIG / 'fig6_roof_recall.png', dpi=300, bbox_inches='tight', pad_inches=.12)
plt.close(fig)

# Figure A.2: preserve every existing dot and sector, remove the crowded
# original bottom legend, and set distinct bands for azimuth ticks and legend.
src = Image.open(FIG / 'fig5_roof_masks_original.png').convert('RGB')
w, h = src.size
out = Image.new('RGB', (w, h + 145), 'white')
# The original azimuth labels finish before the in-figure legend. Keep all
# sky plots and azimuth labels intact, but omit the old legend.
out.paste(src.crop((0, 0, w, 769)), (0, 0))
d = ImageDraw.Draw(out)
font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font_ticks = ImageFont.truetype(font_file, 24)
font_legend = ImageFont.truetype(font_file, 23)
for cx in [343, 1028, 1713]:
    d.text((cx, 785), '180°', fill=(40, 40, 40), anchor='mt', font=font_ticks)
d.text((w // 2, 892),
       'Teal: direct LOS permitted   |   Gray: blocked   |   Dots: real Pixel Watch GPS satellite directions',
       fill=(30, 30, 30), anchor='mt', font=font_legend)
out.save(FIG / 'fig5_roof_masks.png', optimize=True)
print('Corrected Figure 3, Figure A.2, Figure B.3 without changing numeric results or sky-mask dots.')
