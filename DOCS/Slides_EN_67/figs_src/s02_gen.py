import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16

CUDA = '#76B900'
SADGS = '#C83C32'
GAUSS = '#286EBE'
SEC = '#555555'

def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#DDDDDD', linewidth=0.8)
    ax.set_axisbelow(True)

# ---------- s02_2: 32x32 image split into four 16x16 tiles ----------
W = H = 32
B = 16
fig, ax = plt.subplots(figsize=(12, 6.2))
tile_colors = ['#E6F2C2', '#D4EBA0', '#C2E37E', '#AEDB5C']
for ty in range(2):
    for tx in range(2):
        tid = ty * 2 + tx
        ax.add_patch(Rectangle((tx * B, ty * B), B, B, facecolor=tile_colors[tid],
                               edgecolor='none', zorder=1))
        ax.text(tx * B + B / 2, ty * B + B / 2, f'Tile {tid}\nblock ({tx},{ty})\n256 threads',
                ha='center', va='center', fontsize=16, color='black', zorder=3)
# pixel grid (light)
for k in range(W + 1):
    lw = 1.2 if k % B == 0 else 0.3
    col = '#222222' if k % B == 0 else '#BBBBBB'
    ax.plot([k, k], [0, H], color=col, lw=lw, zorder=2)
    ax.plot([0, W], [k, k], color=col, lw=lw, zorder=2)
# highlight one pixel of tile 3
ax.add_patch(Rectangle((B + 15, B + 15), 1, 1, facecolor=SADGS, edgecolor='black', lw=1.2, zorder=4))
ax.annotate('pixel (31,31)\n= thread (15,15)\nof tile 3', xy=(31.5, 31.5), xytext=(34.5, 26),
            fontsize=15, color=SADGS, arrowprops=dict(arrowstyle='->', color=SADGS, lw=1.5),
            va='center', annotation_clip=False)
ax.set_xlim(-0.5, 40)
ax.set_ylim(H + 1.2, -1.2)
ax.set_aspect('equal')
ax.text(W / 2, -2.0, '32 px width  (grid$_x$ = ceil(32/16) = 2)', ha='center', fontsize=15, color=SEC)
ax.text(-2.5, H / 2, '32 px height\n(grid$_y$ = 2)', ha='center', va='center', rotation=90, fontsize=15, color=SEC)
ax.axis('off')
fig.savefig('figs/s02_2.pdf', bbox_inches='tight')
plt.close(fig)

# ---------- s02_5: cumulative SH coefficients per degree ----------
degs = [0, 1, 2, 3]
per = [1, 3, 5, 7]
cum = [1, 4, 9, 16]
fig, ax = plt.subplots(figsize=(12, 6.2))
style(ax)
bars = ax.bar([str(d) for d in degs], cum, color=GAUSS, width=0.6, zorder=3)
for i, (b, p, c) in enumerate(zip(bars, per, cum)):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.35, f'{c} total\n(+{p} new)',
            ha='center', va='bottom', fontsize=16, color='black')
ax.plot([str(d) for d in degs], cum, color=SADGS, marker='o', lw=2.2, zorder=4)
ax.set_xlabel('SH degree $l$', fontsize=18)
ax.set_ylabel('Cumulative coefficients per channel', fontsize=18)
ax.set_ylim(0, 19.5)
ax.set_yticks([0, 4, 9, 16])
ax.spines['left'].set_color('#444444')
ax.text(3.45, 1.2, '3 channels (RGB) x 16 = 48 floats per Gaussian',
        ha='right', fontsize=15, color=SEC)
fig.savefig('figs/s02_5.pdf', bbox_inches='tight')
plt.close(fig)
print('ok')
