import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['axes.axisbelow'] = True

CUDA = '#76B900'
SADGS = '#C83C32'
GAUSS = '#286EBE'
PY = '#F0AA1E'
SEC = '#555555'

# Example from the source (illustrative): Sigma, centre, alpha
S = np.array([[400.0, 180.0], [180.0, 160.0]])
P = np.array([100.0, 84.0])
ALPHA = 0.8
T = 2.0 * np.log(255.0 * ALPHA)          # ~10.64
R3SIG = 67                               # ceil(3*sqrt(lambda_max))
TILE = 16                                # illustrative tile size
GX, GY = 12, 11                          # 12 x 11 tile grid (192 x 176 px)
SI = np.linalg.inv(S)


def draw_grid(ax):
    ax.set_xlim(0, GX * TILE)
    ax.set_ylim(GY * TILE, 0)            # image convention: y down
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, GX * TILE + 1, TILE))
    ax.set_yticks(np.arange(0, GY * TILE + 1, TILE))
    ax.tick_params(labelsize=11, colors=SEC)
    ax.grid(True)


def fill_tiles(ax, tiles, color, alpha=0.45):
    for (i, j) in tiles:
        ax.add_patch(Rectangle((i * TILE, j * TILE), TILE, TILE,
                               facecolor=color, edgecolor='none', alpha=alpha))


def tile_rect(x0f, x1f, y0f, y1f):
    """getRect-style: floor min, ceil max, clamp to grid."""
    x0 = max(0, int(np.floor(x0f / TILE)))
    x1 = min(GX, int(np.ceil(x1f / TILE)))
    y0 = max(0, int(np.floor(y0f / TILE)))
    y1 = min(GY, int(np.ceil(y1f / TILE)))
    return [(i, j) for j in range(y0, y1) for i in range(x0, x1)]


def accutile_tiles(t):
    """Per-tile sampling: a tile is kept if any sample lies inside the level set."""
    out = []
    frac = np.linspace(0, 1, 64)
    for j in range(GY):
        for i in range(GX):
            X, Y = np.meshgrid(i * TILE + frac * TILE, j * TILE + frac * TILE)
            d = np.stack([X - P[0], Y - P[1]], -1)
            m = np.einsum('...i,ij,...j->...', d, SI, d)
            if (m <= t).any():
                out.append((i, j))
    return out


def ellipse_params(t):
    w, V = np.linalg.eigh(S)
    width = 2 * np.sqrt(t * w[1])
    height = 2 * np.sqrt(t * w[0])
    angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return width, height, angle


# ---------- Figure 2: getRect, 3-sigma box on the tile grid ----------
fig, ax = plt.subplots(figsize=(12, 6.2))
draw_grid(ax)
rect_tiles = tile_rect(P[0] - R3SIG, P[0] + R3SIG, P[1] - R3SIG, P[1] + R3SIG)
fill_tiles(ax, rect_tiles, GAUSS, alpha=0.40)
ax.add_patch(Rectangle((P[0] - R3SIG, P[1] - R3SIG), 2 * R3SIG, 2 * R3SIG,
                       fill=False, edgecolor='black', lw=2, ls='--'))
ax.plot(*P, 'o', color='black', ms=8)
ax.annotate('centre p = (100, 84)', xy=P, xytext=(120, 40), fontsize=14,
            arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('exact box [33, 167] x [17, 151]', xy=(167, 151), xytext=(108, 150),
            fontsize=13, color='black')
ax.set_xlabel('x (pixels)', fontsize=15, color=SEC)
ax.set_ylabel('y (pixels)', fontsize=15, color=SEC)
ax.text(96, 170, '9 x 9 = 81 tiles (blue)', ha='center', fontsize=15,
        color=GAUSS, weight='bold', bbox=dict(fc='white', ec='none', alpha=0.85))
fig.savefig('figs/s06_2.pdf', bbox_inches='tight')
plt.close(fig)

# ---------- Figure 3: in_frustum near-plane cull (x-z cross-section) ----------
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(-3, 3)
ax.set_ylim(-0.4, 4.0)
ax.set_aspect('equal')
ax.axhspan(-0.4, 0.2, color='#BBBBBB', alpha=0.45, lw=0)
ax.axhspan(0.2, 4.0, color=CUDA, alpha=0.10, lw=0)
ax.axhline(0.2, color=SADGS, lw=2.5)
ax.text(2.95, 0.27, 'near plane z = 0.2', ha='right', color=SADGS, fontsize=15)
ax.text(2.95, -0.12, 'culled (z <= 0.2)', ha='right', color=SEC, fontsize=14, va='top')
ax.text(-2.95, 3.85, 'kept (z > 0.2)', ha='left', color=CUDA, fontsize=15, weight='bold')
# commented-out NDC +-1.3 test, drawn dotted as NOT applied (tan(fov/2) = 0.5, illustrative)
zz = np.linspace(0.2, 4.0, 50)
ax.plot(1.3 * 0.5 * zz, zz, ls=':', color=SEC, lw=2)
ax.plot(-1.3 * 0.5 * zz, zz, ls=':', color=SEC, lw=2)
ax.text(1.45, 3.1, 'NDC +-1.3 test\n(commented out)', color=SEC, fontsize=13)
ax.plot(0, 0, marker='^', color='black', ms=12)
ax.text(0.1, -0.3, 'camera', fontsize=14, color='black')
pts = [(-1.6, 0.05, 'culled', SADGS), (0.3, 0.12, 'culled', SADGS),
       (2.2, 0.9, 'kept, outside NDC band', GAUSS),
       (0.4, 1.6, 'kept', GAUSS), (-1.7, 2.6, 'kept, outside NDC band', GAUSS)]
for x, z, lab, c in pts:
    ax.plot(x, z, 'o', color=c, ms=11, mec='white', mew=1.5)
    ax.text(x + 0.12, z + 0.1, lab, fontsize=13, color=c)
ax.set_xlabel('x (view space)', fontsize=15, color=SEC)
ax.set_ylabel('z (view space, depth)', fontsize=15, color=SEC)
ax.tick_params(labelsize=12, colors=SEC)
fig.savefig('figs/s06_3.pdf', bbox_inches='tight')
plt.close(fig)

# ---------- Figure 5: opacity level-set ellipses ----------
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_aspect('equal')
ax.set_xlim(10, 190)
ax.set_ylim(0, 170)
ax.set_xlabel('x (pixels)', fontsize=15, color=SEC)
ax.set_ylabel('y (pixels)', fontsize=15, color=SEC)
ax.tick_params(labelsize=12, colors=SEC)
levels = [(0.1, GAUSS), (0.5, PY), (0.8, CUDA), (1.0, SADGS)]
for a, c in levels:
    t = 2.0 * np.log(255.0 * a)
    w, h, ang = ellipse_params(t)
    ax.add_patch(Ellipse(P, w, h, angle=ang, fill=False, ec=c, lw=2.5,
                         label=f'alpha={a}, t={t:.2f}'))
ax.legend(loc='upper left', fontsize=13, frameon=False)
w_, V = np.linalg.eigh(S)
for k in (0, 1):
    v = V[:, k]
    L = np.sqrt(T * w_[k])
    ax.annotate('', xy=P + L * v, xytext=P,
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5, ls='--'))
ax.plot(*P, 'o', color='black', ms=8)
ax.text(P[0] + 4, P[1] - 12, 'p', fontsize=14)
fig.savefig('figs/s06_5.pdf', bbox_inches='tight')
plt.close(fig)

# ---------- Figure 6: 3-sigma box vs AABB vs AccuTile ----------
hx = np.sqrt(T * S[0, 0])
hy = np.sqrt(T * S[1, 1])
box_tiles = tile_rect(P[0] - R3SIG, P[0] + R3SIG, P[1] - R3SIG, P[1] + R3SIG)
aabb_tiles = tile_rect(P[0] - hx, P[0] + hx, P[1] - hy, P[1] + hy)
acc_tiles = accutile_tiles(T)
print('counts', len(box_tiles), len(aabb_tiles), len(acc_tiles))
assert len(box_tiles) == 81 and len(aabb_tiles) == 54 and len(acc_tiles) == 36

fig, axes = plt.subplots(1, 3, figsize=(12, 6.2))
panels = [
    (box_tiles, GAUSS, '3-sigma box: 81 tiles'),
    (aabb_tiles, PY, 'AABB of ellipse: 54 tiles'),
    (acc_tiles, CUDA, 'AccuTile: 36 tiles (est.)'),
]
w, h, ang = ellipse_params(T)
for ax, (tiles, c, title) in zip(axes, panels):
    draw_grid(ax)
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(0, GX * TILE + 1, 4 * TILE))
    ax.set_yticks(np.arange(0, GY * TILE + 1, 4 * TILE))
    fill_tiles(ax, tiles, c, alpha=0.55)
    ax.add_patch(Ellipse(P, w, h, angle=ang, fill=False, ec='black', lw=1.8))
    ax.plot(*P, 'o', color='black', ms=5)
    ax.set_title(title, fontsize=13, color='black')
fig.savefig('figs/s06_6.pdf', bbox_inches='tight')
plt.close(fig)

print('done')
