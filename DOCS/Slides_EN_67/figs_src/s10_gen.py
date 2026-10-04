import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 14

CUDA = '#76B900'
SADGS = '#C83C32'
GAUSS = '#286EBE'
PY = '#F0AA1E'
SEC = '#555555'


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#DDDDDD', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------
# s10_2: eta histogram over simulated views (illustrative)
# ---------------------------------------------------------------
rng = np.random.default_rng(7)
M = 40
eta_max = np.exp(rng.normal(np.log(1.6), 0.45, size=M))  # simulated views
TAU_HIGH, TAU_LOW, SPLIT_R = 1.0, 0.1, 0.8
high_ratio = np.mean(eta_max > TAU_HIGH)
low_ratio = np.mean(eta_max <= TAU_LOW)
print('s10_2 high_ratio=%.3f low_ratio=%.3f' % (high_ratio, low_ratio))

fig, ax = plt.subplots(figsize=(12, 6.2))
bins = np.logspace(-2, 1.5, 36)
counts, edges, patches = ax.hist(eta_max, bins=bins, color=GAUSS, alpha=0.75,
                                 edgecolor='white')
for p, lo in zip(patches, edges[:-1]):
    if lo > TAU_HIGH:
        p.set_facecolor(SADGS)
    elif lo <= TAU_LOW:
        p.set_facecolor('#9AA5B1')
ax.set_xscale('log')
ax.axvline(TAU_LOW, color=SEC, ls='--', lw=2)
ax.axvline(TAU_HIGH, color=SADGS, ls='--', lw=2)
ax.text(TAU_LOW * 1.08, ax.get_ylim()[1] * 0.92, r'$\tau_{low}=0.1$', color=SEC, fontsize=15)
ax.text(TAU_HIGH * 1.08, ax.get_ylim()[1] * 0.92, r'$\tau_{high}=1.0$', color=SADGS, fontsize=15)
ax.set_xlabel(r'$\eta_{\max}$ of one Gaussian in one view (log scale)')
ax.set_ylabel('number of views')
ax.set_title('Simulated $\\eta_{\\max}$ over %d views (illustrative)' % M)
txt = ('high_ratio = %.2f  (split if > %.1f)\n'
       'low_ratio  = %.2f  (prune if > %.1f)') % (high_ratio, SPLIT_R, low_ratio, SPLIT_R)
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=15,
        bbox=dict(boxstyle='round', fc='white', ec='#CCCCCC'))
style(ax)
fig.savefig('figs/s10_2.pdf', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------
# s10_3: anisotropic split counts k_j = ceil(sqrt(max(eta_j,1)))
# ---------------------------------------------------------------
cases = [
    ('Case A', (9.0, 2.2, 0.6)),
    ('Case B', (16.0, 4.0, 1.0)),
    ('Case C', (2.5, 1.0, 0.3)),
]
axes_names = ['x', 'y', 'z']
fig, ax = plt.subplots(figsize=(12, 6.2))
width = 0.25
colors = [GAUSS, CUDA, PY]
x = np.arange(len(cases))
for j in range(3):
    ks = []
    for _, eta in cases:
        ks.append(math.ceil(math.sqrt(max(eta[j], 1.0))))
    bars = ax.bar(x + (j - 1) * width, ks, width, color=colors[j],
                  label='axis %s' % axes_names[j])
    for b, k in zip(bars, ks):
        ax.text(b.get_x() + b.get_width() / 2, k + 0.08, str(k), ha='center', fontsize=15)
labels = []
for name, eta in cases:
    kx = math.ceil(math.sqrt(max(eta[0], 1.0)))
    ky = math.ceil(math.sqrt(max(eta[1], 1.0)))
    kz = math.ceil(math.sqrt(max(eta[2], 1.0)))
    labels.append('%s\n$\\eta$=(%g, %g, %g)\n$N=%d\\cdot%d\\cdot%d=%d$'
                  % (name, eta[0], eta[1], eta[2], kx, ky, kz, kx * ky * kz))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel(r'$k_j$ (slices per axis)')
ax.set_ylim(0, 5)
ax.set_title(r'Split counts $k_j=\lceil\sqrt{\max(\eta_j,1)}\rceil$ and $N=k_xk_yk_z$ (illustrative)')
ax.legend(loc='upper right', frameon=False)
style(ax)
fig.savefig('figs/s10_3.pdf', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------
# s10_5: 3-sigma box vs compact box (example covariance from BOOK/08)
# ---------------------------------------------------------------
S11, S12, S22 = 117.0, 54.0, 36.0
mu = np.array([120.0, 88.0])
MULT = 0.7


def t_of(alpha, mult=MULT):
    return mult * 2.0 * math.log(255.0 * alpha)


t01 = t_of(0.1)
t1 = t_of(1.0)
print('s10_5 t(0.1)=%.3f t(1.0)=%.3f' % (t01, t1))

fig, ax = plt.subplots(figsize=(12, 6.2))
# 3-sigma box (3DGS style)
hx3, hy3 = 3 * math.sqrt(S11), 3 * math.sqrt(S22)
ax.add_patch(Rectangle((mu[0] - hx3, mu[1] - hy3), 2 * hx3, 2 * hy3, fill=False,
                       ec=SEC, lw=2.2, ls=':', label=r'3$\sigma$ box (3DGS): $\pm$%.1f, $\pm$%.1f px' % (hx3, hy3)))
# compact box alpha = 1
hx1, hy1 = math.sqrt(t1 * S11), math.sqrt(t1 * S22)
ax.add_patch(Rectangle((mu[0] - hx1, mu[1] - hy1), 2 * hx1, 2 * hy1, fill=False,
                       ec=PY, lw=2.2, ls='--', label=r'compact box $\alpha$=1: $\pm$%.1f, $\pm$%.1f px' % (hx1, hy1)))
# compact box alpha = 0.1
hx, hy = math.sqrt(t01 * S11), math.sqrt(t01 * S22)
ax.add_patch(Rectangle((mu[0] - hx, mu[1] - hy), 2 * hx, 2 * hy, fill=True,
                       fc=SADGS, alpha=0.12, ec=SADGS, lw=2.4,
                       label=r'compact box $\alpha$=0.1: $\pm$%.1f, $\pm$%.1f px' % (hx, hy)))
# ellipse boundary at alpha = 0.1: Delta^T Sigma'^-1 Delta = t
det = S11 * S22 - S12 ** 2
lam, vec = np.linalg.eigh(np.array([[S11, S12], [S12, S22]]))
width_e = 2 * math.sqrt(t01 * lam[1])
height_e = 2 * math.sqrt(t01 * lam[0])
angle = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
ax.add_patch(Ellipse(mu, width_e, height_e, angle=angle, fill=False, ec=GAUSS, lw=2.4,
                     label=r'ellipse at $\alpha$=0.1 ($t$=%.2f)' % t01))
ax.plot(*mu, 'o', color='black', ms=6)
ax.set_aspect('equal')
ax.set_xlim(mu[0] - 45, mu[0] + 45)
ax.set_ylim(mu[1] - 40, mu[1] + 40)
ax.set_xlabel('u (pixel)')
ax.set_ylabel('v (pixel)')
ax.set_title(r'$t=\mathrm{mult}\cdot 2\ln(255\alpha)$, mult=0.7 (illustrative example)')
ax.legend(loc='lower right', frameon=True, fontsize=13)
style(ax)
fig.savefig('figs/s10_5.pdf', bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------
# s10_6: tile counts 25 -> 15 -> 9 (BOOK/08 example, mult = 0.5)
# ---------------------------------------------------------------
stages = ['Square box', 'Compact box', 'After ellipse\nfilter']
tiles = [25, 15, 9]
fig, ax = plt.subplots(figsize=(12, 6.2))
bars = ax.bar(stages, tiles, color=[SEC, GAUSS, SADGS], width=0.55)
for b, v in zip(bars, tiles):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha='center', fontsize=18, fontweight='bold')
ax.annotate('', xy=(2, 9.5), xytext=(0, 25.5),
            arrowprops=dict(arrowstyle='->', color=SEC, lw=1.5, connectionstyle='arc3,rad=-0.2'))
ax.text(1.0, 22.5, r'$R_{\mathrm{tile}}=9/25=0.36$', ha='center', fontsize=17, color=SADGS)
ax.set_ylabel('tiles touched by one Gaussian ($K_i$)')
ax.set_ylim(0, 30)
ax.set_title('Tile–Gaussian pairs: $P=\\sum_i K_i$ (example, mult = 0.5 as in source)')
style(ax)
ax.grid(axis='x', visible=False)
fig.savefig('figs/s10_6.pdf', bbox_inches='tight')
plt.close(fig)

print('done')
