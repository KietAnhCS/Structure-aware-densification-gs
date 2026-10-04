# Matplotlib figures for sections/s05.tex (slides 1, 3, 7). Run from Slides_EN_67.
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16

CUDA, SADGS, GAUSS, PY, SEC = '#76B900', '#C83C32', '#286EBE', '#F0AA1E', '#555555'


def cov_inv(s1, s2, ang_deg):
    th = math.radians(ang_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    S = R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T
    return np.linalg.inv(S)


def touched_tiles(c, s1, s2, ang, alpha, gx, gy, n=40):
    """Tiles (i=col, j=row) hit by the level set alpha*G >= 1/255, t = 2 ln(255 alpha), mult = 1."""
    t = 2.0 * math.log(255.0 * alpha)
    Si = cov_inv(s1, s2, ang)
    out = []
    for j in range(gy):
        for i in range(gx):
            xs = np.linspace(i, i + 1, n)
            ys = np.linspace(j, j + 1, n)
            X, Y = np.meshgrid(xs, ys)
            d = np.stack([X - c[0], Y - c[1]], -1)
            m = np.einsum('...i,ij,...j->...', d, Si, d)
            if (m <= t).any():
                out.append((i, j))
    return out


# ---------- Figure s05_1: tile grid with tiles_touched ----------
def fig1():
    gx, gy = 8, 5
    gauss = [
        # name, center (tile units), sigma1, sigma2, angle (deg), alpha
        ('G1', (1.7, 1.5), 0.85, 0.45, 20, 0.90),
        ('G2', (4.6, 2.4), 0.45, 0.45, 0, 0.80),
        ('G3', (6.4, 3.8), 0.75, 0.22, -30, 0.50),
        ('G4', (3.1, 3.9), 0.35, 0.30, 0, 0.95),
        ('G5', (5.7, 1.0), 0.55, 0.18, 60, 0.70),
    ]
    counts = np.zeros((gy, gx), dtype=int)
    info = []
    for name, c, s1, s2, ang, a in gauss:
        tl = touched_tiles(c, s1, s2, ang, a, gx, gy)
        for i, j in tl:
            counts[j, i] += 1
        info.append((name, len(tl)))

    fig, ax = plt.subplots(figsize=(12, 6.2))
    cmap = plt.get_cmap('Greens')
    for j in range(gy):
        for i in range(gx):
            k = counts[j, i]
            face = cmap(0.15 + 0.22 * k) if k > 0 else '#FFFFFF'
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=face, edgecolor='#BBBBBB', lw=1))
            tau = j * gx + i
            ax.text(i + 0.07, j + 0.93, f'{tau}', fontsize=10, color=SEC, va='top')
            if k > 0:
                ax.text(i + 0.5, j + 0.42, f'{k}', fontsize=16, ha='center', va='center',
                        color='#1B3A00', fontweight='bold')
    colors = [GAUSS, SADGS, PY, CUDA, '#7A3FA0']
    for (name, c, s1, s2, ang, a), col in zip(gauss, colors):
        t = 2.0 * math.log(255.0 * a)
        w = 2 * math.sqrt(t) * s1
        h = 2 * math.sqrt(t) * s2
        ax.add_patch(Ellipse(c, w, h, angle=ang, fill=False, edgecolor=col, lw=2.6))
        ax.plot(*c, marker='o', color=col, ms=6)
        ax.text(c[0] + 0.12, c[1] + 0.12, name, color=col, fontsize=16, fontweight='bold')
    handles = [plt.Line2D([0], [0], color=col, lw=2.6,
                          label=f'{n}: tiles_touched = {k}') for (n, k), col in zip(info, colors)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=16)
    ax.set_xlim(0, gx)
    ax.set_ylim(0, gy)
    ax.set_aspect('equal')
    ax.set_xticks(range(gx + 1))
    ax.set_yticks(range(gy + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Illustrative 8 x 5 tile grid. Small grey number: tile id. Large number: Gaussians touching the tile',
                 fontsize=12, color=SEC, loc='left')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.savefig('figs/s05_1.pdf', bbox_inches='tight')
    plt.close(fig)
    print('s05_1', info)


# ---------- Figure s05_3: prefix sum table ----------
def fig3():
    rows = [
        ('0', 'A', '4', '4', '0'),
        ('1', 'B', '1', '5', '4'),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.axis('off')
    col_labels = ['i', 'Gaussian', 'tiles_touched[i]', 'offset[i] (inclusive)', 'off_i (exclusive)']
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='upper center', cellLoc='center',
                   colWidths=[0.1, 0.16, 0.24, 0.26, 0.24])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(18)
    tbl.scale(1, 2.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#AAAAAA')
        if r == 0:
            cell.set_facecolor(GAUSS)
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif c == 2:
            cell.set_facecolor('#E6F2CC')
        elif c == 3:
            cell.set_facecolor('#E6EEF8')
        else:
            cell.set_facecolor('white')
    ax.text(0.5, 0.12, 'L = num_rendered = offset[P-1] = 4 + 1 = 5 (key/value pairs)',
            transform=ax.transAxes, ha='center', fontsize=20, color=SADGS, fontweight='bold')
    ax.text(0.5, 0.02, 'Illustrative two-Gaussian batch, P = 2',
            transform=ax.transAxes, ha='center', fontsize=14, color=SEC)
    fig.savefig('figs/s05_3.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------- Figure s05_7: sorted keys and ranges ----------
def fig7():
    keys_rows = [
        ('0', '0', '1.0', 'B', '0x000000003F800000'),
        ('1', '0', '2.0', 'A', '0x0000000040000000'),
        ('2', '1', '2.0', 'A', '0x0000000140000000'),
        ('3', '2', '2.0', 'A', '0x0000000240000000'),
        ('4', '3', '2.0', 'A', '0x0000000340000000'),
    ]
    range_rows = [
        ('0', '[0, 2)', '2', 'B, then A'),
        ('1', '[2, 3)', '1', 'A'),
        ('2', '[3, 4)', '1', 'A'),
        ('3', '[4, 5)', '1', 'A'),
    ]
    fig = plt.figure(figsize=(12, 6.2))
    ax1 = fig.add_axes([0.02, 0.55, 0.96, 0.36])
    ax2 = fig.add_axes([0.02, 0.02, 0.96, 0.36])
    for a in (ax1, ax2):
        a.axis('off')
    t1 = ax1.table(cellText=keys_rows,
                   colLabels=['sorted pos', 'tile tau', 'depth', 'Gaussian', 'key (hex)'],
                   loc='center', cellLoc='center', colWidths=[0.14, 0.13, 0.14, 0.15, 0.44])
    t2 = ax2.table(cellText=range_rows,
                   colLabels=['tile tau', 'ranges[tau]', 'count', 'Gaussians (front to back)'],
                   loc='center', cellLoc='center', colWidths=[0.16, 0.24, 0.16, 0.44])
    for t in (t1, t2):
        t.auto_set_font_size(False)
        t.set_fontsize(16)
        t.scale(1, 1.9)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor('#AAAAAA')
            if r == 0:
                cell.set_facecolor(GAUSS)
                cell.get_text().set_color('white')
                cell.get_text().set_fontweight('bold')
            else:
                cell.set_facecolor('white')
    ax1.set_title('Sorted key/value array: key = tau << 32 | bitcast(depth)', loc='left',
                  fontsize=15, color=SEC)
    ax2.set_title('identifyTileRanges output (end index exclusive)', loc='left', fontsize=15, color=SEC)
    fig.savefig('figs/s05_7.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig1()
    fig3()
    fig7()
