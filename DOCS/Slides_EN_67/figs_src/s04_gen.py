"""Figures for sections/s04.tex (forward pass: conic, radius, per-pixel weight, compositing).

Run from Slides_EN_67:  python figs_src/s04_gen.py
All numbers follow MATH/submodules/diff-gaussian-rasterization_structgs/forward.md.
The example covariance and opacities are illustrative.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#DDDDDD"
plt.rcParams["grid.linewidth"] = 0.8

CUDA = "#76B900"
SADGS = "#C83C32"
GAUSS = "#286EBE"
PY = "#F0AA1E"
GREY = "#555555"

OUT = "figs"
os.makedirs(OUT, exist_ok=True)
FIGSIZE = (12, 6.2)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, os.path.exists(path))


# Illustrative 2x2 covariance used in frames 2 and 4
SIGMA = np.array([[6.0, 2.0], [2.0, 3.0]])
EIG_VALS, EIG_VECS = np.linalg.eigh(SIGMA)          # ascending order
LAM2, LAM1 = EIG_VALS
V1 = EIG_VECS[:, 1]                                  # eigenvector of lam1
V2 = EIG_VECS[:, 0]
RADIUS = int(np.ceil(3 * np.sqrt(LAM1)))
assert np.isclose(LAM1, 7.0) and np.isclose(LAM2, 2.0) and RADIUS == 8

# ---------------------------------------------------------------- s04_2
# Ellipse at 3 sigma (Mahalanobis radius 3) plus eigenvalue bars.
fig, (ax, bx) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.15, 1]})
t = np.linspace(0, 2 * np.pi, 400)
semi1, semi2 = 3 * np.sqrt(LAM1), 3 * np.sqrt(LAM2)
ell = np.outer(V1, semi1 * np.cos(t)) + np.outer(V2, semi2 * np.sin(t))
ax.fill(ell[0], ell[1], color=SADGS, alpha=0.18)
ax.plot(ell[0], ell[1], color=SADGS, lw=2.5, label=r"$k=3$ contour ($3\sqrt{\lambda}$ semi-axes)")
ax.plot([0, semi1 * V1[0]], [0, semi1 * V1[1]], color=SADGS, lw=2)
ax.plot([0, semi2 * V2[0]], [0, semi2 * V2[1]], color=GREY, lw=2)
ax.annotate(r"$3\sqrt{\lambda_1}\approx7.94$", xy=(semi1 * V1[0], semi1 * V1[1]),
            xytext=(-2.6, 5.5), color=SADGS, fontsize=15,
            arrowprops=dict(arrowstyle="-", color=SADGS, lw=1))
ax.plot(0, 0, "o", color=GAUSS, ms=7)
# Square of half-width r = 8 that bounds the ellipse
ax.add_patch(plt.Rectangle((-RADIUS, -RADIUS), 2 * RADIUS, 2 * RADIUS,
                           fill=False, ls="--", lw=2, ec=CUDA))
ax.text(RADIUS - 0.3, -RADIUS - 0.9, r"$r=\lceil 3\sqrt{\lambda_{max}}\rceil=8$",
        color=CUDA, fontsize=15, ha="right")
ax.set_aspect("equal")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(r"Illustrative $\Sigma'=[[6,\,2],[2,\,3]]$", fontsize=16)
ax.legend(loc="upper left", frameon=False, fontsize=13)

vals = [LAM1, LAM2]
labels = [r"$\lambda_1=7$", r"$\lambda_2=2$"]
bars = bx.bar(labels, vals, color=[SADGS, GREY], width=0.5)
for b, v in zip(bars, vals):
    bx.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:g}", ha="center", fontsize=16)
bx.set_ylabel(r"eigenvalue of $\Sigma'$")
bx.set_ylim(0, 8.5)
bx.set_title(r"$\max(\lambda)=7 \Rightarrow \sqrt{7}=2.646,\ 3\sqrt{7}=7.94,\ r=8$", fontsize=15)
bx.grid(axis="x", visible=False)
save(fig, "s04_2.pdf")

# ---------------------------------------------------------------- s04_4
# 2D Gaussian kernel G(d) = exp(-1/2 d^T Sigma^-1 d) as a heatmap with iso-contours.
conic = np.linalg.inv(SIGMA)
xs = np.linspace(-10, 10, 401)
X, Y = np.meshgrid(xs, xs)
Q = conic[0, 0] * X**2 + 2 * conic[0, 1] * X * Y + conic[1, 1] * Y**2
G = np.exp(-0.5 * Q)

fig, ax = plt.subplots(figsize=FIGSIZE)
im = ax.imshow(G, origin="lower", extent=[-10, 10, -10, 10], cmap="Blues", vmin=0, vmax=1)
cs = ax.contour(X, Y, Q, levels=[1, 4, 9], colors=[SADGS, SADGS, SADGS],
                linewidths=[2, 2, 2.5], linestyles=["-", "--", "-"])
ax.clabel(cs, fmt={1: r"$k=1$", 4: r"$k=2$", 9: r"$k=3$"}, fontsize=15, inline=True)
ax.plot(0, 0, "o", color=GAUSS, ms=7)
ax.set_xlabel(r"$d_x = xy_x - x_{pix}$ (pixels)")
ax.set_ylabel(r"$d_y$ (pixels)")
ax.set_title(r"$G(d)=\exp(-\frac{1}{2}\, d^{T}\Sigma'^{-1} d)$, "
             r"illustrative $\Sigma'$", fontsize=16)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("G (peak = 1 at the centre)")
ax.text(5.8, -8.6, r"$G=e^{-0.5}=0.61$ at $k=1$" "\n" r"$G=e^{-2}=0.14$ at $k=2$" "\n"
        r"$G=e^{-4.5}=0.011$ at $k=3$", color="black", fontsize=14,
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
save(fig, "s04_4.pdf")

# ---------------------------------------------------------------- s04_5
# alpha(d) = min(0.99, o * coef * G(d)), skipped if alpha < 1/255.
# Illustrative isotropic Sigma' = diag(4,4), coef = 1, one horizontal line of pixels.
d = np.linspace(0, 10, 1001)
s2 = 4.0
thr = 1 / 255
fig, ax = plt.subplots(figsize=FIGSIZE)
for o, col, lab in [(0.9, GAUSS, r"$o=0.9$"), (0.999, SADGS, r"$o=0.999$")]:
    alpha = np.minimum(0.99, o * np.exp(-d**2 / (2 * s2)))
    ax.plot(d, alpha, color=col, lw=2.8, label=lab + r" (illustrative)")
    cut = d[np.argmax(alpha < thr)] if np.any(alpha < thr) else None
    if cut is not None:
        ax.plot([cut], [thr], "o", color=col, ms=7)
        ax.annotate(f"cut-off $\\approx{cut:.1f}$ px", xy=(cut, thr), xytext=(cut + 0.8, 0.12),
                    color=col, fontsize=14, arrowprops=dict(arrowstyle="-", color=col, lw=1))
ax.axhline(thr, color=GREY, ls="--", lw=1.5)
ax.text(9.9, thr + 0.02, r"$1/255$ skip threshold", color=GREY, ha="right", fontsize=14)
ax.axhline(0.99, color=PY, ls=":", lw=2)
ax.text(9.9, 0.99 - 0.07, r"$\alpha_{max}=0.99$ cap", color=PY, ha="right", fontsize=14)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"distance $|d|$ from the Gaussian centre (pixels)")
ax.set_ylabel(r"$\alpha = \min(0.99,\ o\,G(d))$")
ax.set_title(r"Per-pixel opacity along a line, $\Sigma'=\mathrm{diag}(4,4)$, illustrative", fontsize=16)
ax.legend(loc="upper right", frameon=False, fontsize=14, bbox_to_anchor=(0.98, 0.9))
save(fig, "s04_5.pdf")

# ---------------------------------------------------------------- s04_6
# Front-to-back compositing with the forward.md section 10.2 example (illustrative).
alphas = np.array([0.6, 0.9 * np.exp(-0.5), 0.8 * np.exp(-3.125)])
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
bg = np.array([0.1, 0.1, 0.1])

T = [1.0]
for a in alphas:
    T.append(T[-1] * (1 - a))
T = np.array(T)                       # T_0 .. T_3
weights = alphas * T[:-1]             # alpha_i * T_{i-1}
C = (colors * weights[:, None]).sum(axis=0)
C_final = C + T[-1] * bg
print("T:", np.round(T, 5), "weights:", np.round(weights, 5))
print("C:", np.round(C, 5), "C_final:", np.round(C_final, 5))
assert np.allclose(np.round(C_final, 4), [0.6175, 0.2359, 0.0239], atol=1e-4)
assert np.isclose(round(T[-1], 5), 0.17526, atol=2e-5)
assert np.isclose(weights.sum() + T[-1], 1.0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.1, 1]})
idx = np.arange(4)
ax.step(idx, T, where="post", color=SADGS, lw=2.5)
ax.plot(idx, T, "o", color=SADGS, ms=8)
for i, v in enumerate(T):
    ax.text(i + 0.12, v + 0.05, f"$T_{i}={v:.4f}$" if i < 3 else f"$T_3={v:.5f}$", fontsize=14)
ax.axhline(1e-4, color=GREY, ls="--", lw=1.3)
ax.text(2.9, 0.035, r"early stop $T<10^{-4}$", color=GREY, fontsize=13, ha="right")
ax.set_xticks(idx)
ax.set_xticklabels(["start", "after G1", "after G2", "after G3"], fontsize=14)
ax.set_ylim(0, 1.15)
ax.set_ylabel(r"transmittance $T$")
ax.set_title("Transmittance drops with each Gaussian", fontsize=16)

names = ["G1", "G2", "G3"]
cols = [colors[0], colors[1], colors[2]]
bottom = 0.0
for n, w, c in zip(names, weights, cols):
    bx.bar(["pixel"], [w], bottom=bottom, color=c, width=0.45, edgecolor="white")
    bx.text(0.27, bottom + w / 2, f"{n}: {w:.4f}", va="center", fontsize=14)
    bottom += w
bx.bar(["pixel"], [T[-1]], bottom=bottom, color=bg, width=0.45, edgecolor="white")
bx.text(0.27, bottom + T[-1] / 2, f"background: {T[-1]:.4f}", va="center", fontsize=14)
bx.set_ylim(0, 1.05)
bx.set_xlim(-0.5, 1.2)
bx.set_ylabel(r"weight $\alpha_i T_{i-1}$ (sums to 1)")
bx.set_title(rf"Pixel $(50,50)$: $C=({C_final[0]:.4f},{C_final[1]:.4f},{C_final[2]:.4f})$", fontsize=15)
bx.grid(False)
save(fig, "s04_6.pdf")
