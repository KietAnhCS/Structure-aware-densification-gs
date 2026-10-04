"""Figures for section s01 (matplotlib). Run from Slides_EN_67: python figs_src/s01_gen.py"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16

BLUE, GREEN, RED, YEL, GRAY = "#286EBE", "#76B900", "#C83C32", "#F0AA1E", "#555555"
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ASSETS = os.path.join(BASE, "..", "assets")
OUT = os.path.join(BASE, "figs")


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)


def load_board():
    with open(os.path.join(ASSETS, "leaderboard.csv"), newline="", encoding="utf-8") as h:
        return {r["scene"]: r for r in csv.DictReader(h)}


def term_values(row):
    """Score components from the leaderboard row: 0.4(1-LPIPS), 0.3*SSIM, 0.3*PSNR_norm."""
    return (0.4 * (1 - float(row["lpips"])),
            0.3 * float(row["ssim"]),
            0.3 * float(row["psnr_norm"]))


# ---- s01_5: score breakdown for drjohnson (values from pipeline/score.py formula) ----
def fig_s01_5():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    a, b, c = 0.4 * (1 - 0.3293), 0.3 * 0.8677, 0.3 * 0.9136
    total = a + b + c
    labels = ["0.4(1 − LPIPS)\nLPIPS = 0.3293", "0.3 · SSIM\nSSIM = 0.8677", "0.3 · PSNR_norm\nPSNR = 27.408 dB"]
    parts = [(a, BLUE), (b, GREEN), (c, YEL)]
    left = 0.0
    for (val, col), lab in zip(parts, labels):
        ax.barh([0], [val], left=left, color=col, height=0.5, edgecolor="white")
        ax.text(left + val / 2, 0, f"{val:.4f}", ha="center", va="center", color="black", fontsize=18, fontweight="bold")
        ax.text(left + val / 2, 0.42, lab, ha="center", va="bottom", color=GRAY, fontsize=14)
        left += val
    ax.axvline(total, color=RED, linestyle="--", linewidth=1.5)
    ax.text(total, -0.45, f"Score = {total:.4f}", ha="right", va="top", color=RED, fontsize=18, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Contribution to Score (0 to 1)")
    style(ax)
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.set_title("drjohnson: Score = 0.2683 + 0.2603 + 0.2741", fontsize=18, loc="left")
    fig.savefig(os.path.join(OUT, "s01_5.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---- s01_6: stacked bars per scene + MEAN ----
def fig_s01_6():
    board = load_board()
    order = ["drjohnson", "playroom", "train", "truck", "MEAN"]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    xs = range(len(order))
    bottoms = [0.0] * len(order)
    names = ["0.4(1 − LPIPS)", "0.3 · SSIM", "0.3 · PSNR_norm"]
    cols = [BLUE, GREEN, YEL]
    comps = [term_values(board[s]) for s in order]
    for k in range(3):
        vals = [comps[i][k] for i in range(len(order))]
        ax.bar(xs, vals, bottom=bottoms, color=cols[k], width=0.6, label=names[k], edgecolor="white")
        bottoms = [bottoms[i] + vals[i] for i in range(len(order))]
    for i, s in enumerate(order):
        ax.text(i, bottoms[i] + 0.012, f"{float(board[s]['score']):.4f}", ha="center", va="bottom",
                fontsize=17, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(order, fontsize=17)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    style(ax)
    fig.savefig(os.path.join(OUT, "s01_6.pdf"), bbox_inches='tight')
    plt.close(fig)


# ---- s01_7: drjohnson live training curves ----
def fig_s01_7():
    rows = []
    with open(os.path.join(ASSETS, "history.csv"), newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if r["scene"] == "drjohnson":
                rows.append(r)
    it = [int(r["iter"]) for r in rows]
    psnr = [float(r["psnr"]) for r in rows]
    ssim = [float(r["ssim"]) for r in rows]
    lpips = [float(r["lpips"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 6.2))
    specs = [(psnr, "PSNR (dB)", BLUE, "higher is better"),
             (ssim, "SSIM", GREEN, "higher is better"),
             (lpips, "LPIPS", RED, "lower is better")]
    for ax, (ys, ylab, col, note) in zip(axes, specs):
        ax.plot(it, ys, marker="o", color=col, linewidth=2.5, markersize=6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}\n({note})", fontsize=14)
        ax.set_xticks([1000, 3000, 5000, 7000])
        style(ax)
    # annotate the dips at 3000 and 6000 on PSNR
    for i, ps in zip(it, psnr):
        if i in (3000, 6000):
            axes[0].annotate(f"{ps:.1f} dB", (i, ps), textcoords="offset points", xytext=(8, -18),
                             color=RED, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s01_7.pdf"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_s01_5()
    fig_s01_6()
    fig_s01_7()
    print("wrote s01_5, s01_6, s01_7")
