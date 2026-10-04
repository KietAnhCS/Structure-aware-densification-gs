import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16

# Illustrative alpha values (same pixel as the T-recovery figure)
alpha = np.array([0.30, 0.50, 0.20, 0.60, 0.10, 0.40, 0.05, 0.70, 0.20, 0.30, 0.10, 0.02])
N = len(alpha)

T_prev = np.empty(N)          # T_{i-1} for each Gaussian i
T = 1.0
for i in range(N):
    T_prev[i] = T
    T = T * (1.0 - alpha[i])
T_final = T

# dL/dc_i = alpha_i * T_{i-1} * dL/dC  -> per-Gaussian weight
w = alpha * T_prev
w_bg = T_final               # weight of background color bg in C_final
print("sum of Gaussian weights:", w.sum(), " + T_final =", w.sum() + w_bg)

labels = [f'G{i+1}' for i in range(N)] + ['bg']
vals = np.concatenate([w, [w_bg]])
colors = ['#286EBE'] * N + ['#555555']

fig, ax = plt.subplots(figsize=(12, 6.2))
bars = ax.bar(labels, vals, color=colors, width=0.7)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}', ha='center', fontsize=12, color='#555555')

ax.set_ylabel(r'Weight in pixel color, $\alpha_i T_{i-1}$ (or $T_{final}$ for bg)', color='#555555')
ax.set_ylim(0, max(vals) * 1.2)
ax.grid(True, axis='y', color='#DDDDDD', lw=0.8)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=14)
ax.text(N - 0.5, max(vals) * 1.05,
        r'$\sum_i \alpha_i T_{i-1} + T_{final} = 1$ (illustrative values)',
        ha='right', fontsize=15, color='#555555')

fig.savefig('figs/s07_4.pdf', bbox_inches='tight')
print("saved figs/s07_4.pdf")
