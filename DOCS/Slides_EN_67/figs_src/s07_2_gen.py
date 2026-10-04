import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16

# Illustrative alpha values for 12 depth-sorted Gaussians at one pixel (front to back)
alpha = np.array([0.30, 0.50, 0.20, 0.60, 0.10, 0.40, 0.05, 0.70, 0.20, 0.30, 0.10, 0.02])
N = len(alpha)

# Forward pass: T_i = T_{i-1} * (1 - alpha_i), T_0 = 1. Only T_final is kept.
T_fwd = np.empty(N + 1)
T_fwd[0] = 1.0
for i in range(N):
    T_fwd[i + 1] = T_fwd[i] * (1.0 - alpha[i])
T_final = T_fwd[-1]

# Backward pass: start from T_final and recover T_{i-1} = T_i / (1 - alpha_i)
T_rec = np.empty(N + 1)
T_rec[N] = T_final
T = T_final
for i in range(N, 0, -1):
    T = T / (1.0 - alpha[i - 1])
    T_rec[i - 1] = T

max_err = np.max(np.abs(T_rec - T_fwd))
print("T_final =", T_final, " max |recovered - forward| =", max_err)

x = np.arange(N + 1)
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.plot(x, T_fwd, '-o', color='#286EBE', lw=2.5, ms=8, label='Forward: $T_i$ stored in forward pass (illustrative)')
ax.plot(x, T_rec, 'o', mfc='white', mec='#76B900', mew=2.5, ms=14, label='Backward: $T_{i-1}=T_i/(1-\\alpha_i)$ recomputed')
ax.axhline(T_final, color='#C83C32', ls='--', lw=1.5)
ax.text(N - 0.2, T_final + 0.03, '$T_{final}$', color='#C83C32', ha='right', fontsize=16)

ax.set_xlabel('Gaussian index $i$ (front to back)', color='#555555')
ax.set_ylabel('Transmittance $T$', color='#555555')
ax.set_xticks(x)
ax.set_ylim(0, 1.05)
ax.grid(True, color='#DDDDDD', lw=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper right', fontsize=15)
ax.text(0.2, 0.08, f'max round-off error (float64): {max_err:.1e}', color='#555555', fontsize=15)

fig.savefig('figs/s07_2.pdf', bbox_inches='tight')
print("saved figs/s07_2.pdf")
