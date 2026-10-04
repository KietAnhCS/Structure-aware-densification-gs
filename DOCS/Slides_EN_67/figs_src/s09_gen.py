import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16

CUDA = '#76B900'
SECOND = '#555555'
GAUSS = '#286EBE'

# Illustrative example (adam_and_bindings.md, section 5): theta0=1, g1=0.5, g2=-0.2,
# lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, visible at both steps.
B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 0.01
grads = [0.5, -0.2]

def run(bias_correction):
    theta, m, v = 1.0, 0.0, 0.0
    thetas = [theta]
    dthetas = []
    for t, g in enumerate(grads, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        if bias_correction:
            mh = m / (1 - B1 ** t)
            vh = v / (1 - B2 ** t)
        else:
            mh, vh = m, v
        d = -LR * mh / (np.sqrt(vh) + EPS)
        theta += d
        thetas.append(theta)
        dthetas.append(d)
    return np.array(thetas), np.array(dthetas)

th_code, d_code = run(False)
th_std, d_std = run(True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.2))

steps = np.arange(len(th_code))
ax1.plot(steps, th_code, 'o-', color=CUDA, lw=2.5, ms=9, label='custom CUDA Adam (no bias correction)')
ax1.plot(steps, th_std, 's--', color=GAUSS, lw=2.5, ms=9, label='standard Adam (with bias correction)')
for s, a, b in zip(steps, th_code, th_std):
    if s > 0:
        ax1.annotate(f"{a:.6f}", (s, a), textcoords='offset points', xytext=(10, -16), color=CUDA, fontsize=13)
        ax1.annotate(f"{b:.6f}", (s, b), textcoords='offset points', xytext=(10, 8), color=GAUSS, fontsize=13)
ax1.set_xlabel('step t')
ax1.set_ylabel(r'parameter $\theta_t$')
ax1.set_title('Two steps, same gradients (illustrative)', fontsize=17)
ax1.set_xticks(steps)
ax1.legend(frameon=False, fontsize=13, loc='lower left')
ax1.grid(True, color='#DDDDDD', lw=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

t = np.arange(1, 5001)
ax2.semilogx(t, 1 - B1 ** t, color=SECOND, lw=2.5, label=r'$1-\beta_1^t$')
ax2.semilogx(t, 1 - B2 ** t, color=CUDA, lw=2.5, label=r'$1-\beta_2^t$')
ax2.axhline(0.99, color='#C83C32', lw=1.5, ls=':')
ax2.text(1.2, 0.97, '0.99', color='#C83C32', fontsize=13)
ax2.set_xlabel('step t (log scale)')
ax2.set_ylabel('bias-correction denominator')
ax2.set_ylim(0, 1.05)
ax2.set_title('Correction matters only in early steps', fontsize=17)
ax2.legend(frameon=False, fontsize=13, loc='center right')
ax2.grid(True, which='both', color='#DDDDDD', lw=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.savefig('figs/s09_1.pdf', bbox_inches='tight')
print('theta code:', th_code)
print('theta std :', th_std)
print('delta code:', d_code)
print('delta std :', d_std)
