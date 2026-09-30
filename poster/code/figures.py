"""Poster figures. Print size is the figsize, so point sizes here are the
point sizes on the printed A1 poster."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.stats import norm

from pt_sim import tempered_mode_weight

SEO_C = '#3b5bdb'
DEO_C = '#e8590c'
INK = '#1a1c1f'
MUTED = '#6c757d'
ACCENT = '#7048a8'
GRID = '#d8dade'

plt.rcParams.update({
    'font.family': 'Liberation Sans',
    'font.size': 15,
    'axes.titlesize': 17,
    'axes.titleweight': 'bold',
    'axes.labelsize': 15.5,
    'axes.edgecolor': '#adb5bd',
    'axes.linewidth': 1.0,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2.0,
})

COL = 6.41          # poster column width, inches
DPI = 300
T = np.load('traces.npz')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f'fig_{name}.png', dpi=DPI, bbox_inches='tight',
                pad_inches=0.04)
    plt.close(fig)
    print('wrote fig_%s.png' % name)


# ---------------------------------------------------------------- FIG 1
def fig_index_process():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 3.0), sharey=True)
    n = 8000
    for ax, scheme, c in [(axes[0], 'SEO', SEO_C), (axes[1], 'DEO', DEO_C)]:
        idx = T[f'idx_{scheme}'][:n]
        for r in range(3):
            ax.plot(idx[:, r], color=c, lw=0.85, alpha=0.8)
        ax.set_title(f'{scheme}  ' + ('(reversible)' if scheme == 'SEO'
                                      else '(non-reversible)'),
                     color=c, pad=7, fontsize=16)
        ax.set_xlabel('scan')
        grid(ax)
        ax.set_xticks([0, 4000, 8000])
        ax.set_xticklabels(['0', '4k', '8k'])
        ax.axhline(99, color=MUTED, lw=1.0, ls=':')
    axes[0].set_ylabel('chain index')
    axes[0].set_ylim(-4, 108)
    axes[0].set_yticks([0, 50, 99])
    axes[0].set_yticklabels(['0','50','99'],
                            fontsize=12.5)
    fig.tight_layout()
    save(fig, '1_index')


# ---------------------------------------------------------------- FIG 2
def fig_speedup():
    """Theory from the measured swap rates; measurement from the 6-seed sweep
    at the headline target, which has far better round trip statistics."""
    def theory(swap):
        r = 1 - np.asarray(swap)
        E = np.sum(r / np.maximum(swap, 1e-12))
        return 1 / (2 * len(swap) + 2 * E), 1 / (2 + 2 * E), r.sum()

    tS, _, LamS = theory(T['swap_SEO'])
    _, tD, LamD = theory(T['swap_DEO'])
    thy = [tS, tD]

    R = json.load(open('res_inhom.json'))['inhom']
    emp, err = [], []
    for scheme in ['SEO', 'DEO']:
        row = [e for e in R if e['scheme'] == scheme
               and abs(e['s2'] - 0.40) < 1e-9][0]
        v = np.array([r['tau'] for r in row['reps']])
        emp.append(v.mean()); err.append(v.std())

    fig, ax = plt.subplots(figsize=(COL, 3.1))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, thy, w, color=[SEO_C, DEO_C], alpha=0.28,
           edgecolor=[SEO_C, DEO_C], lw=1.6, hatch='///',
           label='theory, Syed et al. Cor. 1')
    ax.bar(x + w / 2, emp, w, yerr=err, capsize=5, color=[SEO_C, DEO_C],
           error_kw=dict(ecolor=INK, lw=1.4), label='measured')
    for xi in range(2):
        ax.annotate('%.4f' % thy[xi], (xi - w / 2, thy[xi]), ha='center',
                    va='bottom', fontsize=13, color=MUTED, xytext=(0, 4),
                    textcoords='offset points')
        ax.annotate('%.4f' % emp[xi], (xi + w / 2, emp[xi] + err[xi]),
                    ha='center', va='bottom', fontsize=13.5,
                    fontweight='bold', xytext=(0, 5),
                    textcoords='offset points')
    ax.set_xticks(x)
    ax.set_xticklabels(['SEO (reversible)', 'DEO (non-reversible)'],
                       fontsize=14.5)
    ax.set_ylabel(r'round trip rate $\tau$')
    ax.set_ylim(0, max(thy + emp) * 1.60)
    ax.set_xlim(-0.62, 1.62)
    grid(ax)
    ax.legend(loc='upper left', fontsize=13, ncol=1)
    ax.annotate('%.1f$\\times$ more round trips' % (emp[1] / emp[0]),
                (1, max(thy + emp) * 1.30), ha='center', va='center',
                fontsize=16, fontweight='bold', color=DEO_C)
    ax.set_title(r'$\hat\Lambda$ = %.1f (SEO)   %.1f (DEO)' % (LamS, LamD),
                 fontsize=14.5, color=MUTED, fontweight='normal',
                 loc='right', pad=6)
    fig.tight_layout()
    save(fig, '2_speedup')


# ---------------------------------------------------------------- FIG 3
def fig_failure():
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.05), sharex=True)
    grid_x = np.linspace(-2.6, 2.6, 900)
    s1, s2 = 0.3, 0.4
    true = 0.5 * norm.pdf(grid_x, 1, s1) + 0.5 * norm.pdf(grid_x, -1, s2)
    for ax, scheme, c in [(axes[0], 'SEO', SEO_C), (axes[1], 'DEO', DEO_C)]:
        tr = T[f'trace_inhomog_{scheme}'][4000:]
        ax.hist(tr, bins=110, range=(-2.6, 2.6), density=True,
                color=c, alpha=0.85, edgecolor='none')
        ax.plot(grid_x, true, color=INK, lw=2.2, ls='--',
                label='target density')
        occ = (tr > 1 - 2 * (s1 / (s1 + s2))).mean()
        ax.set_title(f'{scheme}   ' + ('(reversible)' if scheme == 'SEO'
                                       else '(non-reversible)'),
                     fontsize=15.5, color=c, loc='left', pad=6)
        ax.annotate(f'mode A weight  {occ:.3f}\ntruth  0.500',
                    (0.985, 0.93), xycoords='axes fraction', ha='right',
                    va='top', fontsize=14, color=INK,
                    bbox=dict(boxstyle='round,pad=0.42', fc='#f1f3f5',
                              ec=c, lw=1.4))
        ax.set_ylabel('density')
        ax.set_yticks([])
        grid(ax)
    axes[0].legend(loc='upper left', fontsize=13.5)
    axes[1].set_xlabel('coordinate 1   (mode B at $-1$, mode A at $+1$)')
    fig.suptitle('$d = 30$,  $s_1 = 0.3$,  $s_2 = 0.4$,  25 000 scans',
                 fontsize=16, fontweight='bold', y=1.005)
    fig.tight_layout()
    save(fig, '3_failure')


# ---------------------------------------------------------------- FIG 4
def fig_closed_form():
    fig, ax = plt.subplots(figsize=(COL, 2.95))
    b = np.linspace(0, 1, 400)
    shades = ['#c5cae9', '#8b9adc', '#5570cf', SEO_C]
    for d, c in zip([5, 10, 20, 30], shades):
        w = tempered_mode_weight(b, d, 0.3, 0.4)
        ax.plot(b, w, color=c, lw=2.6, label=f'$d = {d}$')
    ax.axhline(0.5, color=INK, ls='--', lw=1.6)
    ax.annotate('balanced (0.5)', (0.985, 0.62), ha='right', va='bottom',
                fontsize=13.5, color=INK)
    ax.axvline(0.01, color=ACCENT, lw=1.8, ls=':')
    ax.annotate('reference\nchain', (0.045, 2.2e-5), color=ACCENT,
                fontsize=13.5, fontweight='bold', va='bottom')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 2.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'inverse temperature $\beta$')
    ax.set_ylabel(r'$\pi_\beta(\mathrm{mode\ A})$')
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    grid(ax)
    ax.legend(loc='lower right', fontsize=13.5, ncol=2, columnspacing=1.1,
              handlelength=1.5)
    fig.tight_layout()
    save(fig, '4_closedform')


# ---------------------------------------------------------------- FIG 5
def fig_sweep():
    R = json.load(open('res_inhom.json'))['inhom']
    fig, axes = plt.subplots(3, 1, figsize=(COL, 6.1), sharex=True)
    ratios = sorted({round(e['s2'] / e['s1'], 3) for e in R})

    def series(scheme, key):
        rows = sorted([e for e in R if e['scheme'] == scheme],
                      key=lambda e: e['s2'])
        x = [e['s2'] / e['s1'] for e in rows]
        v = np.array([[r[key] for r in e['reps']] for e in rows])
        return np.array(x), v.mean(1), v.std(1)

    for scheme, c in [('SEO', SEO_C), ('DEO', DEO_C)]:
        x, m, s = series(scheme, 'tau')
        axes[0].errorbar(x, m, yerr=s, color=c, lw=2.6, marker='o', ms=8,
                         capsize=4, label=scheme)
        for ax, key, floor in [(axes[1], 'switch_rate', 2e-5),
                               (axes[2], 'occ', 2e-4)]:
            x, m, s = series(scheme, key)
            m = np.maximum(m, floor)
            lo = np.maximum(m - s, floor * 1.05)
            ax.errorbar(x, m, yerr=[m - lo, s], color=c, lw=2.6, marker='o',
                        ms=8, capsize=4, label=scheme)

    axes[0].set_ylabel('round trip rate\n' + r'$\tau$')
    axes[0].set_ylim(0, 0.030)
    axes[0].legend(loc='center left', ncol=2, fontsize=13.5,
                   bbox_to_anchor=(0.02, 0.52))
    axes[0].set_title('the efficiency metric: unchanged', fontsize=15.5,
                      loc='left', color=MUTED)

    axes[1].set_yscale('log')
    axes[1].set_ylabel('mode changes\nper scan')
    axes[1].set_title('what actually matters: collapses', fontsize=15.5,
                      loc='left', color=MUTED)

    axes[2].set_yscale('log')
    axes[2].axhline(0.5, color=INK, ls='--', lw=1.6)
    axes[2].annotate('truth 0.5', (1.005, 0.62), ha='left', va='bottom',
                     fontsize=13, color=INK)
    axes[2].set_ylabel('mode A weight\nrecovered')
    axes[2].set_ylim(1e-4, 3.2)
    axes[2].set_xlabel(r'inhomogeneity  $s_2/s_1$      ($d=30$)')
    axes[2].set_title('and so does the answer', fontsize=15.5, loc='left',
                      color=MUTED)
    for ax in axes:
        grid(ax)
    axes[2].set_xticks(ratios)
    axes[2].set_xticklabels(['%.2f' % r for r in ratios])
    fig.tight_layout(h_pad=1.3)
    save(fig, '5_sweep')


# ---------------------------------------------------------------- FIG 6
def fig_dimension():
    R = json.load(open('res_dim.json'))['dim']
    FLOOR = 8e-6
    fig, ax = plt.subplots(figsize=(COL, 3.3))
    for scheme, c, mk in [('SEO', SEO_C, 'o'), ('DEO', DEO_C, 's')]:
        rows = sorted([e for e in R if e['scheme'] == scheme],
                      key=lambda e: e['d'])
        d = np.array([e['d'] for e in rows])
        sw = np.array([np.mean([r['switch_rate'] for r in e['reps']])
                       for e in rows])
        tau = np.array([np.mean([r['tau'] for r in e['reps']]) for e in rows])
        obs = sw > 0
        ax.plot(d, np.where(obs, sw, FLOOR), color=c, lw=2.6, marker=mk,
                ms=9, label=f'{scheme}  mode changes', zorder=3)
        if (~obs).any():
            ax.plot(d[~obs], np.full((~obs).sum(), FLOOR), marker=mk, ms=9,
                    mfc='white', mec=c, mew=2.2, ls='none', zorder=4)
            ax.annotate('none\nobserved', (d[~obs][0], FLOOR), ha='right',
                        va='bottom', fontsize=12.5, color=c,
                        fontweight='bold', xytext=(-8, 8),
                        textcoords='offset points')
        ax.plot(d, tau, color=c, lw=2.0, ls=':', marker=mk, ms=6, alpha=0.6,
                label=f'{scheme}  round trips ' + r'$\tau$')
    ax.set_yscale('log')
    ax.set_ylim(FLOOR * 0.55, 1.1)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlabel(r'dimension $d$      ($s_1=0.3,\ s_2=0.4$)')
    ax.set_ylabel('rate per scan')
    ax.legend(loc='lower left', fontsize=11.5, ncol=1, handlelength=1.8,
              borderpad=0.2, labelspacing=0.3)
    grid(ax)
    fig.tight_layout()
    save(fig, '6_dimension')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
    fns = {'1': fig_index_process, '2': fig_speedup, '3': fig_failure,
           '4': fig_closed_form, '5': fig_sweep, '6': fig_dimension}
    for k in which:
        fns[k]()
