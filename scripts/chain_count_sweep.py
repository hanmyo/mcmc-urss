"""Chain-count sweep for SEO and DEO parallel tempering.

Checks the round trip rate against the theory of Syed et al. (2022) for
N in {25, 50, 100, 200, 400} chains, 3 seeds each. Results go to
../results/chain_count_sweep.json.

Same algorithm as the notebook, with two changes:
  * log densities. At d=30, pi(x) underflows to 0.0 for beta below ~0.02, so
    (pi(y)/pi(x))**beta becomes 0/0 = nan and min(1, nan) returns 1: the
    hottest chains would accept every move.
  * vectorised over chains, roughly 200x faster than a per-chain loop.
"""
import numpy as np, time, json
from pathlib import Path
from numpy.random import default_rng

# ------------------------------------------------ parameters to sweep
D, S1, S2   = 30, 0.3, 0.4      # benchmark target
HOTTEST     = 0.01              # beta_min
RADIUS      = 0.15              # proposal radius
T_EXPLORE   = 8                 # exploration moves per scan
N_SCANS     = 20_000            # scans per run
N_LIST      = [25, 50, 100, 200, 400]
SEEDS       = [11, 22, 33]      # 3 independent repeats
# 5 chain counts x 2 schemes x 3 seeds = 30 runs


def make_log_target(d, s1, s2):
    """log of  0.5 N(+1_d, s1^2 I) + 0.5 N(-1_d, s2^2 I),  computed stably."""
    nu1, nu2 = np.ones(d), -np.ones(d)
    c1 = -0.5 * d * np.log(2 * np.pi * s1 * s1) + np.log(0.5)
    c2 = -0.5 * d * np.log(2 * np.pi * s2 * s2) + np.log(0.5)

    def log_target(x):                      # x: (..., d) -> (...,)
        q1 = c1 - 0.5 * np.sum((x - nu1) ** 2, axis=-1) / (s1 * s1)
        q2 = c2 - 0.5 * np.sum((x - nu2) ** 2, axis=-1) / (s2 * s2)
        m = np.maximum(q1, q2)
        return m + np.log(np.exp(q1 - m) + np.exp(q2 - m))
    return log_target


def round_trip_stats(index_process):
    """A round trip is target -> reference -> target by one
    replica. Returns per-replica counts."""
    n_scans, num_temps = index_process.shape
    n_round_trips = np.zeros(num_temps, dtype=int)
    for r in range(num_temps):
        traj = index_process[:, r]
        visits = np.flatnonzero((traj == 0) | (traj == num_temps - 1))
        last_end = -1
        for i in visits:
            end = 0 if traj[i] == 0 else 1
            if end == 0 and last_end == 1:
                n_round_trips[r] += 1
            last_end = end
    return n_round_trips


def parallel_tempering(beta, n_scans, n_explore, d, s1, s2, scheme,
                       radius=RADIUS, seed=0):
    rng = default_rng(seed)
    log_target = make_log_target(d, s1, s2)
    N = len(beta)
    beta = np.asarray(beta, float)

    x = -np.ones((N, d))                     # all chains start in mode B
    lp = log_target(x)
    step = (radius / np.sqrt(beta))[:, None]  # 1/sqrt(beta) scaling

    n_accept = np.zeros(N)
    n_swap_prop = np.zeros(N - 1)
    n_swap_acc = np.zeros(N - 1)
    rep_at = np.arange(N)
    index_process = np.zeros((n_scans, N), dtype=np.int32)

    for i in range(n_scans):
        # ---- exploration: local MH at every temperature ----
        for _ in range(n_explore):
            y = x + step * rng.standard_normal((N, d))
            lp_y = log_target(y)
            # log form of  min(1, (pi(y)/pi(x))**beta)
            acc = np.log(rng.random(N)) < beta * (lp_y - lp)
            x[acc], lp[acc] = y[acc], lp_y[acc]
            n_accept += acc

        # ---- swap: the ONLY line where SEO and DEO differ ----
        start = int(rng.random() > 0.5) if scheme == 'SEO' else i % 2
        j = np.arange(start, N - 1, 2)
        if j.size:
            # log form of the swap ratio A_j
            log_ratio = (beta[j + 1] - beta[j]) * (lp[j] - lp[j + 1])
            acc = np.log(rng.random(j.size)) < log_ratio
            n_swap_prop[j] += 1
            n_swap_acc[j] += acc
            js = j[acc]
            if js.size:
                x[[js, js + 1]] = x[[js + 1, js]]
                lp[[js, js + 1]] = lp[[js + 1, js]]
                rep_at[[js, js + 1]] = rep_at[[js + 1, js]]

        index_process[i, rep_at] = np.arange(N)

    swap = n_swap_acc / np.maximum(n_swap_prop, 1)
    return dict(tau=round_trip_stats(index_process).sum() / n_scans,
                swap=swap, accept=n_accept / (n_scans * n_explore))


# ------------------------------------------------ time estimate first
_t = time.time()
parallel_tempering(np.geomspace(1, HOTTEST, 50), 200, T_EXPLORE,
                   D, S1, S2, 'DEO', seed=0)
_unit = (time.time() - _t) / (50 * 200)          # sec per chain-scan
_total = _unit * sum(N_LIST) * 2 * len(SEEDS) * N_SCANS
print('calibration: %.1f us per chain-scan  ->  whole sweep %.0f min\n'
      % (1e6 * _unit, _total / 60), flush=True)

# ------------------------------------------------ run the sweep
rows, t0 = [], time.time()
for N in N_LIST:
    beta = np.geomspace(1, HOTTEST, N)
    rec = {'N': N}
    for scheme in ('SEO', 'DEO'):
        taus, lams, Es = [], [], []
        for sd in SEEDS:
            r = parallel_tempering(beta, N_SCANS, T_EXPLORE, D, S1, S2,
                                   scheme, seed=sd)
            sw = np.asarray(r['swap'])
            taus.append(r['tau'])
            lams.append(float((1 - sw).sum()))                    # Lambda-hat
            Es.append(float(np.sum((1 - sw) / np.maximum(sw, 1e-12))))  # E
        rec[scheme] = dict(tau=float(np.mean(taus)), tau_sd=float(np.std(taus)),
                           Lam=float(np.mean(lams)), E=float(np.mean(Es)))
    E_bar = 0.5 * (rec['SEO']['E'] + rec['DEO']['E'])
    rec['E'] = E_bar
    rec['tau_SEO_theory'] = 1 / (2 * N + 2 * E_bar)
    rec['tau_DEO_theory'] = 1 / (2 + 2 * E_bar)
    rows.append(rec)
    print('N=%3d  Lam %5.2f/%5.2f  E %5.2f | SEO %.5f (th %.5f) | '
          'DEO %.5f (th %.5f)  [%.0f min]'
          % (N, rec['SEO']['Lam'], rec['DEO']['Lam'], E_bar,
             rec['SEO']['tau'], rec['tau_SEO_theory'],
             rec['DEO']['tau'], rec['tau_DEO_theory'],
             (time.time() - t0) / 60), flush=True)
    json.dump(rows, open(Path(__file__).parent.parent / 'results' / 'chain_count_sweep.json', 'w'), indent=1)

print('\ndone in %.0f min -> results/chain_count_sweep.json\n' % ((time.time() - t0) / 60))

# ------------------------------------------------ LaTeX rows for the poster
print('%% paste these rows into the table')
for r in rows:
    s, dd = r['SEO'], r['DEO']
    meas_s = ('$<0.1^{\\dagger}$' if s['tau'] * N_SCANS < 1
              else '$%.2f\\pm%.2f$' % (1e3 * s['tau'], 1e3 * s['tau_sd']))
    print('%3d & %5.2f & %5.2f & %5.2f & %5.2f & %s & %4.1f & $%.2f\\pm%.2f$ \\\\'
          % (r['N'], s['Lam'], dd['Lam'], r['E'], 1e3 * r['tau_SEO_theory'],
             meas_s, 1e3 * r['tau_DEO_theory'],
             1e3 * dd['tau'], 1e3 * dd['tau_sd']))
lam = rows[-1]['DEO']['Lam']
print('%% asymptote: tau_bar = 1/(2+2*Lambda) = %.1f x 10^-3' % (1e3 / (2 + 2 * lam)))
