"""
Vectorised SEO / DEO parallel tempering on a two-component Gaussian mixture.

Reproduces the sampler in the notebook (early version) but in log space and vectorised
over chains, so a run that took hours takes seconds.

Target:  pi(x) = 0.5 N(+1_d, s1^2 I) + 0.5 N(-1_d, s2^2 I)
"""
import numpy as np
from numpy.random import default_rng


def make_log_target(d, s1, s2):
    nu1 = np.ones(d)
    nu2 = -np.ones(d)
    c1 = -0.5 * d * np.log(2 * np.pi * s1 * s1) + np.log(0.5)
    c2 = -0.5 * d * np.log(2 * np.pi * s2 * s2) + np.log(0.5)

    def log_target(x):
        """x: (..., d) -> (...,)"""
        q1 = c1 - 0.5 * np.sum((x - nu1) ** 2, axis=-1) / (s1 * s1)
        q2 = c2 - 0.5 * np.sum((x - nu2) ** 2, axis=-1) / (s2 * s2)
        m = np.maximum(q1, q2)
        return m + np.log(np.exp(q1 - m) + np.exp(q2 - m))

    return log_target


def round_trip_stats(index_process):
    """index_process[i, r] = chain occupied by replica r at end of scan i.
    Chain 0 = target (beta=1), chain N-1 = reference. Returns per-replica counts."""
    n_samples, num_temps = index_process.shape
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
                       radius=0.15, seed=0, record_coord=1, x0=None):
    """scheme: 'SEO' (random even/odd) or 'DEO' (deterministic alternation)."""
    rng = default_rng(seed)
    log_target = make_log_target(d, s1, s2)
    N = len(beta)
    beta = np.asarray(beta, dtype=float)

    x = -np.ones((N, d)) if x0 is None else x0.copy()
    lp = log_target(x)                       # (N,)

    step = radius / np.sqrt(beta)            # temperature-scaled proposal scale
    step = step[:, None]

    n_accept = np.zeros(N)
    n_swap_prop = np.zeros(N - 1)
    n_swap_acc = np.zeros(N - 1)

    rep_at = np.arange(N)                    # rep_at[j] = replica label on chain j
    index_process = np.zeros((n_scans, N), dtype=np.int32)
    trace = np.zeros(n_scans)                # coordinate of the target chain

    for i in range(n_scans):
        # ---- exploration (local MH at each temperature) ----
        for _ in range(n_explore):
            y = x + step * rng.standard_normal((N, d))
            lp_y = log_target(y)
            log_alpha = beta * (lp_y - lp)
            acc = np.log(rng.random(N)) < log_alpha
            x[acc] = y[acc]
            lp[acc] = lp_y[acc]
            n_accept += acc

        # ---- swap move ----
        if scheme == 'SEO':
            start = int(rng.random() > 0.5)
        else:
            start = i % 2
        j = np.arange(start, N - 1, 2)
        if j.size:
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
        trace[i] = x[0, record_coord]

    n_round_trips = round_trip_stats(index_process)
    return dict(
        trace=trace,
        accept=n_accept / (n_scans * n_explore),
        swap=n_swap_acc / np.maximum(n_swap_prop, 1),
        round_trip_rate=n_round_trips.sum() / n_scans,
        index_process=index_process,
    )


def mode_A_occupancy(trace, s1, s2):
    """Fraction of target-chain samples in mode A (+1), using the notebook's
    variance-weighted midpoint threshold."""
    thresh = 1 - 2 * (s1 / (s1 + s2))
    return (trace > thresh).mean()


def tempered_mode_weight(beta, d, s1, s2):
    """Closed form: w_beta(A) = 1 / (1 + (s2/s1)^{d(1-beta)}) under the
    Laplace/well-separated-modes approximation."""
    beta = np.asarray(beta, dtype=float)
    log_odds = d * (1 - beta) * np.log(s1 / s2)
    return 1.0 / (1.0 + np.exp(-log_odds))


def communication_barrier(swap_rate):
    return float(np.sum(1 - np.asarray(swap_rate)))


def mode_switches(trace, s1, s2, margin=0.5):
    """Hysteresis crossing count: a switch is registered only once the chain is
    convincingly inside the other mode, so within-mode wiggle is not counted."""
    hiA, hiB = 1 - margin * s1, -1 + margin * s2
    state, n = None, 0
    for v in trace:
        if v > hiA:
            if state == 'B':
                n += 1
            state = 'A'
        elif v < hiB:
            if state == 'A':
                n += 1
            state = 'B'
    return n
