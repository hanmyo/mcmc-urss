"""Two sweeps, selected by argv[1]:  'dim' or 'inhom'."""
import json, sys, time
import numpy as np
from pt_sim import (parallel_tempering, mode_A_occupancy, mode_switches,
                    communication_barrier, tempered_mode_weight)

N_CHAINS, HOTTEST, SCANS, EXPLORE = 100, 0.01, 20000, 8
beta = np.geomspace(1, HOTTEST, N_CHAINS)


def run(d, s1, s2, scheme, seed):
    r = parallel_tempering(beta, SCANS, EXPLORE, d=d, s1=s1, s2=s2,
                           scheme=scheme, seed=seed)
    tr = r["trace"]
    return dict(tau=float(r["round_trip_rate"]),
                switch_rate=mode_switches(tr, s1, s2) / SCANS,
                occ=float(mode_A_occupancy(tr[int(.2 * SCANS):], s1, s2)),
                Lambda=communication_barrier(r["swap"]))


def sweep(name, grid, seeds):
    out, t0 = [], time.time()
    for d, s1, s2 in grid:
        for scheme in ["SEO", "DEO"]:
            reps = [run(d, s1, s2, scheme, sd) for sd in seeds]
            out.append(dict(d=d, s1=s1, s2=s2, scheme=scheme, reps=reps,
                            w=float(tempered_mode_weight(HOTTEST, d, s1, s2))))
            print("%s d=%d s2=%.3f %s: tau=%.4f switch=%.4f occ=%.3f +/- %.3f "
                  "[%ds]" % (name, d, s2, scheme,
                             np.mean([r['tau'] for r in reps]),
                             np.mean([r['switch_rate'] for r in reps]),
                             np.mean([r['occ'] for r in reps]),
                             np.std([r['occ'] for r in reps]),
                             time.time() - t0), flush=True)
            json.dump({"config": dict(scans=SCANS, seeds=seeds, n_chains=N_CHAINS),
                       name: out}, open("res_%s.json" % name, "w"), indent=1)
    print("DONE", time.time() - t0, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'dim':
        sweep('dim', [(d, 0.30, 0.40) for d in [5, 10, 15, 20, 25, 30]],
              [11, 22, 33, 44])
    else:
        sweep('inhom', [(30, 0.30, s2) for s2 in
                        [0.30, 0.325, 0.35, 0.375, 0.40]],
              [41, 42, 43, 44, 45, 46])
