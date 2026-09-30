"""Traces needed for the poster figures, cached to npz."""
import numpy as np
from pt_sim import parallel_tempering

beta = np.geomspace(1, 0.01, 100)
store = {}

# index process, equal chain count, inhomogeneous target
for scheme in ['SEO', 'DEO']:
    r = parallel_tempering(beta, 3000, 8, d=30, s1=0.3, s2=0.4,
                           scheme=scheme, seed=7)
    store[f'idx_{scheme}'] = r['index_process'][:, :6].astype(np.int16)
    store[f'swap_{scheme}'] = r['swap']
    store[f'tau_{scheme}'] = r['round_trip_rate']
    print(scheme, 'tau', r['round_trip_rate'], flush=True)

# marginal traces: homogeneous vs inhomogeneous, DEO
for tag, (s1, s2) in {'homog': (0.3, 0.3), 'inhomog': (0.3, 0.4)}.items():
    for scheme in ['SEO', 'DEO']:
        r = parallel_tempering(beta, 20000, 8, d=30, s1=s1, s2=s2,
                               scheme=scheme, seed=5)
        store[f'trace_{tag}_{scheme}'] = r['trace']
        print(tag, scheme, 'tau', r['round_trip_rate'], flush=True)

np.savez_compressed('traces.npz', **store)
print('saved')
