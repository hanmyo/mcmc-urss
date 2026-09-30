# Poster code

A vectorised, log-space version of the sampler, plus the sweeps and figures
used on the poster.

I wrote this after finding two problems with the first version of the notebook:

1. **Underflow.** The notebook used to form `(pi(y)/pi(x))**beta` from raw
   densities. At `d = 30`, `pi(x)` is exactly `0.0` on any chain with `beta`
   below about 0.02 (the hottest ~15% of a 100-chain ladder). The ratio then
   becomes `0/0 = nan`, and since `min(1, nan)` is `1` in Python those chains
   accepted every proposal. Everything here works with log densities
   instead: `log(u) < beta * (lp_y - lp_x)`.
2. **Speed.** Exploration moves are applied to all chains at once as an
   `(N, d)` array. A run that took ~15 hours in the old notebook takes ~16
   seconds here, which is what made the sweeps possible.

Apart from that the algorithm is unchanged: `radius / sqrt(beta)` proposals,
even/odd swaps, the same index process bookkeeping and round trip definition.
(The current notebook is also in log space and vectorised, but it uses a
different reference and proposal scaling, so its numbers don't match these
exactly.)

| file | |
|---|---|
| `pt_sim.py` | sampler, round trip and mode switch counting, closed-form tempered mode weight |
| `make_traces.py` | the long single runs the figures need, saved to `traces.npz` |
| `sweep.py` | `python sweep.py inhom` (vary `s2` at `d = 30`) and `python sweep.py dim` (vary `d`), saved to `res_*.json` |
| `figures.py` | the six poster figures, `python figures.py [1-6]` |

To regenerate the figures, run these from this folder:

```bash
python make_traces.py      # ~1 min
python sweep.py inhom      # ~17 min
python sweep.py dim        # ~11 min
python figures.py
```

and copy the resulting `fig_*.png` into `../latex/figs/`.

## Where the mode imbalance comes from

With well-separated modes, a Laplace approximation of the tempered density
gives

    pi_beta(A) / pi_beta(B) = (s1/s2)^(d(1-beta))

Near mode i, `pi(x)^beta` is proportional to
`[0.5 (2 pi s_i^2)^(-d/2)]^beta exp(-beta |x - nu_i|^2 / (2 s_i^2))`, which is
a Gaussian with variance `s_i^2 / beta`. Integrating gives a mass proportional to
`(s_i^2)^(-d beta/2) (s_i^2/beta)^(d/2)`. The `beta` factors cancel in the
ratio, leaving `(s1/s2)^(d(1-beta))`.

At `beta = 1` the ratio is 1, as it should be. At `beta = 0.01`, `d = 30` and
`s1/s2 = 3/4` it is about `2e-4`, so the hottest chain almost never sees mode
A, and the imbalance grows exponentially in `d`.
