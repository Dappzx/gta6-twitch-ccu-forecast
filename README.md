# GTA VI Twitch CCV forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dappzx/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dappzx/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast how
many people watch Grand Theft Auto VI live on Twitch in each hour of its
release day (2026-11-19 UTC), hours 2–24, as five quantiles per hour.

## How it works

`main.py` (Python 3.13, standard library only) reads one case
`/input/<id>.in` and writes `/output/<id>.out`. On every run it fits the
model fresh from the case's own `comparables` — 42 past big game launches'
hourly Twitch curves.

The key fact the model uses: a launch day's shape depends strongly on its
scale. Across the comparable launches, corr(log hour-1 viewers, log
evening-ratio) is about −0.95 — a launch that is already huge at midnight
stays flat through the day, while a launch with a tiny hour 1 explodes
later as regions unlock. So for each hour the model predicts the
log-ratio to hour 1 two ways and blends them (log space, 0.7 / 0.3):

- **local-linear**: per hour, regress log-ratio on log(hour-1) over the 12
  nearest launches by log(hour-1), median-centred on the residuals;
- **kNN**: per hour, the median log-ratio of the 5 nearest launches.

The quantile spread is the root-mean-square of the two components'
residual spreads. The effective log(hour-1) is capped at the pool maximum
plus ln(1.5), so a launch bigger than anything in the pool is forecast
from the biggest launches' shapes instead of extrapolating the log-log
slope past the data. Early-access-contaminated launches and the target's
own pre-release history are excluded from the pool. With too few
comparables it falls back to a global log-log regression, then to the
median ratio curve, then to the flat house baseline — it never crashes on
a case.

## Local results (practice cases, mean pinball loss on log scale)

| case | this entry | house baseline |
| ---- | ---------: | -------------: |
| p00 Marvel's Wolverine | 0.219 | 0.744 |
| p01 Red Dead Redemption 2 | 0.155 | 0.287 |
| p02 God of War Ragnarök | 0.117 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.199 | 0.303 |
| **mean** | **0.173** | **0.387** |

Deterministic: two runs on the same input produce byte-identical output.
Runs in ~0.05 s per case (limit 8 s). No network, no randomness.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
