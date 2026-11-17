"""gta6-twitch-ccu-forecast-01 entry: launch-day Twitch CCV forecaster.

Forecasts Grand Theft Auto VI's Twitch category viewers for hours 2-24 of
release day (2026-11-19 UTC) as five quantiles per hour, from everything
recorded up to the end of hour 1.

Model (fit fresh on every run from the case's own `comparables`, stdlib only):
  For each past big launch, take its release-day hourly curve and form the
  log-ratio of each hour's viewers to its hour-1 viewers. The shape of a
  launch day depends strongly on its scale: the bigger hour 1, the flatter
  the day (corr(log hour-1, log evening-ratio) ~ -0.95 across the pool), so
  the model conditions on hour 1's print two ways and blends them:
    - local-linear: per hour, regress log-ratio on log(hour-1) over the 12
      nearest launches by log(hour-1), median-centred on the residuals;
    - kNN: per hour, the median log-ratio of the 5 nearest launches.
  Blend (log space): 0.7 * local-linear + 0.3 * kNN for the median, and the
  root-mean-square of the two residual spreads for the quantile width.
  The effective log(hour-1) is capped at the pool maximum plus ln(1.5), so a
  launch bigger than anything in the pool is forecast from the biggest
  launches' shapes instead of extrapolating the log-log slope past the data.

  Pool cleaning: the target's own pre-release history entry and any
  early-access-contaminated launch are excluded; a launch needs a positive
  hour-1 print. If too few comparables survive, fall back to a global
  log-log regression, then to the unconditional median ratio curve, then to
  the house baseline (flat at hour-1 with a widening spread).

  Deterministic: no randomness, no wall-clock, no network; the same input
  file always produces the same output bytes. Runs in well under 1 s.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. It reads every <id>.in in the input
directory (one case at a time) and writes <id>.out to the output directory:
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, 23 rows.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Local-linear neighbourhood size, kNN neighbourhood size, blend weight.
K_LL = 12
K_NN = 5
W_LL = 0.7
# Cap on extrapolation: effective log(hour-1) never exceeds the pool's
# maximum log(hour-1) by more than this (ln 1.5).
LOG_CAP = math.log(1.5)


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (same as the starter's round2)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_day(comp):
    """Hour -> viewers for the 24 UTC hours of the comparable's release day."""
    try:
        ds = parse_ts(comp["day_start"])
    except Exception:
        return {}
    out = {}
    for p in comp.get("hourly") or []:
        try:
            t = parse_ts(p["from"])
            v = p["viewers"]
        except Exception:
            continue
        if v is None or v <= 0:
            continue
        if ds <= t < ds + timedelta(hours=24):
            out[int((t - ds).total_seconds() // 3600) + 1] = v
    return out


def clean_pool(comparables):
    """(comparable, release-day dict) pairs usable for fitting."""
    out = []
    for c in comparables or []:
        name = c.get("name") or ""
        if "Grand Theft Auto VI" in name:
            continue  # the target's own pre-release history, not a launch
        if "early-access" in (c.get("unlock") or ""):
            continue  # early-access-contaminated launch day
        rd = release_day(c)
        if 1 not in rd or rd[1] <= 0:
            continue
        out.append((c, rd))
    return out


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def baseline_quantiles(p1, h):
    """House baseline: flat at hour-1 with a widening log spread."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]


def median_ratio_forecast(p1, pool):
    """Unconditional median log-ratio curve with residual spreads."""
    out = {}
    for h in range(2, 25):
        lr = [math.log(rd[h] / rd[1]) for _, rd in pool if h in rd]
        if not lr:
            out[h] = baseline_quantiles(p1, h)
            continue
        m = statistics.median(lr)
        s = statistics.pstdev([x - m for x in lr]) or 0.15
        out[h] = [round2(max(1.0, p1 * math.exp(m + s * z))) for z in Z]
    return out


def global_loglog_forecast(p1, pool):
    """Global per-hour log-ratio ~ log(hour-1) regression."""
    lh1 = math.log(p1)
    out = {}
    for h in range(2, 25):
        xs, ys = [], []
        for _, rd in pool:
            if h in rd:
                xs.append(math.log(rd[1]))
                ys.append(math.log(rd[h] / rd[1]))
        if len(xs) < 3:
            out[h] = baseline_quantiles(p1, h)
            continue
        a, b = ols(xs, ys)
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        s = statistics.pstdev(resid) or 0.15
        mu = a + b * lh1
        out[h] = [round2(max(1.0, p1 * math.exp(mu + s * z))) for z in Z]
    return out


def local_linear_component(p1, pool, cap_lh1):
    """Per-hour (mu, sigma) from a local-linear fit on the K_LL nearest launches."""
    lh1 = min(math.log(p1), cap_lh1)
    nbrs = sorted(pool, key=lambda cr: abs(math.log(cr[1][1]) - math.log(p1)))
    nbrs = nbrs[:K_LL]
    mu, sigma = {}, {}
    for h in range(2, 25):
        xs, ys = [], []
        for _, rd in nbrs:
            if h in rd:
                xs.append(math.log(rd[1]))
                ys.append(math.log(rd[h] / rd[1]))
        if len(xs) < 3:
            mu[h] = None
            sigma[h] = None
            continue
        a, b = ols(xs, ys)
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        mu[h] = a + b * lh1 + statistics.median(resid)
        sigma[h] = statistics.pstdev(resid) or 0.15
    return mu, sigma


def knn_component(p1, pool):
    """Per-hour (mu, sigma) from the median log-ratio of the K_NN nearest launches."""
    nbrs = sorted(pool, key=lambda cr: abs(math.log(cr[1][1]) - math.log(p1)))
    nbrs = nbrs[:K_NN]
    mu, sigma = {}, {}
    for h in range(2, 25):
        lr = [math.log(rd[h] / rd[1]) for _, rd in nbrs if h in rd]
        if not lr:
            mu[h] = None
            sigma[h] = None
            continue
        m = statistics.median(lr)
        mu[h] = m
        sigma[h] = statistics.pstdev([x - m for x in lr]) or 0.15
    return mu, sigma


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None or first.get("viewers") is None:
        # No hour-1 print: fall back to the last pre-checkpoint hourly point.
        hist = case.get("history_hourly") or []
        p1 = max(1, hist[-1]["viewers"]) if hist else 1
    else:
        p1 = max(1, first["viewers"])

    pool = clean_pool(case.get("comparables"))
    if len(pool) >= K_LL:
        cap_lh1 = max(math.log(rd[1]) for _, rd in pool) + LOG_CAP
        mu_ll, sig_ll = local_linear_component(p1, pool, cap_lh1)
        mu_nn, sig_nn = knn_component(p1, pool)
        hours = []
        for row in case["forecast"]:
            h = row["hour"]
            if mu_ll.get(h) is None or mu_nn.get(h) is None:
                hours.append({"hour": h, "q": baseline_quantiles(p1, h)})
                continue
            mu = W_LL * mu_ll[h] + (1 - W_LL) * mu_nn[h]
            sigma = math.sqrt(W_LL * sig_ll[h] ** 2 + (1 - W_LL) * sig_nn[h] ** 2)
            qs = []
            prev = 0.0
            for z in Z:
                q = round2(min(1e9, max(1.0, p1 * math.exp(mu + sigma * z))))
                q = max(q, prev)  # keep quantiles non-decreasing after rounding
                qs.append(q)
                prev = q
            hours.append({"hour": h, "q": qs})
        return {"hours": hours}
    if len(pool) >= 5:
        full = global_loglog_forecast(p1, pool)
        return {"hours": [{"hour": row["hour"], "q": full[row["hour"]]}
                          for row in case["forecast"]]}
    if len(pool) >= 3:
        med = median_ratio_forecast(p1, pool)
        return {"hours": [{"hour": row["hour"], "q": med[row["hour"]]}
                          for row in case["forecast"]]}
    return {"hours": [{"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
                      for row in case["forecast"]]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash on a case: fall back to the house baseline shape.
            first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
            p1 = max(1, first["viewers"]) if first and first.get("viewers") else 1
            forecast = {"hours": [{"hour": row["hour"],
                                   "q": baseline_quantiles(p1, row["hour"])}
                                  for row in case.get("forecast", [])]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
