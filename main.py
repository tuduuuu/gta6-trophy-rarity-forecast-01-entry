"""gta6-trophy-rarity-forecast-01 entry: trend + dilution forecaster (Python 3.13, stdlib only).

For each trophy, fit a least-squares line of logit(earn rate) against
x = ln(1 + hours since 2026-11-19T00:00:00Z) over the last 24 hours of
Exophase reads, extrapolate it to the case target for the median, and widen
the quantiles with a spread that combines a random-walk term
(0.5 logit units per sqrt day) with the fit's residual spread, floored at
0.5 logit units so a flat launch-night history still carries real
uncertainty. A player-growth dilution term pushes medians down when the
tracked-player count is growing (new players have earned few trophies).

I/O contract (see kit RULES.txt): read every <id>.in in
$MAKEFUN_INPUT_DIR (else /input/), write one <id>.out per case to
$MAKEFUN_OUTPUT_DIR (else /output/):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, five finite numbers 0-100, non-decreasing, <= 65,536 bytes.
Deterministic: no randomness, files processed in name order.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RW_SIGMA = 0.5          # random-walk spread, logit units per sqrt day
MIN_SIGMA = 0.5         # floor on total logit spread
DILUTION_K = 0.5        # weight on the player-growth dilution term
WINDOW = timedelta(hours=24)
RATE_MIN = 0.01
RATE_MAX = 99.99


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp, tolerating a trailing Z or offset."""
    if isinstance(text, datetime):
        return text
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clamp_rate(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return min(RATE_MAX, max(RATE_MIN, v))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    # Clamp so a wild spread can never overflow math.exp: the result is
    # already indistinguishable from 0 or 100 long before this bound.
    lg = min(700.0, max(-700.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (same convention as the house starter)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def hours_since_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def x_of(dt):
    return math.log1p(hours_since_epoch(dt))


def least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line ys ~ xs."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def fit_window(history, checkpoint):
    """Reads inside [checkpoint - 24h, checkpoint]; all reads if fewer."""
    windowed = [r for r in history if checkpoint - WINDOW <= r["_at"] <= checkpoint]
    return windowed if windowed else list(history)


def forecast_trophy(points, x_target, horizon_days, players_trend, window_last_x):
    """points: list of (x, rate) usable reads for one trophy, oldest first.
    players_trend: (intercept, slope) of ln(players) vs x over the window,
    or None. window_last_x: x of the window's last read, the anchor for the
    dilution extrapolation (a trophy whose own last usable read is stale
    must not get an inflated dilution). Returns the five quantiles."""
    if not points:
        median = RATE_MIN
        sigma = MIN_SIGMA
    else:
        xs = [p[0] for p in points]
        ys = [logit(p[1]) for p in points]
        if len(points) >= 3:
            intercept, slope = least_squares(xs, ys)
            median_logit = intercept + slope * x_target
            resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
            dof = max(len(resid) - 2, 1)
            resid_sigma = math.sqrt(sum(r * r for r in resid) / dof)
        else:
            median_logit = ys[-1]
            resid_sigma = 0.0
        sigma = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2 + resid_sigma ** 2)
        sigma = max(sigma, MIN_SIGMA)
        if players_trend is not None:
            p_intercept, p_slope = players_trend
            # Extrapolated log-growth of tracked players to the target,
            # relative to the window's last read (not this trophy's last
            # usable read, which may be stale).
            log_growth = (p_intercept + p_slope * x_target) - (p_intercept + p_slope * window_last_x)
            if log_growth > 0.0:
                # Dilution only ever pushes rates down, and never by more
                # than 1 logit unit, so a fast-growing player base cannot
                # collapse a trophy's median to the floor on its own.
                median_logit -= min(DILUTION_K * log_growth, 1.0)
        median = inv_logit(median_logit)
        median = min(RATE_MAX, max(RATE_MIN, median))
    qs = []
    prev = None
    for z in Z:
        q = round4(inv_logit(logit(median) + z * sigma))
        q = min(100.0, max(0.0, q))
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def fallback_forecast(case):
    """Last rates with the house-baseline random-walk spread; never crashes."""
    horizon_days = max(case.get("horizon_minutes", 0) or 0, 0) / 1440.0
    sigma = RW_SIGMA * math.sqrt(horizon_days)
    history = case.get("history") or []
    last_rates = history[-1].get("rates") if history else None
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = None
        if last_rates is not None and i < len(last_rates):
            rate = clamp_rate(last_rates[i])
        if rate is None:
            rate = RATE_MIN
        lg = logit(rate)
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(lg + z * sigma))
            q = min(100.0, max(0.0, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    if not history or not trophies:
        return fallback_forecast(case)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0) or 0, 0) / 1440.0
    x_target = x_of(target)

    reads = []
    for raw in history:
        try:
            at = parse_time(raw["at"])
        except (KeyError, TypeError, ValueError):
            continue
        reads.append({"_at": at, "players": raw.get("players_tracked"),
                      "rates": raw.get("rates") or []})
    if not reads:
        return fallback_forecast(case)
    reads.sort(key=lambda r: r["_at"])
    window = fit_window(reads, checkpoint)

    # Player-growth trend over the fitted window (log-linear in x).
    players_trend = None
    px, py = [], []
    for r in window:
        p = r["players"]
        if p is not None and float(p) > 0:
            px.append(x_of(r["_at"]))
            py.append(math.log(float(p)))
    if len(px) >= 2:
        players_trend = least_squares(px, py)

    rows = []
    window_last_x = x_of(window[-1]["_at"])
    for i, trophy in enumerate(trophies):
        points = []
        for r in window:
            rates = r["rates"]
            if i < len(rates):
                rate = clamp_rate(rates[i])
                if rate is not None:
                    points.append((x_of(r["_at"]), rate))
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(points, x_target, horizon_days,
                                          players_trend, window_last_x)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # A broken case file still gets a valid-shaped fallback if the
            # trophies can be read at all; otherwise an empty trophy list.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_forecast(case)
            except Exception:
                forecast = {"trophies": []}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
