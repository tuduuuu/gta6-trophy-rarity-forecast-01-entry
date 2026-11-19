# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tuduuuu/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tuduuuu/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python stdlib-only forecaster for GTA VI trophy earn rates on Exophase at 2026-11-26T00:00:00Z. For each of the 55 trophies it fits a least-squares line of logit(earn rate) against ln(1 + hours since 2026-11-19T00:00Z) over the last 24 hours of reads, extrapolates the line to the target for the median, and widens the five quantiles (5/25/50/75/95%) with a spread combining a random-walk term (0.5 logit units per sqrt day) and the fit's residual spread, floored at 0.5 logit units so a flat launch-night history still carries real uncertainty. A player-growth dilution term pulls medians down when the tracked-player count is growing, since new players have earned few trophies. The program reads every case in the input directory and writes one deterministic forecast file per case; it never crashes on a case, falling back to last rates with the house-baseline spread.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=43zj5sq1&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
