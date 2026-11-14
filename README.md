# Polymarket FAA outage attribution forecaster

Entry for make.fun competition `polymarket-faa-attribution-02`: forecast the
Yes price of each of the 4 FAA outage attribution markets at every minute of
the 4 hours after a checkpoint, as 5 quantiles (0.05 / 0.25 / 0.5 / 0.75 /
0.95) per minute.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Python 3.13, standard library only, no network, deterministic, ~60 ms/case.

## Model

Per market, fitted only from that market's own prices in the case file:

- **Median = last price.** Prediction-market prices are martingales; the last
  price is the best point forecast. (Tested mean-reversion of the median on
  the practice backtest: far worse.)
- **Spread:** `sd(h) = s * sqrt((1 - exp(-2 k h)) / (2k) + a h)` with
  `s` = RMS of the last 120 one-minute price changes (floor 0.0005, cap 0.01),
  `k = 0.02`/min (Ornstein–Uhlenbeck variance saturation: a bounded price
  cannot diffuse like `sqrt(h)` for 4 hours), `a = 0.05` (small random-walk
  share as insurance against a genuine regime shift).
- **Quantiles:** `clamp(x + z_tau * sd(h), 0, 1)`, rounded to 6 decimals.
- **FAA coherence guard:** "attributed to Iran" implies "attributed to a
  foreign actor", so P(Iran) ≤ P(foreign actor). If the two checkpoints
  violate that, both medians are set to their midpoint (slug-gated; never
  fires on other events).

## Backtest (practice cases, answers fetched from Polymarket's public CLOB price history)

| model | mean pinball loss |
|---|---|
| house baseline (fixed σ = 0.0025) | 0.3636 |
| **this entry** | **0.2697** |

Beats the baseline in **every one of the six practice cases**
(0.470 / 0.317 / 0.344 / 0.232 / 0.206 / 0.116 vs baseline
0.54 / 0.38 / 0.41 / 0.37 / 0.30 / 0.24). The scoring pipeline was validated
by reproducing the house baseline's published practice loss (0.363589) exactly.

Caveat: the practice is the "Next CEO of NVIDIA?" event, not the FAA family,
so the final margin will differ. The model uses no per-market constants and
no fitted drift, so it transfers by construction — but the FAA markets are
news-driven and jump-prone, which is why the spread adapts to each market's
recent realized volatility instead of using a fixed sigma.

## Files

- `main.py` — the entry (only file the grader runs).
- `LICENSE` — MIT.
- `dev/` — backtest scripts and fetched practice answers (development only;
  **exclude from the submitted repo**).
