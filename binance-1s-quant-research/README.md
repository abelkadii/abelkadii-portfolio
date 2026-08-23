# Binance 1-Second Turning-Point Research

A compact, reproducible quantitative-research project for studying short-horizon turning points in Binance spot markets from **1-second OHLCV candles**.

The goal is not to claim a profitable strategy. The goal is to demonstrate a research process that is useful for building one:

- acquire high-frequency market data;
- construct strictly causal features;
- define retrospective peak/valley ground truth separately from live signals;
- evaluate candidate signals out of sample;
- inspect forward excursions and failure cases;
- keep the research surface clean enough to later plug into a faster execution stack.

This showcase is intentionally standalone. It does not depend on the private internals of my larger Market Brain research codebase.

## Research question

Given only information available at second `t`, can we identify states that are disproportionately likely to precede a short-horizon local reversal?

The included baseline studies combinations of:

- short/medium momentum;
- return z-scores;
- realized volatility;
- rolling range position;
- distance from recent extrema;
- volume surprise;
- short-term reversal pressure.

The baseline is deliberately interpretable. It is meant to establish whether there is signal worth modeling before adding more complex ML.

## Pipeline

```text
Binance 1s klines
      |
      v
causal feature matrix --------------+
      |                              |
      v                              v
live peak/valley scores        retrospective labels
      |                              |
      +-------------> walk-forward evaluation
                              |
                              v
                 precision / recall / hit rate /
                 forward-return & excursion stats
```

A key rule is that **future data is allowed only in label construction and evaluation**, never in features used by the candidate signal.

## Quick start

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Download a small BTCUSDT 1-second sample:

```bash
python download_binance_1s.py \
  --symbol BTCUSDT \
  --start 2026-08-01T00:00:00Z \
  --end   2026-08-01T06:00:00Z \
  --output data/btcusdt_1s.parquet
```

Run the research baseline:

```bash
python run_research.py \
  --input data/btcusdt_1s.parquet \
  --horizon 30 \
  --move-bps 8 \
  --output-dir outputs/btc_30s
```

The run writes:

- `metrics.json` — out-of-sample signal metrics;
- `signals.parquet` — timestamps, causal features, labels and signal flags;
- `diagnostic.png` — price with predicted peak/valley candidates;
- console summary with train/test counts and forward-return statistics.

## Project structure

```text
.
├── README.md
├── METHODOLOGY.md
├── requirements.txt
├── download_binance_1s.py
├── quant_research.py
├── run_research.py
└── tests/
    └── test_causality.py
```

## What makes the evaluation useful

### 1. Causal feature construction

Every live feature is computed from current/past observations only. The test suite explicitly checks that changing future prices does not change already-computed features.

### 2. Labels are treated as hindsight

A local peak/valley label necessarily uses future information. That is acceptable for training/evaluation ground truth, but those labels are never fed into the live feature calculation.

### 3. Chronological holdout

The baseline calibrates score thresholds on the earlier segment and reports results on the later segment. Random shuffling is intentionally avoided.

### 4. No profit claims

Precision, recall, forward returns and excursion statistics are research diagnostics. They are not a backtest with fees, slippage, queue position, execution latency or risk sizing.

## Why 1-second candles are useful — and what they are not

One-second candles are useful for rapid event/reversal research and for building a causal signal pipeline. They are **not sufficient for classical low-latency HFT**. A serious execution system may later need:

- trade-by-trade data;
- L2/order-book imbalance and queue dynamics;
- event-time rather than candle-time features;
- realistic fees/slippage/fill modeling;
- latency measurements;
- a lower-latency execution component (for example Rust) around a Python research layer.

That separation is intentional: Python is excellent for research iteration; production execution can be optimized only after the research justifies it.

## Extensions I would test next

1. multi-horizon labels (5s / 15s / 30s / 60s);
2. volatility-conditioned thresholds rather than fixed basis-point moves;
3. cross-timescale momentum exhaustion and regime features;
4. signed volume / trade imbalance if raw trades are available;
5. order-book imbalance and liquidity withdrawal features;
6. purged walk-forward validation for learned models;
7. explicit transaction-cost and fill simulation;
8. specialized models for setup detection vs entry timing instead of one monolithic predictor.

## Background

This public-facing research package was extracted from a larger personal quantitative-research project, **Market Brain**, which explores high-frequency crypto data, stochastic market simulation, multiscale latent-state modeling, volatility/tail dynamics, causal signal design and leakage-aware evaluation.

The broader project is research-oriented: it focuses on understanding market state and building testable models rather than presenting unverified profitability claims.
