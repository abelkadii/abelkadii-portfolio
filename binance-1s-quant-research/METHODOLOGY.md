# Methodology

## Objective

The baseline asks a narrow question: can strictly causal information available at second `t` identify market states with an elevated probability of a short-horizon reversal?

This is a research problem, not an execution claim. The project separates three concerns that are often mixed together:

1. **features** — information that could be computed live;
2. **labels** — retrospective ground truth that may use future observations;
3. **evaluation** — chronological testing of whether live scores align with later outcomes.

## Data

The downloader uses Binance public spot klines and stores the standard 1-second OHLCV fields, including price, volume, trade count and taker-buy volume.

## Causal features

All feature windows end at `t`.

- one-second log return;
- trailing realized volatility;
- current return normalized by prior volatility;
- short/medium trailing momentum;
- current price position inside a trailing high/low envelope;
- volume surprise via a trailing z-score;
- interpretable reversal pressure combining recent extension with current counter-move.

The peak and valley scores are deliberately simple weighted baselines. A more flexible model should earn its complexity by improving held-out behavior.

## Retrospective labels

Peak and valley labels are intentionally hindsight-based. For horizon `H`, a peak is a local high around `t` followed by a configured downward excursion; a valley is the symmetric local low followed by an upward excursion.

Future information is therefore permitted in labels and evaluation, but never in live features.

## Threshold calibration

The dataset is split chronologically. The earlier segment calibrates score thresholds by quantile; the later segment is held out for reporting. Random shuffling is avoided.

For learned models, the natural extension is rolling or purged walk-forward validation with an embargo large enough to prevent overlapping-label leakage.

## Metrics

The baseline reports separately for peaks and valleys:

- signal count and signal rate;
- label count and base rate;
- precision;
- recall;
- lift over unconditional label rate;
- mean/median signed forward return;
- favorable and adverse excursion statistics.

## Important limitations

This project does not model commissions, spread, slippage, partial fills, queue position, network/exchange latency, order-book reconstruction, market impact, position sizing, or portfolio risk.

The output is therefore **signal research**, not a claim of executable PnL.

## Research progression

A disciplined next sequence would be:

1. establish stable 1-second baselines;
2. test regime-conditioned behavior;
3. replace fixed move labels with volatility-scaled event labels;
4. add raw-trade/order-book features;
5. test statistical/ML models with purged walk-forward evaluation;
6. build realistic execution simulation;
7. move latency-critical components to lower-latency code only if measured bottlenecks justify it.
