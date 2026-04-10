# CONFIGURATION REQUIRED 

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from datetime import datetime

# Fix random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

START_DATE = "2005-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

WINDOW      = 60
FORWARD     = 126
REBALANCE   = 126
TRAIN_YEARS = 5
N_EPOCHS    = 5

TOP_PERCENT      = 0.2
TRANSACTION_COST = 0.001
INITIAL_CAPITAL  = 1_000_000
BATCH_SIZE       = 32

DEVICE = "cpu"


# DATA


def get_nifty50():
    return [
        "RELIANCE.NS","HDFCBANK.NS","TCS.NS","INFY.NS",
        "ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS",
        "ITC.NS","BHARTIARTL.NS","LT.NS",
        "ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
        "SUNPHARMA.NS","NESTLEIND.NS",
        "POWERGRID.NS","ONGC.NS","NTPC.NS",
        "M&M.NS","SBILIFE.NS"
    ]

def download_prices(tickers):
    return yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True
    )


# FEATURE ENGINEERING


def build_sample(data, ticker, idx, index_close):

    df = data[ticker]

    if idx - WINDOW < 0 or idx + FORWARD >= len(df):
        return None

    close   = df["Close"]
    returns = close.pct_change(fill_method=None)

    mom6     = close.pct_change(21, fill_method=None)
    mom12    = close.pct_change(63, fill_method=None)
    vol3     = returns.rolling(21).std()
    sharpe6  = (
        returns.rolling(42).mean() /
        (returns.rolling(42).std() + 1e-8)
    )

    index_mom    = index_close.reindex(close.index).pct_change(21, fill_method=None)
    rel_strength = mom6 - index_mom
    vol_growth   = df["Volume"].pct_change(21, fill_method=None)

    features = pd.DataFrame({
        "ret":        returns,
        "mom6":       mom6,
        "mom12":      mom12,
        "vol3":       vol3,
        "sharpe6":    sharpe6,
        "rel_str":    rel_strength,
        "vol_growth": vol_growth
    })

    window_slice = features.iloc[idx - WINDOW : idx].copy()
    window_slice = window_slice.ffill().bfill()

    for col in ["ret","mom6","mom12","vol3","sharpe6","rel_str","vol_growth"]:
        if col not in window_slice.columns:
            window_slice[col] = 0.0

    window_slice = window_slice[["ret","mom6","mom12","vol3","sharpe6","rel_str","vol_growth"]]

    values = window_slice.values.astype(np.float32)
    values = np.where(np.isinf(values), np.nan, values)

    if np.isnan(values).any():
        return None

    future_return = close.iloc[idx + FORWARD] / close.iloc[idx] - 1
    if not np.isfinite(future_return):
        return None

    return values, float(future_return)


# CROSS SECTION


def build_cross_section(data, idx, index_close):

    X_list, Y_list, tickers = [], [], []

    for ticker in data.columns.get_level_values(0).unique():
        sample = build_sample(data, ticker, idx, index_close)
        if sample is None:
            continue
        X, Y = sample
        X_list.append(X)
        Y_list.append(Y)
        tickers.append(ticker)

    if len(X_list) < 5:
        return None

    X_np = np.array(X_list, dtype=np.float32)
    Y_np = np.array(Y_list, dtype=np.float32)

    X_np = np.where(np.isinf(X_np), np.nan, X_np)
    valid_mask = ~np.isnan(X_np).any(axis=(1, 2))
    X_np    = X_np[valid_mask]
    Y_np    = Y_np[valid_mask]
    tickers = [t for i, t in enumerate(tickers) if valid_mask[i]]

    if len(X_np) < 5:
        return None

    mean = np.nanmean(X_np, axis=0)
    std  = np.nanstd(X_np, axis=0)
    std  = np.where(std < 1e-8, 1e-8, std)
    X_np = (X_np - mean) / std
    X_np = np.clip(X_np, -10.0, 10.0)

    valid_mask2 = ~np.isnan(X_np).any(axis=(1, 2))
    X_np    = X_np[valid_mask2]
    Y_np    = Y_np[valid_mask2]
    tickers = [t for i, t in enumerate(tickers) if valid_mask2[i]]

    if len(X_np) < 5:
        return None

    return X_np, Y_np, tickers


# INFERENCE-ONLY CROSS SECTION (no future return needed)


def build_cross_section_live(data, idx, index_close):
    """
    Builds features for a given idx without requiring future returns.
    Uses a larger lookback buffer to ensure rolling windows are stable.
    """
    X_list, tickers = [], []

    # Use a larger buffer so rolling windows are fully warmed up
    BUFFER = max(WINDOW + 63, WINDOW * 2)

    for ticker in data.columns.get_level_values(0).unique():

        df = data[ticker]

        # Need enough history before idx for rolling windows to be stable
        if idx - BUFFER < 0 or idx > len(df):
            continue

        close   = df["Close"]
        returns = close.pct_change(fill_method=None)

        mom6     = close.pct_change(21, fill_method=None)
        mom12    = close.pct_change(63, fill_method=None)
        vol3     = returns.rolling(21).std()
        sharpe6  = (
            returns.rolling(42).mean() /
            (returns.rolling(42).std() + 1e-8)
        )

        index_mom    = index_close.reindex(close.index).pct_change(21, fill_method=None)
        rel_strength = mom6 - index_mom
        vol_growth   = df["Volume"].pct_change(21, fill_method=None)

        features = pd.DataFrame({
            "ret":        returns,
            "mom6":       mom6,
            "mom12":      mom12,
            "vol3":       vol3,
            "sharpe6":    sharpe6,
            "rel_str":    rel_strength,
            "vol_growth": vol_growth
        })

        # Take the last WINDOW rows up to idx (exclusive)
        window_slice = features.iloc[idx - WINDOW : idx].copy()

        if len(window_slice) < WINDOW:
            continue

        window_slice = window_slice.ffill().bfill()

        for col in ["ret","mom6","mom12","vol3","sharpe6","rel_str","vol_growth"]:
            if col not in window_slice.columns:
                window_slice[col] = 0.0

        window_slice = window_slice[["ret","mom6","mom12","vol3","sharpe6","rel_str","vol_growth"]]

        values = window_slice.values.astype(np.float32)
        values = np.where(np.isinf(values), np.nan, values)

        # Fill any remaining NaNs with column median instead of rejecting
        for col_i in range(values.shape[1]):
            col_vals = values[:, col_i]
            nan_mask = np.isnan(col_vals)
            if nan_mask.all():
                values[:, col_i] = 0.0
            elif nan_mask.any():
                median = np.nanmedian(col_vals)
                values[nan_mask, col_i] = median

        if np.isnan(values).any():
            continue

        X_list.append(values)
        tickers.append(ticker)

    if len(X_list) < 5:
        return None

    X_np = np.array(X_list, dtype=np.float32)
    X_np = np.where(np.isinf(X_np), np.nan, X_np)

    valid_mask = ~np.isnan(X_np).any(axis=(1, 2))
    X_np    = X_np[valid_mask]
    tickers = [t for i, t in enumerate(tickers) if valid_mask[i]]

    if len(X_np) < 5:
        return None

    mean = np.nanmean(X_np, axis=0)
    std  = np.nanstd(X_np, axis=0)
    std  = np.where(std < 1e-8, 1e-8, std)
    X_np = (X_np - mean) / std
    X_np = np.clip(X_np, -10.0, 10.0)

    valid_mask2 = ~np.isnan(X_np).any(axis=(1, 2))
    X_np    = X_np[valid_mask2]
    tickers = [t for i, t in enumerate(tickers) if valid_mask2[i]]

    if len(X_np) < 5:
        return None

    return X_np, tickers


# MODEL


class TemporalTransformer(nn.Module):

    def __init__(self, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)


# LOSS


def ranking_loss(pred, target):
    diff_pred   = pred.unsqueeze(1)   - pred.unsqueeze(0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    label = torch.sign(diff_target)
    return F.relu(-label * diff_pred).mean()


# BACKTEST + EVALUATION


def run_backtest(data):

    capital          = INITIAL_CAPITAL
    portfolio_values = [capital]

    ic_scores        = []
    hit_rates        = []
    precision_scores = []
    period_log       = []

    index_data  = yf.download("^NSEI", start=START_DATE, end=END_DATE,
                               auto_adjust=True)
    index_close = index_data["Close"].squeeze()

    index_fwd_returns = index_close.pct_change(FORWARD, fill_method=None)

    model     = TemporalTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    all_dates = data.index
    rebalance_points = list(range(
        WINDOW + TRAIN_YEARS * 252,
        len(all_dates) - FORWARD,
        REBALANCE
    ))

    for t in rebalance_points:

        print(f"\nRebalance Date: {all_dates[t].date()}")

        # Training 
        train_start   = max(WINDOW, t - TRAIN_YEARS * 252)
        train_indices = [
            i for i in range(train_start, t, 5)
            if i + FORWARD <= t
        ]

        cross_sections = []
        for idx in train_indices:
            cs = build_cross_section(data, idx, index_close)
            if cs is None:
                continue
            cross_sections.append((cs[0], cs[1]))

        print(f"  Training cross-sections: {len(cross_sections)}")

        if len(cross_sections) > 0:
            model.train()
            for epoch in range(N_EPOCHS):
                perm       = np.random.permutation(len(cross_sections))
                epoch_loss = 0.0
                n_batches  = 0

                for batch_start in range(0, len(perm), BATCH_SIZE):
                    batch_idx = perm[batch_start : batch_start + BATCH_SIZE]

                    X_batch = torch.tensor(
                        np.concatenate([cross_sections[i][0] for i in batch_idx], axis=0),
                        dtype=torch.float32
                    )
                    Y_batch = torch.tensor(
                        np.concatenate([cross_sections[i][1] for i in batch_idx], axis=0),
                        dtype=torch.float32
                    )

                    pred = model(X_batch)
                    loss = ranking_loss(pred, Y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches  += 1

                    del X_batch, Y_batch, pred

                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{N_EPOCHS}  loss={avg_loss:.4f}")

        # Inference
        cs_test = build_cross_section(data, t, index_close)
        if cs_test is None:
            print("  No test cross-section — skipping.")
            continue

        X_np, Y_np, tickers = cs_test
        X = torch.tensor(X_np, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            scores = model(X).numpy()

        n = len(scores)
        k = max(1, int(n * TOP_PERCENT))

        long_idx     = np.argsort(scores)[-k:]
        long_tickers = [tickers[i] for i in long_idx]
        long_return  = np.mean(Y_np[long_idx])
        net_return   = long_return - TRANSACTION_COST

        # Metric 1: IC
        ic, _ = spearmanr(scores, Y_np)
        ic_scores.append(ic)

        # Metric 2: Hit Rate vs Index 
        rebal_date = all_dates[t]
        try:
            idx_return = float(index_fwd_returns.loc[rebal_date])
            beat_index = bool(long_return > idx_return)
        except Exception:
            idx_return = float("nan")
            beat_index = None

        hit_rates.append(beat_index)

        # Metric 3: Top-K Precision 
        actual_top_k    = set(np.argsort(Y_np)[-k:])
        predicted_top_k = set(long_idx)
        precision       = len(actual_top_k & predicted_top_k) / k
        precision_scores.append(precision)

        capital *= (1 + net_return)
        portfolio_values.append(capital)

        period_log.append({
            "date"      : rebal_date.date(),
            "picks"     : long_tickers,
            "long_ret"  : round(long_return, 4),
            "idx_ret"   : round(idx_return, 4) if np.isfinite(idx_return) else None,
            "beat_index": beat_index,
            "ic"        : round(ic, 4),
            "precision" : round(precision, 4),
            "capital"   : round(capital, 2),
        })

        print(f"  Top picks  : {long_tickers}")
        print(f"  Long ret   : {long_return:.4f}  Net: {net_return:.4f}  "
              f"Capital: {capital:,.0f}")
        print(f"  IC         : {ic:.4f}")
        print(f"  Precision  : {precision:.2%}  "
              f"(random baseline: {TOP_PERCENT:.2%})")
        print(f"  Beat index : {beat_index}  "
              f"(index fwd ret: {idx_return:.4f})")

    return portfolio_values, period_log, model, index_close


# LIVE RECOMMENDATION


def get_live_recommendation(data, model, index_close):
    """
    Walks back from the last row to find a valid live cross-section.
    Uses a larger buffer and median imputation to handle recent NaNs.
    """
    print("\n" + "="*45)
    print("LIVE RECOMMENDATION (TODAY'S PICKS)")
    print("="*45)

    cs      = None
    found_t = None

    # Walk back up to 10 days — with buffer+median fix this should succeed quickly
    for t in range(len(data.index) - 1, len(data.index) - 11, -1):
        cs = build_cross_section_live(data, t, index_close)
        if cs is not None:
            found_t = t
            break

    if cs is None or found_t is None:
        print("\n  Could not build live cross-section.")
        print("  Possible causes:")
        print("  - Too many NaN values in recent data")
        print("  - Insufficient price history for some tickers")
        print("  - Try running again after market hours for complete data")
        return

    X_np, tickers = cs
    as_of_date    = data.index[found_t].strftime("%Y-%m-%d")

    X = torch.tensor(X_np, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(X).numpy()

    n = len(scores)
    k = max(1, int(n * TOP_PERCENT))

    long_idx     = np.argsort(scores)[-k:]
    long_tickers = [tickers[i] for i in long_idx]
    long_scores  = scores[long_idx]

    # Softmax weights for position sizing
    weights    = np.exp(long_scores) / np.exp(long_scores).sum()
    hold_until = (
        pd.Timestamp(as_of_date) + pd.offsets.BDay(FORWARD)
    ).strftime("%Y-%m-%d")

    print(f"\n  As of         : {as_of_date}")
    print(f"  Hold until    : {hold_until}  (~{FORWARD} trading days / 6 months)")
    print(f"  Starting cap  : ₹{INITIAL_CAPITAL:,.0f}")
    print(f"  Stocks scored : {n}")
    print(f"  Picks (top {k})")
    print()
    print(f"  {'Rank':<6} {'Stock':<20} {'Score':>8} "
          f"{'Weight':>8} {'Allocate (₹)':>15}")
    print(f"  {'-'*62}")

    for rank, (ticker, score, weight) in enumerate(
        zip(long_tickers[::-1], long_scores[::-1], weights[::-1]), 1
    ):
        allocation = weight * INITIAL_CAPITAL
        print(f"  {rank:<6} {ticker:<20} {score:>8.4f} "
              f"{weight:>7.1%} {allocation:>15,.0f}")

    print()
    print(f"  Notes:")
    print(f"  - Transaction cost : {TRANSACTION_COST:.1%} per trade")
    print(f"  - Rebalance every  : {FORWARD} trading days (~6 months)")
    print(f"  - Weights          : softmax of model scores (not equal-weight)")
    print(f"  - Past performance does not guarantee future results")

    # Full ranking table
    print(f"\n  --- Full ranking (all {n} stocks) ---")
    print(f"  {'Rank':<6} {'Stock':<20} {'Score':>8} {'Signal':>10}")
    print(f"  {'-'*48}")

    sorted_idx = np.argsort(scores)[::-1]
    for rank, i in enumerate(sorted_idx, 1):
        signal = "BUY  ✓" if i in set(long_idx) else "hold"
        print(f"  {rank:<6} {tickers[i]:<20} {scores[i]:>8.4f} {signal:>10}")


# METRICS

def compute_metrics(values):

    values    = np.array(values, dtype=float)
    returns   = np.diff(values) / values[:-1]
    n_periods = len(returns)

    if n_periods < 2:
        return {"Error": "Not enough data points to compute metrics."}

    periods_per_year = 252 / REBALANCE
    cagr  = (values[-1] / values[0]) ** (periods_per_year / n_periods) - 1
    std   = np.std(returns, ddof=1)
    sharpe = (
        (np.mean(returns) / std) * np.sqrt(periods_per_year)
        if std > 0 else float("nan")
    )
    peak   = np.maximum.accumulate(values)
    max_dd = np.max((peak - values) / peak)

    return {
        "Final Capital" : f"₹{values[-1]:,.0f}",
        "CAGR"          : f"{cagr:.2%}",
        "Sharpe"        : f"{sharpe:.3f}",
        "Max Drawdown"  : f"{max_dd:.2%}",
        "Periods"       : n_periods,
    }


# ACCURACY REPORT


def print_accuracy_report(period_log, ic_scores, hit_rates, precision_scores):

    valid_hits = [h for h in hit_rates if h is not None]

    print("\n" + "="*45)
    print("ACCURACY REPORT")
    print("="*45)
    print(f"  Periods evaluated     : {len(ic_scores)}")

    print(f"\n  --- Ranking Quality (IC) ---")
    print(f"  Mean IC               : {np.mean(ic_scores):.4f}")
    print(f"  IC Std                : {np.std(ic_scores):.4f}")
    ic_std = np.std(ic_scores)
    icir   = np.mean(ic_scores) / ic_std if ic_std > 0 else float("nan")
    print(f"  ICIR                  : {icir:.4f}")
    print(f"  IC > 0                : "
          f"{sum(1 for x in ic_scores if x > 0)}/{len(ic_scores)} periods")

    print(f"\n  --- Hit Rate vs Nifty 50 ---")
    if valid_hits:
        print(f"  Beat index            : "
              f"{sum(valid_hits)}/{len(valid_hits)} "
              f"({100*sum(valid_hits)/len(valid_hits):.1f}%)")

    print(f"\n  --- Top-K Precision ---")
    print(f"  Mean Precision        : {np.mean(precision_scores):.2%}")
    print(f"  Random Baseline       : {TOP_PERCENT:.2%}")
    lift = np.mean(precision_scores) / TOP_PERCENT if TOP_PERCENT > 0 else float("nan")
    print(f"  Precision Lift        : {lift:.2f}x over random")

    print(f"\n  --- Period Log ---")
    print(f"  {'Date':<12} {'Picks':<45} {'IC':>7} {'Prec':>7} {'Beat':>6}")
    print(f"  {'-'*82}")
    for p in period_log:
        picks_str = ", ".join([t.replace(".NS", "") for t in p["picks"]])
        beat_str  = (
            "✓" if p["beat_index"] is True
            else ("?" if p["beat_index"] is None else "✗")
        )
        print(f"  {str(p['date']):<12} {picks_str:<45} "
              f"{p['ic']:>7.4f} {p['precision']:>7.2%} {beat_str:>6}")


# saving live data to csv

def save_live_results_to_csv():
    tickers = get_nifty50()
    data = download_prices(tickers)

    _, _, model, index_close = run_backtest(data)

    cs = build_cross_section_live(data, len(data.index)-1, index_close)

    if cs is None:
        print("No live data available")
        return

    X_np, tickers = cs
    X = torch.tensor(X_np, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(X).numpy()

    df = pd.DataFrame({
        "Stock": tickers,
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    df.to_csv("model1_output.csv", index=False)
    print("Saved latest recommendations!")

# main function

def main():

    tickers = get_nifty50()

    print("Downloading data...")
    print(f"Period: {START_DATE} → {END_DATE}")
    data = download_prices(tickers)

    print("\nRunning backtest + evaluation...")
    portfolio, period_log, model, index_close = run_backtest(data)

    if len(portfolio) <= 1:
        print("No results generated.")
        return

    # Performance metrics
    print("\n" + "="*45)
    print("BACKTEST RESULTS")
    print("="*45)
    for k, v in compute_metrics(portfolio).items():
        print(f"  {k:<18}: {v}")

    # Accuracy report
    ic_scores        = [p["ic"]         for p in period_log]
    hit_rates        = [p["beat_index"] for p in period_log]
    precision_scores = [p["precision"]  for p in period_log]

    print_accuracy_report(period_log, ic_scores, hit_rates, precision_scores)

    # Live recommendation
    get_live_recommendation(data, model, index_close)
    save_live_results_to_csv()


if __name__ == "__main__":
    main()