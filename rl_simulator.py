"""
rl_simulator.py — Reinforcement Learning Trading Simulator
Uses Stable-Baselines3 (PPO / A2C / DQN) on a custom Gymnasium environment
to learn and backtest an automated trading strategy on Indian NSE stocks.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf
import math
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line and signal line, return histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram


def compute_bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute Bollinger Band width as (upper - lower) / middle."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return width


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features needed for the RL observation space.
    Returns a clean DataFrame with no NaN rows.
    """
    feat = pd.DataFrame(index=df.index)
    feat["close"] = df["Close"].values
    feat["norm_close"] = feat["close"] / feat["close"].iloc[0]

    # RSI
    feat["rsi"] = compute_rsi(feat["close"], 14) / 100.0

    # Moving averages
    feat["ma20"] = feat["close"].rolling(20).mean()
    feat["ma50"] = feat["close"].rolling(50).mean()
    feat["close_ma20_ratio"] = feat["close"] / feat["ma20"].replace(0, np.nan)
    feat["close_ma50_ratio"] = feat["close"] / feat["ma50"].replace(0, np.nan)

    # Bollinger Band width
    feat["bb_width"] = compute_bollinger_width(feat["close"], 20)

    # MACD histogram (normalised by price)
    feat["macd_hist"] = compute_macd(feat["close"]) / feat["close"].replace(0, np.nan)

    # Volume ratio
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        avg_vol = vol.rolling(20).mean().replace(0, np.nan)
        feat["vol_ratio"] = (vol / avg_vol).clip(upper=3.0)
    else:
        feat["vol_ratio"] = 1.0

    # 5-day return
    feat["ret_5d"] = feat["close"].pct_change(5)

    # Drop NaN rows (from rolling windows)
    feat = feat.dropna()
    return feat


# ═══════════════════════════════════════════════════════════════════════════
#  GYMNASIUM TRADING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

class StockTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for single-stock trading.

    Observation (10 dims):
        0: normalised close price
        1: RSI / 100
        2: close / MA20
        3: close / MA50
        4: Bollinger width
        5: MACD histogram (norm)
        6: volume ratio (capped 3.0)
        7: 5-day return
        8: position flag (0=cash, 1=holding)
        9: unrealised PnL %

    Actions: 0=SELL, 1=HOLD, 2=BUY
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        transaction_fee_pct: float = 0.001,
    ):
        super().__init__()
        self.features_df = features_df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee_pct = transaction_fee_pct
        self.n_steps = len(self.features_df)

        # Action: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)

        # Observation: 10 continuous values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares_held = 0
        self.buy_price = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.features_df.iloc[self.current_step]
        price = row["close"]

        position_flag = 1.0 if self.shares_held > 0 else 0.0
        unrealised_pnl = 0.0
        if self.shares_held > 0 and self.buy_price > 0:
            unrealised_pnl = (price - self.buy_price) / self.buy_price

        obs = np.array(
            [
                row["norm_close"],
                row["rsi"],
                row["close_ma20_ratio"],
                row["close_ma50_ratio"],
                row["bb_width"],
                row["macd_hist"],
                row["vol_ratio"],
                row["ret_5d"],
                position_flag,
                unrealised_pnl,
            ],
            dtype=np.float32,
        )
        # Replace any NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        return obs

    def _get_price(self) -> float:
        return float(self.features_df.iloc[self.current_step]["close"])

    def _portfolio_value_now(self) -> float:
        price = self._get_price()
        return self.cash + self.shares_held * price

    def step(self, action: int):
        price = self._get_price()
        old_value = self._portfolio_value_now()

        date_val = (
            self.features_df.index[self.current_step]
            if hasattr(self.features_df.index[self.current_step], "strftime")
            else self.current_step
        )

        # ── Execute action ────────────────────────────────────────────
        if action == 2 and self.shares_held == 0:
            # BUY — go all-in
            cost_per_share = price * (1 + self.fee_pct)
            max_shares = int(self.cash / cost_per_share)
            if max_shares > 0:
                self.shares_held = max_shares
                self.cash -= max_shares * cost_per_share
                self.buy_price = price
                self.trades.append(
                    {
                        "step": self.current_step,
                        "date": str(date_val),
                        "action": "BUY",
                        "price": round(price, 2),
                        "shares": max_shares,
                        "value": round(self._portfolio_value_now(), 2),
                    }
                )

        elif action == 0 and self.shares_held > 0:
            # SELL — exit entire position
            revenue = self.shares_held * price * (1 - self.fee_pct)
            self.cash += revenue
            pnl_pct = (price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0
            self.trades.append(
                {
                    "step": self.current_step,
                    "date": str(date_val),
                    "action": "SELL",
                    "price": round(price, 2),
                    "shares": self.shares_held,
                    "value": round(self._portfolio_value_now(), 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                }
            )
            self.shares_held = 0
            self.buy_price = 0.0

        # ── Advance step ──────────────────────────────────────────────
        self.current_step += 1
        new_price = self._get_price()
        new_value = self._portfolio_value_now()
        self.equity_curve.append(new_value)

        # ── Reward: position-aware directional signal ─────────────────
        # Daily price change (signed)
        daily_return = (new_price - price) / price if price > 0 else 0.0

        reward = 0.0

        if self.shares_held > 0:
            # Holding shares: reward tracks price movement (scaled up)
            reward = daily_return * 100.0
        else:
            # Sitting in cash: constant drag so agent doesn't idle forever
            reward = -0.1
            # Extra penalty if market went up and we missed it
            if daily_return > 0:
                reward -= daily_return * 50.0
            else:
                # Small reward for correctly avoiding a down day
                reward += abs(daily_return) * 30.0

        # Bonus for completing a profitable SELL trade
        if action == 0 and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade.get("pnl_pct", 0) > 0:
                reward += last_trade["pnl_pct"] * 0.5  # Bonus per % profit

        # Drawdown penalty
        self.peak_value = max(self.peak_value, new_value)
        drawdown = (self.peak_value - new_value) / self.peak_value if self.peak_value > 0 else 0
        if drawdown > 0.05:
            reward -= drawdown * 10.0

        reward = float(reward)

        # ── Done check ────────────────────────────────────────────────
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        if terminated:
            # Force close position at end
            if self.shares_held > 0:
                price = self._get_price()
                revenue = self.shares_held * price * (1 - self.fee_pct)
                self.cash += revenue
                self.trades.append(
                    {
                        "step": self.current_step,
                        "date": str(date_val),
                        "action": "SELL (FORCED)",
                        "price": round(price, 2),
                        "shares": self.shares_held,
                        "value": round(self._portfolio_value_now(), 2),
                    }
                )
                self.shares_held = 0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN + BACKTEST PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def train_and_backtest(
    ticker: str,
    period: str = "2y",
    algorithm: str = "PPO",
    timesteps: int = 10_000,
    initial_balance: float = 100_000.0,
    fee_pct: float = 0.001,
    train_split: float = 0.8,
    progress_callback=None,
) -> dict:
    """
    Full pipeline: download data → compute features → train RL agent → backtest.

    Returns dict with:
        - train_metrics, test_metrics
        - equity_curve (test), buy_hold_curve (test)
        - trades (test)
        - model summary
    """

    # ── 1. Download data ──────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.05, f"Downloading {ticker} data ({period})...")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df is None or df.empty or len(df) < 100:
        raise ValueError(
            f"Insufficient data for {ticker}. Need at least 100 trading days, got {len(df) if df is not None else 0}."
        )

    # ── 2. Compute features ───────────────────────────────────────────
    if progress_callback:
        progress_callback(0.10, "Computing technical indicators...")

    features = prepare_features(df)
    if len(features) < 80:
        raise ValueError(
            f"After computing indicators, only {len(features)} data points remain. Need at least 80."
        )

    # ── 3. Train/Test split ───────────────────────────────────────────
    split_idx = int(len(features) * train_split)
    train_df = features.iloc[:split_idx].copy()
    test_df = features.iloc[split_idx:].copy()

    if len(train_df) < 60:
        raise ValueError(f"Training set too small ({len(train_df)} rows). Use a longer period.")
    if len(test_df) < 20:
        raise ValueError(f"Test set too small ({len(test_df)} rows). Use a longer period.")

    if progress_callback:
        progress_callback(
            0.15,
            f"Data split: {len(train_df)} train / {len(test_df)} test days",
        )

    # ── 4. Create training environment ────────────────────────────────
    def make_train_env():
        return StockTradingEnv(train_df, initial_balance, fee_pct)

    vec_env = DummyVecEnv([make_train_env])

    # ── 5. Select and train algorithm ─────────────────────────────────
    algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    AlgoClass = algo_map.get(algorithm.upper(), PPO)

    if progress_callback:
        progress_callback(0.20, f"Training {algorithm} agent for {timesteps:,} timesteps...")

    # Hyperparams tuned for trading: higher entropy for exploration,
    # larger batch for stable gradients
    algo_kwargs = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "verbose": 0,
        "learning_rate": 3e-4,
        "device": "cpu",
    }
    if algorithm.upper() in ("PPO", "A2C"):
        algo_kwargs["ent_coef"] = 0.1  # High entropy to force trade exploration
    if algorithm.upper() == "PPO":
        algo_kwargs["n_steps"] = min(128, len(train_df) - 1)  # Faster updates
        algo_kwargs["batch_size"] = 64
        algo_kwargs["gamma"] = 0.99
        algo_kwargs["gae_lambda"] = 0.95
        algo_kwargs["clip_range"] = 0.2
    if algorithm.upper() == "DQN":
        algo_kwargs["exploration_fraction"] = 0.3  # Explore 30% of training
        algo_kwargs["exploration_final_eps"] = 0.05

    model = AlgoClass(**algo_kwargs)

    # Train in chunks to report progress
    chunk_size = max(timesteps // 10, 1000)
    trained = 0
    while trained < timesteps:
        this_chunk = min(chunk_size, timesteps - trained)
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)
        trained += this_chunk
        pct = 0.20 + 0.50 * (trained / timesteps)
        if progress_callback:
            progress_callback(
                min(pct, 0.70),
                f"Training {algorithm}... {trained:,}/{timesteps:,} steps",
            )

    vec_env.close()

    # ── 6. Backtest on TEST data ──────────────────────────────────────
    if progress_callback:
        progress_callback(0.75, "Running backtest on out-of-sample data...")

    test_env = StockTradingEnv(test_df, initial_balance, fee_pct)
    obs, _ = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated

    # ── 7. Compute metrics ────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.85, "Computing performance metrics...")

    equity = test_env.equity_curve
    trades = test_env.trades

    # Buy & Hold benchmark
    test_prices = test_df["close"].values
    bh_shares = int(initial_balance / (test_prices[0] * (1 + fee_pct)))
    bh_leftover = initial_balance - bh_shares * test_prices[0] * (1 + fee_pct)
    buy_hold_curve = [bh_shares * p + bh_leftover for p in test_prices]
    # Pad to same length as equity curve
    while len(buy_hold_curve) < len(equity):
        buy_hold_curve.append(buy_hold_curve[-1])

    # Returns
    final_equity = equity[-1]
    rl_return = (final_equity - initial_balance) / initial_balance * 100
    bh_final = buy_hold_curve[-1]
    bh_return = (bh_final - initial_balance) / initial_balance * 100

    # Sharpe Ratio (annualised, using daily equity returns)
    equity_arr = np.array(equity)
    daily_returns = np.diff(equity_arr) / equity_arr[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (peak - equity_arr) / np.where(peak > 0, peak, 1)
    max_drawdown = float(np.max(drawdowns)) * 100

    # Win Rate
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    if sell_trades:
        wins = sum(1 for t in sell_trades if t.get("pnl_pct", 0) > 0)
        win_rate = wins / len(sell_trades) * 100
    else:
        win_rate = 0.0

    # Dates for x-axis
    if hasattr(test_df.index, "strftime"):
        test_dates = test_df.index.strftime("%Y-%m-%d").tolist()
    else:
        test_dates = list(range(len(test_df)))

    if progress_callback:
        progress_callback(1.0, "Done — Simulation complete!")

    return {
        "ticker": ticker,
        "algorithm": algorithm,
        "timesteps": timesteps,
        "period": period,
        "initial_balance": initial_balance,
        "fee_pct": fee_pct,
        "train_days": len(train_df),
        "test_days": len(test_df),
        "equity_curve": equity,
        "buy_hold_curve": buy_hold_curve[:len(equity)],
        "test_dates": test_dates[:len(equity)],
        "trades": trades,
        "metrics": {
            "rl_return": round(rl_return, 2),
            "bh_return": round(bh_return, 2),
            "alpha": round(rl_return - bh_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 2),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 1),
            "final_value": round(final_equity, 2),
        },
    }
