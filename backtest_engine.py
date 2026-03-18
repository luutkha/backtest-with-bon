"""
Backtesting Engine
- Entry signals generated from 1h candle data
- Trade execution and price validation using 1m candle data
- Position management with TP/SL/trailing stop on 1m candles
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExitPriority(Enum):
    CONSERVATIVE = "conservative"  # SL first
    AGGRESSIVE = "aggressive"      # TP first


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeResult:
    """Result of a single trade"""
    symbol: str
    side: PositionSide
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    hold_bars: int
    exit_reason: str  # "tp", "sl", "trailing_stop", "manual"
    leverage: float = 1.0


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000.0
    fee_rate: float = 0.0004  # Binance taker fee (0.04%)
    slippage: float = 0.0     # Slippage in percentage
    exit_priority: ExitPriority = ExitPriority.CONSERVATIVE
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.0  # Trailing stop percentage
    tp_pct: float = 0.02  # Take profit percentage (e.g., 0.02 = 2%)
    sl_pct: float = 0.01  # Stop loss percentage (e.g., 0.01 = 1%)
    position_size_pct: float = 1.0  # Position size as % of capital
    leverage: float = 1.0  # Leverage multiplier (e.g., 1.0 = no leverage, 3.0 = 3x)
    start_time: Optional[int] = None  # Start timestamp in ms
    end_time: Optional[int] = None    # End timestamp in ms


@dataclass
class Position:
    """Current open position"""
    symbol: str
    side: PositionSide
    entry_time: int
    entry_price: float
    quantity: float
    leverage: float = 1.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    trailing_stop_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    bars_held: int = 0


@dataclass
class Signal:
    """Trading signal from strategy"""
    time: int
    action: str  # "long", "short", "close_long", "close_short"
    price: float
    metadata: dict = field(default_factory=dict)


class BacktestEngine:
    """Main backtesting engine"""

    def __init__(
        self,
        config: BacktestConfig,
        data_dir: str,
        symbol: str,
        strategy: Callable
    ):
        self.config = config
        self.data_dir = data_dir
        self.symbol = symbol
        self.strategy = strategy

        self.h1_data: Optional[pd.DataFrame] = None
        self.m1_data: Optional[pd.DataFrame] = None

        self.capital = config.initial_capital
        self.position: Optional[Position] = None
        self.trades: list[TradeResult] = []

        self._load_data()

    def _load_data(self):
        """Load 1h and 1m data with time filtering"""
        h1_path = os.path.join(self.data_dir, "1h", f"{self.symbol}.csv")
        m1_path = os.path.join(self.data_dir, "5m", f"{self.symbol}.csv")

        logger.info(f"Loading data from {h1_path} and {m1_path}")

        self.h1_data = pd.read_csv(h1_path)
        self.m1_data = pd.read_csv(m1_path)

        # Convert timestamp to datetime for easier handling
        self.h1_data['datetime'] = pd.to_datetime(self.h1_data['openTime'], unit='ms')
        self.m1_data['datetime'] = pd.to_datetime(self.m1_data['openTime'], unit='ms')

        # Sort by time ascending (data might be in descending order)
        self.h1_data = self.h1_data.sort_values('openTime').reset_index(drop=True)
        self.m1_data = self.m1_data.sort_values('openTime').reset_index(drop=True)

        # Apply time filter if specified
        if self.config.start_time is not None:
            self.h1_data = self.h1_data[self.h1_data['openTime'] >= self.config.start_time]
            self.m1_data = self.m1_data[self.m1_data['openTime'] >= self.config.start_time]

        if self.config.end_time is not None:
            self.h1_data = self.h1_data[self.h1_data['openTime'] <= self.config.end_time]
            self.m1_data = self.m1_data[self.m1_data['openTime'] <= self.config.end_time]

        # Reset index after filtering
        self.h1_data = self.h1_data.reset_index(drop=True)
        self.m1_data = self.m1_data.reset_index(drop=True)

        # Get date range
        start_date = self.h1_data['datetime'].iloc[0] if len(self.h1_data) > 0 else "N/A"
        end_date = self.h1_data['datetime'].iloc[-1] if len(self.h1_data) > 0 else "N/A"

        logger.info(f"Loaded {len(self.h1_data)} 1h candles, {len(self.m1_data)} 5m candles")
        logger.info(f"Period: {start_date} to {end_date}")

    def _calculate_fees(self, price: float, quantity: float) -> float:
        """Calculate trading fees"""
        return price * quantity * self.config.fee_rate

    def _apply_slippage(self, price: float, side: PositionSide) -> float:
        """Apply slippage to entry/exit price"""
        if side == PositionSide.LONG:
            return price * (1 + self.config.slippage)
        else:
            return price * (1 - self.config.slippage)

    def _get_entry_price_from_m1(self, h1_signal: Signal) -> float:
        """
        Get actual entry price from 1m candles after 1h signal.
        Uses the open of the first 1m candle after the 1h close.
        """
        signal_time = h1_signal.time
        m1_after_signal = self.m1_data[self.m1_data['openTime'] > signal_time]

        if len(m1_after_signal) == 0:
            return self._apply_slippage(h1_signal.price, PositionSide.LONG if h1_signal.action == "long" else PositionSide.SHORT)

        first_m1 = m1_after_signal.iloc[0]
        entry_price = first_m1['open']

        side = PositionSide.LONG if h1_signal.action == "long" else PositionSide.SHORT
        return self._apply_slippage(entry_price, side)

    def _check_exit_conditions(self, m1_candle: pd.Series) -> Tuple[bool, str, float]:
        """
        Check if any exit condition is met using intrabar high/low.
        Returns: (should_exit, reason, exit_price)
        """
        if self.position is None:
            return False, "", 0.0

        high = m1_candle['high']
        low = m1_candle['low']

        if self.position.side == PositionSide.LONG:
            self.position.highest_price = max(self.position.highest_price, high)

            # Check trailing stop for long
            if self.config.trailing_stop_enabled:
                self.position.trailing_stop_price = self.position.highest_price * (1 - self.config.trailing_stop_pct)
                if low <= self.position.trailing_stop_price:
                    return True, "trailing_stop", self.position.trailing_stop_price

            # Check TP/SL for long
            if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                if low <= self.position.sl_price:
                    return True, "sl", self.position.sl_price
                if high >= self.position.tp_price:
                    return True, "tp", self.position.tp_price
            else:
                if high >= self.position.tp_price:
                    return True, "tp", self.position.tp_price
                if low <= self.position.sl_price:
                    return True, "sl", self.position.sl_price

        else:  # SHORT
            self.position.lowest_price = min(self.position.lowest_price, low)

            # Check trailing stop for short
            if self.config.trailing_stop_enabled:
                self.position.trailing_stop_price = self.position.lowest_price * (1 + self.config.trailing_stop_pct)
                if high >= self.position.trailing_stop_price:
                    return True, "trailing_stop", self.position.trailing_stop_price

            # Check TP/SL for short
            if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                if high >= self.position.sl_price:
                    return True, "sl", self.position.sl_price
                if low <= self.position.tp_price:
                    return True, "tp", self.position.tp_price
            else:
                if low <= self.position.tp_price:
                    return True, "tp", self.position.tp_price
                if high >= self.position.sl_price:
                    return True, "sl", self.position.sl_price

        return False, "", 0.0

    def _close_position(self, exit_time: int, exit_price: float, reason: str):
        """Close current position"""
        if self.position is None:
            return

        # Calculate PnL (with leverage)
        if self.position.side == PositionSide.LONG:
            pnl = (exit_price - self.position.entry_price) * self.position.quantity * self.position.leverage
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.quantity * self.position.leverage

        pnl_pct = pnl / (self.position.entry_price * self.position.quantity) * 100

        # Calculate fees (entry + exit)
        entry_fees = self._calculate_fees(self.position.entry_price, self.position.quantity)
        exit_fees = self._calculate_fees(exit_price, self.position.quantity)
        total_fees = entry_fees + exit_fees

        # Update capital
        self.capital += pnl - total_fees

        # Record trade
        trade = TradeResult(
            symbol=self.symbol,
            side=self.position.side,
            entry_time=self.position.entry_time,
            entry_price=self.position.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            quantity=self.position.quantity,
            pnl=pnl - total_fees,
            pnl_pct=pnl_pct,
            fees=total_fees,
            hold_bars=self.position.bars_held,
            exit_reason=reason,
            leverage=self.position.leverage
        )
        self.trades.append(trade)

        logger.info(
            f"CLOSED {self.position.side.value.upper()} | "
            f"Entry: {self.position.entry_price:.4f} | Exit: {exit_price:.4f} | "
            f"PnL: {pnl - total_fees:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
        )

        self.position = None

    def _open_position(self, signal: Signal):
        """Open new position"""
        if self.position is not None:
            logger.warning("Position already open, skipping entry signal")
            return

        # Get entry price from 1m data
        entry_price = self._get_entry_price_from_m1(signal)

        # Calculate position size (accounting for leverage)
        # With leverage, we can use more capital, but risk more
        position_value = self.capital * self.config.position_size_pct * self.config.leverage
        quantity = position_value / entry_price

        # Calculate TP/SL prices
        if signal.action == "long":
            side = PositionSide.LONG
            tp_price = entry_price * (1 + self.config.tp_pct)
            sl_price = entry_price * (1 - self.config.sl_pct)
        else:
            side = PositionSide.SHORT
            tp_price = entry_price * (1 - self.config.tp_pct)
            sl_price = entry_price * (1 + self.config.sl_pct)

        # Create position
        self.position = Position(
            symbol=self.symbol,
            side=side,
            entry_time=signal.time,
            entry_price=entry_price,
            quantity=quantity,
            leverage=self.config.leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            bars_held=0
        )

        # Deduct entry fee
        entry_fee = self._calculate_fees(entry_price, quantity)
        self.capital -= entry_fee

        logger.info(
            f"OPENED {side.value.upper()} ({self.config.leverage}x) | "
            f"Entry: {entry_price:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f} | "
            f"Quantity: {quantity:.6f}"
        )

    def run(self):
        """Run backtest"""
        logger.info(f"Starting backtest for {self.symbol}")
        logger.info(f"Config: initial_capital={self.config.initial_capital}, "
                   f"leverage={self.config.leverage}x, fee_rate={self.config.fee_rate}, "
                   f"tp_pct={self.config.tp_pct*100}%, sl_pct={self.config.sl_pct*100}%, "
                   f"exit_priority={self.config.exit_priority.value}")

        # Get signals from strategy
        signals = self.strategy(self.h1_data)

        # Process each 1h candle and its corresponding 1m candles
        h1_idx = 0
        m1_idx = 0

        while h1_idx < len(self.h1_data):
            h1_candle = self.h1_data.iloc[h1_idx]
            h1_time = h1_candle['openTime']

            # Process 1m candles until we reach the next 1h candle
            next_h1_time = self.h1_data.iloc[h1_idx + 1]['openTime'] if h1_idx + 1 < len(self.h1_data) else float('inf')

            # Find the range of 1m candles for this 1h period
            m1_in_period = self.m1_data[
                (self.m1_data['openTime'] >= h1_time) &
                (self.m1_data['openTime'] < next_h1_time)
            ]

            # Process each 1m candle in the period
            for _, m1_candle in m1_in_period.iterrows():
                m1_idx += 1

                # Check exit conditions if we have a position
                if self.position is not None:
                    should_exit, reason, exit_price = self._check_exit_conditions(m1_candle)
                    if should_exit:
                        self._close_position(m1_candle['openTime'], exit_price, reason)

            # After processing all 1m candles in the 1h period,
            # check if there's a signal at the close of this 1h candle
            for signal in signals:
                if signal.time == h1_time:
                    if signal.action in ["long", "short"]:
                        self._open_position(signal)
                    elif signal.action == "close_long" and self.position and self.position.side == PositionSide.LONG:
                        m1_after = self.m1_data[self.m1_data['openTime'] > h1_time]
                        if len(m1_after) > 0:
                            exit_price = m1_after.iloc[0]['open']
                            self._close_position(m1_after.iloc[0]['openTime'], exit_price, "manual")
                    elif signal.action == "close_short" and self.position and self.position.side == PositionSide.SHORT:
                        m1_after = self.m1_data[self.m1_data['openTime'] > h1_time]
                        if len(m1_after) > 0:
                            exit_price = m1_after.iloc[0]['open']
                            self._close_position(m1_after.iloc[0]['openTime'], exit_price, "manual")

            # Update bars held count
            if self.position is not None:
                self.position.bars_held += 1

            h1_idx += 1

        # Close any remaining position at the end
        if self.position is not None:
            if len(self.m1_data) > 0:
                last_m1 = self.m1_data.iloc[-1]
                self._close_position(last_m1['openTime'], last_m1['close'], "end_of_data")
            elif len(self.h1_data) > 0:
                last_h1 = self.h1_data.iloc[-1]
                self._close_position(last_h1['openTime'], last_h1['close'], "end_of_data")

        self._print_summary()

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trades:
            return 0.0

        equity_curve = [self.config.initial_capital]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_consecutive_wins_losses(self) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses"""
        if not self.trades:
            return 0, 0

        max_consecutive_win = 0
        max_consecutive_loss = 0
        current_win = 0
        current_loss = 0

        for trade in self.trades:
            if trade.pnl > 0:
                current_win += 1
                current_loss = 0
                max_consecutive_win = max(max_consecutive_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_consecutive_loss = max(max_consecutive_loss, current_loss)

        return max_consecutive_win, max_consecutive_loss

    def _calculate_profit_factor(self) -> float:
        """Calculate Profit Factor (gross profit / gross loss)"""
        if not self.trades:
            return 0.0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_risk_reward(self) -> float:
        """Calculate average Risk Reward ratio (avg win / avg loss)"""
        if not self.trades:
            return 0.0

        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]

        if not wins or not losses:
            return 0.0

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        return avg_win / avg_loss if avg_loss > 0 else 0.0

    def _print_summary(self):
        """Print detailed backtest summary"""
        if not self.trades:
            logger.info("No trades executed")
            return

        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        long_trades = [t for t in self.trades if t.side == PositionSide.LONG]
        short_trades = [t for t in self.trades if t.side == PositionSide.SHORT]

        long_wins = [t for t in long_trades if t.pnl > 0]
        short_wins = [t for t in short_trades if t.pnl > 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_fees = sum(t.fees for t in self.trades)

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        long_win_rate = len(long_wins) / len(long_trades) * 100 if long_trades else 0
        short_win_rate = len(short_wins) / len(short_trades) * 100 if short_trades else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        long_avg_profit = np.mean([t.pnl for t in long_wins]) if long_wins else 0
        short_avg_profit = np.mean([t.pnl for t in short_wins]) if short_wins else 0

        long_rr = long_avg_profit / abs(np.mean([t.pnl for t in long_trades if t.pnl < 0])) if long_trades and any(t.pnl < 0 for t in long_trades) else 0
        short_rr = short_avg_profit / abs(np.mean([t.pnl for t in short_trades if t.pnl < 0])) if short_trades and any(t.pnl < 0 for t in short_trades) else 0

        max_dd = self._calculate_max_drawdown()
        max_consecutive_win, max_consecutive_loss = self._calculate_consecutive_wins_losses()
        pf = self._calculate_profit_factor()
        rr = self._calculate_risk_reward()

        logger.info("=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Leverage: {self.config.leverage}x")
        logger.info(f"Period: {self.h1_data['datetime'].iloc[0]} to {self.h1_data['datetime'].iloc[-1]}")
        logger.info("-" * 70)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info("-" * 70)
        logger.info(f"Total PnL: {total_pnl:.2f}")
        logger.info(f"Total Fees: {total_fees:.2f}")
        logger.info(f"Final Capital: {self.capital:.2f}")
        logger.info(f"Return: {(self.capital - self.config.initial_capital) / self.config.initial_capital * 100:.2f}%")
        logger.info("-" * 70)
        logger.info(f"Max Drawdown: {max_dd:.2f}%")
        logger.info(f"Profit Factor: {pf:.2f}")
        logger.info(f"Risk Reward: {rr:.2f}")
        logger.info(f"Max Consecutive Win: {max_consecutive_win}")
        logger.info(f"Max Consecutive Loss: {max_consecutive_loss}")
        logger.info("-" * 70)
        logger.info(f"Long Trades: {len(long_trades)} | Long Win Rate: {long_win_rate:.2f}%")
        logger.info(f"Short Trades: {len(short_trades)} | Short Win Rate: {short_win_rate:.2f}%")
        logger.info(f"Long Avg Profit: {long_avg_profit:.2f}")
        logger.info(f"Short Avg Profit: {short_avg_profit:.2f}")
        logger.info(f"Long RR: {long_rr:.2f}")
        logger.info(f"Short RR: {short_rr:.2f}")
        logger.info("=" * 70)

    def get_results(self) -> dict:
        """Get backtest results as dictionary"""
        if not self.trades:
            return {}

        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        long_trades = [t for t in self.trades if t.side == PositionSide.LONG]
        short_trades = [t for t in self.trades if t.side == PositionSide.SHORT]
        long_wins = [t for t in long_trades if t.pnl > 0]
        short_wins = [t for t in short_trades if t.pnl > 0]

        long_trades_loss = [t for t in long_trades if t.pnl < 0]
        short_trades_loss = [t for t in short_trades if t.pnl < 0]

        long_avg_profit = np.mean([t.pnl for t in long_wins]) if long_wins else 0
        short_avg_profit = np.mean([t.pnl for t in short_wins]) if short_wins else 0

        long_rr = long_avg_profit / abs(np.mean([t.pnl for t in long_trades_loss])) if long_trades_loss and long_avg_profit > 0 else 0
        short_rr = short_avg_profit / abs(np.mean([t.pnl for t in short_trades_loss])) if short_trades_loss and short_avg_profit > 0 else 0

        return {
            "symbol": self.symbol,
            "leverage": self.config.leverage,
            "period_start": str(self.h1_data['datetime'].iloc[0]) if len(self.h1_data) > 0 else "N/A",
            "period_end": str(self.h1_data['datetime'].iloc[-1]) if len(self.h1_data) > 0 else "N/A",
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": total_trades - len(winning_trades),
            "win_rate": len(winning_trades) / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl": sum(t.pnl for t in self.trades),
            "total_fees": sum(t.fees for t in self.trades),
            "final_capital": self.capital,
            "return_pct": (self.capital - self.config.initial_capital) / self.config.initial_capital * 100,
            "max_drawdown": self._calculate_max_drawdown(),
            "profit_factor": self._calculate_profit_factor(),
            "risk_reward": self._calculate_risk_reward(),
            "max_consecutive_win": self._calculate_consecutive_wins_losses()[0],
            "max_consecutive_loss": self._calculate_consecutive_wins_losses()[1],
            "long_trades": len(long_trades),
            "long_wins": len(long_wins),
            "long_win_rate": len(long_wins) / len(long_trades) * 100 if long_trades else 0,
            "short_trades": len(short_trades),
            "short_wins": len(short_wins),
            "short_win_rate": len(short_wins) / len(short_trades) * 100 if short_trades else 0,
            "long_avg_profit": long_avg_profit,
            "short_avg_profit": short_avg_profit,
            "long_rr": long_rr,
            "short_rr": short_rr,
            "trades": self.trades
        }
