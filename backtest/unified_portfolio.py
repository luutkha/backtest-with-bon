"""
Unified Portfolio Backtest Engine (Optimized)

A backtest engine that treats all symbols as part of a single portfolio:
- Single capital pool shared across all symbols
- Only one position per symbol at a time
- Processes all symbols simultaneously, respecting capital constraints
- Optimized for speed using vectorized operations
"""

import logging
from typing import Optional, List, Callable, Dict, Any, Tuple, Set
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .data.data_loader import DataLoader, DataConfig
from .signals.signal_generator import SignalGenerator, Signal, SignalType
from .execution.execution_engine import (
    ExecutionEngine, ExecutionConfig, IntradaySimulator,
    PositionSide, ExitPriority, Trade
)
from .analytics.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class UnifiedPortfolioConfig:
    """Configuration for unified portfolio backtest"""
    initial_capital: float = 100000.0
    max_positions: int = 10
    position_size_pct: float = 0.1
    fee_rate: float = 0.0004
    slippage: float = 0.0001
    tp_pct: float = 0.02
    sl_pct: float = 0.01
    leverage: float = 1.0
    exit_priority: ExitPriority = ExitPriority.CONSERVATIVE
    verbose: bool = True


class UnifiedPortfolioBacktest:
    """
    Unified portfolio backtest engine - optimized version.

    Processes symbols in batches at 1h candle boundaries for efficiency.
    """

    def __init__(
        self,
        config: UnifiedPortfolioConfig,
        strategy: Callable,
        data_dir: str,
    ):
        self.config = config
        self.strategy = strategy
        self.data_dir = data_dir

        # Execution engine
        exec_config = ExecutionConfig(
            fee_rate=config.fee_rate,
            slippage=config.slippage,
            exit_priority=config.exit_priority,
            tp_pct=config.tp_pct,
            sl_pct=config.sl_pct,
            leverage=config.leverage,
        )
        self.execution_engine = ExecutionEngine(exec_config)

        # Data loader
        self.data_loader = DataLoader(DataConfig(base_path=data_dir))

        # State
        self.symbols: List[str] = []
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.symbol_m1_data: Dict[str, pd.DataFrame] = {}
        self.all_trades: List[Trade] = []
        self.capital: float = config.initial_capital

    def load_data(self, symbols: List[str]) -> None:
        """Load data for all symbols"""
        self.symbols = symbols

        if self.config.verbose:
            logger.info(f"Loading data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                h1_data, m1_data = self.data_loader.load_pair_data(
                    symbol=symbol,
                    base_path=self.data_dir,
                )
                h1_data, m1_data = self.data_loader.align_timeframes(h1_data, m1_data)
                self.symbol_data[symbol] = h1_data
                self.symbol_m1_data[symbol] = m1_data
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")

        if self.config.verbose:
            logger.info(f"Loaded {len(self.symbol_data)} symbols")

    def run_backtest(self) -> Dict[str, Any]:
        """Run unified portfolio backtest - optimized"""
        if not self.symbol_data:
            raise ValueError("No data loaded")

        # Find common time range (intersection of all symbols)
        common_start = max(df['opentime'].min() for df in self.symbol_data.values())
        common_end = min(df['opentime'].max() for df in self.symbol_data.values())

        if self.config.verbose:
            logger.info(f"Running unified portfolio backtest...")
            logger.info(f"Capital: ${self.config.initial_capital:,.2f}")
            logger.info(f"Max positions: {self.config.max_positions}")

        # Pre-generate signals for each symbol
        signal_generators = {}
        symbol_signals = {}  # symbol -> {timestamp -> signal}

        for symbol in self.symbol_data:
            signal_generators[symbol] = SignalGenerator(strategy_func=self.strategy)
            try:
                signals = self.strategy(self.symbol_data[symbol])
                # Convert to dict by timestamp
                sig_dict = {}
                for sig in signals:
                    if sig.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
                        sig_dict[sig.timestamp] = sig
                symbol_signals[symbol] = sig_dict
            except Exception as e:
                logger.warning(f"Error generating signals for {symbol}: {e}")
                symbol_signals[symbol] = {}

        # Get all unique 1h timestamps
        all_timestamps = set()
        for df in self.symbol_data.values():
            all_timestamps.update(df['opentime'].values)
        all_timestamps = sorted([t for t in all_timestamps if common_start <= t <= common_end])

        # State
        open_positions: Dict[str, dict] = {}  # symbol -> position info

        # Process each 1h timestamp
        for ts_idx, ts in enumerate(all_timestamps):
            # Check exits for open positions
            positions_to_close = []
            for symbol, pos in open_positions.items():
                m1_data = self.symbol_m1_data.get(symbol)
                if m1_data is None:
                    continue

                # Get next timestamp boundaries
                next_ts = all_timestamps[ts_idx + 1] if ts_idx + 1 < len(all_timestamps) else float('inf')

                # Get m1 candles in this period
                m1_in_period = m1_data[
                    (m1_data['opentime'] > ts) &
                    (m1_data['opentime'] <= next_ts)
                ]

                if len(m1_in_period) == 0:
                    pos['bars_held'] += 1
                    continue

                highs = m1_in_period['high'].values
                lows = m1_in_period['low'].values
                exit_price = None
                exit_reason = None

                if pos['side'] == PositionSide.LONG:
                    pos['highest'] = max(pos['highest'], highs.max())

                    if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                        if lows.min() <= pos['sl']:
                            exit_price = pos['sl']
                            exit_reason = 'sl'
                        elif highs.max() >= pos['tp']:
                            exit_price = pos['tp']
                            exit_reason = 'tp'
                    else:
                        if highs.max() >= pos['tp']:
                            exit_price = pos['tp']
                            exit_reason = 'tp'
                        elif lows.min() <= pos['sl']:
                            exit_price = pos['sl']
                            exit_reason = 'sl'
                else:  # SHORT
                    pos['lowest'] = min(pos['lowest'], lows.min())

                    if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                        if highs.max() >= pos['sl']:
                            exit_price = pos['sl']
                            exit_reason = 'sl'
                        elif lows.min() <= pos['tp']:
                            exit_price = pos['tp']
                            exit_reason = 'tp'
                    else:
                        if lows.min() <= pos['tp']:
                            exit_price = pos['tp']
                            exit_reason = 'tp'
                        elif highs.max() >= pos['sl']:
                            exit_price = pos['sl']
                            exit_reason = 'sl'

                if exit_price is not None:
                    # Calculate PnL
                    if pos['side'] == PositionSide.LONG:
                        pnl = (exit_price - pos['entry_price']) * pos['quantity'] * self.config.leverage
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity'] * self.config.leverage

                    exit_fee = self.execution_engine.calculate_fee(exit_price, pos['quantity'])
                    pnl -= exit_fee

                    # Return capital + PnL
                    allocated = pos['entry_price'] * pos['quantity'] / self.config.leverage
                    self.capital += allocated + pnl

                    trade = Trade(
                        symbol=symbol,
                        side=pos['side'],
                        entry_time=pos['entry_time'],
                        entry_price=pos['entry_price'],
                        exit_time=int(m1_in_period.iloc[0]['opentime']),
                        exit_price=exit_price,
                        quantity=pos['quantity'],
                        pnl=pnl,
                        pnl_pct=(pnl / allocated * 100) if allocated > 0 else 0,
                        fees=exit_fee,
                        hold_bars=pos['bars_held'],
                        exit_reason=exit_reason,
                        leverage=self.config.leverage,
                    )
                    self.all_trades.append(trade)
                    positions_to_close.append(symbol)

                pos['bars_held'] += 1

            # Close positions
            for symbol in positions_to_close:
                del open_positions[symbol]

            # Process entries if we have available slots
            if len(open_positions) < self.config.max_positions and self.capital > 0:
                available = self.capital * self.config.position_size_pct
                slots = self.config.max_positions - len(open_positions)

                # Find signals at current timestamp
                for symbol in self.symbol_data:
                    if symbol in open_positions:
                        continue

                    sigs = symbol_signals.get(symbol, {})
                    if ts not in sigs:
                        continue

                    sig = sigs[ts]

                    # Determine side
                    if sig.signal_type == SignalType.LONG_ENTRY:
                        side = PositionSide.LONG
                    elif sig.signal_type == SignalType.SHORT_ENTRY:
                        side = PositionSide.SHORT
                    else:
                        continue

                    # Calculate position
                    entry_price = self.execution_engine.apply_slippage(sig.price, side)
                    per_slot_value = available / slots
                    quantity = per_slot_value / entry_price

                    tp_price, sl_price = self.execution_engine.calculate_tp_sl(entry_price, side)

                    # Deduct entry fee
                    entry_fee = self.execution_engine.calculate_fee(entry_price, quantity)
                    allocated = entry_price * quantity / self.config.leverage

                    open_positions[symbol] = {
                        'side': side,
                        'entry_time': ts,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'tp': tp_price,
                        'sl': sl_price,
                        'highest': entry_price,
                        'lowest': entry_price,
                        'bars_held': 0,
                    }

                    # Deduct capital
                    self.capital -= (allocated + entry_fee)
                    slots -= 1

                    if slots <= 0:
                        break

        # Close remaining positions
        end_ts = all_timestamps[-1] if all_timestamps else 0
        for symbol, pos in open_positions.items():
            m1_data = self.symbol_m1_data.get(symbol)
            if m1_data is not None and len(m1_data) > 0:
                exit_price = m1_data.iloc[-1]['close']
            else:
                exit_price = pos['entry_price']

            if pos['side'] == PositionSide.LONG:
                pnl = (exit_price - pos['entry_price']) * pos['quantity'] * self.config.leverage
            else:
                pnl = (pos['entry_price'] - exit_price) * pos['quantity'] * self.config.leverage

            exit_fee = self.execution_engine.calculate_fee(exit_price, pos['quantity'])
            pnl -= exit_fee

            allocated = pos['entry_price'] * pos['quantity'] / self.config.leverage
            self.capital += allocated + pnl

            trade = Trade(
                symbol=symbol,
                side=pos['side'],
                entry_time=pos['entry_time'],
                entry_price=pos['entry_price'],
                exit_time=int(end_ts),
                exit_price=exit_price,
                quantity=pos['quantity'],
                pnl=pnl,
                pnl_pct=(pnl / allocated * 100) if allocated > 0 else 0,
                fees=exit_fee,
                hold_bars=pos['bars_held'],
                exit_reason='end_of_data',
                leverage=self.config.leverage,
            )
            self.all_trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics()

        return {
            'config': {
                'initial_capital': self.config.initial_capital,
                'max_positions': self.config.max_positions,
                'position_size_pct': self.config.position_size_pct,
                'symbols': len(self.symbol_data),
            },
            'metrics': metrics,
            'trades': self.all_trades,
            'final_capital': self.capital,
        }

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio metrics"""
        if not self.all_trades:
            return {'total_trades': 0, 'total_pnl': 0, 'return_pct': 0}

        trades_df = pd.DataFrame([
            {'pnl': t.pnl, 'pnl_pct': t.pnl_pct, 'fees': t.fees, 'hold_bars': t.hold_bars}
            for t in self.all_trades
        ])

        metrics = {}
        metrics['total_trades'] = len(self.all_trades)
        metrics['total_pnl'] = trades_df['pnl'].sum()
        metrics['return_pct'] = (self.capital - self.config.initial_capital) / self.config.initial_capital * 100
        metrics['final_capital'] = self.capital
        metrics['total_fees'] = trades_df['fees'].sum()

        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        metrics['winning_trades'] = len(winning)
        metrics['losing_trades'] = len(losing)
        metrics['win_rate'] = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        metrics['avg_win'] = winning['pnl'].mean() if len(winning) > 0 else 0
        metrics['avg_loss'] = losing['pnl'].mean() if len(losing) > 0 else 0

        gross_profit = winning['pnl'].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        equity_curve = [self.config.initial_capital]
        for pnl in trades_df['pnl']:
            equity_curve.append(equity_curve[-1] + pnl)

        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100
        metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0

        metrics['avg_holding_bars'] = trades_df['hold_bars'].mean() if len(trades_df) > 0 else 0

        return metrics

    def print_summary(self) -> None:
        """Print backtest summary"""
        if not self.all_trades:
            logger.info("No trades executed")
            return

        m = self._calculate_metrics()

        logger.info("=" * 60)
        logger.info("UNIFIED PORTFOLIO BACKTEST")
        logger.info("=" * 60)
        logger.info(f"Symbols: {len(self.symbol_data)}")
        logger.info(f"Capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Max Positions: {self.config.max_positions}")
        logger.info("-" * 60)
        logger.info(f"Total Trades: {m['total_trades']}")
        logger.info(f"Win Rate: {m['win_rate']:.1f}%")
        logger.info("-" * 60)
        logger.info(f"Total PnL: ${m['total_pnl']:,.2f}")
        logger.info(f"Return: {m['return_pct']:.2f}%")
        logger.info(f"Final Capital: ${self.capital:,.2f}")
        logger.info(f"Max DD: {m['max_drawdown']:.2f}%")
        logger.info(f"Profit Factor: {m['profit_factor']:.2f}")
        logger.info("=" * 60)
