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
    # Phase 1 fixes (backward compatible defaults)
    use_cooldown_after_sl: bool = False   # Issue 1: No re-entry at same timestamp after SL
    use_actual_exit_price: bool = False    # Issue 2: Exit price = candle low/high, not SL level
    # Phase 2 fix
    sl_close_beyond: bool = False          # Issue 3: Require candle close beyond SL level
    # Phase 3 fix
    mark_to_market: bool = False           # Issue 5: Mark-to-market equity at each 1h candle

    def validate(self) -> bool:
        """
        Validate the unified portfolio configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if self.tp_pct <= 0:
            raise ValueError(f"tp_pct must be positive, got {self.tp_pct}")

        if self.sl_pct <= 0:
            raise ValueError(f"sl_pct must be positive, got {self.sl_pct}")

        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")

        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")

        if self.leverage < 1:
            raise ValueError(f"leverage must be >= 1, got {self.leverage}")

        if self.position_size_pct <= 0 or self.position_size_pct > 1.0:
            raise ValueError(f"position_size_pct must be between 0 and 1, got {self.position_size_pct}")

        if self.fee_rate < 0:
            raise ValueError(f"fee_rate must be non-negative, got {self.fee_rate}")

        if self.slippage < 0:
            raise ValueError(f"slippage must be non-negative, got {self.slippage}")

        return True


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

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

        # State
        self.symbols: List[str] = []
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.symbol_m1_data: Dict[str, pd.DataFrame] = {}
        self.all_trades: List[Trade] = []
        self.capital: float = config.initial_capital
        self.equity_df: Optional[pd.DataFrame] = None
        self.equity_timeline: List[Tuple[int, float]] = []  # (timestamp, equity) for MTM

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
        cooldown_symbols: Set[str] = set()  # Symbols in cooldown after SL (Issue 1 fix)

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
                opens = m1_in_period['open'].values
                closes = m1_in_period['close'].values
                exit_price = None
                exit_reason = None
                exit_fill_time = None

                if pos['side'] == PositionSide.LONG:
                    pos['highest'] = max(pos['highest'], highs.max())

                    if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                        # Check SL first - find FIRST candle where low <= SL
                        for i, low in enumerate(lows):
                            # Issue 3 fix: Optionally require close beyond SL
                            if self.config.sl_close_beyond:
                                # Require candle close to go below SL
                                if closes[i] <= pos['sl']:
                                    exit_price = closes[i] if self.config.use_actual_exit_price else pos['sl']
                                    exit_reason = 'sl'
                                    exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                    break
                            elif low <= pos['sl']:
                                # Issue 2 fix: Use actual candle low for SL (market order)
                                exit_price = low if self.config.use_actual_exit_price else pos['sl']
                                exit_reason = 'sl'
                                exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                break
                        # Check TP second - find FIRST candle where high >= TP
                        if exit_price is None:
                            for i, high in enumerate(highs):
                                if high >= pos['tp']:
                                    # TP uses limit price (correct as-is)
                                    exit_price = pos['tp']
                                    exit_reason = 'tp'
                                    exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                    break
                    else:
                        # AGGRESSIVE: Check TP first
                        for i, high in enumerate(highs):
                            if high >= pos['tp']:
                                exit_price = pos['tp']
                                exit_reason = 'tp'
                                exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                break
                        if exit_price is None:
                            for i, low in enumerate(lows):
                                # Issue 3 fix: Optionally require close beyond SL
                                if self.config.sl_close_beyond:
                                    if closes[i] <= pos['sl']:
                                        exit_price = closes[i] if self.config.use_actual_exit_price else pos['sl']
                                        exit_reason = 'sl'
                                        exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                        break
                                elif low <= pos['sl']:
                                    # Issue 2 fix: Use actual candle low for SL (market order)
                                    exit_price = low if self.config.use_actual_exit_price else pos['sl']
                                    exit_reason = 'sl'
                                    exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                    break
                else:  # SHORT
                    pos['lowest'] = min(pos['lowest'], lows.min())

                    if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                        # Check SL first - find FIRST candle where high >= SL
                        for i, high in enumerate(highs):
                            # Issue 3 fix: Optionally require close beyond SL
                            if self.config.sl_close_beyond:
                                # Require candle close to go above SL
                                if closes[i] >= pos['sl']:
                                    exit_price = closes[i] if self.config.use_actual_exit_price else pos['sl']
                                    exit_reason = 'sl'
                                    exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                    break
                            elif high >= pos['sl']:
                                # Issue 2 fix: Use actual candle high for SL (market order)
                                exit_price = high if self.config.use_actual_exit_price else pos['sl']
                                exit_reason = 'sl'
                                exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                break
                        # Check TP second - find FIRST candle where low <= TP
                        if exit_price is None:
                            for i, low in enumerate(lows):
                                if low <= pos['tp']:
                                    # TP uses limit price (correct as-is)
                                    exit_price = pos['tp']
                                    exit_reason = 'tp'
                                    exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                    break
                    else:
                        # AGGRESSIVE: Check TP first
                        for i, low in enumerate(lows):
                            if low <= pos['tp']:
                                exit_price = pos['tp']
                                exit_reason = 'tp'
                                exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                break
                        if exit_price is None:
                            for i, high in enumerate(highs):
                                # Issue 3 fix: Optionally require close beyond SL
                                if self.config.sl_close_beyond:
                                    if closes[i] >= pos['sl']:
                                        exit_price = closes[i] if self.config.use_actual_exit_price else pos['sl']
                                        exit_reason = 'sl'
                                        exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                        break
                                elif high >= pos['sl']:
                                    # Issue 2 fix: Use actual candle high for SL (market order)
                                    exit_price = high if self.config.use_actual_exit_price else pos['sl']
                                    exit_reason = 'sl'
                                    exit_fill_time = int(m1_in_period.iloc[i]['opentime'])
                                    break

                if exit_price is not None:
                    # Calculate PnL
                    if pos['side'] == PositionSide.LONG:
                        pnl = (exit_price - pos['entry_price']) * pos['quantity'] * self.config.leverage
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity'] * self.config.leverage

                    exit_fee = self.execution_engine.calculate_fee(exit_price, pos['quantity'])
                    entry_fee = pos.get('entry_fee', 0)

                    # Deduct both entry and exit fees from PnL
                    total_fees = entry_fee + exit_fee
                    pnl -= total_fees

                    # Return capital + PnL
                    allocated = pos['entry_price'] * pos['quantity'] / self.config.leverage
                    self.capital += allocated + pnl

                    # Calculate actual hold time based on fill times
                    actual_hold_ms = exit_fill_time - pos['entry_time']
                    actual_hold_bars = int(actual_hold_ms / (60 * 60 * 1000))  # Convert to hours

                    # Wait time for TP/SL is from entry to exit
                    wait_tp_sl_ms = actual_hold_ms

                    trade = Trade(
                        symbol=symbol,
                        side=pos['side'],
                        entry_time=pos['entry_time'],
                        entry_price=pos['entry_price'],
                        exit_time=exit_fill_time,
                        exit_price=exit_price,
                        quantity=pos['quantity'],
                        pnl=pnl,
                        pnl_pct=(pnl / allocated * 100) if allocated > 0 else 0,
                        fees=total_fees,
                        hold_bars=actual_hold_bars,
                        exit_reason=exit_reason,
                        leverage=self.config.leverage,
                        tp_price=pos.get('tp', 0.0),
                        sl_price=pos.get('sl', 0.0),
                        entry_signal_time=pos.get('signal_time', pos['entry_time']),
                    )
                    self.all_trades.append(trade)
                    positions_to_close.append(symbol)
                    # Issue 1 fix: Add to cooldown if closed by SL
                    if exit_reason == 'sl' and self.config.use_cooldown_after_sl:
                        cooldown_symbols.add(symbol)
                else:
                    # Only increment bars held if position was not closed this period
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
                    # Issue 1 fix: Skip symbols in cooldown after SL
                    if self.config.use_cooldown_after_sl and symbol in cooldown_symbols:
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

                    # Get the first 5m candle after the signal for actual entry price and time
                    m1_data = self.symbol_m1_data.get(symbol)
                    if m1_data is None:
                        continue

                    # Find first 5m candle with opentime > ts (the 1h signal timestamp)
                    m1_after = m1_data[m1_data['opentime'] > ts]
                    if len(m1_after) == 0:
                        continue

                    first_m1_candle = m1_after.iloc[0]
                    actual_entry_time = int(first_m1_candle['opentime'])
                    # Use actual 5m candle open price for entry (with slippage)
                    actual_entry_price = self.execution_engine.apply_slippage(
                        float(first_m1_candle['open']), side
                    )

                    per_slot_value = available / slots
                    quantity = per_slot_value / actual_entry_price

                    tp_price, sl_price = self.execution_engine.calculate_tp_sl(actual_entry_price, side)

                    # Calculate entry fee
                    entry_fee = self.execution_engine.calculate_fee(actual_entry_price, quantity)
                    allocated = actual_entry_price * quantity / self.config.leverage

                    open_positions[symbol] = {
                        'side': side,
                        'entry_time': actual_entry_time,  # Actual 5m candle timestamp
                        'entry_price': actual_entry_price,
                        'quantity': quantity,
                        'tp': tp_price,
                        'sl': sl_price,
                        'highest': actual_entry_price,
                        'lowest': actual_entry_price,
                        'bars_held': 0,
                        'signal_time': ts,  # Original 1h signal timestamp
                        'entry_fee': entry_fee,  # Track entry fee for deduction at exit
                    }

                    # Deduct only the allocated margin (fees will be deducted at exit)
                    self.capital -= allocated
                    slots -= 1

                    if slots <= 0:
                        break

            # Issue 5 fix: Mark-to-market equity at each 1h candle
            if self.config.mark_to_market and open_positions:
                # Calculate unrealized PnL for open positions
                unrealized_pnl = 0.0
                allocated_margin = 0.0
                for symbol, pos in open_positions.items():
                    # Get current price at this 1h candle
                    h1_data = self.symbol_data.get(symbol)
                    if h1_data is not None:
                        h1_row = h1_data[h1_data['opentime'] == ts]
                        if len(h1_row) > 0:
                            current_price = h1_row.iloc[0]['close']
                            if pos['side'] == PositionSide.LONG:
                                unrealized_pnl += (current_price - pos['entry_price']) * pos['quantity'] * self.config.leverage
                            else:
                                unrealized_pnl += (pos['entry_price'] - current_price) * pos['quantity'] * self.config.leverage
                    # Track allocated margin (capital tied up in this position)
                    allocated_margin += pos['entry_price'] * pos['quantity'] / self.config.leverage
                # MTM equity = cash + allocated margin + unrealized PnL
                mtm_equity = self.capital + allocated_margin + unrealized_pnl
                self.equity_timeline.append((ts, mtm_equity))

            # Issue 1 fix: Clear cooldown at end of timestamp iteration
            # Cooldown symbols that were NOT re-entered this timestamp can trade next timestamp
            if self.config.use_cooldown_after_sl:
                cooldown_symbols.clear()

        # Close remaining positions at end of data
        for symbol, pos in open_positions.items():
            m1_data = self.symbol_m1_data.get(symbol)
            if m1_data is not None and len(m1_data) > 0:
                exit_price = m1_data.iloc[-1]['close']
                exit_fill_time = int(m1_data.iloc[-1]['opentime'])
            else:
                exit_price = pos['entry_price']
                exit_fill_time = int(pos['entry_time'])

            if pos['side'] == PositionSide.LONG:
                pnl = (exit_price - pos['entry_price']) * pos['quantity'] * self.config.leverage
            else:
                pnl = (pos['entry_price'] - exit_price) * pos['quantity'] * self.config.leverage

            exit_fee = self.execution_engine.calculate_fee(exit_price, pos['quantity'])
            entry_fee = pos.get('entry_fee', 0)

            # Deduct both entry and exit fees from PnL
            total_fees = entry_fee + exit_fee
            pnl -= total_fees

            allocated = pos['entry_price'] * pos['quantity'] / self.config.leverage
            self.capital += allocated + pnl

            # Calculate actual hold time
            actual_hold_ms = exit_fill_time - pos['entry_time']
            actual_hold_bars = int(actual_hold_ms / (60 * 60 * 1000))

            trade = Trade(
                symbol=symbol,
                side=pos['side'],
                entry_time=pos['entry_time'],
                entry_price=pos['entry_price'],
                exit_time=exit_fill_time,
                exit_price=exit_price,
                quantity=pos['quantity'],
                pnl=pnl,
                pnl_pct=(pnl / allocated * 100) if allocated > 0 else 0,
                fees=total_fees,
                hold_bars=actual_hold_bars,
                exit_reason='end_of_data',
                leverage=self.config.leverage,
                tp_price=pos.get('tp', 0.0),
                sl_price=pos.get('sl', 0.0),
                entry_signal_time=pos.get('signal_time', pos['entry_time']),
            )
            self.all_trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Create trades DataFrame
        trades_df = self._trades_to_dataframe()

        return {
            'config': {
                'initial_capital': self.config.initial_capital,
                'max_positions': self.config.max_positions,
                'position_size_pct': self.config.position_size_pct,
                'symbols': len(self.symbol_data),
            },
            'metrics': metrics,
            'trades': self.all_trades,
            'trades_df': trades_df,
            'equity_df': self.equity_df,
            'final_capital': self.capital,
        }

    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades list to DataFrame"""
        if not self.all_trades:
            return pd.DataFrame()

        records = []
        for trade in self.all_trades:
            # Calculate status (TP or SL or OTHER)
            status = 'OTHER'
            if trade.exit_reason == 'tp':
                status = 'TP'
            elif trade.exit_reason == 'sl':
                status = 'SL'

            # Calculate durations
            wait_filled_ms = trade.entry_time - trade.entry_signal_time if trade.entry_signal_time > 0 else 0
            wait_tp_sl_ms = trade.exit_time - trade.entry_time if trade.exit_time > trade.entry_time else 0

            # Calculate ROI (return on investment)
            roi = (trade.pnl / (trade.entry_price * trade.quantity)) * 100 if trade.entry_price * trade.quantity > 0 else 0

            records.append({
                'symbol': trade.symbol,
                'side': trade.side.value if hasattr(trade.side, 'value') else trade.side,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'tp_price': trade.tp_price,
                'sl_price': trade.sl_price,
                'quantity': trade.quantity,
                'wait_filled_ms': wait_filled_ms,
                'wait_tp_sl_ms': wait_tp_sl_ms,
                'roi_pct': roi,
                'profit': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'fees': trade.fees,
                'hold_bars': trade.hold_bars,
                'status': status,
                'exit_reason': trade.exit_reason,
                'leverage': trade.leverage,
            })

        return pd.DataFrame(records)

    def export_trades_csv(self, filepath: str) -> None:
        """Export trades to CSV file

        Args:
            filepath: Path to output CSV file
        """
        trades_df = self._trades_to_dataframe()
        if not trades_df.empty:
            trades_df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(trades_df)} trades to {filepath}")

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio metrics using MetricsCalculator"""
        if not self.all_trades:
            return {'total_trades': 0, 'total_pnl': 0, 'return_pct': 0}

        # Build equity curve DataFrame for MetricsCalculator
        equity_curve = [self.config.initial_capital]

        # Issue 5 fix: Use MTM equity timeline if available
        if self.config.mark_to_market and self.equity_timeline:
            # Use mark-to-market equity at each 1h candle
            timestamps = [0] + [t[0] for t in self.equity_timeline]
            equity_values = [self.config.initial_capital] + [t[1] for t in self.equity_timeline]
            self.equity_df = pd.DataFrame({
                'timestamp': timestamps,
                'equity': equity_values
            })
        else:
            # Original: equity only at trade close points
            for trade in self.all_trades:
                equity_curve.append(equity_curve[-1] + trade.pnl)

            # Create equity DataFrame with timestamps
            timestamps = [0] + [t.exit_time for t in self.all_trades]
            self.equity_df = pd.DataFrame({
                'timestamp': timestamps,
                'equity': equity_curve
            })

        # Use MetricsCalculator to calculate all metrics
        metrics = self.metrics_calculator.calculate_all(
            trades=self.all_trades,
            equity_curve=self.equity_df,
            initial_capital=self.config.initial_capital,
        )

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
