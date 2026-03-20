"""
Execution Layer - Trade lifecycle simulation using intrabar (1m/5m) candles.
Responsibilities:
- After a 1h signal closes, enter using next available intrabar candle
- Simulate realistic execution price
- Process each intrabar candle sequentially
- Support long/short, take profit, stop loss, trailing stop
- Support fee per side and configurable slippage model

Intrabar rules:
- TP/SL must trigger using intrabar high/low
- If TP and SL hit in same candle: configurable execution priority
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """Reason for position exit"""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    ENDT_OF_DATA = "end_of_data"
    SIGNAL_EXIT = "signal_exit"


class ExitPriority(Enum):
    """TP/SL execution priority when both hit in same candle"""
    CONSERVATIVE = "conservative"  # SL first
    AGGRESSIVE = "aggressive"      # TP first


class SlippageModel(Enum):
    """Slippage model types"""
    FIXED = "fixed"           # Fixed percentage slippage
    VOLUME_BASED = "volume"   # Slippage based on volume (placeholder)
    RANDOM = "random"         # Random slippage within range


class OrderType(Enum):
    """Order types"""
    MARKET = "market"   # Market order - execute immediately
    LIMIT = "limit"     # Limit order - execute when price reaches level


@dataclass
class PendingOrder:
    """Represents a pending limit order waiting to be filled"""
    symbol: str
    side: PositionSide
    order_time: int           # Timestamp when order was placed
    limit_price: float        # Price at which order should execute
    quantity: float
    leverage: float = 1.0
    bars_waited: int = 0
    tp_price: float = 0.0
    sl_price: float = 0.0


@dataclass
class ExecutionConfig:
    """Execution layer configuration"""
    fee_rate: float = 0.0004           # Fee per side (0.04% = Binance taker)
    slippage: float = 0.0              # Slippage percentage
    slippage_model: SlippageModel = SlippageModel.FIXED
    exit_priority: ExitPriority = ExitPriority.CONSERVATIVE
    tp_pct: float = 0.02               # Take profit percentage
    sl_pct: float = 0.01               # Stop loss percentage
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.0
    leverage: float = 1.0
    # Limit order settings
    order_type: OrderType = OrderType.MARKET
    limit_order_offset: float = 0.001   # Limit order price offset (0.1% away from market)
    limit_order_timeout_bars: int = 3   # Max bars to wait for limit order fill (0 = wait indefinitely)


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
class FillInfo:
    """Information about a trade fill"""
    timestamp: int
    price: float
    quantity: float
    fee: float
    slippage_applied: float


@dataclass
class LimitOrder:
    """Pending limit order"""
    symbol: str
    side: PositionSide
    order_type: OrderType
    limit_price: float  # Price at which to execute
    quantity: float
    created_time: int
    expires_at: Optional[int] = None  # Optional expiration time
    filled: bool = False
    cancelled: bool = False


@dataclass
class Trade:
    """Completed trade record"""
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
    exit_reason: str
    leverage: float = 1.0
    entry_fill: Optional[FillInfo] = None
    exit_fill: Optional[FillInfo] = None


class ExecutionEngine:
    """
    Handles trade execution and intrabar simulation.
    Stateless regarding signals - receives signals and executes them.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config

    def calculate_position_size(
        self,
        capital: float,
        position_size_pct: float,
        entry_price: float
    ) -> float:
        """
        Calculate position size based on capital and position size percentage.

        Args:
            capital: Available capital
            position_size_pct: Position size as decimal (e.g., 0.95 = 95%)
            entry_price: Entry price

        Returns:
            Position quantity
        """
        position_value = capital * position_size_pct * self.config.leverage
        quantity = position_value / entry_price
        return quantity

    def calculate_tp_sl(
        self,
        entry_price: float,
        side: PositionSide
    ) -> Tuple[float, float]:
        """
        Calculate take profit and stop loss prices.

        Args:
            entry_price: Entry price
            side: Position side

        Returns:
            Tuple of (tp_price, sl_price)
        """
        if side == PositionSide.LONG:
            tp_price = entry_price * (1 + self.config.tp_pct)
            sl_price = entry_price * (1 - self.config.sl_pct)
        else:  # SHORT
            tp_price = entry_price * (1 - self.config.tp_pct)
            sl_price = entry_price * (1 + self.config.sl_pct)

        return tp_price, sl_price

    def apply_slippage(
        self,
        price: float,
        side: PositionSide,
        volume: float = 0.0  # For future volume-based models
    ) -> float:
        """
        Apply slippage to price based on configured model.

        Args:
            price: Base price
            side: Position side
            volume: Trading volume (for volume-based models)

        Returns:
            Price after slippage
        """
        if self.config.slippage_model == SlippageModel.FIXED:
            slippage = self.config.slippage
        elif self.config.slippage_model == SlippageModel.RANDOM:
            slippage = np.random.uniform(0, self.config.slippage * 2)
        else:
            slippage = self.config.slippage

        if side == PositionSide.LONG:
            # Buy: slippage increases entry price
            return price * (1 + slippage)
        else:
            # Sell: slippage decreases entry price
            return price * (1 - slippage)

    def calculate_fee(self, price: float, quantity: float) -> float:
        """Calculate trading fee"""
        return price * quantity * self.config.fee_rate

    def calculate_limit_price(self, market_price: float, side: PositionSide) -> float:
        """
        Calculate limit order price based on configured offset.

        For LONG: limit price below market (buy dip)
        For SHORT: limit price above market (sell rip)

        Args:
            market_price: Current market price
            side: Position side

        Returns:
            Limit order price
        """
        offset = self.config.limit_order_offset
        if side == PositionSide.LONG:
            # Buy limit: place below market
            return market_price * (1 - offset)
        else:
            # Sell limit: place above market
            return market_price * (1 + offset)

    def get_entry_price(
        self,
        signal_price: float,
        signal_time: int,
        intrabar_data: pd.DataFrame,
        side: PositionSide
    ) -> Tuple[float, FillInfo]:
        """
        Get actual entry price from intrabar data after signal.

        Uses the open of the first intrabar candle after signal time.

        Args:
            signal_price: Signal price from 1h candle
            signal_time: Signal timestamp (1h candle openTime)
            intrabar_data: Intraday dataframe (5m)
            side: Position side

        Returns:
            Tuple of (entry_price, fill_info)
        """
        # Find first intrabar candle after signal
        intrabar_after = intrabar_data[intrabar_data['opentime'] > signal_time]

        if len(intrabar_after) == 0:
            # No intrabar data after signal - use signal price
            entry_price = self.apply_slippage(signal_price, side)
            fee = self.calculate_fee(entry_price, 1.0)  # Placeholder quantity
            return entry_price, FillInfo(
                timestamp=signal_time,
                price=entry_price,
                quantity=0,
                fee=fee,
                slippage_applied=self.config.slippage
            )

        first_bar = intrabar_after.iloc[0]
        entry_price = self.apply_slippage(first_bar['open'], side)

        fee = self.calculate_fee(entry_price, 1.0)  # Quantity not yet known

        return entry_price, FillInfo(
            timestamp=int(first_bar['opentime']),
            price=entry_price,
            quantity=0,
            fee=fee,
            slippage_applied=self.config.slippage
        )

    def check_exit_conditions(
        self,
        position: Position,
        intrabar_candle: pd.Series
    ) -> Tuple[bool, ExitReason, float]:
        """
        Check if any exit condition is met using intrabar high/low.

        Args:
            position: Current position
            intrabar_candle: Current intrabar candle

        Returns:
            Tuple of (should_exit, reason, exit_price)
        """
        if position is None:
            return False, ExitReason.MANUAL, 0.0

        high = intrabar_candle['high']
        low = intrabar_candle['low']

        if position.side == PositionSide.LONG:
            # Update highest price for trailing stop
            position.highest_price = max(position.highest_price, high)

            # Check trailing stop first if enabled
            if self.config.trailing_stop_enabled:
                position.trailing_stop_price = position.highest_price * (1 - self.config.trailing_stop_pct)
                if low <= position.trailing_stop_price:
                    return True, ExitReason.TRAILING_STOP, position.trailing_stop_price

            # Check TP/SL based on priority
            if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                # SL first (conservative - assume worst case)
                if low <= position.sl_price:
                    return True, ExitReason.STOP_LOSS, position.sl_price
                if high >= position.tp_price:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
            else:
                # TP first (aggressive - assume best case)
                if high >= position.tp_price:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
                if low <= position.sl_price:
                    return True, ExitReason.STOP_LOSS, position.sl_price

        else:  # SHORT
            # Update lowest price for trailing stop
            position.lowest_price = min(position.lowest_price, low)

            # Check trailing stop first if enabled
            if self.config.trailing_stop_enabled:
                position.trailing_stop_price = position.lowest_price * (1 + self.config.trailing_stop_pct)
                if high >= position.trailing_stop_price:
                    return True, ExitReason.TRAILING_STOP, position.trailing_stop_price

            # Check TP/SL based on priority
            if self.config.exit_priority == ExitPriority.CONSERVATIVE:
                if high >= position.sl_price:
                    return True, ExitReason.STOP_LOSS, position.sl_price
                if low <= position.tp_price:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
            else:
                if low <= position.tp_price:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
                if high >= position.sl_price:
                    return True, ExitReason.STOP_LOSS, position.sl_price

        return False, ExitReason.MANUAL, 0.0

    def calculate_pnl(
        self,
        position: Position,
        exit_price: float
    ) -> Tuple[float, float]:
        """
        Calculate PnL for a trade.

        Args:
            position: Closed position
            exit_price: Exit price

        Returns:
            Tuple of (pnl, pnl_pct)
        """
        if position.side == PositionSide.LONG:
            raw_pnl = (exit_price - position.entry_price) * position.quantity * position.leverage
        else:  # SHORT
            raw_pnl = (position.entry_price - exit_price) * position.quantity * position.leverage

        # PnL as percentage of position value
        position_value = position.entry_price * position.quantity
        pnl_pct = (raw_pnl / position_value) * 100 if position_value > 0 else 0

        return raw_pnl, pnl_pct

    def create_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_time: int,
        entry_price: float,
        quantity: float
    ) -> Position:
        """Create a new position with calculated TP/SL"""
        tp_price, sl_price = self.calculate_tp_sl(entry_price, side)

        position = Position(
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            quantity=quantity,
            leverage=self.config.leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            bars_held=0
        )

        return position


class IntradaySimulator:
    """
    Simulates trade execution within 1h candle periods using intrabar data.
    Optimized with vectorized operations instead of iterrows().
    Supports both market and limit orders.
    """

    def __init__(self, execution_engine: ExecutionEngine):
        self.execution = execution_engine
        self.pending_orders: List[PendingOrder] = []

    def create_limit_order(
        self,
        symbol: str,
        side: PositionSide,
        order_time: int,
        limit_price: float,
        quantity: float,
        tp_price: float = 0.0,
        sl_price: float = 0.0
    ) -> PendingOrder:
        """Create a pending limit order"""
        order = PendingOrder(
            symbol=symbol,
            side=side,
            order_time=order_time,
            limit_price=limit_price,
            quantity=quantity,
            leverage=self.execution.config.leverage,
            bars_waited=0,
            tp_price=tp_price,
            sl_price=sl_price
        )
        self.pending_orders.append(order)
        return order

    def _check_limit_orders(
        self,
        high: float,
        low: float,
        open_price: float
    ) -> List[Tuple[PendingOrder, float]]:
        """
        Check pending limit orders and fill any that meet conditions.

        For LIMIT orders:
        - LONG: Fill when price goes below or to limit_price (buy low)
        - SHORT: Fill when price goes above or to limit_price (sell high)

        Returns:
            List of filled orders with their fill prices
        """
        filled = []
        config = self.execution.config

        for order in self.pending_orders[:]:
            # Check timeout
            if config.limit_order_timeout_bars > 0:
                order.bars_waited += 1
                if order.bars_waited >= config.limit_order_timeout_bars:
                    self.pending_orders.remove(order)
                    continue

            # Check fill condition
            if order.side == PositionSide.LONG:
                # Fill if price touched or went below limit price
                if low <= order.limit_price:
                    # Fill at limit price or better (lower)
                    fill_price = min(order.limit_price, low, open_price)
                    filled.append((order, fill_price))
                    self.pending_orders.remove(order)
            else:  # SHORT
                # Fill if price touched or went above limit price
                if high >= order.limit_price:
                    # Fill at limit price or better (higher)
                    fill_price = max(order.limit_price, high, open_price)
                    filled.append((order, fill_price))
                    self.pending_orders.remove(order)

        return filled

    def cancel_orders_for_symbol(self, symbol: str) -> None:
        """Cancel all pending orders for a symbol"""
        self.pending_orders = [o for o in self.pending_orders if o.symbol != symbol]

    def has_pending_orders(self, symbol: str = None) -> bool:
        """Check if there are pending orders"""
        if symbol is None:
            return len(self.pending_orders) > 0
        return any(o.symbol == symbol for o in self.pending_orders)

    def _execute_filled_limit_order(
        self,
        order: PendingOrder,
        fill_price: float,
        h1_data: pd.DataFrame,
        trades: List[Trade],
        capital: float,
        current_time: int
    ) -> Tuple[Optional[Position], float]:
        """Execute a filled limit order and create a position"""
        # Cancel any other pending orders for this symbol to prevent duplicates
        self.cancel_orders_for_symbol(order.symbol)

        position = Position(
            symbol=order.symbol,
            side=order.side,
            entry_time=current_time,
            entry_price=fill_price,
            quantity=order.quantity,
            leverage=order.leverage,
            tp_price=order.tp_price,
            sl_price=order.sl_price,
            highest_price=fill_price,
            lowest_price=fill_price,
            bars_held=0
        )

        # Deduct entry fee
        entry_fee = self.execution.calculate_fee(fill_price, order.quantity)
        capital -= entry_fee

        return position, capital

    def simulate(
        self,
        h1_data: pd.DataFrame,
        m1_data: pd.DataFrame,
        long_signals: pd.Series,
        short_signals: pd.Series,
        long_exit_signals: Optional[pd.Series] = None,
        short_exit_signals: Optional[pd.Series] = None,
        capital: float = 10000.0,
        position_size_pct: float = 0.95
    ) -> Tuple[List[Trade], float]:
        """
        Run intrabar simulation using optimized vectorized approach.

        Args:
            h1_data: 1-hour OHLCV data
            m1_data: Intraday OHLCV data (5m)
            long_signals: Boolean series of long entry signals
            short_signals: Boolean series of short entry signals
            long_exit_signals: Optional long exit signals
            short_exit_signals: Optional short exit signals
            capital: Starting capital
            position_size_pct: Position size as decimal

        Returns:
            Tuple of (list of trades, final capital)
        """
        trades = []
        position: Optional[Position] = None

        # Reset pending orders for each backtest run
        self.pending_orders = []

        # Pre-compute signal lookup dictionaries for O(1) access
        long_signal_dict = dict(zip(long_signals.index, long_signals.values))
        short_signal_dict = dict(zip(short_signals.index, short_signals.values))

        long_exit_times = set(long_exit_signals[long_exit_signals].index) if long_exit_signals is not None else set()
        short_exit_times = set(short_exit_signals[short_exit_signals].index) if short_exit_signals is not None else set()

        # Get h1 timestamps as numpy array for fast access
        h1_timestamps = h1_data['opentime'].values
        h1_closes = h1_data['close'].values
        h1_len = len(h1_data)

        # Get m1 data as numpy arrays for fast access
        m1_timestamps = m1_data['opentime'].values
        m1_opens = m1_data['open'].values
        m1_highs = m1_data['high'].values
        m1_lows = m1_data['low'].values

        # Pre-filter m1 data into chunks by 1h periods (vectorized)
        # Instead of filtering each time, build index ranges once
        m1_h1_indices = np.searchsorted(m1_timestamps, h1_timestamps)

        # Process each 1h candle
        for h1_idx in range(h1_len):
            h1_time = h1_timestamps[h1_idx]
            h1_close = h1_closes[h1_idx]

            # Get m1 candle indices for this 1h period
            start_idx = m1_h1_indices[h1_idx]
            end_idx = m1_h1_indices[h1_idx + 1] if h1_idx + 1 < h1_len else len(m1_timestamps)

            # Check for pending limit order fills before checking exits
            if len(self.pending_orders) > 0 and end_idx > start_idx:
                period_highs = m1_highs[start_idx:end_idx]
                period_lows = m1_lows[start_idx:end_idx]
                period_opens = m1_opens[start_idx:end_idx]
                period_times = m1_timestamps[start_idx:end_idx]

                # Check each pending order
                filled_orders = self._check_limit_orders(
                    high=period_highs.max(),
                    low=period_lows.min(),
                    open_price=period_opens[0] if len(period_opens) > 0 else h1_close
                )

                for order, fill_price in filled_orders:
                    if position is None:
                        position, capital = self._execute_filled_limit_order(
                            order, fill_price, h1_data, trades, capital, h1_time
                        )

            # Check if there's an open position and we have intrabar data
            if position is not None and end_idx > start_idx:
                # Get arrays for this period
                period_highs = m1_highs[start_idx:end_idx]
                period_lows = m1_lows[start_idx:end_idx]
                period_opens = m1_opens[start_idx:end_idx]
                period_times = m1_timestamps[start_idx:end_idx]

                # Vectorized TP/SL check
                should_exit, reason, exit_price = self._check_exit_vectorized(
                    position, period_highs, period_lows, period_times
                )

                if should_exit:
                    # Calculate PnL
                    pnl, pnl_pct = self.execution.calculate_pnl(position, exit_price)
                    entry_fee = self.execution.calculate_fee(position.entry_price, position.quantity)
                    exit_fee = self.execution.calculate_fee(exit_price, position.quantity)
                    total_fees = entry_fee + exit_fee

                    trade = Trade(
                        symbol=position.symbol,
                        side=position.side,
                        entry_time=position.entry_time,
                        entry_price=position.entry_price,
                        exit_time=int(period_times[0]),  # First bar that triggered exit
                        exit_price=exit_price,
                        quantity=position.quantity,
                        pnl=pnl - total_fees,
                        pnl_pct=pnl_pct,
                        fees=total_fees,
                        hold_bars=position.bars_held,
                        exit_reason=reason.value,
                        leverage=position.leverage
                    )
                    trades.append(trade)
                    capital += pnl - total_fees
                    position = None

            # After processing intrabar candles, check for entry signals at close of this 1h candle
            if position is None:
                order_type = self.execution.config.order_type

                # Check long entry
                if long_signal_dict.get(h1_time, False):
                    side = PositionSide.LONG
                    quantity = self.execution.calculate_position_size(
                        capital, position_size_pct, h1_close
                    )

                    if order_type == OrderType.LIMIT:
                        # Create pending limit order
                        limit_price = self.execution.calculate_limit_price(h1_close, side)
                        tp_price, sl_price = self.execution.calculate_tp_sl(h1_close, side)
                        self.create_limit_order(
                            symbol=h1_data.get('symbol', 'UNKNOWN'),
                            side=side,
                            order_time=h1_time,
                            limit_price=limit_price,
                            quantity=quantity,
                            tp_price=tp_price,
                            sl_price=sl_price
                        )
                    else:
                        # Market order - execute immediately
                        entry_price = self.execution.apply_slippage(h1_close, side)
                        position = self.execution.create_position(
                            symbol=h1_data.get('symbol', 'UNKNOWN'),
                            side=side,
                            entry_time=h1_time,
                            entry_price=entry_price,
                            quantity=quantity
                        )
                        entry_fee = self.execution.calculate_fee(entry_price, quantity)
                        capital -= entry_fee

                # Check short entry
                elif short_signal_dict.get(h1_time, False):
                    side = PositionSide.SHORT
                    quantity = self.execution.calculate_position_size(
                        capital, position_size_pct, h1_close
                    )

                    if order_type == OrderType.LIMIT:
                        # Create pending limit order
                        limit_price = self.execution.calculate_limit_price(h1_close, side)
                        tp_price, sl_price = self.execution.calculate_tp_sl(h1_close, side)
                        self.create_limit_order(
                            symbol=h1_data.get('symbol', 'UNKNOWN'),
                            side=side,
                            order_time=h1_time,
                            limit_price=limit_price,
                            quantity=quantity,
                            tp_price=tp_price,
                            sl_price=sl_price
                        )
                    else:
                        # Market order - execute immediately
                        entry_price = self.execution.apply_slippage(h1_close, side)
                        position = self.execution.create_position(
                            symbol=h1_data.get('symbol', 'UNKNOWN'),
                            side=side,
                            entry_time=h1_time,
                            entry_price=entry_price,
                            quantity=quantity
                        )
                        entry_fee = self.execution.calculate_fee(entry_price, quantity)
                        capital -= entry_fee

            # Check exit signals (manual exit)
            if position is not None:
                exit_triggered = False
                exit_price = 0.0
                exit_reason_str = ""

                if position.side == PositionSide.LONG and h1_time in long_exit_times:
                    if end_idx < len(m1_timestamps):
                        exit_price = self.execution.apply_slippage(m1_opens[end_idx], PositionSide.SHORT)
                        pnl, pnl_pct = self.execution.calculate_pnl(position, exit_price)
                        exit_fee = self.execution.calculate_fee(exit_price, position.quantity)
                        exit_reason_str = ExitReason.SIGNAL_EXIT.value
                        exit_triggered = True

                elif position.side == PositionSide.SHORT and h1_time in short_exit_times:
                    if end_idx < len(m1_timestamps):
                        exit_price = self.execution.apply_slippage(m1_opens[end_idx], PositionSide.LONG)
                        pnl, pnl_pct = self.execution.calculate_pnl(position, exit_price)
                        exit_fee = self.execution.calculate_fee(exit_price, position.quantity)
                        exit_reason_str = ExitReason.SIGNAL_EXIT.value
                        exit_triggered = True

                if exit_triggered:
                    trade = Trade(
                        symbol=position.symbol,
                        side=position.side,
                        entry_time=position.entry_time,
                        entry_price=position.entry_price,
                        exit_time=int(m1_timestamps[end_idx]),
                        exit_price=exit_price,
                        quantity=position.quantity,
                        pnl=pnl - exit_fee,
                        pnl_pct=pnl_pct,
                        fees=exit_fee,
                        hold_bars=position.bars_held,
                        exit_reason=exit_reason_str,
                        leverage=position.leverage
                    )
                    trades.append(trade)
                    capital += pnl - exit_fee
                    position = None

            # Update bars held count
            if position is not None:
                position.bars_held += 1

        # Close any remaining position at end of data
        if position is not None and len(m1_timestamps) > 0:
            last_m1_idx = len(m1_timestamps) - 1
            exit_price = m1_opens[last_m1_idx]  # Use open of last candle

            pnl, pnl_pct = self.execution.calculate_pnl(position, exit_price)
            exit_fee = self.execution.calculate_fee(exit_price, position.quantity)

            trade = Trade(
                symbol=position.symbol,
                side=position.side,
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                exit_time=int(m1_timestamps[last_m1_idx]),
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=pnl - exit_fee,
                pnl_pct=pnl_pct,
                fees=exit_fee,
                hold_bars=position.bars_held,
                exit_reason=ExitReason.ENDT_OF_DATA.value,
                leverage=position.leverage
            )
            trades.append(trade)
            capital += pnl - exit_fee

        return trades, capital

    def _check_exit_vectorized(
        self,
        position: Position,
        period_highs: np.ndarray,
        period_lows: np.ndarray,
        period_times: np.ndarray
    ) -> Tuple[bool, ExitReason, float]:
        """
        Vectorized exit condition check - find first bar that hits TP/SL.
        """
        if position.side == PositionSide.LONG:
            # Update highest price
            max_high = np.max(period_highs)
            position.highest_price = max(position.highest_price, max_high)

            # Check trailing stop first if enabled
            if self.execution.config.trailing_stop_enabled:
                position.trailing_stop_price = position.highest_price * (1 - self.execution.config.trailing_stop_pct)
                trailing_idx = np.where(period_lows <= position.trailing_stop_price)[0]
                if len(trailing_idx) > 0:
                    return True, ExitReason.TRAILING_STOP, position.trailing_stop_price

            # Check TP/SL based on priority
            if self.execution.config.exit_priority == ExitPriority.CONSERVATIVE:
                # SL first
                sl_idx = np.where(period_lows <= position.sl_price)[0]
                if len(sl_idx) > 0:
                    return True, ExitReason.STOP_LOSS, position.sl_price
                tp_idx = np.where(period_highs >= position.tp_price)[0]
                if len(tp_idx) > 0:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
            else:
                # TP first (aggressive)
                tp_idx = np.where(period_highs >= position.tp_price)[0]
                if len(tp_idx) > 0:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
                sl_idx = np.where(period_lows <= position.sl_price)[0]
                if len(sl_idx) > 0:
                    return True, ExitReason.STOP_LOSS, position.sl_price
        else:  # SHORT
            # Update lowest price
            min_low = np.min(period_lows)
            position.lowest_price = min(position.lowest_price, min_low)

            # Check trailing stop first if enabled
            if self.execution.config.trailing_stop_enabled:
                position.trailing_stop_price = position.lowest_price * (1 + self.execution.config.trailing_stop_pct)
                trailing_idx = np.where(period_highs >= position.trailing_stop_price)[0]
                if len(trailing_idx) > 0:
                    return True, ExitReason.TRAILING_STOP, position.trailing_stop_price

            # Check TP/SL based on priority
            if self.execution.config.exit_priority == ExitPriority.CONSERVATIVE:
                # SL first
                sl_idx = np.where(period_highs >= position.sl_price)[0]
                if len(sl_idx) > 0:
                    return True, ExitReason.STOP_LOSS, position.sl_price
                tp_idx = np.where(period_lows <= position.tp_price)[0]
                if len(tp_idx) > 0:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
            else:
                # TP first (aggressive)
                tp_idx = np.where(period_lows <= position.tp_price)[0]
                if len(tp_idx) > 0:
                    return True, ExitReason.TAKE_PROFIT, position.tp_price
                sl_idx = np.where(period_highs >= position.sl_price)[0]
                if len(sl_idx) > 0:
                    return True, ExitReason.STOP_LOSS, position.sl_price

        return False, ExitReason.MANUAL, 0.0
