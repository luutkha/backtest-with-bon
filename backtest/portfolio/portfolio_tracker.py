"""
Portfolio Layer - Capital accounting and position tracking.
Responsibilities:
- Track balance, realized PnL, unrealized PnL
- Equity curve generation
- Position sizing models (fixed size, percentage risk)

Support:
- fixed size: Fixed quantity per trade
- percentage risk: Risk-based position sizing
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionSizeModel(Enum):
    """Position sizing models"""
    FIXED = "fixed"              # Fixed quantity per trade
    FIXED_PERCENT = "fixed_pct"  # Fixed percentage of capital
    RISK_BASED = "risk_based"    # Risk-based sizing


@dataclass
class PortfolioConfig:
    """Portfolio configuration"""
    initial_capital: float = 10000.0
    position_size_model: PositionSizeModel = PositionSizeModel.FIXED_PERCENT
    position_size_pct: float = 0.95    # For FIXED_PERCENT model
    risk_per_trade: float = 0.02       # For RISK_BASED model (2% risk)
    max_position_pct: float = 1.0       # Max position size as % of capital


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time"""
    timestamp: int
    equity: float
    cash: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PositionState:
    """Current position state"""
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: int
    unrealized_pnl: float = 0.0


class PortfolioTracker:
    """
    Tracks portfolio capital, positions, and equity curve.
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.initial_capital = config.initial_capital
        self.cash = config.initial_capital
        self.realized_pnl = 0.0
        self.position: Optional[PositionState] = None

        # Equity curve
        self.equity_curve: List[PortfolioSnapshot] = []

        # Trade history
        self.trades: List[Dict[str, Any]] = []

    def get_equity(self, current_price: float = 0.0) -> float:
        """Calculate current equity"""
        position_value = 0.0

        if self.position is not None and current_price > 0:
            if self.position.side == "long":
                position_value = (current_price - self.position.entry_price) * self.position.quantity
            else:
                position_value = (self.position.entry_price - current_price) * self.position.quantity

        return self.cash + self.realized_pnl + position_value

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.position is None or current_price <= 0:
            return 0.0

        if self.position.side == "long":
            return (current_price - self.position.entry_price) * self.position.quantity
        else:
            return (self.position.entry_price - current_price) * self.position.quantity

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        leverage: float = 1.0
    ) -> float:
        """
        Calculate position size based on configured model.

        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            leverage: Leverage multiplier

        Returns:
            Position quantity
        """
        if self.config.position_size_model == PositionSizeModel.FIXED:
            # Fixed quantity - return 1 unit (or some base)
            return 1.0

        elif self.config.position_size_model == PositionSizeModel.FIXED_PERCENT:
            # Fixed percentage of capital
            available = self.cash * self.config.position_size_pct
            position_value = available * leverage
            quantity = position_value / entry_price
            return quantity

        elif self.config.position_size_model == PositionSizeModel.RISK_BASED:
            # Risk-based: risk = |entry - sl| * quantity * leverage
            # quantity = risk_amount / |entry - sl|
            risk_amount = self.cash * self.config.risk_per_trade
            price_risk = abs(entry_price - stop_loss_price)

            if price_risk == 0:
                return 0.0

            quantity = risk_amount / price_risk
            return quantity

        return 0.0

    def open_position(
        self,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: int,
        entry_fee: float
    ) -> None:
        """Open a new position"""
        if self.position is not None:
            logger.warning("Cannot open position - already in position")
            return

        # Deduct entry fee from cash
        self.cash -= entry_fee

        self.position = PositionState(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time,
            unrealized_pnl=0.0
        )

        logger.info(f"Opened {side} position: price={entry_price}, qty={quantity}")

    def close_position(
        self,
        exit_price: float,
        exit_time: int,
        exit_fee: float,
        reason: str
    ) -> Dict[str, Any]:
        """Close current position and return trade record"""
        if self.position is None:
            return {}

        # Calculate PnL
        if self.position.side == "long":
            pnl = (exit_price - self.position.entry_price) * self.position.quantity
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.quantity

        # Deduct exit fee
        pnl -= exit_fee

        # Update realized PnL
        self.realized_pnl += pnl

        # Update cash
        self.cash += (exit_price * self.position.quantity) + pnl

        # Create trade record
        trade = {
            'entry_time': self.position.entry_time,
            'exit_time': exit_time,
            'side': self.position.side,
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'quantity': self.position.quantity,
            'pnl': pnl,
            'exit_reason': reason,
            'realized_pnl': self.realized_pnl,
            'cash': self.cash
        }

        self.trades.append(trade)

        logger.info(
            f"Closed {self.position.side} position: "
            f"entry={self.position.entry_price}, exit={exit_price}, "
            f"pnl={pnl:.2f}, reason={reason}"
        )

        self.position = None

        return trade

    def has_position(self) -> bool:
        """Check if currently in a position"""
        return self.position is not None

    def record_snapshot(self, timestamp: int, current_price: float = 0.0) -> None:
        """Record a portfolio snapshot for equity curve"""
        unrealized = self.get_unrealized_pnl(current_price) if current_price > 0 else 0.0
        equity = self.get_equity(current_price)
        position_value = self.position.quantity * self.position.entry_price if self.position else 0.0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            position_value=position_value,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl
        )

        self.equity_curve.append(snapshot)

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()

        records = [
            {
                'timestamp': s.timestamp,
                'equity': s.equity,
                'cash': s.cash,
                'position_value': s.position_value,
                'unrealized_pnl': s.unrealized_pnl,
                'realized_pnl': s.realized_pnl
            }
            for s in self.equity_curve
        ]

        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')

        return df

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.cash,
            'total_realized_pnl': self.realized_pnl,
            'total_return': self.cash - self.initial_capital,
            'return_pct': ((self.cash - self.initial_capital) / self.initial_capital) * 100,
            'num_trades': len(self.trades)
        }
