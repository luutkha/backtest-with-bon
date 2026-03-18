"""
Backtest Configuration Module
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

from ..execution.execution_engine import ExitPriority, SlippageModel
from ..portfolio.portfolio_tracker import PositionSizeModel


class Timeframe(Enum):
    """Supported timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class BacktestConfig:
    """
    Main backtest configuration.
    Combines all layer configurations into one place.
    """

    # Data config
    data_dir: str = ""
    symbol: str = "BTCUSDT"
    h1_timeframe: str = "1h"
    intrabar_timeframe: str = "5m"
    start_time: Optional[int] = None  # milliseconds
    end_time: Optional[int] = None    # milliseconds

    # Execution config
    initial_capital: float = 10000.0
    fee_rate: float = 0.0004           # 0.04% (Binance taker)
    slippage: float = 0.0001           # 0.01%
    exit_priority: ExitPriority = ExitPriority.CONSERVATIVE
    tp_pct: float = 0.02              # 2%
    sl_pct: float = 0.01              # 1%
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.0
    leverage: float = 1.0

    # Position sizing
    position_size_model: PositionSizeModel = PositionSizeModel.FIXED_PERCENT
    position_size_pct: float = 0.95
    risk_per_trade: float = 0.02      # 2% risk per trade

    # Performance options
    verbose: bool = True              # Enable logging
    skip_validation: bool = False      # Skip data validation for speed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timeframe': self.h1_timeframe,
            'start_time': str(self.start_time) if self.start_time else 'N/A',
            'end_time': str(self.end_time) if self.end_time else 'N/A',
            'initial_capital': self.initial_capital,
            'leverage': self.leverage,
            'fee_rate': self.fee_rate,
            'tp_pct': self.tp_pct,
            'sl_pct': self.sl_pct,
            'position_size_pct': self.position_size_pct,
            'exit_priority': self.exit_priority.value,
        }
