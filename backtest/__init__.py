"""
Backtesting Framework

A production-grade backtesting framework for futures trading strategies.

Architecture:
- Data Layer: Market data loading and validation
- Signal Layer: Strategy signal generation
- Execution Layer: Trade execution with intrabar simulation
- Portfolio Layer: Capital and position tracking
- Analytics Layer: Performance metrics calculation
- Reporting Layer: Output generation

Usage:
    from backtest import BacktestEngine
    from backtest.config import BacktestConfig

    config = BacktestConfig(
        data_dir="./data",
        symbol="BTCUSDT",
        initial_capital=10000,
    )

    engine = BacktestEngine(config, strategy_function)
    results = engine.run_backtest()
"""

from .backtest_engine import BacktestEngine
from .config import BacktestConfig
from .signals.signal_generator import Signal, SignalType, SignalData
from .execution.execution_engine import Trade, PositionSide, ExitReason
from .unified_portfolio import UnifiedPortfolioBacktest, UnifiedPortfolioConfig

__version__ = "1.0.0"

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'Signal',
    'SignalType',
    'SignalData',
    'Trade',
    'PositionSide',
    'ExitReason',
    'UnifiedPortfolioBacktest',
    'UnifiedPortfolioConfig',
]
