"""
BacktestEngine - Main orchestrator for the backtesting framework.

Coordinates all layers:
- Data Layer: Loads and validates market data
- Signal Layer: Generates trading signals
- Execution Layer: Simulates trade execution with intrabar data
- Portfolio Layer: Tracks capital and positions
- Analytics Layer: Calculates performance metrics
- Reporting Layer: Generates output reports
"""

import logging
from typing import Optional, List, Callable, Dict, Any, Tuple

import pandas as pd
import numpy as np

from .data.data_loader import DataLoader, DataConfig
from .signals.signal_generator import SignalGenerator, Signal, SignalData, SignalType
from .execution.execution_engine import (
    ExecutionEngine, ExecutionConfig, IntradaySimulator,
    PositionSide, ExitPriority, SlippageModel
)
from .portfolio.portfolio_tracker import PortfolioTracker, PortfolioConfig, PositionSizeModel
from .analytics.metrics_calculator import MetricsCalculator
from .reporting.report_generator import ReportGenerator
from .config import BacktestConfig

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine that orchestrates all layers.

    Usage:
        config = BacktestConfig(data_dir="./data", symbol="BTCUSDT", ...)
        engine = BacktestEngine(config, strategy_function)
        results = engine.run()
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Optional[Callable] = None,
    ):
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration
            strategy: Strategy function that takes h1_data and returns List[Signal]
        """
        self.config = config
        self.strategy = strategy

        # Initialize layers
        self._init_layers()

        # Data placeholders
        self.h1_data: Optional[pd.DataFrame] = None
        self.m1_data: Optional[pd.DataFrame] = None
        self.signal_data: Optional[SignalData] = None

        # Results placeholders
        self.trades: List = []
        self.equity_df: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, Any] = {}
        self.final_capital: float = config.initial_capital

    def _init_layers(self):
        """Initialize all layers with configuration"""

        # Data Layer
        self.data_loader = DataLoader(
            DataConfig(base_path=self.config.data_dir)
        )

        # Signal Layer
        self.signal_generator = SignalGenerator(strategy_func=self.strategy)

        # Execution Layer
        exec_config = ExecutionConfig(
            fee_rate=self.config.fee_rate,
            slippage=self.config.slippage,
            exit_priority=self.config.exit_priority,
            tp_pct=self.config.tp_pct,
            sl_pct=self.config.sl_pct,
            trailing_stop_enabled=self.config.trailing_stop_enabled,
            trailing_stop_pct=self.config.trailing_stop_pct,
            leverage=self.config.leverage,
        )
        self.execution_engine = ExecutionEngine(exec_config)
        self.intraday_simulator = IntradaySimulator(self.execution_engine)

        # Portfolio Layer
        portfolio_config = PortfolioConfig(
            initial_capital=self.config.initial_capital,
            position_size_model=self.config.position_size_model,
            position_size_pct=self.config.position_size_pct,
            risk_per_trade=self.config.risk_per_trade,
        )
        self.portfolio = PortfolioTracker(portfolio_config)

        # Analytics Layer
        self.metrics_calculator = MetricsCalculator()

        # Reporting Layer
        self.report_generator = ReportGenerator()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load market data.

        Returns:
            Tuple of (h1_data, m1_data)
        """
        if self.config.verbose:
            logger.info(f"Loading data for {self.config.symbol}")

        self.h1_data, self.m1_data = self.data_loader.load_pair_data(
            symbol=self.config.symbol,
            base_path=self.config.data_dir,
            start_time=self.config.start_time,
            end_time=self.config.end_time,
        )

        # Add symbol column for reference
        self.h1_data['symbol'] = self.config.symbol

        # Validate data (skip if configured)
        if not self.config.skip_validation:
            h1_validation = self.data_loader.validate_schema(self.h1_data)
            if not h1_validation.is_valid:
                logger.warning(f"1h data validation issues: {h1_validation.issues}")

            m1_validation = self.data_loader.validate_schema(self.m1_data)
            if not m1_validation.is_valid:
                logger.warning(f"5m data validation issues: {m1_validation.issues}")

        # Align timeframes
        self.h1_data, self.m1_data = self.data_loader.align_timeframes(
            self.h1_data, self.m1_data
        )

        if self.config.verbose:
            logger.info(
                f"Data loaded: {len(self.h1_data)} 1h candles, "
                f"{len(self.m1_data)} 5m candles"
            )

        return self.h1_data, self.m1_data

    def generate_signals(self, params: Optional[Dict[str, Any]] = None) -> SignalData:
        """
        Generate trading signals from 1h data.

        Args:
            params: Strategy parameters

        Returns:
            SignalData object
        """
        if self.h1_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.strategy is None:
            raise ValueError("No strategy provided")

        if self.config.verbose:
            logger.info("Generating signals")

        # Run strategy
        signals = self.strategy(self.h1_data, **(params or {}))

        # Convert to SignalData
        self.signal_data = self.signal_generator._signals_to_series(
            self.h1_data, signals
        )

        if self.config.verbose:
            logger.info(
                f"Signals generated: {self.signal_data.long_entry.sum()} long, "
                f"{self.signal_data.short_entry.sum()} short"
            )

        return self.signal_data

    def run_backtest(
        self,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete backtest.

        Args:
            params: Strategy parameters

        Returns:
            Dictionary with all results
        """
        # Load data
        self.load_data()

        # Generate signals
        self.generate_signals(params)

        # Run simulation
        if self.config.verbose:
            logger.info("Running intrabar simulation")

        self.trades, self.final_capital = self.intraday_simulator.simulate(
            h1_data=self.h1_data,
            m1_data=self.m1_data,
            long_signals=self.signal_data.long_entry,
            short_signals=self.signal_data.short_entry,
            long_exit_signals=self.signal_data.long_exit,
            short_exit_signals=self.signal_data.short_exit,
            capital=self.config.initial_capital,
            position_size_pct=self.config.position_size_pct,
        )

        if self.config.verbose:
            logger.info(f"Simulation complete: {len(self.trades)} trades executed")

        # Generate equity curve
        self.equity_df = self.report_generator.generate_equity_dataframe(
            trades=self.trades,
            h1_data=self.h1_data,
            initial_capital=self.config.initial_capital,
        )

        # Calculate metrics
        self.metrics = self.metrics_calculator.calculate_all(
            trades=self.trades,
            equity_curve=self.equity_df,
            initial_capital=self.config.initial_capital,
        )

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get all backtest results"""
        trades_df = self.report_generator.generate_trades_dataframe(self.trades)

        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'trades': self.trades,
            'trades_df': trades_df,
            'equity_df': self.equity_df,
            'final_capital': self.final_capital,
        }

    def save_results(self, output_dir: str, prefix: str = "backtest") -> None:
        """Save results to files"""
        if not self.trades:
            logger.warning("No trades to save")
            return

        trades_df = self.report_generator.generate_trades_dataframe(self.trades)

        self.report_generator.save_results(
            trades_df=trades_df,
            equity_df=self.equity_df,
            metrics=self.metrics,
            output_dir=output_dir,
            prefix=prefix,
        )

    def print_summary(self) -> None:
        """Print backtest summary to console"""
        self.report_generator.print_summary(
            metrics=self.metrics,
            config=self.config.to_dict()
        )

    # ===== Convenience Methods =====

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], strategy: Callable) -> 'BacktestEngine':
        """Create BacktestEngine from dictionary"""
        config = BacktestConfig(**config_dict)
        return cls(config=config, strategy=strategy)


# ===== Factory Functions =====

def create_ma_crossover_backtest(
    data_dir: str,
    symbol: str,
    fast_period: int = 20,
    slow_period: int = 50,
    **kwargs
) -> BacktestEngine:
    """
    Factory function to create a moving average crossover backtest.

    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        fast_period: Fast MA period
        slow_period: Slow MA period
        **kwargs: Additional configuration

    Returns:
        Configured BacktestEngine
    """
    from backtest.signals.signal_generator import moving_average_crossover_strategy

    config = BacktestConfig(
        data_dir=data_dir,
        symbol=symbol,
        **kwargs
    )

    def strategy(data, fp=fast_period, sp=slow_period):
        return moving_average_crossover_strategy(data, fp, sp)

    return BacktestEngine(config=config, strategy=strategy)


def create_rsi_backtest(
    data_dir: str,
    symbol: str,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    **kwargs
) -> BacktestEngine:
    """
    Factory function to create an RSI strategy backtest.

    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        rsi_period: RSI period
        oversold: Oversold threshold
        overbought: Overbought threshold
        **kwargs: Additional configuration

    Returns:
        Configured BacktestEngine
    """
    from backtest.signals.signal_generator import rsi_strategy

    config = BacktestConfig(
        data_dir=data_dir,
        symbol=symbol,
        **kwargs
    )

    def strategy(data, rp=rsi_period, os=oversold, ob=overbought):
        return rsi_strategy(data, rp, os, ob)

    return BacktestEngine(config=config, strategy=strategy)
