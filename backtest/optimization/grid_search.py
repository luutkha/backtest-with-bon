"""
Grid Search Optimization for Backtest Strategies.

Performs exhaustive parameter search over a grid of hyperparameters.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from itertools import product
import pandas as pd

from ..unified_portfolio import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from ..signals.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class ParameterGridSearch:
    """
    Grid search optimizer for strategy parameters.

    Generates all combinations of parameters and runs backtests to find
    the optimal configuration based on a target metric.
    """

    def __init__(
        self,
        strategy: Callable,
        data_dir: str,
        symbols: List[str],
        param_grid: Dict[str, List[Any]],
        base_config: Optional[UnifiedPortfolioConfig] = None,
        target_metric: str = 'sharpe_ratio',
        verbose: bool = True,
    ):
        """
        Initialize grid search optimizer.

        Args:
            strategy: Strategy function to optimize
            data_dir: Path to data directory
            symbols: List of symbols to backtest
            param_grid: Dict of parameter names to list of values to test
            base_config: Base configuration (uses defaults if None)
            target_metric: Metric to optimize (default: sharpe_ratio)
            verbose: Enable logging
        """
        self.strategy = strategy
        self.data_dir = data_dir
        self.symbols = symbols
        self.param_grid = param_grid
        self.base_config = base_config or UnifiedPortfolioConfig()
        self.target_metric = target_metric
        self.verbose = verbose

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from the grid."""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        combinations = []
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            combinations.append(params)

        return combinations

    def _create_strategy_with_params(self, params: Dict[str, Any]) -> Callable:
        """
        Create a strategy function with fixed parameters.

        Returns a wrapped strategy that applies the given parameters.
        """
        # Get the original strategy's parameters
        original_func = self.strategy

        def wrapped_strategy(data):
            # Pass params to the strategy function if it accepts them
            try:
                return original_func(data, **params)
            except TypeError:
                # If strategy doesn't accept params, try calling with just data
                return original_func(data)

        return wrapped_strategy

    def run(self) -> pd.DataFrame:
        """
        Run grid search over all parameter combinations.

        Returns:
            DataFrame with results sorted by target_metric (descending)
        """
        combinations = self._generate_param_combinations()
        total = len(combinations)

        if self.verbose:
            logger.info(f"Starting grid search with {total} parameter combinations")

        results = []

        for i, params in enumerate(combinations):
            if self.verbose:
                logger.info(f"Testing {i+1}/{total}: {params}")

            # Create strategy with current params
            strategy_func = self._create_strategy_with_params(params)

            # Create backtest engine
            engine = UnifiedPortfolioBacktest(
                config=self.base_config,
                strategy=strategy_func,
                data_dir=self.data_dir,
            )

            # Run backtest
            try:
                engine.load_data(self.symbols)
                result = engine.run_backtest()

                # Extract metrics
                metrics = result['metrics']
                metrics_row = {
                    'params': params,
                    'total_trades': metrics.get('total_trades', 0),
                    'winning_trades': metrics.get('winning_trades', 0),
                    'losing_trades': metrics.get('losing_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_pnl': metrics.get('total_pnl', 0),
                    'return_pct': metrics.get('return_pct', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'sortino_ratio': metrics.get('sortino_ratio', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'final_capital': metrics.get('final_capital', self.base_config.initial_capital),
                }

                # Add target metric value
                metrics_row[self.target_metric] = metrics.get(self.target_metric, 0)

                results.append(metrics_row)

            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to run backtest with {params}: {e}")
                results.append({
                    'params': params,
                    'total_trades': 0,
                    'error': str(e),
                })

        # Convert to DataFrame and sort
        df = pd.DataFrame(results)

        if len(df) > 0 and 'total_trades' in df.columns:
            # Filter out failed runs
            df = df[df.get('error', pd.Series([''] * len(df))) == '']

            if len(df) > 0:
                # Sort by target metric (descending)
                df = df.sort_values(self.target_metric, ascending=False)

        if self.verbose:
            logger.info(f"Grid search complete. {len(df)} valid results.")
            if len(df) > 0:
                best = df.iloc[0]
                logger.info(f"Best {self.target_metric}: {best[self.target_metric]:.4f}")
                logger.info(f"Best params: {best['params']}")

        return df

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Run grid search and return the best parameters.

        Returns:
            Dict of best parameters, or None if no valid results
        """
        results = self.run()
        if len(results) > 0:
            return results.iloc[0]['params']
        return None


# Example parameter grids for common strategies
STREAK_GRID = {
    'consecutive_candles': [3, 4, 5, 6],
    'atr_window_min': [1.0, 2.0],
    'atr_window_max': [4.0, 6.0, 8.0],
    'risk_reward_ratio_sl': [0.3, 0.5, 0.7],
    'risk_reward_ratio_tp': [1.0, 1.5, 2.0],
}

RSI_GRID = {
    'rsi_period': [7, 14, 21],
    'oversold': [20, 30, 40],
    'overbought': [60, 70, 80],
}

MA_CROSSOVER_GRID = {
    'fast_period': [10, 15, 20, 25],
    'slow_period': [40, 50, 60, 70],
}

MACD_GRID = {
    'fast_period': [8, 12, 16],
    'slow_period': [20, 26, 32],
    'signal_period': [6, 9, 12],
}
