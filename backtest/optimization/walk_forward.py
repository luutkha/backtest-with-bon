"""
Walk-Forward Analysis for Backtest Strategies.

Performs rolling window optimization and validation to test strategy robustness.
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
import pandas as pd
import numpy as np

from ..unified_portfolio import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from .grid_search import ParameterGridSearch

logger = logging.getLogger(__name__)


class WalkForwardAnalysis:
    """
    Walk-forward analysis optimizer.

    Splits data into training and testing windows, optimizes parameters
    on training data, and validates on test data. Uses sliding windows
    to test strategy robustness over time.
    """

    def __init__(
        self,
        strategy: Callable,
        data_dir: str,
        symbols: List[str],
        param_grid: Dict[str, List[Any]],
        base_config: Optional[UnifiedPortfolioConfig] = None,
        train_pct: float = 0.7,
        steps: int = 5,
        step_size_pct: float = None,
        target_metric: str = 'sharpe_ratio',
        verbose: bool = True,
    ):
        """
        Initialize walk-forward analysis.

        Args:
            strategy: Strategy function to optimize
            data_dir: Path to data directory
            symbols: List of symbols to backtest
            param_grid: Dict of parameter names to list of values to test
            base_config: Base configuration (uses defaults if None)
            train_pct: Percentage of data for training (default: 0.7)
            steps: Number of walk-forward steps (default: 5)
            step_size_pct: Step size as percentage of total (default: auto-calculated)
            target_metric: Metric to optimize (default: sharpe_ratio)
            verbose: Enable logging
        """
        self.strategy = strategy
        self.data_dir = data_dir
        self.symbols = symbols
        self.param_grid = param_grid
        self.base_config = base_config or UnifiedPortfolioConfig()
        self.train_pct = train_pct
        self.steps = steps
        self.step_size_pct = step_size_pct
        self.target_metric = target_metric
        self.verbose = verbose

        # Will be set during run
        self.train_results: List[pd.DataFrame] = []
        self.test_results: List[Dict[str, Any]] = []
        self.best_params_per_step: List[Dict[str, Any]] = []

    def _calculate_windows(
        self,
        timestamps: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate train/test window boundaries.

        Returns:
            List of (train_start, train_end, test_start, test_end) indices
        """
        n = len(timestamps)

        # Calculate step size
        if self.step_size_pct is None:
            # Default: use remaining data divided by steps
            # First window starts at 0, ends at train_pct
            # Each step moves forward by (1 - train_pct) / steps
            self.step_size_pct = (1 - self.train_pct) / self.steps

        windows = []
        for step in range(self.steps):
            # Calculate window boundaries
            train_end = int(n * (self.train_pct + step * self.step_size_pct))
            test_start = train_end
            test_end = int(n * (self.train_pct + (step + 1) * self.step_size_pct))

            # Ensure minimum sizes
            train_start = 0
            if step > 0:
                # Sliding window: use previous test start as train start
                prev_test_start = windows[-1][2]
                train_start = prev_test_start

            # Ensure valid window sizes
            if train_end - train_start < n * 0.1:  # Min 10% for training
                continue
            if test_end - test_start < n * 0.05:  # Min 5% for testing
                test_end = n

            windows.append((train_start, train_end, test_start, test_end))

        return windows

    def _create_dated_strategy(
        self,
        strategy: Callable,
        start_time: int,
        end_time: int,
    ) -> Callable:
        """
        Create a strategy that filters data by date range.

        Returns a wrapped strategy that only processes data in the given range.
        """
        original_strategy = strategy

        def dated_strategy(data: pd.DataFrame) -> List:
            # Filter data by time range
            filtered = data[
                (data['opentime'] >= start_time) &
                (data['opentime'] <= end_time)
            ]
            if len(filtered) == 0:
                return []
            return original_strategy(filtered)

        return dated_strategy

    def run(self) -> Dict[str, Any]:
        """
        Run walk-forward analysis.

        Returns:
            Dict with train_results, test_results, and summary statistics
        """
        if self.verbose:
            logger.info(f"Starting walk-forward analysis with {self.steps} steps")
            logger.info(f"Train/Test split: {self.train_pct*100}%/{(1-self.train_pct)*100}%")

        # Load all data to get timestamps
        engine = UnifiedPortfolioBacktest(
            config=self.base_config,
            strategy=self.strategy,
            data_dir=self.data_dir,
        )
        engine.load_data(self.symbols)

        # Get common timestamps across all symbols
        all_timestamps = set()
        for symbol, df in engine.symbol_data.items():
            all_timestamps.update(df['opentime'].values)

        timestamps = sorted(list(all_timestamps))

        if len(timestamps) == 0:
            raise ValueError("No data available for walk-forward analysis")

        # Calculate windows
        windows = self._calculate_windows(np.array(timestamps))

        if self.verbose:
            logger.info(f"Created {len(windows)} walk-forward windows")

        self.train_results = []
        self.test_results = []
        self.best_params_per_step = []

        for step, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if self.verbose:
                logger.info(f"\n--- Step {step + 1}/{len(windows)} ---")
                logger.info(f"Train: {pd.Timestamp(timestamps[train_start], unit='ms')} to {pd.Timestamp(timestamps[train_end-1], unit='ms')}")
                logger.info(f"Test: {pd.Timestamp(timestamps[test_start], unit='ms')} to {pd.Timestamp(timestamps[test_end-1], unit='ms')}")

            # Get time boundaries
            train_start_time = int(timestamps[train_start])
            train_end_time = int(timestamps[train_end - 1])
            test_start_time = int(timestamps[test_start])
            test_end_time = int(timestamps[test_end - 1])

            # Create dated strategy for training
            train_strategy = self._create_dated_strategy(
                self.strategy, train_start_time, train_end_time
            )

            # Run grid search on training data
            optimizer = ParameterGridSearch(
                strategy=train_strategy,
                data_dir=self.data_dir,
                symbols=self.symbols,
                param_grid=self.param_grid,
                base_config=self.base_config,
                target_metric=self.target_metric,
                verbose=self.verbose,
            )

            train_df = optimizer.run()
            self.train_results.append(train_df)

            # Get best params from training
            if len(train_df) > 0:
                best_params = train_df.iloc[0]['params']
                best_train_metric = train_df.iloc[0][self.target_metric]
            else:
                best_params = {}
                best_train_metric = 0

            self.best_params_per_step.append(best_params)

            if self.verbose:
                logger.info(f"Best train params: {best_params}")
                logger.info(f"Best train {self.target_metric}: {best_train_metric:.4f}")

            # Test on validation period with best params
            test_strategy = self._create_dated_strategy(
                self.strategy, test_start_time, test_end_time
            )

            # Create strategy with best params for testing
            def test_strategy_func(data):
                try:
                    return test_strategy(data, **best_params)
                except TypeError:
                    return test_strategy(data)

            test_engine = UnifiedPortfolioBacktest(
                config=self.base_config,
                strategy=test_strategy_func,
                data_dir=self.data_dir,
            )

            try:
                test_engine.load_data(self.symbols)
                test_result = test_engine.run_backtest()
                test_metrics = test_result['metrics']

                test_row = {
                    'step': step + 1,
                    'train_start': train_start_time,
                    'train_end': train_end_time,
                    'test_start': test_start_time,
                    'test_end': test_end_time,
                    'best_params': best_params,
                    'train_metric': best_train_metric,
                    'test_total_trades': test_metrics.get('total_trades', 0),
                    'test_win_rate': test_metrics.get('win_rate', 0),
                    'test_total_pnl': test_metrics.get('total_pnl', 0),
                    'test_return_pct': test_metrics.get('return_pct', 0),
                    'test_max_drawdown': test_metrics.get('max_drawdown', 0),
                    'test_sharpe_ratio': test_metrics.get('sharpe_ratio', 0),
                    'test_final_capital': test_metrics.get('final_capital', self.base_config.initial_capital),
                }

                # Add target metric
                test_row[f'test_{self.target_metric}'] = test_metrics.get(self.target_metric, 0)

                self.test_results.append(test_row)

                if self.verbose:
                    logger.info(f"Test {self.target_metric}: {test_row[f'test_{self.target_metric}']:.4f}")
                    logger.info(f"Test Return: {test_row['test_return_pct']:.2f}%")

            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to run test for step {step + 1}: {e}")
                self.test_results.append({
                    'step': step + 1,
                    'error': str(e),
                })

        # Calculate summary statistics
        summary = self._calculate_summary()

        return {
            'train_results': self.train_results,
            'test_results': self.test_results,
            'best_params_per_step': self.best_params_per_step,
            'summary': summary,
        }

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics across all steps."""
        if not self.test_results:
            return {}

        # Filter out failed steps
        valid_results = [r for r in self.test_results if 'error' not in r]

        if not valid_results:
            return {'error': 'All steps failed'}

        test_df = pd.DataFrame(valid_results)

        summary = {
            'total_steps': len(valid_results),
            'avg_test_return': test_df['test_return_pct'].mean(),
            'std_test_return': test_df['test_return_pct'].std(),
            'avg_test_sharpe': test_df['test_sharpe_ratio'].mean(),
            'avg_test_drawdown': test_df['test_max_drawdown'].mean(),
            'avg_test_trades': test_df['test_total_trades'].mean(),
            'avg_win_rate': test_df['test_win_rate'].mean(),
            'consistency_score': (test_df['test_return_pct'] > 0).sum() / len(test_df) * 100,
        }

        # Calculate robustness score
        # Higher score = more consistent parameters across steps
        if self.best_params_per_step:
            param_consistency = {}
            for param_name in self.param_grid.keys():
                values = [p.get(param_name) for p in self.best_params_per_step if p]
                if values:
                    # Count most common value
                    from collections import Counter
                    most_common_count = Counter(values).most_common(1)[0][1]
                    param_consistency[param_name] = most_common_count / len(values) * 100

            summary['param_consistency'] = param_consistency

        return summary

    def print_summary(self) -> None:
        """Print walk-forward analysis summary."""
        if not self.test_results:
            logger.info("No results to display")
            return

        logger.info("=" * 70)
        logger.info("WALK-FORWARD ANALYSIS SUMMARY")
        logger.info("=" * 70)

        for i, result in enumerate(self.test_results):
            if 'error' in result:
                logger.info(f"Step {i+1}: FAILED - {result['error']}")
                continue

            logger.info(f"\n--- Step {i+1} ---")
            logger.info(f"Train {self.target_metric}: {result['train_metric']:.4f}")
            logger.info(f"Test Return: {result['test_return_pct']:.2f}%")
            logger.info(f"Test {self.target_metric}: {result[f'test_{self.target_metric}']:.4f}")
            logger.info(f"Test Trades: {result['test_total_trades']}")
            logger.info(f"Best Params: {result['best_params']}")

        # Print aggregate summary
        summary = self._calculate_summary()
        if summary and 'error' not in summary:
            logger.info("\n" + "=" * 70)
            logger.info("AGGREGATE METRICS")
            logger.info("=" * 70)
            logger.info(f"Steps Completed: {summary['total_steps']}")
            logger.info(f"Avg Test Return: {summary['avg_test_return']:.2f}%")
            logger.info(f"Std Test Return: {summary['std_test_return']:.2f}%")
            logger.info(f"Avg Test Sharpe: {summary['avg_test_sharpe']:.4f}")
            logger.info(f"Avg Test Drawdown: {summary['avg_test_drawdown']:.2f}%")
            logger.info(f"Consistency Score: {summary['consistency_score']:.1f}%")
            logger.info("=" * 70)

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get test results as a DataFrame.

        Returns:
            DataFrame with test results for each step
        """
        if not self.test_results:
            return pd.DataFrame()

        return pd.DataFrame([
            r for r in self.test_results if 'error' not in r
        ])
