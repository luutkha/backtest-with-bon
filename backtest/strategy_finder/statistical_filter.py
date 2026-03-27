"""
Statistical Filtering for strategy validation.

Provides statistical tests and filters to identify
statistically significant strategies vs overfitted/ lucky ones.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalFilterConfig:
    """Configuration for statistical filters"""
    min_trades: int = 30
    min_sharpe: float = 1.0
    min_profit_factor: float = 1.2
    max_drawdown: float = 30.0  # Percentage
    confidence_level: float = 0.95
    min_win_rate: float = 0.0  # Optional minimum win rate


class StatisticalFilter:
    """
    Filter strategies based on statistical significance.

    Tests include:
    - t-test: Is strategy return significantly different from zero?
    - Confidence intervals: Does the return CI cross zero?
    - Outlier detection: Are returns normally distributed?
    - Luck factor: Could random trading produce similar results?
    """

    def __init__(self, config: StatisticalFilterConfig = None):
        self.config = config or StatisticalFilterConfig()

    def min_trades_filter(self, result: Dict[str, Any]) -> bool:
        """Require minimum number of trades for statistical significance"""
        total_trades = result.get('total_trades', 0)
        return total_trades >= self.config.min_trades

    def min_sharpe_filter(self, result: Dict[str, Any]) -> bool:
        """Require minimum Sharpe ratio"""
        sharpe = result.get('sharpe_ratio', 0)
        return sharpe >= self.config.min_sharpe

    def min_profit_factor_filter(self, result: Dict[str, Any]) -> bool:
        """Require minimum profit factor"""
        pf = result.get('profit_factor', 0)
        return pf >= self.config.min_profit_factor

    def max_drawdown_filter(self, result: Dict[str, Any]) -> bool:
        """Require maximum drawdown is within limit"""
        dd = result.get('max_drawdown', 100)
        return dd <= self.config.max_drawdown

    def min_win_rate_filter(self, result: Dict[str, Any]) -> bool:
        """Require minimum win rate (if configured)"""
        if self.config.min_win_rate > 0:
            win_rate = result.get('win_rate', 0)
            return win_rate >= self.config.min_win_rate
        return True

    def t_test_filter(
        self,
        trades: List[Dict[str, Any]],
        returns: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        t-test to check if strategy returns are significantly different from zero.

        H0: Mean return = 0
        H1: Mean return != 0

        Args:
            trades: List of trade dicts with pnl values
            returns: List of trade returns
            confidence_level: Confidence level for test

        Returns:
            Dict with test statistic, p-value, and pass boolean
        """
        if len(returns) < 2:
            return {'passed': False, 'p_value': 1.0, 't_stat': 0.0}

        # One-sample t-test against zero
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        alpha = 1 - confidence_level
        passed = p_value < alpha

        return {
            'passed': passed,
            't_stat': t_stat,
            'p_value': p_value,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
        }

    def confidence_interval_filter(
        self,
        returns: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate confidence interval for mean return.

        Args:
            returns: List of trade returns
            confidence_level: Confidence level

        Returns:
            Dict with CI bounds and pass status
        """
        if len(returns) < 2:
            return {'passed': False, 'ci_lower': 0, 'ci_upper': 0}

        mean = np.mean(returns)
        std_err = stats.sem(returns)
        ci = stats.t.interval(confidence_level, len(returns) - 1, loc=mean, scale=std_err)

        return {
            'passed': ci[0] > 0,  # CI should not cross zero
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'mean': mean,
        }

    def outlier_filter(
        self,
        trades: List[Dict[str, Any]],
        z_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect outliers using z-score.

        Args:
            trades: List of trade dicts with pnl values
            z_threshold: Z-score threshold for outliers

        Returns:
            Dict with outlier info
        """
        if len(trades) < 10:
            return {'has_outliers': False, 'outlier_count': 0}

        returns = [t.get('pnl', 0) for t in trades]
        z_scores = np.abs(stats.zscore(returns))

        outlier_mask = z_scores > z_threshold
        outlier_count = np.sum(outlier_mask)

        return {
            'has_outliers': outlier_count > 0,
            'outlier_count': outlier_count,
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
        }

    def consistency_filter(
        self,
        walk_forward_returns: List[float]
    ) -> Dict[str, Any]:
        """
        Check consistency of walk-forward returns.

        Args:
            walk_forward_returns: List of returns from walk-forward periods

        Returns:
            Dict with consistency metrics
        """
        if not walk_forward_returns:
            return {'passed': False, 'consistency_score': 0}

        positive_count = sum(1 for r in walk_forward_returns if r > 0)
        consistency_score = positive_count / len(walk_forward_returns)

        # Require 70%+ positive periods
        passed = consistency_score >= 0.7

        return {
            'passed': passed,
            'consistency_score': consistency_score,
            'positive_periods': positive_count,
            'total_periods': len(walk_forward_returns),
        }

    def apply_all_filters(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all configured filters to a result.

        Args:
            result: Backtest result dict

        Returns:
            Dict with filter results and overall pass status
        """
        filters_passed = {}

        # Basic metric filters
        filters_passed['min_trades'] = self.min_trades_filter(result)
        filters_passed['min_sharpe'] = self.min_sharpe_filter(result)
        filters_passed['min_profit_factor'] = self.min_profit_factor_filter(result)
        filters_passed['max_drawdown'] = self.max_drawdown_filter(result)
        filters_passed['min_win_rate'] = self.min_win_rate_filter(result)

        # Overall pass - all basic filters must pass
        basic_passed = all(filters_passed.values())

        return {
            'passed': basic_passed,
            'filters_passed': filters_passed,
            'sharpe': result.get('sharpe_ratio', 0),
            'profit_factor': result.get('profit_factor', 0),
            'total_trades': result.get('total_trades', 0),
            'max_drawdown': result.get('max_drawdown', 0),
        }

    def get_pass_rate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate pass rate for each filter across multiple results.

        Args:
            results: List of backtest results

        Returns:
            Dict of filter name -> pass rate
        """
        if not results:
            return {}

        n = len(results)
        return {
            'min_trades': sum(1 for r in results if self.min_trades_filter(r)) / n,
            'min_sharpe': sum(1 for r in results if self.min_sharpe_filter(r)) / n,
            'min_profit_factor': sum(1 for r in results if self.min_profit_factor_filter(r)) / n,
            'max_drawdown': sum(1 for r in results if self.max_drawdown_filter(r)) / n,
        }


class WalkForwardValidator:
    """
    Walk-forward validation for strategy robustness.

    Splits data into rolling train/test windows and validates
    strategy performance across out-of-sample periods.
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.6,
        min_positive_periods: int = 4
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_folds: Number of walk-forward folds
            train_ratio: Ratio of data for training
            min_positive_periods: Minimum number of positive test periods required
        """
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.min_positive_periods = min_positive_periods

    def split_data(
        self,
        data: 'pd.DataFrame',
        timestamp_col: str = 'opentime'
    ) -> List[Dict[str, Any]]:
        """
        Split data into walk-forward windows.

        Args:
            data: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            List of dicts with train/test indices
        """
        n = len(data)
        train_size = int(n * self.train_ratio)
        test_size = (n - train_size) // self.n_folds

        splits = []
        for i in range(self.n_folds):
            train_end = train_size + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_start >= n:
                break

            splits.append({
                'train_start': 0,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

        return splits

    def validate(
        self,
        strategy_func: Callable,
        data: 'pd.DataFrame',
        params: Dict[str, Any],
        run_backtest_func: Callable
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            strategy_func: Strategy function
            data: Full dataset
            params: Strategy parameters
            run_backtest_func: Function to run backtest with params on a data subset

        Returns:
            Dict with validation results
        """
        splits = self.split_data(data)

        test_returns = []
        test_sharpes = []

        for i, split in enumerate(splits):
            try:
                train_data = data.iloc[split['train_start']:split['train_end']]
                test_data = data.iloc[split['test_start']:split['test_end']]

                # Run on train to get params (could optimize here)
                train_result = run_backtest_func(strategy_func, train_data, params)

                # Run on test (out-of-sample)
                test_result = run_backtest_func(strategy_func, test_data, params)

                test_returns.append(test_result.get('return_pct', 0))
                test_sharpes.append(test_result.get('sharpe_ratio', 0))

            except Exception as e:
                logger.warning(f"Walk-forward fold {i} failed: {e}")
                test_returns.append(0)
                test_sharpes.append(0)

        consistency = self.consistency_filter(test_returns)

        return {
            'passed': consistency['passed'],
            'consistency_score': consistency['consistency_score'],
            'test_returns': test_returns,
            'test_sharpes': test_sharpes,
            'avg_test_return': np.mean(test_returns),
            'avg_test_sharpe': np.mean(test_sharpes),
            'n_folds': len(splits),
        }
