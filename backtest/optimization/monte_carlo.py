"""
Monte Carlo Simulation for backtest results.

Provides resampling of trades to generate distribution of outcomes
with confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Performs Monte Carlo simulation on backtest trades.

    Resamples trades with replacement to generate distribution
    of possible outcomes and confidence intervals.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        num_simulations: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            initial_capital: Starting capital
            num_simulations: Number of Monte Carlo iterations
            random_seed: Random seed for reproducibility
        """
        self.initial_capital = initial_capital
        self.num_simulations = num_simulations

        if random_seed is not None:
            np.random.seed(random_seed)

    def run_simulation(
        self,
        trades: List[Dict[str, Any]],
        risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on trades.

        Args:
            trades: List of trade dictionaries with 'pnl' key
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (annualized)

        Returns:
            Dictionary with simulation results and statistics
        """
        if not trades:
            return self._empty_results()

        # Extract PnL values - handle both dict and dataclass
        pnls = np.array([t.pnl if hasattr(t, 'pnl') else t['pnl'] for t in trades])
        num_trades = len(pnls)

        # Run simulations
        final_capitals = []
        max_drawdowns = []
        returns = []
        win_rates = []
        sharpe_ratios = []

        for _ in range(self.num_simulations):
            # Resample trades with replacement
            resampled_pnls = np.random.choice(pnls, size=num_trades, replace=True)

            # Calculate equity curve
            equity = self.initial_capital + np.cumsum(resampled_pnls)
            final_capitals.append(equity[-1])

            # Calculate return
            ret = (equity[-1] - self.initial_capital) / self.initial_capital
            returns.append(ret)

            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            max_drawdowns.append(np.max(drawdown))

            # Calculate win rate
            wins = np.sum(resampled_pnls > 0)
            win_rates.append(wins / num_trades)

            # Calculate Sharpe ratio for this simulation
            period_returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([ret])
            if len(period_returns) > 0 and np.std(period_returns) > 0:
                sharpe = (np.mean(period_returns) - risk_free_rate / num_trades) / np.std(period_returns) * np.sqrt(num_trades)
            else:
                sharpe = 0.0
            sharpe_ratios.append(sharpe)

        final_capitals = np.array(final_capitals)
        max_drawdowns = np.array(max_drawdowns)
        returns = np.array(returns)
        win_rates = np.array(win_rates)
        sharpe_ratios = np.array(sharpe_ratios)

        # Calculate statistics
        return self._calculate_statistics(
            final_capitals,
            max_drawdowns,
            returns,
            win_rates,
            sharpe_ratios
        )

    def run_bootstrap(
        self,
        trades: List[Dict[str, Any]],
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Run bootstrap simulation (sample size equals original).

        Args:
            trades: List of trade dictionaries
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Dictionary with bootstrap results
        """
        if not trades:
            return self._empty_results()

        pnls = np.array([t['pnl'] for t in trades])
        n_trades = len(pnls)

        bootstrap_final_capitals = []
        bootstrap_max_drawdowns = []

        for _ in range(n_bootstrap):
            # Sample with replacement, same size as original
            sample = np.random.choice(pnls, size=n_trades, replace=True)
            equity = self.initial_capital + np.cumsum(sample)
            bootstrap_final_capitals.append(equity[-1])

            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            bootstrap_max_drawdowns.append(np.max(drawdown))

        return {
            'bootstrap_final_capitals': bootstrap_final_capitals,
            'bootstrap_max_drawdowns': bootstrap_max_drawdowns,
            'bootstrap_mean_capital': np.mean(bootstrap_final_capitals),
            'bootstrap_median_capital': np.median(bootstrap_final_capitals),
            'bootstrap_mean_drawdown': np.mean(bootstrap_max_drawdowns),
        }

    def run_block_bootstrap(
        self,
        trades: List[Dict[str, Any]],
        block_size: int = 5,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Run block bootstrap to preserve trade sequence correlations.

        Args:
            trades: List of trade dictionaries
            block_size: Size of blocks to resample
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Dictionary with block bootstrap results
        """
        if not trades or block_size < 1:
            return self._empty_results()

        pnls = np.array([t['pnl'] for t in trades])
        n_trades = len(pnls)

        # Create blocks
        n_blocks = int(np.ceil(n_trades / block_size))

        block_final_capitals = []
        block_max_drawdowns = []

        for _ in range(n_bootstrap):
            # Randomly select blocks with replacement
            block_indices = np.random.randint(0, n_blocks, size=n_blocks)

            # Construct resampled sequence
            resampled = []
            for block_idx in block_indices:
                start = block_idx * block_size
                end = min(start + block_size, n_trades)
                resampled.extend(pnls[start:end])

            resampled = np.array(resampled[:n_trades])  # Truncate to original size
            equity = self.initial_capital + np.cumsum(resampled)
            block_final_capitals.append(equity[-1])

            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            block_max_drawdowns.append(np.max(drawdown))

        return {
            'block_bootstrap_final_capitals': block_final_capitals,
            'block_bootstrap_max_drawdowns': block_max_drawdowns,
            'block_bootstrap_mean_capital': np.mean(block_final_capitals),
            'block_bootstrap_median_capital': np.median(block_final_capitals),
        }

    def _calculate_statistics(
        self,
        final_capitals: np.ndarray,
        max_drawdowns: np.ndarray,
        returns: np.ndarray,
        win_rates: np.ndarray,
        sharpe_ratios: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate statistics with confidence intervals"""

        def confidence_interval(data: np.ndarray, confidence: float = 0.95):
            """Calculate confidence interval"""
            alpha = (1 - confidence) / 2
            lower = np.percentile(data, alpha * 100)
            upper = np.percentile(data, (1 - alpha) * 100)
            return lower, upper

        result = {
            # Final capital statistics
            'mean_final_capital': float(np.mean(final_capitals)),
            'median_final_capital': float(np.median(final_capitals)),
            'std_final_capital': float(np.std(final_capitals)),
            'min_final_capital': float(np.min(final_capitals)),
            'max_final_capital': float(np.max(final_capitals)),
            'capital_ci_95': tuple(map(float, confidence_interval(final_capitals, 0.95))),
            'capital_ci_99': tuple(map(float, confidence_interval(final_capitals, 0.99))),

            # Return statistics
            'mean_return': float(np.mean(returns)),
            'median_return': float(np.median(returns)),
            'std_return': float(np.std(returns)),
            'return_ci_95': tuple(map(float, confidence_interval(returns, 0.95))),

            # Drawdown statistics
            'mean_max_drawdown': float(np.mean(max_drawdowns)),
            'median_max_drawdown': float(np.median(max_drawdowns)),
            'std_max_drawdown': float(np.std(max_drawdowns)),
            'max_drawdown_ci_95': tuple(map(float, confidence_interval(max_drawdowns, 0.95))),

            # Win rate statistics
            'mean_win_rate': float(np.mean(win_rates)),
            'median_win_rate': float(np.median(win_rates)),
            'win_rate_ci_95': tuple(map(float, confidence_interval(win_rates, 0.95))),

            # Probability metrics
            'prob_profit': float(np.mean(final_capitals > self.initial_capital)),
            'prob_loss': float(np.mean(final_capitals < self.initial_capital)),
            'prob_reaching_target': float(np.mean(returns >= 0.1)),  # 10% target

            # Distribution percentiles
            'percentile_5_capital': float(np.percentile(final_capitals, 5)),
            'percentile_25_capital': float(np.percentile(final_capitals, 25)),
            'percentile_50_capital': float(np.percentile(final_capitals, 50)),
            'percentile_75_capital': float(np.percentile(final_capitals, 75)),
            'percentile_95_capital': float(np.percentile(final_capitals, 95)),
        }

        # Add Sharpe ratio statistics if provided
        if sharpe_ratios is not None and len(sharpe_ratios) > 0:
            result.update({
                'mean_sharpe_ratio': float(np.mean(sharpe_ratios)),
                'median_sharpe_ratio': float(np.median(sharpe_ratios)),
                'std_sharpe_ratio': float(np.std(sharpe_ratios)),
                'min_sharpe_ratio': float(np.min(sharpe_ratios)),
                'max_sharpe_ratio': float(np.max(sharpe_ratios)),
                'sharpe_ratio_ci_95': tuple(map(float, confidence_interval(sharpe_ratios, 0.95))),
                'percentile_5_sharpe': float(np.percentile(sharpe_ratios, 5)),
                'percentile_25_sharpe': float(np.percentile(sharpe_ratios, 25)),
                'percentile_50_sharpe': float(np.percentile(sharpe_ratios, 50)),
                'percentile_75_sharpe': float(np.percentile(sharpe_ratios, 75)),
                'percentile_95_sharpe': float(np.percentile(sharpe_ratios, 95)),
                'prob_sharpe_positive': float(np.mean(sharpe_ratios > 0)),
            })

        return result

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'mean_final_capital': 0.0,
            'median_final_capital': 0.0,
            'std_final_capital': 0.0,
            'min_final_capital': 0.0,
            'max_final_capital': 0.0,
            'capital_ci_95': (0.0, 0.0),
            'capital_ci_99': (0.0, 0.0),
            'mean_return': 0.0,
            'median_return': 0.0,
            'std_return': 0.0,
            'return_ci_95': (0.0, 0.0),
            'mean_max_drawdown': 0.0,
            'median_max_drawdown': 0.0,
            'std_max_drawdown': 0.0,
            'max_drawdown_ci_95': (0.0, 0.0),
            'mean_win_rate': 0.0,
            'median_win_rate': 0.0,
            'win_rate_ci_95': (0.0, 0.0),
            'prob_profit': 0.0,
            'prob_loss': 0.0,
            'prob_reaching_target': 0.0,
            'percentile_5_capital': 0.0,
            'percentile_25_capital': 0.0,
            'percentile_50_capital': 0.0,
            'percentile_75_capital': 0.0,
            'percentile_95_capital': 0.0,
            'mean_sharpe_ratio': 0.0,
            'median_sharpe_ratio': 0.0,
            'std_sharpe_ratio': 0.0,
            'min_sharpe_ratio': 0.0,
            'max_sharpe_ratio': 0.0,
            'sharpe_ratio_ci_95': (0.0, 0.0),
            'percentile_5_sharpe': 0.0,
            'percentile_25_sharpe': 0.0,
            'percentile_50_sharpe': 0.0,
            'percentile_75_sharpe': 0.0,
            'percentile_95_sharpe': 0.0,
            'prob_sharpe_positive': 0.0,
        }

    def generate_equity_curves(
        self,
        trades: List[Dict[str, Any]],
        n_curves: int = 100
    ) -> np.ndarray:
        """
        Generate sample equity curves for visualization.

        Args:
            trades: List of trade dictionaries
            n_curves: Number of equity curves to generate

        Returns:
            Array of equity curves (n_curves x num_trades+1)
        """
        if not trades:
            return np.array([])

        pnls = np.array([t['pnl'] for t in trades])
        num_trades = len(pnls)

        curves = np.zeros((n_curves, num_trades + 1))
        curves[:, 0] = self.initial_capital

        for i in range(n_curves):
            resampled = np.random.choice(pnls, size=num_trades, replace=True)
            curves[i, 1:] = self.initial_capital + np.cumsum(resampled)

        return curves
