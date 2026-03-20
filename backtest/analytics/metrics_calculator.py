"""
Analytics Layer - Performance metrics calculation using vectorbt.
Returns: total pnl, winrate, expectancy, sharpe ratio, max drawdown, profit factor, average holding time
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import vectorbt
import vectorbt as vbt


class MetricsCalculator:
    """
    Calculates performance metrics for backtest results using vectorbt.
    """

    def __init__(self, use_vectorbt: bool = True):
        self.use_vectorbt = use_vectorbt

    def calculate_all(
        self,
        trades: List,
        equity_curve: Optional[pd.DataFrame] = None,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics using vectorbt.

        Args:
            trades: List of Trade objects or dicts
            equity_curve: Optional equity curve DataFrame
            initial_capital: Starting capital

        Returns:
            Dictionary of metrics
        """
        if not trades:
            return self._empty_metrics()

        # Convert to DataFrame if needed
        if isinstance(trades[0], dict):
            trades_df = pd.DataFrame(trades)
        else:
            trades_df = self._trades_to_dataframe(trades)

        metrics = {}

        # Basic metrics
        metrics.update(self._basic_metrics(trades_df, initial_capital))

        # Win/Loss metrics
        metrics.update(self._winloss_metrics(trades_df))

        # Drawdown metrics
        metrics.update(self._drawdown_metrics(trades_df, initial_capital))

        # Risk metrics using vectorbt
        metrics.update(self._vectorbt_risk_metrics(equity_curve))

        # VaR/CVaR and Tail Ratio
        metrics.update(self._risk_extreme_metrics(equity_curve))

        # Trade statistics
        metrics.update(self._trade_statistics(trades_df))

        return metrics

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'expectancy': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_holding_bars': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_trade_duration': 0.0,
            'return_pct': 0.0,
        }

    def _trades_to_dataframe(self, trades: List) -> pd.DataFrame:
        """Convert Trade objects to DataFrame"""
        records = []
        for trade in trades:
            record = {
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'side': trade.side.value if hasattr(trade.side, 'value') else trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'fees': trade.fees,
                'hold_bars': trade.hold_bars,
                'exit_reason': trade.exit_reason,
                'leverage': trade.leverage
            }
            records.append(record)

        return pd.DataFrame(records)

    def _basic_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Calculate basic metrics"""
        total_pnl = trades_df['pnl'].sum()
        final_capital = initial_capital + total_pnl
        return_pct = (total_pnl / initial_capital) * 100

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'return_pct': return_pct,
            'total_trades': len(trades_df),
            'total_fees': trades_df['fees'].sum(),
        }

    def _winloss_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate win/loss metrics"""
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total = win_count + loss_count

        win_rate = (win_count / total * 100) if total > 0 else 0.0

        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if win_count > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if loss_count > 0 else 0.0

        # Expectancy
        if total > 0:
            expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        else:
            expectancy = 0.0

        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if win_count > 0 else 0.0
        gross_loss = abs(losing_trades['pnl'].sum()) if loss_count > 0 else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Consecutive wins/losses
        max_consecutive_wins = self._max_consecutive(trades_df['pnl'] > 0)
        max_consecutive_losses = self._max_consecutive(trades_df['pnl'] <= 0)

        # Long/Short breakdown
        long_trades = trades_df[trades_df['side'] == 'long']
        short_trades = trades_df[trades_df['side'] == 'short']

        long_wins = len(long_trades[long_trades['pnl'] > 0])
        short_wins = len(short_trades[short_trades['pnl'] > 0])

        return {
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'long_trades': len(long_trades),
            'long_wins': long_wins,
            'long_win_rate': (long_wins / len(long_trades) * 100) if len(long_trades) > 0 else 0.0,
            'short_trades': len(short_trades),
            'short_wins': short_wins,
            'short_win_rate': (short_wins / len(short_trades) * 100) if len(short_trades) > 0 else 0.0,
        }

    def _max_consecutive(self, conditions: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        if len(conditions) == 0:
            return 0

        arr = conditions.astype(int).values
        max_consecutive = 0
        current_consecutive = 0

        for val in arr:
            if val:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _drawdown_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        equity = [initial_capital]
        for pnl in trades_df['pnl']:
            equity.append(equity[-1] + pnl)

        equity = np.array(equity)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100

        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        max_drawdown_dollar = np.max(running_max - equity)

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_dollar': max_drawdown_dollar,
        }

    def _vectorbt_risk_metrics(self, equity_curve: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate risk metrics using vectorbt"""
        if equity_curve is None or len(equity_curve) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility': 0.0,
            }

        try:
            # Create returns from equity curve
            returns = equity_curve['equity'].pct_change().fillna(0).values
            equity = equity_curve['equity'].values

            if len(returns) == 0:
                return {
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'volatility': 0.0,
                }

            # Infer annualization factor from data timeframe
            if 'datetime' in equity_curve.columns or len(equity_curve.index) > 0:
                # Try to infer timeframe from index
                if isinstance(equity_curve.index, pd.DatetimeIndex):
                    # Calculate average time between observations
                    if len(equity_curve) > 1:
                        time_diffs = equity_curve.index.to_series().diff().dropna()
                        avg_diff_hours = time_diffs.mean().total_seconds() / 3600
                        if avg_diff_hours > 0:
                            # Annualization: hours per year / avg observation interval
                            annualization = np.sqrt(8760 / avg_diff_hours)
                        else:
                            annualization = np.sqrt(8760)
                    else:
                        annualization = np.sqrt(8760)
                else:
                    annualization = np.sqrt(8760)
            else:
                # Default to hourly
                annualization = np.sqrt(8760)

            # Mean return and std
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Sharpe Ratio
            sharpe = (mean_return / std_return) * annualization if std_return > 0 else 0.0

            # Sortino (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino = (mean_return / downside_std) * annualization if downside_std > 0 else 0.0
            else:
                sortino = float('inf')

            # Calmar Ratio (return / max drawdown)
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

            # Use proper total return calculation (end / start - 1)
            total_return = equity[-1] / equity[0] - 1
            calmar = (total_return / max_dd) if max_dd > 0 else 0.0

            # Volatility (annualized)
            volatility = std_return * annualization * 100

            return {
                'sharpe_ratio': float(sharpe),
                'sortino_ratio': float(sortino),
                'calmar_ratio': float(calmar),
                'volatility': float(volatility),
            }
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility': 0.0,
            }

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of returns
            confidence: Confidence level (default 0.95)

        Returns:
            VaR as a positive number representing potential loss
        """
        if len(returns) == 0:
            return 0.0

        # Use historical method - percentile of returns
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)  # Return as positive value

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        CVaR is the expected return beyond the VaR threshold.

        Args:
            returns: Array of returns
            confidence: Confidence level (default 0.95)

        Returns:
            CVaR as a positive number representing expected loss beyond VaR
        """
        if len(returns) == 0:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        # Average of all returns below VaR
        cvar = returns[returns <= var].mean()
        return abs(cvar) if not np.isnan(cvar) else 0.0

    def calculate_tail_ratio(
        self,
        equity_curve: np.ndarray
    ) -> float:
        """
        Calculate Tail Ratio.

        Ratio of 95th percentile return to 5th percentile return.
        Values > 1 indicate positive skew (more extreme gains than losses).

        Args:
            equity_curve: Array of equity values

        Returns:
            Tail ratio (95th / 5th percentile)
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate returns from equity curve
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return 0.0

        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)

        if percentile_5 == 0:
            return float('inf') if percentile_95 > 0 else 0.0

        return percentile_95 / abs(percentile_5)

    def _risk_extreme_metrics(self, equity_curve: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate VaR, CVaR, and Tail Ratio metrics"""
        if equity_curve is None or len(equity_curve) == 0:
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'tail_ratio': 0.0,
            }

        try:
            # Calculate returns
            equity = equity_curve['equity'].values
            returns = np.diff(equity) / equity[:-1]
            returns = returns[~np.isnan(returns)]

            if len(returns) == 0:
                return {
                    'var_95': 0.0,
                    'var_99': 0.0,
                    'cvar_95': 0.0,
                    'cvar_99': 0.0,
                    'tail_ratio': 0.0,
                }

            # VaR at 95% and 99% confidence
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)

            # CVaR at 95% and 99% confidence
            cvar_95 = self.calculate_cvar(returns, 0.95)
            cvar_99 = self.calculate_cvar(returns, 0.99)

            # Tail Ratio
            tail_ratio = self.calculate_tail_ratio(equity)

            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'tail_ratio': float(tail_ratio),
            }
        except Exception as e:
            logger.warning(f"Error calculating extreme risk metrics: {e}")
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'tail_ratio': 0.0,
            }

    def _trade_statistics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade statistics"""
        # Average holding time
        avg_holding_bars = trades_df['hold_bars'].mean() if len(trades_df) > 0 else 0.0

        # Trade duration in hours (assuming 1h candles)
        if 'exit_time' in trades_df.columns and 'entry_time' in trades_df.columns:
            durations = (trades_df['exit_time'] - trades_df['entry_time']) / (1000 * 60 * 60)
            avg_duration_hours = durations.mean() if len(durations) > 0 else 0.0
            max_duration_hours = durations.max() if len(durations) > 0 else 0.0
            min_duration_hours = durations.min() if len(durations) > 0 else 0.0
        else:
            avg_duration_hours = 0.0
            max_duration_hours = 0.0
            min_duration_hours = 0.0

        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict() if 'exit_reason' in trades_df.columns else {}

        # Average leverage
        avg_leverage = trades_df['leverage'].mean() if 'leverage' in trades_df.columns else 1.0

        return {
            'avg_holding_bars': avg_holding_bars,
            'avg_trade_duration': avg_duration_hours,
            'max_trade_duration': max_duration_hours,
            'min_trade_duration': min_duration_hours,
            'avg_leverage': avg_leverage,
            'exit_reasons': exit_reasons,
        }
