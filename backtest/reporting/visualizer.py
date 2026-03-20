"""
Visualization utilities for backtest results using matplotlib.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualization disabled.")


class BacktestVisualizer:
    """
    Visualization tools for backtest results.

    Provides equity curves, drawdown charts, trade distributions, and more.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            style: matplotlib style to use
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")

        self.figsize = figsize

        # Apply style
        try:
            plt.style.use(style)
        except:
            plt.style.use('ggplot')

    def plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        title: str = "Equity Curve",
        show_drawdown: bool = True,
        ax=None,
    ) -> Optional[Any]:
        """
        Plot equity curve with optional drawdown.

        Args:
            equity_df: DataFrame with 'equity' column (and optionally 'drawdown')
            title: Plot title
            show_drawdown: Show drawdown on secondary axis
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Plot equity
        ax.plot(equity_df.index, equity_df['equity'], 'b-', linewidth=1.5, label='Equity')

        # Add drawdown on secondary axis
        if show_drawdown and 'drawdown' in equity_df.columns:
            ax2 = ax.twinx()
            ax2.fill_between(equity_df.index, 0, -equity_df['drawdown'],
                            alpha=0.3, color='red', label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        if hasattr(equity_df.index, 'year'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

        return ax

    def plot_drawdown(
        self,
        equity_df: pd.DataFrame,
        title: str = "Drawdown",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot drawdown chart.

        Args:
            equity_df: DataFrame with 'drawdown' column
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        if 'drawdown' not in equity_df.columns:
            # Calculate drawdown if not present
            equity = equity_df['equity']
            running_max = equity.cummax()
            drawdown = (running_max - equity) / running_max * 100
        else:
            drawdown = equity_df['drawdown']

        ax.fill_between(equity_df.index, 0, -drawdown, alpha=0.5, color='red')
        ax.plot(equity_df.index, -drawdown, 'r-', linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        if hasattr(equity_df.index, 'year'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

        return ax

    def plot_returns_distribution(
        self,
        trades: List,
        title: str = "Returns Distribution",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot histogram of trade returns.

        Args:
            trades: List of Trade objects
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Extract returns
        returns = [t.pnl_pct for t in trades if hasattr(t, 'pnl_pct')]

        if not returns:
            return ax

        # Separate wins and losses
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        # Plot histogram
        ax.hist(returns, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

        # Add vertical lines for mean
        if returns:
            mean_return = np.mean(returns)
            ax.axvline(mean_return, color='green', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_return:.2f}%')

        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_trade_pnl(
        self,
        trades: List,
        title: str = "Trade PnL",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot individual trade PnL as bar chart.

        Args:
            trades: List of Trade objects
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Extract PnL
        pnls = [t.pnl for t in trades if hasattr(t, 'pnl')]

        if not pnls:
            return ax

        trade_nums = range(1, len(pnls) + 1)
        colors = ['green' if p > 0 else 'red' for p in pnls]

        ax.bar(trade_nums, pnls, color=colors, alpha=0.7, edgecolor='black')

        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xlabel('Trade Number')
        ax.set_ylabel('PnL ($)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        return ax

    def plot_cumulative_returns(
        self,
        trades: List,
        initial_capital: float = 10000,
        title: str = "Cumulative Returns",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot cumulative returns over time.

        Args:
            trades: List of Trade objects
            initial_capital: Starting capital
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Get exit times and cumulative returns
        exit_times = []
        cumulative = [0]

        for t in trades:
            if hasattr(t, 'exit_time') and hasattr(t, 'pnl'):
                exit_times.append(t.exit_time)
                cumulative.append(cumulative[-1] + t.pnl)

        if not exit_times:
            return ax

        # Convert to percentage
        cumulative_pct = [(c / initial_capital) * 100 for c in cumulative]

        # Create datetime index
        dates = pd.to_datetime(exit_times, unit='ms')
        dates = pd.DatetimeIndex([dates[0] - pd.Timedelta(hours=1)] + list(dates))

        ax.plot(dates, cumulative_pct, 'b-', linewidth=1.5)
        ax.fill_between(dates, 0, cumulative_pct, alpha=0.3, color='blue')

        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        return ax

    def plot_monthly_returns(
        self,
        trades: List,
        title: str = "Monthly Returns",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot heatmap of monthly returns.

        Args:
            trades: List of Trade objects
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Extract returns by month
        returns_by_month = {}

        for t in trades:
            if hasattr(t, 'exit_time') and hasattr(t, 'pnl'):
                date = pd.to_datetime(t.exit_time, unit='ms')
                month_key = (date.year, date.month)
                returns_by_month[month_key] = returns_by_month.get(month_key, 0) + t.pnl

        if not returns_by_month:
            return ax

        # Create DataFrame
        df = pd.DataFrame([
            {'year': k[0], 'month': k[1], 'pnl': v}
            for k, v in returns_by_month.items()
        ])

        if df.empty:
            return ax

        # Pivot for heatmap
        pivot = df.pivot(index='year', columns='month', values='pnl')

        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='PnL ($)')

        # Set labels
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)])

        ax.set_title(title)

        return ax

    def plot_win_loss_ratio(
        self,
        metrics: Dict[str, Any],
        title: str = "Win/Loss Analysis",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot win/loss breakdown as pie chart.

        Args:
            metrics: Metrics dictionary
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        winning = metrics.get('winning_trades', 0)
        losing = metrics.get('losing_trades', 0)

        if winning + losing == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return ax

        sizes = [winning, losing]
        labels = [f'Wins\n{winning}', f'Losses\n{losing}']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title(title)

        return ax

    def plot_exit_reasons(
        self,
        metrics: Dict[str, Any],
        title: str = "Exit Reasons",
        ax=None,
    ) -> Optional[Any]:
        """
        Plot breakdown of exit reasons.

        Args:
            metrics: Metrics dictionary with 'exit_reasons'
            title: Plot title
            ax: Optional axes object

        Returns:
            matplotlib axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        exit_reasons = metrics.get('exit_reasons', {})

        if not exit_reasons:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center')
            return ax

        reasons = list(exit_reasons.keys())
        counts = list(exit_reasons.values())

        # Sort by count
        sorted_pairs = sorted(zip(counts, reasons), reverse=True)
        counts, reasons = zip(*sorted_pairs)

        colors = plt.cm.Set3(range(len(reasons)))

        ax.barh(reasons, counts, color=colors, edgecolor='black')
        ax.set_xlabel('Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

        return ax

    def plot_metrics_summary(
        self,
        metrics: Dict[str, Any],
        title: str = "Performance Metrics",
    ) -> Optional[Any]:
        """
        Create a summary dashboard of key metrics.

        Args:
            metrics: Metrics dictionary
            title: Plot title

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Key metrics to display
        key_metrics = [
            ('Total Trades', f"{metrics.get('total_trades', 0)}"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.1f}%"),
            ('Total PnL', f"${metrics.get('total_pnl', 0):,.2f}"),
            ('Return %', f"{metrics.get('return_pct', 0):.2f}%"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Avg Win', f"${metrics.get('avg_win', 0):,.2f}"),
        ]

        # Display metrics as text
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off')

        text_str = " | ".join([f"{k}: {v}" for k, v in key_metrics])
        ax_text.text(0.5, 0.5, text_str, ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Win/Loss pie
        ax_pie = fig.add_subplot(gs[1, 0])
        self.plot_win_loss_ratio(metrics, ax=ax_pie)

        # Exit reasons
        ax_exit = fig.add_subplot(gs[1, 1])
        self.plot_exit_reasons(metrics, ax=ax_exit)

        # Risk metrics (if available)
        ax_risk = fig.add_subplot(gs[1, 2])
        ax_risk.axis('off')

        var_95 = metrics.get('var_95', 0)
        cvar_95 = metrics.get('cvar_95', 0)
        tail_ratio = metrics.get('tail_ratio', 0)

        risk_text = f"VaR (95%): {var_95:.4f}\n"
        risk_text += f"CVaR (95%): {cvar_95:.4f}\n"
        risk_text += f"Tail Ratio: {tail_ratio:.2f}"

        ax_risk.text(0.5, 0.5, risk_text, ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Holdings distribution
        ax_hold = fig.add_subplot(gs[2, :])
        if 'avg_holding_bars' in metrics:
            ax_hold.axis('off')
            hold_text = f"Avg Holding Bars: {metrics.get('avg_holding_bars', 0):.1f}\n"
            hold_text += f"Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.2f} hours\n"
            hold_text += f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}\n"
            hold_text += f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}"
            ax_hold.text(0.5, 0.5, hold_text, ha='center', va='center',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        fig.suptitle(title, fontsize=14, fontweight='bold')

        return fig

    def plot_all(
        self,
        trades: List,
        equity_df: pd.DataFrame,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create a full dashboard with all plots.

        Args:
            trades: List of Trade objects
            equity_df: Equity curve DataFrame
            metrics: Metrics dictionary
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        # Equity curve
        ax_equity = fig.add_subplot(gs[0, :])
        self.plot_equity_curve(equity_df, ax=ax_equity)

        # Drawdown
        ax_dd = fig.add_subplot(gs[1, 0])
        self.plot_drawdown(equity_df, ax=ax_dd)

        # Returns distribution
        ax_dist = fig.add_subplot(gs[1, 1])
        self.plot_returns_distribution(trades, ax=ax_dist)

        # Trade PnL
        ax_pnl = fig.add_subplot(gs[2, 0])
        self.plot_trade_pnl(trades, ax=ax_pnl)

        # Win/Loss pie
        ax_pie = fig.add_subplot(gs[2, 1])
        self.plot_win_loss_ratio(metrics, ax=ax_pie)

        fig.suptitle('Backtest Results Dashboard', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved dashboard to {save_path}")

        return fig

    def plot_trades(
        self,
        price_data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        title: str = "Trades Visualization",
        show_markers: bool = True,
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Plot price data with trade entry/exit markers.

        Args:
            price_data: DataFrame with OHLC data (open, high, low, close)
            trades: List of trade dictionaries with entry_time, exit_time, side, entry_price, exit_price
            title: Plot title
            show_markers: Whether to show entry/exit markers
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot price
        ax.plot(price_data.index, price_data['close'], label='Close', linewidth=1)

        # Plot trades
        long_entries = []
        long_exits = []
        short_entries = []
        short_exits = []

        for trade in trades:
            entry_idx = None
            exit_idx = None

            # Find indices
            for i, idx in enumerate(price_data.index):
                ts = price_data.loc[idx, 'opentime'] if 'opentime' in price_data.columns else idx
                if ts == trade.get('entry_time'):
                    entry_idx = i
                if ts == trade.get('exit_time'):
                    exit_idx = i

            if entry_idx is not None and exit_idx is not None:
                if trade.get('side') == 'long':
                    long_entries.append(entry_idx)
                    long_exits.append(exit_idx)
                else:
                    short_entries.append(entry_idx)
                    short_exits.append(exit_idx)

                # Draw trade line
                if show_markers:
                    ax.axvline(x=entry_idx, color='green' if trade.get('side') == 'long' else 'red',
                              alpha=0.3, linestyle='--', linewidth=0.5)
                    ax.axvline(x=exit_idx, color='blue', alpha=0.3, linestyle='--', linewidth=0.5)

        if show_markers:
            # Mark long entries
            if long_entries:
                entry_prices = [price_data.iloc[i]['close'] for i in long_entries]
                ax.scatter(long_entries, entry_prices, color='green', marker='^', s=100,
                          label='Long Entry', zorder=5)

            # Mark long exits
            if long_exits:
                exit_prices = [price_data.iloc[i]['close'] for i in long_exits]
                ax.scatter(long_exits, exit_prices, color='red', marker='v', s=100,
                          label='Long Exit', zorder=5)

            # Mark short entries
            if short_entries:
                entry_prices = [price_data.iloc[i]['close'] for i in short_entries]
                ax.scatter(short_entries, entry_prices, color='red', marker='v', s=100,
                          label='Short Entry', zorder=5)

            # Mark short exits
            if short_exits:
                exit_prices = [price_data.iloc[i]['close'] for i in short_exits]
                ax.scatter(short_exits, exit_prices, color='green', marker='^', s=100,
                          label='Short Exit', zorder=5)

            ax.legend(loc='upper left')

        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved trades plot to {save_path}")

        return fig

    def plot_optimization_results(
        self,
        grid_results: Dict[str, Any],
        metric: str = 'sharpe_ratio',
        title: str = "Optimization Results",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Plot optimization results from grid search.

        Args:
            grid_results: Dictionary with optimization results containing:
                - 'params': list of parameter combinations
                - 'results': list of metric results for each param
            metric: Metric to plot (e.g., 'sharpe_ratio', 'return_pct')
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        params = grid_results.get('params', [])
        results = grid_results.get('results', [])

        if not params or not results:
            logger.warning("No optimization results to plot")
            return None

        # Extract metric values
        metric_values = [r.get(metric, 0) for r in results]

        # Check if 2D grid (two parameters)
        if len(params) > 0 and len(params[0]) == 2:
            # 2D heatmap
            param_names = list(params[0].keys())
            param1_values = sorted(set(p[param_names[0]] for p in params))
            param2_values = sorted(set(p[param_names[1]] for p in params))

            # Create meshgrid
            X, Y = np.meshgrid(param1_values, param2_values)
            Z = np.zeros_like(X)

            # Fill Z with metric values
            for i, p in enumerate(params):
                idx1 = param1_values.index(p[param_names[0]])
                idx2 = param2_values.index(p[param_names[1]])
                Z[idx2, idx1] = metric_values[i]

            fig, ax = plt.subplots(figsize=self.figsize)
            im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_title(f"{title} - {metric}")
            plt.colorbar(im, ax=ax, label=metric)

            # Mark best parameters
            best_idx = np.argmax(metric_values)
            best_params = params[best_idx]
            ax.scatter(best_params[param_names[0]], best_params[param_names[1]],
                      color='red', marker='*', s=300, label=f'Best: {best_params}')
            ax.legend()

        else:
            # 1D line plot
            fig, ax = plt.subplots(figsize=self.figsize)

            # Get parameter name and values
            if len(params) > 0:
                param_name = list(params[0].keys())[0] if params[0] else 'Parameter'
            else:
                param_name = 'Parameter'

            param_values = [p.get(param_name, i) for i, p in enumerate(params)]

            ax.plot(param_values, metric_values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.set_title(f"{title} - {metric}")
            ax.grid(True, alpha=0.3)

            # Mark best
            best_idx = np.argmax(metric_values)
            best_value = param_values[best_idx]
            ax.scatter([best_value], [metric_values[best_idx]], color='red', s=150, zorder=5,
                      label=f'Best: {param_name}={best_value:.4f}, {metric}={metric_values[best_idx]:.4f}')
            ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved optimization plot to {save_path}")

        return fig

    def save_plot(
        self,
        fig,
        path: str,
        dpi: int = 150,
    ) -> None:
        """
        Save figure to file.

        Args:
            fig: matplotlib figure
            path: Output path
            dpi: Resolution
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {path}")

    def close_all(self) -> None:
        """Close all open figures."""
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')
