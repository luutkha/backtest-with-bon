"""
Reporting Layer - Generate structured outputs.
Returns: trades dataframe, equity dataframe, summary metrics dict
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates structured outputs from backtest results.
    """

    def __init__(self):
        pass

    def generate_trades_dataframe(
        self,
        trades: List,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Generate trades DataFrame.

        Args:
            trades: List of Trade objects or dicts
            include_metadata: Include additional metadata columns

        Returns:
            DataFrame of all trades
        """
        if not trades:
            return pd.DataFrame()

        # Convert to records
        records = []
        for trade in trades:
            if hasattr(trade, '__dict__'):
                record = trade.__dict__.copy()
                # Convert enums to strings
                if hasattr(trade, 'side') and hasattr(trade.side, 'value'):
                    record['side'] = trade.side.value
                if hasattr(trade, 'exit_reason'):
                    record['exit_reason'] = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason
            else:
                record = trade.copy()

            # Add datetime columns
            if 'entry_time' in record:
                record['entry_datetime'] = pd.to_datetime(record['entry_time'], unit='ms')
            if 'exit_time' in record:
                record['exit_datetime'] = pd.to_datetime(record['exit_time'], unit='ms')

            # Calculate duration
            if 'entry_time' in record and 'exit_time' in record:
                record['duration_minutes'] = (record['exit_time'] - record['entry_time']) / (1000 * 60)

            records.append(record)

        df = pd.DataFrame(records)

        # Reorder columns
        priority_cols = [
            'entry_datetime', 'exit_datetime', 'side', 'entry_price', 'exit_price',
            'quantity', 'pnl', 'pnl_pct', 'fees', 'exit_reason', 'hold_bars'
        ]

        cols = [c for c in priority_cols if c in df.columns]
        remaining = [c for c in df.columns if c not in priority_cols]
        df = df[cols + remaining]

        return df

    def generate_equity_dataframe(
        self,
        trades: List,
        h1_data: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Generate equity curve DataFrame.

        Args:
            trades: List of trades
            h1_data: 1h OHLCV data (for timestamps)
            initial_capital: Starting capital

        Returns:
            DataFrame with equity curve
        """
        # Start with initial capital
        equity_curve = [initial_capital]
        timestamps = [h1_data['opentime'].iloc[0] - 3600000]  # Before first candle

        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
            timestamps.append(trade.exit_time)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity_curve
        })

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['equity_pct'] = (df['equity'] / initial_capital - 1) * 100
        df['drawdown'] = (df['equity'].cummax() - df['equity']) / df['equity'].cummax() * 100

        df = df.set_index('datetime')

        return df

    def generate_summary_metrics(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate summary metrics dictionary.

        Args:
            metrics: Calculated metrics
            config: Configuration used for backtest

        Returns:
            Summary dictionary
        """
        summary = {
            'backtest_info': {
                'symbol': config.get('symbol', 'UNKNOWN'),
                'timeframe': config.get('timeframe', '1h'),
                'start_time': config.get('start_time', 'N/A'),
                'end_time': config.get('end_time', 'N/A'),
                'generated_at': datetime.now().isoformat(),
            },
            'configuration': {
                'initial_capital': config.get('initial_capital', 10000),
                'leverage': config.get('leverage', 1.0),
                'fee_rate': config.get('fee_rate', 0.0004),
                'tp_pct': config.get('tp_pct', 0.02),
                'sl_pct': config.get('sl_pct', 0.01),
                'position_size_pct': config.get('position_size_pct', 0.95),
            },
            'performance': {
                'total_return': metrics.get('total_pnl', 0),
                'return_pct': metrics.get('return_pct', 0),
                'final_capital': metrics.get('final_capital', config.get('initial_capital', 10000)),
            },
            'trading': {
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': metrics.get('winning_trades', 0),
                'losing_trades': metrics.get('losing_trades', 0),
                'win_rate': metrics.get('win_rate', 0),
            },
            'risk': {
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'profit_factor': metrics.get('profit_factor', 0),
            },
            'execution': {
                'avg_holding_bars': metrics.get('avg_holding_bars', 0),
                'avg_trade_duration': metrics.get('avg_trade_duration', 0),
                'total_fees': metrics.get('total_fees', 0),
            }
        }

        return summary

    def save_results(
        self,
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        metrics: Dict[str, Any],
        output_dir: str,
        prefix: str = "backtest"
    ) -> None:
        """
        Save results to files.

        Args:
            trades_df: Trades DataFrame
            equity_df: Equity curve DataFrame
            metrics: Summary metrics
            output_dir: Output directory
            prefix: Filename prefix
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save trades
        trades_path = os.path.join(output_dir, f"{prefix}_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved trades to {trades_path}")

        # Save equity curve
        equity_path = os.path.join(output_dir, f"{prefix}_equity.csv")
        equity_df.to_csv(equity_path)
        logger.info(f"Saved equity curve to {equity_path}")

        # Save metrics
        metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_path}")

    def print_summary(self, metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Print formatted summary to console"""
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        print(f"\nSymbol: {config.get('symbol', 'UNKNOWN')}")
        print(f"Timeframe: {config.get('timeframe', '1h')}")
        print(f"Period: {config.get('start_time', 'N/A')} to {config.get('end_time', 'N/A')}")

        print("\n--- Configuration ---")
        print(f"Initial Capital: ${config.get('initial_capital', 10000):,.2f}")
        print(f"Leverage: {config.get('leverage', 1.0)}x")
        print(f"Fee Rate: {config.get('fee_rate', 0.0004)*100:.4f}%")
        print(f"TP/SL: {config.get('tp_pct', 0.02)*100:.1f}% / {config.get('sl_pct', 0.01)*100:.1f}%")

        print("\n--- Performance ---")
        print(f"Total Return: ${metrics.get('total_pnl', 0):,.2f} ({metrics.get('return_pct', 0):.2f}%)")
        print(f"Final Capital: ${metrics.get('final_capital', 0):,.2f}")
        print(f"Total Fees: ${metrics.get('total_fees', 0):,.2f}")

        print("\n--- Trading Statistics ---")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Winning: {metrics.get('winning_trades', 0)}")
        print(f"Losing: {metrics.get('losing_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        print("\n--- Risk Metrics ---")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Expectancy: ${metrics.get('expectancy', 0):.2f}")

        print("\n--- Trade Details ---")
        print(f"Avg Win: ${metrics.get('avg_win', 0):,.2f}")
        print(f"Avg Loss: ${metrics.get('avg_loss', 0):,.2f}")
        print(f"Avg Holding Bars: {metrics.get('avg_holding_bars', 0):.1f}")
        print(f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
        print(f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")

        if metrics.get('long_trades', 0) > 0:
            print(f"\n--- Long/Short Breakdown ---")
            print(f"Long Trades: {metrics.get('long_trades', 0)} ({metrics.get('long_win_rate', 0):.1f}% win rate)")
            print(f"Short Trades: {metrics.get('short_trades', 0)} ({metrics.get('short_win_rate', 0):.1f}% win rate)")

        print("\n" + "=" * 70)
