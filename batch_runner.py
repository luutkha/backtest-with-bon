"""
Batch Backtest Runner (Unified Portfolio Mode)
Run backtest on multiple symbols as a single unified portfolio
"""

import os
import pandas as pd
from typing import List, Callable, Dict, Any
import logging
import argparse
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.unified_portfolio import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from backtest.signals import rsi_strategy, moving_average_crossover_strategy, macd_strategy
from backtest.signals.streak_breakout_strategy import streak_breakout_strategy
from backtest.execution import ExitPriority

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Strategy map
STRATEGIES = {
    # 'rsi': rsi_strategy,
    # 'ma': moving_average_crossover_strategy,
    # 'macd': macd_strategy,
    'streak': streak_breakout_strategy,
}


def run_unified_backtest(
    symbols: List[str],
    data_dir: str,
    strategy: Callable,
    config: UnifiedPortfolioConfig,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run unified portfolio backtest across all symbols"""
    try:
        engine = UnifiedPortfolioBacktest(
            config=config,
            strategy=strategy,
            data_dir=data_dir,
        )
        engine.load_data(symbols)
        results = engine.run_backtest()

        if verbose:
            engine.print_summary()

        return {
            'total_trades': results['metrics'].get('total_trades', 0),
            'winning_trades': results['metrics'].get('winning_trades', 0),
            'losing_trades': results['metrics'].get('losing_trades', 0),
            'win_rate': results['metrics'].get('win_rate', 0),
            'total_pnl': results['metrics'].get('total_pnl', 0),
            'return_pct': results['metrics'].get('return_pct', 0),
            'final_capital': results['final_capital'],
            'max_drawdown': results['metrics'].get('max_drawdown', 0),
            'sharpe_ratio': results['metrics'].get('sharpe_ratio', 0),
            'profit_factor': results['metrics'].get('profit_factor', 0),
            'avg_win': results['metrics'].get('avg_win', 0),
            'avg_loss': results['metrics'].get('avg_loss', 0),
            'avg_holding_bars': results['metrics'].get('avg_holding_bars', 0),
            'total_fees': results['metrics'].get('total_fees', 0),
        }
    except Exception as e:
        return {'error': str(e)}


def get_available_symbols(data_dir: str, timeframe: str = "1h") -> List[str]:
    """Get list of available symbols from data directory"""
    path = os.path.join(data_dir, timeframe)
    if not os.path.exists(path):
        return []
    files = [f.replace(".csv", "") for f in os.listdir(path) if f.endswith(".csv")]
    return sorted(files)


def run_batch(
    symbols: List[str],
    data_dir: str,
    strategy_name: str,
    config: UnifiedPortfolioConfig,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run unified portfolio backtest on multiple symbols"""

    strategy = STRATEGIES.get(strategy_name)
    if not strategy:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    result = run_unified_backtest(symbols, data_dir, strategy, config, verbose)

    if 'error' in result:
        logger.warning(f"Unified portfolio backtest failed: {result['error']}")
        return []

    # Return single result since it's one unified portfolio
    return [result]


def print_batch_summary(results: List[Dict[str, Any]]):
    """Print unified portfolio summary"""
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r and r.get('total_trades', 0) > 0]

    if not valid_results:
        logger.info("No valid results to summarize")
        return

    # Unified portfolio returns single aggregated result
    r = valid_results[0]

    logger.info("=" * 70)
    logger.info("UNIFIED PORTFOLIO BACKTEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Trades: {r['total_trades']}")
    logger.info(f"Win Rate: {r['win_rate']:.1f}%")
    logger.info(f"Winning Trades: {r['winning_trades']}")
    logger.info(f"Losing Trades: {r['losing_trades']}")
    logger.info("-" * 70)
    logger.info(f"Total PnL: ${r['total_pnl']:,.2f}")
    logger.info(f"Return: {r['return_pct']:.2f}%")
    logger.info(f"Final Capital: ${r['final_capital']:,.2f}")
    logger.info(f"Max Drawdown: {r['max_drawdown']:.2f}%")
    logger.info(f"Profit Factor: {r['profit_factor']:.2f}")
    logger.info(f"Total Fees: ${r['total_fees']:,.2f}")
    logger.info(f"Avg Win: ${r['avg_win']:,.2f}")
    logger.info(f"Avg Loss: ${r['avg_loss']:,.2f}")
    logger.info("=" * 70)


def save_results_to_csv(results: List[Dict[str, Any]], filename: str = "batch_results.csv"):
    """Save unified portfolio results to CSV"""
    clean_results = [r for r in results if 'error' not in r]
    if clean_results:
        df = pd.DataFrame(clean_results)
        # Add symbol count for reference
        df['symbols'] = len([r for r in results if 'error' not in r])
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Batch backtest runner (Unified Portfolio Mode)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to test (default: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT)')
    parser.add_argument('--all', action='store_true',
                        help='Run on all available symbols')
    parser.add_argument('--strategy', default='rsi',
                        choices=['rsi', 'ma', 'macd', 'streak'],
                        help='Strategy to use')
    parser.add_argument('--data-dir', default=r'C:\Personals\Code\backtest-with-bon',
                        help='Data directory')
    parser.add_argument('--tp', type=float, default=0.02,
                        help='Take profit %')
    parser.add_argument('--sl', type=float, default=0.01,
                        help='Stop loss %')
    parser.add_argument('--leverage', type=float, default=1.0)
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital for unified portfolio (default: 100000)')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Maximum simultaneous positions (default: 10)')
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as % of available capital (default: 0.1 = 10%)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    DATA_DIR = args.data_dir

    # Get symbols
    if args.all:
        symbols = get_available_symbols(DATA_DIR)
        logger.info(f"Found {len(symbols)} symbols")
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

    logger.info(f"Running {args.strategy} strategy on {len(symbols)} symbols as unified portfolio")
    logger.info(f"Capital: ${args.capital:,.2f}, Max Positions: {args.max_positions}")
    logger.info("-" * 50)

    # Unified Portfolio Config
    config = UnifiedPortfolioConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        position_size_pct=args.position_size,
        fee_rate=0.0004,
        slippage=0.0001,
        tp_pct=args.tp,
        sl_pct=args.sl,
        leverage=args.leverage,
        exit_priority=ExitPriority.CONSERVATIVE,
        verbose=not args.quiet,
    )

    # Run
    results = run_batch(
        symbols=symbols,
        data_dir=DATA_DIR,
        strategy_name=args.strategy,
        config=config,
        verbose=not args.quiet
    )

    # Summary
    if results:
        print_batch_summary(results)

    # Save
    if args.output:
        save_results_to_csv(results, args.output)
    elif results:
        save_results_to_csv(results, "batch_results.csv")


if __name__ == "__main__":
    main()
