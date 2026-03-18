"""
Batch Backtest Runner
Run backtest on multiple symbols at once
"""

import os
import pandas as pd
from typing import List, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import BacktestEngine, BacktestConfig
from backtest.signals import rsi_strategy, moving_average_crossover_strategy, macd_strategy
from backtest.execution import ExitPriority

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Strategy map
STRATEGIES = {
    'rsi': rsi_strategy,
    'ma': moving_average_crossover_strategy,
    'macd': macd_strategy,
}


def run_backtest(
    symbol: str,
    data_dir: str,
    strategy: Callable,
    base_config: BacktestConfig
) -> Dict[str, Any]:
    """Run backtest for a single symbol"""
    try:
        # Create new config for this symbol
        import copy
        config = copy.copy(base_config)
        config.data_dir = data_dir
        config.symbol = symbol

        engine = BacktestEngine(
            config=config,
            strategy=strategy
        )
        results = engine.run_backtest()
        metrics = results['metrics']
        return {
            'symbol': symbol,
            'total_trades': metrics.get('total_trades', 0),
            'winning_trades': metrics.get('winning_trades', 0),
            'losing_trades': metrics.get('losing_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_pnl': metrics.get('total_pnl', 0),
            'return_pct': metrics.get('return_pct', 0),
            'final_capital': results['final_capital'],
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0),
            'avg_holding_bars': metrics.get('avg_holding_bars', 0),
            'total_fees': metrics.get('total_fees', 0),
            'long_trades': metrics.get('long_trades', 0),
            'long_win_rate': metrics.get('long_win_rate', 0),
            'short_trades': metrics.get('short_trades', 0),
            'short_win_rate': metrics.get('short_win_rate', 0),
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


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
    config: BacktestConfig,
    parallel: bool = True,
    max_workers: int = 4,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run backtest on multiple symbols"""

    strategy = STRATEGIES.get(strategy_name)
    if not strategy:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    results = []

    # Store original config values
    orig_data_dir = config.data_dir
    orig_symbol = config.symbol

    if parallel and len(symbols) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_backtest, symbol, data_dir, strategy, config): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if 'error' in result:
                        logger.warning(f"  {symbol}: ERROR - {result['error']}")
                    elif verbose:
                        logger.info(
                            f"  {symbol}: {result['total_trades']} trades, "
                            f"Return: {result['return_pct']:.2f}%, "
                            f"Win: {result['win_rate']:.1f}%"
                        )
                except Exception as e:
                    logger.warning(f"  {symbol}: EXCEPTION - {e}")
    else:
        for symbol in symbols:
            result = run_backtest(symbol, data_dir, strategy, config)
            results.append(result)
            if 'error' in result:
                logger.warning(f"  {symbol}: ERROR - {result['error']}")
            elif verbose:
                logger.info(
                    f"  {symbol}: {result['total_trades']} trades, "
                    f"Return: {result['return_pct']:.2f}%, "
                    f"Win: {result['win_rate']:.1f}%"
                )

    return results


def print_batch_summary(results: List[Dict[str, Any]]):
    """Print aggregated batch summary"""
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r and r.get('total_trades', 0) > 0]

    if not valid_results:
        logger.info("No valid results to summarize")
        return

    total_trades = sum(r['total_trades'] for r in valid_results)
    total_pnl = sum(r['total_pnl'] for r in valid_results)
    total_fees = sum(r['total_fees'] for r in valid_results)
    profitable_symbols = sum(1 for r in valid_results if r['total_pnl'] > 0)
    losing_symbols = sum(1 for r in valid_results if r['total_pnl'] <= 0)

    all_max_dd = [r['max_drawdown'] for r in valid_results]
    all_pf = [r['profit_factor'] for r in valid_results if r['profit_factor'] > 0]

    # Best and worst
    best = max(valid_results, key=lambda x: x['total_pnl']) if valid_results else None
    worst = min(valid_results, key=lambda x: x['total_pnl']) if valid_results else None

    logger.info("=" * 70)
    logger.info("BATCH BACKTEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Symbols Tested: {len(valid_results)}")
    logger.info(f"Profitable: {profitable_symbols} ({profitable_symbols/len(valid_results)*100:.1f}%)")
    logger.info(f"Losing: {losing_symbols} ({losing_symbols/len(valid_results)*100:.1f}%)")
    logger.info("-" * 70)
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Total PnL: ${total_pnl:,.2f}")
    logger.info(f"Total Fees: ${total_fees:,.2f}")
    if valid_results:
        logger.info(f"Avg PnL per Symbol: ${total_pnl/len(valid_results):,.2f}")
    if best:
        logger.info(f"Best: {best['symbol']} = ${best['total_pnl']:.2f}")
    if worst:
        logger.info(f"Worst: {worst['symbol']} = ${worst['total_pnl']:.2f}")
    if all_max_dd:
        logger.info(f"Avg Max DD: {sum(all_max_dd)/len(all_max_dd):.2f}%")
    if all_pf:
        logger.info(f"Avg Profit Factor: {sum(all_pf)/len(all_pf):.2f}")
    logger.info("=" * 70)

    # Top 5
    logger.info("\n=== TOP 5 ===")
    for i, r in enumerate(sorted(valid_results, key=lambda x: x['total_pnl'], reverse=True)[:5], 1):
        logger.info(f"  {i}. {r['symbol']}: ${r['total_pnl']:.2f} ({r['return_pct']:.1f}%)")

    logger.info("\n=== BOTTOM 5 ===")
    for i, r in enumerate(sorted(valid_results, key=lambda x: x['total_pnl'])[:5], 1):
        logger.info(f"  {i}. {r['symbol']}: ${r['total_pnl']:.2f} ({r['return_pct']:.1f}%)")


def save_results_to_csv(results: List[Dict[str, Any]], filename: str = "batch_results.csv"):
    """Save batch results to CSV"""
    clean_results = [r for r in results if 'error' not in r]
    if clean_results:
        df = pd.DataFrame(clean_results)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Batch backtest runner')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to test (default: BTCUSDT, ETHUSDT, BNBUSDT)')
    parser.add_argument('--all', action='store_true',
                        help='Run on all available symbols')
    parser.add_argument('--strategy', default='rsi',
                        choices=['rsi', 'ma', 'macd'],
                        help='Strategy to use')
    parser.add_argument('--data-dir', default=r'C:\Personals\Code\backtest-with-bon',
                        help='Data directory')
    parser.add_argument('--tp', type=float, default=0.02,
                        help='Take profit %')
    parser.add_argument('--sl', type=float, default=0.01,
                        help='Stop loss %')
    parser.add_argument('--leverage', type=float, default=1.0)
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file')
    parser.add_argument('--parallel', action='store_true', default=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress individual symbol output')

    args = parser.parse_args()

    DATA_DIR = args.data_dir

    # Get symbols
    if args.all:
        symbols = get_available_symbols(DATA_DIR)
        logger.info(f"Found {len(symbols)} symbols")
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    logger.info(f"Running {args.strategy} strategy on {len(symbols)} symbols")
    logger.info("-" * 50)

    # Config - enable verbose unless quiet mode, skip validation for speed
    config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=0.0004,
        slippage=0.0001,
        exit_priority=ExitPriority.CONSERVATIVE,
        tp_pct=args.tp,
        sl_pct=args.sl,
        leverage=args.leverage,
        position_size_pct=0.95,
        verbose=not args.quiet,
        skip_validation=args.quiet,  # Skip validation in quiet mode for speed
    )

    # Run
    results = run_batch(
        symbols=symbols,
        data_dir=DATA_DIR,
        strategy_name=args.strategy,
        config=config,
        parallel=args.parallel,
        max_workers=args.workers,
        verbose=not args.quiet
    )

    # Summary
    print_batch_summary(results)

    # Save
    if args.output:
        save_results_to_csv(results, args.output)
    else:
        save_results_to_csv(results, "batch_results.csv")


if __name__ == "__main__":
    main()
