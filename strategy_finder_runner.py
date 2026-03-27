#!/usr/bin/env python
"""
Strategy Finder Runner - Automated profitable strategy discovery.

Usage:
    # Find best strategies across all symbols
    python strategy_finder_runner.py --all

    # Run on specific symbols
    python strategy_finder_runner.py --symbols BTCUSDT ETHUSDT

    # Run specific strategies
    python strategy_finder_runner.py --all --strategies RSI MACD

    # Custom filtering
    python strategy_finder_runner.py --all --min-sharpe 1.5 --min-trades 50

    # Output options
    python strategy_finder_runner.py --all --output results.csv --top-k 10
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from itertools import product

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.strategy_finder import (
    SymbolScanConfig,
    get_available_symbols,
    validate_symbol_data,
    StrategyRanker,
    StrategyResult,
    GeneticOptimizer,
    StatisticalFilter,
    get_all_templates,
    get_templates_by_names,
    build_strategy_with_params,
)
from backtest.strategy_finder.statistical_filter import StatisticalFilterConfig
from backtest.strategy_finder.strategy_ranker import OutputFormatter
from backtest.unified_portfolio import UnifiedPortfolioConfig
from backtest.unified_portfolio import UnifiedPortfolioBacktest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_strategy_on_symbol(
    symbol: str,
    strategy_template,
    param_combination: dict,
    config: UnifiedPortfolioConfig,
    data_dir: str
) -> StrategyResult:
    """
    Run a single strategy with given params on a symbol.

    Args:
        symbol: Symbol name
        strategy_template: StrategyTemplate object
        param_combination: Dict of parameters
        config: UnifiedPortfolioConfig
        data_dir: Data directory

    Returns:
        StrategyResult object
    """
    def wrapped_strategy(data):
        return strategy_template.strategy_func(data, **param_combination)

    try:
        engine = UnifiedPortfolioBacktest(
            config=config,
            strategy=wrapped_strategy,
            data_dir=data_dir,
        )
        engine.load_data([symbol])
        result = engine.run_backtest()
        metrics = result.get('metrics', {})

        return StrategyResult(
            symbol=symbol,
            strategy_name=strategy_template.name,
            params=param_combination,
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            losing_trades=metrics.get('losing_trades', 0),
            win_rate=metrics.get('win_rate', 0),
            total_pnl=metrics.get('total_pnl', 0),
            return_pct=metrics.get('return_pct', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            sortino_ratio=metrics.get('sortino_ratio', 0),
            profit_factor=metrics.get('profit_factor', 0),
            expectancy=metrics.get('expectancy', 0),
            long_trades=metrics.get('long_trades', 0),
            short_trades=metrics.get('short_trades', 0),
            statistical_passed=False,  # Will be set later
        )
    except Exception as e:
        logger.debug(f"Failed {symbol}/{strategy_template.name}/{param_combination}: {e}")
        return StrategyResult(
            symbol=symbol,
            strategy_name=strategy_template.name,
            params=param_combination,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            return_pct=0,
            max_drawdown=100,
            sharpe_ratio=-999,
            sortino_ratio=0,
            profit_factor=0,
            expectancy=0,
            long_trades=0,
            short_trades=0,
        )


def grid_search_strategy(
    symbol: str,
    strategy_template,
    param_grid: dict,
    config: UnifiedPortfolioConfig,
    data_dir: str,
    max_combos: int = 1000
) -> list:
    """
    Run grid search for a strategy on a symbol.

    Args:
        symbol: Symbol name
        strategy_template: StrategyTemplate object
        param_grid: Parameter ranges
        config: UnifiedPortfolioConfig
        data_dir: Data directory
        max_combos: Max combinations to test

    Returns:
        List of StrategyResult objects
    """
    results = []

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(product(*param_values))

    if len(combos) > max_combos:
        logger.warning(f"Too many combos ({len(combos)}), sampling {max_combos}")
        import random
        combos = random.sample(combos, max_combos)

    for combo in combos:
        params = dict(zip(param_names, combo))
        result = run_strategy_on_symbol(symbol, strategy_template, params, config, data_dir)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Automated Strategy Finder')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'],
                        help='Symbols to test')
    parser.add_argument('--all', action='store_true',
                        help='Run on all available symbols')
    parser.add_argument('--strategies', nargs='+',
                        choices=['RSI', 'MA_CROSSOVER', 'MACD', 'BOLLINGER_BANDS', 'STREAK_BREAKOUT', 'STOCHASTIC', 'CONSECUTIVE_CANDLE'],
                        help='Strategies to test (default: all)')
    parser.add_argument('--data-dir', default=r'C:\Personals\Code\backtest-with-bon',
                        help='Data directory')
    parser.add_argument('--output', default='strategy_finder_results.csv',
                        help='Output CSV file')
    parser.add_argument('--summary', default='strategy_finder_summary.txt',
                        help='Summary text file')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top K strategies per symbol')
    parser.add_argument('--min-trades', type=int, default=30,
                        help='Minimum trades for filtering')
    parser.add_argument('--min-sharpe', type=float, default=1.0,
                        help='Minimum Sharpe ratio')
    parser.add_argument('--min-pf', type=float, default=1.2,
                        help='Minimum profit factor')
    parser.add_argument('--max-dd', type=float, default=30.0,
                        help='Maximum drawdown %%')
    parser.add_argument('--max-combos', type=int, default=500,
                        help='Max parameter combinations per symbol')
    parser.add_argument('--parallel', type=int, default=4,
                        help='Parallel workers')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce logging')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Get symbols
    if args.all:
        scan_config = SymbolScanConfig(data_dir=args.data_dir)
        symbols = get_available_symbols(scan_config)
        logger.info(f"Found {len(symbols)} symbols")
    else:
        symbols = args.symbols

    if not symbols:
        logger.error("No symbols to test")
        return

    # Get strategy templates
    if args.strategies:
        templates = get_templates_by_names(args.strategies)
    else:
        templates = list(get_all_templates().values())

    logger.info(f"Testing {len(templates)} strategies: {[t.name for t in templates]}")

    # Base config
    base_config = UnifiedPortfolioConfig(
        initial_capital=10000,
        max_positions=5,
        position_size_pct=0.1,
        tp_pct=0.02,
        sl_pct=0.01,
        verbose=False,
    )

    # Statistical filter
    stat_filter = StatisticalFilter(
        StatisticalFilterConfig(
            min_trades=args.min_trades,
            min_sharpe=args.min_sharpe,
            min_profit_factor=args.min_pf,
            max_drawdown=args.max_dd,
        )
    )

    # Ranker
    ranker = StrategyRanker(
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        min_profit_factor=args.min_pf,
        max_drawdown=args.max_dd,
    )

    # Run optimization
    all_results = []
    start_time = time.time()

    total_tasks = len(symbols) * len(templates)
    completed = 0

    logger.info(f"Starting optimization: {len(symbols)} symbols x {len(templates)} strategies")

    for symbol in symbols:
        # Validate symbol data
        validation = validate_symbol_data(args.data_dir, symbol, min_candles=100)
        if not validation['valid']:
            logger.debug(f"Skipping {symbol}: {validation['errors']}")
            continue

        for template in templates:
            # Get parameter grid
            param_grid = template.param_ranges

            # Run grid search
            results = grid_search_strategy(
                symbol=symbol,
                strategy_template=template,
                param_grid=param_grid,
                config=base_config,
                data_dir=args.data_dir,
                max_combos=args.max_combos,
            )

            # Apply filters
            for result in results:
                filter_result = stat_filter.apply_all_filters({
                    'total_trades': result.total_trades,
                    'sharpe_ratio': result.sharpe_ratio,
                    'profit_factor': result.profit_factor,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                })
                result.statistical_passed = filter_result['passed']

            all_results.extend(results)
            completed += 1

            if completed % 10 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_tasks - completed) / 60
                logger.info(f"Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) - ETA: {eta:.1f} min")

    elapsed = time.time() - start_time
    logger.info(f"Optimization complete in {elapsed/60:.1f} minutes")
    logger.info(f"Total results: {len(all_results)}")

    # Filter and rank
    significant = ranker.filter_significant(all_results)
    logger.info(f"Strategies passing filters: {len(significant)}")

    # Rank by composite score
    ranked = ranker.composite_rank(significant)

    # Output
    formatter = OutputFormatter(ranker)
    formatter.to_csv(ranked, args.output, include_params=True)
    formatter.to_summary_txt(ranked, args.summary, top_n=args.top_k)
    formatter.print_summary(ranked, top_n=args.top_k)

    logger.info(f"\nResults saved to:")
    logger.info(f"  CSV: {args.output}")
    logger.info(f"  Summary: {args.summary}")


if __name__ == '__main__':
    main()
