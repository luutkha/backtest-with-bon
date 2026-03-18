"""
Unified Portfolio Batch Runner

Runs backtest on multiple symbols as a single unified portfolio:
- Single capital pool shared across all symbols
- Only one position per symbol at a time
- Realistic portfolio management
"""

import os
import argparse
import logging
from typing import List

from backtest import BacktestConfig
from backtest.unified_portfolio import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from backtest.signals import rsi_strategy, moving_average_crossover_strategy, macd_strategy
from backtest.execution import ExitPriority

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

STRATEGIES = {
    'rsi': rsi_strategy,
    'ma': moving_average_crossover_strategy,
    'macd': macd_strategy,
}


def get_available_symbols(data_dir: str, timeframe: str = "1h") -> List[str]:
    """Get list of available symbols from data directory"""
    path = os.path.join(data_dir, timeframe)
    if not os.path.exists(path):
        return []
    files = [f.replace(".csv", "") for f in os.listdir(path) if f.endswith(".csv")]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Unified portfolio backtest')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to test')
    parser.add_argument('--all', action='store_true',
                        help='Run on all available symbols')
    parser.add_argument('--strategy', default='rsi',
                        choices=['rsi', 'ma', 'macd'],
                        help='Strategy to use')
    parser.add_argument('--data-dir', default=r'C:\Personals\Code\backtest-with-bon',
                        help='Data directory')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital for portfolio')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Maximum simultaneous positions')
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as % of available capital (default: 0.1 = 10%)')
    parser.add_argument('--tp', type=float, default=0.02,
                        help='Take profit %')
    parser.add_argument('--sl', type=float, default=0.01,
                        help='Stop loss %')
    parser.add_argument('--leverage', type=float, default=1.0)
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

    # Get strategy
    strategy = STRATEGIES.get(args.strategy)
    if not strategy:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Create unified portfolio config
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

    # Create and run backtest
    engine = UnifiedPortfolioBacktest(
        config=config,
        strategy=strategy,
        data_dir=DATA_DIR,
    )

    # Load data
    engine.load_data(symbols)

    # Run backtest
    results = engine.run_backtest()

    # Print summary
    engine.print_summary()

    logger.info(f"\nResults: {results['metrics']}")


if __name__ == "__main__":
    main()
