"""
Strategy Finder - Automated strategy discovery and optimization.

This package provides tools to:
- Scan available symbols
- Test multiple strategy patterns per symbol
- Optimize parameters via grid search and genetic algorithms
- Validate with walk-forward analysis
- Filter by statistical significance
- Output ranked strategies
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolScanConfig:
    """Configuration for symbol scanning"""
    data_dir: str
    h1_subdir: str = "1h"
    m5_subdir: str = "5m"
    min_candles: int = 100  # Minimum candles per symbol


def get_available_symbols(config: SymbolScanConfig) -> List[str]:
    """
    Find all symbols that have both 1h and 5m data.

    Args:
        config: SymbolScanConfig with data_dir and subdirectories

    Returns:
        List of symbol names (without .csv extension)
    """
    h1_dir = os.path.join(config.data_dir, config.h1_subdir)
    m5_dir = os.path.join(config.data_dir, config.m5_subdir)

    if not os.path.exists(h1_dir):
        logger.warning(f"1h directory not found: {h1_dir}")
        return []

    if not os.path.exists(m5_dir):
        logger.warning(f"5m directory not found: {m5_dir}")
        return []

    h1_files = glob.glob(os.path.join(h1_dir, "*.csv"))
    m5_files = glob.glob(os.path.join(m5_dir, "*.csv"))

    h1_symbols = {os.path.splitext(os.path.basename(f))[0] for f in h1_files}
    m5_symbols = {os.path.splitext(os.path.basename(f))[0] for f in m5_files}

    common_symbols = sorted(h1_symbols & m5_symbols)
    logger.info(f"Found {len(common_symbols)} symbols with both 1h and 5m data")
    return common_symbols


def validate_symbol_data(data_dir: str, symbol: str, min_candles: int = 100) -> Dict[str, Any]:
    """
    Validate symbol data quality.

    Args:
        data_dir: Data directory
        symbol: Symbol name
        min_candles: Minimum required candles

    Returns:
        Dict with validation results
    """
    h1_path = os.path.join(data_dir, "1h", f"{symbol}.csv")
    m5_path = os.path.join(data_dir, "5m", f"{symbol}.csv")

    result = {
        'symbol': symbol,
        'valid': False,
        'h1_candles': 0,
        'm5_candles': 0,
        'h1_date_range': None,
        'm5_date_range': None,
        'errors': []
    }

    # Check 1h data
    if not os.path.exists(h1_path):
        result['errors'].append("Missing 1h data")
        return result

    try:
        df_h1 = pd.read_csv(h1_path)
        result['h1_candles'] = len(df_h1)

        if len(df_h1) < min_candles:
            result['errors'].append(f"Insufficient 1h candles: {len(df_h1)} < {min_candles}")

        if 'opentime' in df_h1.columns and 'close' in df_h1.columns:
            result['h1_date_range'] = (df_h1['opentime'].min(), df_h1['opentime'].max())
    except Exception as e:
        result['errors'].append(f"Error reading 1h data: {e}")
        return result

    # Check 5m data
    if not os.path.exists(m5_path):
        result['errors'].append("Missing 5m data")
        return result

    try:
        df_m5 = pd.read_csv(m5_path)
        result['m5_candles'] = len(df_m5)

        if len(df_m5) < min_candles:
            result['errors'].append(f"Insufficient 5m candles: {len(df_m5)} < {min_candles}")

        if 'opentime' in df_m5.columns and 'close' in df_m5.columns:
            result['m5_date_range'] = (df_m5['opentime'].min(), df_m5['opentime'].max())
    except Exception as e:
        result['errors'].append(f"Error reading 5m data: {e}")
        return result

    # Check overlapping time range
    if result['h1_date_range'] and result['m5_date_range']:
        h1_start, h1_end = result['h1_date_range']
        m5_start, m5_end = result['m5_date_range']

        # Check if there's any overlap
        if h1_end < m5_start or m5_end < h1_start:
            result['errors'].append("No overlapping time range between 1h and 5m data")

    if not result['errors']:
        result['valid'] = True

    return result


def batch_symbols(symbols: List[str], batch_size: int = 10) -> List[List[str]]:
    """
    Split symbols into batches for parallel processing.

    Args:
        symbols: List of symbol names
        batch_size: Number of symbols per batch

    Returns:
        List of symbol batches
    """
    batches = []
    for i in range(0, len(symbols), batch_size):
        batches.append(symbols[i:i + batch_size])
    return batches


def run_single_symbol_backtest(
    symbol: str,
    strategy_func: Callable,
    params: Dict[str, Any],
    config: 'UnifiedPortfolioConfig',
    data_dir: str
) -> Dict[str, Any]:
    """
    Run backtest for a single symbol with given strategy and params.

    Args:
        symbol: Symbol name
        strategy_func: Strategy function
        params: Strategy parameters
        config: UnifiedPortfolioConfig
        data_dir: Data directory

    Returns:
        Dict with symbol, params, and metrics
    """
    from backtest.unified_portfolio import UnifiedPortfolioBacktest

    # Create strategy wrapper that applies params
    def wrapped_strategy(data):
        return strategy_func(data, **params)

    try:
        engine = UnifiedPortfolioBacktest(
            config=config,
            strategy=wrapped_strategy,
            data_dir=data_dir,
        )
        engine.load_data([symbol])
        result = engine.run_backtest()

        metrics = result.get('metrics', {})

        return {
            'symbol': symbol,
            'params': params,
            'total_trades': metrics.get('total_trades', 0),
            'winning_trades': metrics.get('winning_trades', 0),
            'losing_trades': metrics.get('losing_trades', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_pnl': metrics.get('total_pnl', 0),
            'return_pct': metrics.get('return_pct', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'sortino_ratio': metrics.get('sortino_ratio', 0),
            'calmar_ratio': metrics.get('calmar_ratio', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'expectancy': metrics.get('expectancy', 0),
            'avg_trade_duration': metrics.get('avg_trade_duration', 0),
            'long_trades': metrics.get('long_trades', 0),
            'short_trades': metrics.get('short_trades', 0),
            'total_fees': metrics.get('total_fees', 0),
            'final_capital': result.get('final_capital', config.initial_capital),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'params': params,
            'success': False,
            'error': str(e)
        }


def parallel_backtest(
    symbols: List[str],
    strategy_func: Callable,
    param_combinations: List[Dict[str, Any]],
    config: 'UnifiedPortfolioConfig',
    data_dir: str,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Run backtests in parallel across symbols and parameter combinations.

    Args:
        symbols: List of symbols
        strategy_func: Strategy function
        param_combinations: List of parameter dicts to test
        config: UnifiedPortfolioConfig
        data_dir: Data directory
        max_workers: Max parallel workers

    Returns:
        List of result dicts
    """
    results = []

    # Create all (symbol, params) pairs
    tasks = [(symbol, params) for symbol in symbols for params in param_combinations]

    logger.info(f"Running {len(tasks)} backtests with {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_symbol_backtest, symbol, strategy_func, params, config, data_dir): (symbol, params)
            for symbol, params in tasks
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                symbol, params = futures[future]
                results.append({
                    'symbol': symbol,
                    'params': params,
                    'success': False,
                    'error': str(e)
                })

    return results


# Export main classes and functions
from .strategies import (
    StrategyTemplate,
    RSI_TEMPLATE,
    MA_CROSSOVER_TEMPLATE,
    MACD_TEMPLATE,
    BOLLINGER_BANDS_TEMPLATE,
    STREAK_BREAKOUT_TEMPLATE,
    STOCHASTIC_TEMPLATE,
    get_all_templates,
    get_templates_by_names,
    build_strategy_with_params
)

from .genetic_optimizer import GeneticOptimizer

from .statistical_filter import StatisticalFilter

from .strategy_ranker import StrategyRanker, StrategyResult

__all__ = [
    'SymbolScanConfig',
    'get_available_symbols',
    'validate_symbol_data',
    'batch_symbols',
    'run_single_symbol_backtest',
    'parallel_backtest',
    'StrategyTemplate',
    'RSI_TEMPLATE',
    'MA_CROSSOVER_TEMPLATE',
    'MACD_TEMPLATE',
    'BOLLINGER_BANDS_TEMPLATE',
    'STREAK_BREAKOUT_TEMPLATE',
    'STOCHASTIC_TEMPLATE',
    'get_all_templates',
    'get_templates_by_names',
    'build_strategy_with_params',
    'GeneticOptimizer',
    'StatisticalFilter',
    'StrategyRanker',
    'StrategyResult',
]
