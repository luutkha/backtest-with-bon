"""
Strategy Optimizer - Find the best strategy across all symbols
Tests multiple strategies with various parameters to identify optimal configuration.
"""

import os
import glob
import time
import json
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np

from backtest import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from backtest.signals import (
    rsi_strategy,
    moving_average_crossover_strategy,
    macd_strategy,
    streak_breakout_strategy,
    ma_streak_strategy,
)


def get_all_symbols(data_dir: str) -> list:
    """Get all symbols that have both 1h and 5m data"""
    h1_dir = os.path.join(data_dir, "1h")
    m5_dir = os.path.join(data_dir, "5m")

    h1_files = glob.glob(os.path.join(h1_dir, "*.csv"))
    m5_files = glob.glob(os.path.join(m5_dir, "*.csv"))

    h1_symbols = {os.path.splitext(os.path.basename(f))[0] for f in h1_files}
    m5_symbols = {os.path.splitext(os.path.basename(f))[0] for f in m5_files}

    common_symbols = sorted(h1_symbols & m5_symbols)
    return common_symbols


def run_strategy_backtest(
    symbols: list,
    strategy_func,
    strategy_name: str,
    config_params: dict,
    data_dir: str,
    capital: float = 100_000,
    max_positions: int = 20,
    position_size_pct: float = 0.1,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    leverage: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Run backtest for a specific strategy configuration"""

    config = UnifiedPortfolioConfig(
        initial_capital=capital,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        leverage=leverage,
        verbose=verbose,
    )

    try:
        engine = UnifiedPortfolioBacktest(
            config=config,
            strategy=strategy_func,
            data_dir=data_dir,
        )
        engine.load_data(symbols)
        results = engine.run_backtest()

        return {
            'strategy': strategy_name,
            'params': config_params,
            'metrics': results['metrics'],
            'final_capital': results['final_capital'],
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'strategy': strategy_name,
            'params': config_params,
            'metrics': None,
            'final_capital': 0,
            'success': False,
            'error': str(e)
        }


def calculate_score(metrics: dict) -> float:
    """
    Calculate composite score for strategy comparison.
    Higher is better. Weighs multiple factors:
    - Sharpe ratio (most important)
    - Win rate
    - Return
    - Max drawdown (penalized)
    - Profit factor
    """
    if metrics is None:
        return -999999

    sharpe = metrics.get('sharpe_ratio', 0) or 0
    win_rate = metrics.get('win_rate', 0) or 0
    ret_pct = metrics.get('return_pct', 0) or 0
    max_dd = metrics.get('max_drawdown', 100) or 100
    profit_factor = metrics.get('profit_factor', 0) or 0
    total_trades = metrics.get('total_trades', 0) or 0

    # Need minimum trades for statistical significance
    if total_trades < 30:
        return -999999

    # Normalize and weight
    # Sharpe: most important (weight 40%)
    sharpe_score = sharpe * 40

    # Win rate: weight 20%
    win_rate_score = (win_rate - 50) * 2 * 20  # +20 for >50%, -20 for <50%

    # Return: weight 15%
    ret_score = min(ret_pct, 100) * 0.15

    # Max drawdown: penalty weight 15% (lower is better)
    dd_penalty = max(0, (max_dd - 20) / 20) * 15  # Start penalizing above 20%

    # Profit factor: weight 10%
    pf_score = min(profit_factor, 5) * 2 * 10  # Cap at 5

    total_score = sharpe_score + win_rate_score + ret_score - dd_penalty + pf_score

    return total_score


def main():
    DATA_DIR = r"C:\Personals\Code\backtest-with-bon"

    # Get all symbols
    print("Scanning for symbols...")
    symbols = get_all_symbols(DATA_DIR)
    print(f"Found {len(symbols)} symbols with 1h + 5m data")

    # Strategy configurations to test
    strategy_configs = [
        # RSI Strategy variations
        {
            'name': 'RSI_14_30_70',
            'func': rsi_strategy,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
        },
        {
            'name': 'RSI_7_25_75',
            'func': rsi_strategy,
            'params': {'rsi_period': 7, 'oversold': 25, 'overbought': 75}
        },
        {
            'name': 'RSI_21_35_65',
            'func': rsi_strategy,
            'params': {'rsi_period': 21, 'oversold': 35, 'overbought': 65}
        },

        # MA Crossover variations
        {
            'name': 'MA_20_50',
            'func': moving_average_crossover_strategy,
            'params': {'fast_period': 20, 'slow_period': 50}
        },
        {
            'name': 'MA_10_30',
            'func': moving_average_crossover_strategy,
            'params': {'fast_period': 10, 'slow_period': 30}
        },
        {
            'name': 'MA_50_200',
            'func': moving_average_crossover_strategy,
            'params': {'fast_period': 50, 'slow_period': 200}
        },

        # MACD variations
        {
            'name': 'MACD_12_26_9',
            'func': macd_strategy,
            'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        },
        {
            'name': 'MACD_8_17_9',
            'func': macd_strategy,
            'params': {'fast_period': 8, 'slow_period': 17, 'signal_period': 9}
        },
        {
            'name': 'MACD_19_39_9',
            'func': macd_strategy,
            'params': {'fast_period': 19, 'slow_period': 39, 'signal_period': 9}
        },

        # Streak Breakout variations
        {
            'name': 'Streak_3_1_6',
            'func': streak_breakout_strategy,
            'params': {'consecutive_candles': 3, 'atr_window_min': 1.0, 'atr_window_max': 6.0, 'risk_reward_ratio_sl': 0.5, 'risk_reward_ratio_tp': 1.5}
        },
        {
            'name': 'Streak_4_1_6',
            'func': streak_breakout_strategy,
            'params': {'consecutive_candles': 4, 'atr_window_min': 1.0, 'atr_window_max': 6.0, 'risk_reward_ratio_sl': 0.5, 'risk_reward_ratio_tp': 1.5}
        },
        {
            'name': 'Streak_5_1_6',
            'func': streak_breakout_strategy,
            'params': {'consecutive_candles': 5, 'atr_window_min': 1.0, 'atr_window_max': 6.0, 'risk_reward_ratio_sl': 0.5, 'risk_reward_ratio_tp': 1.5}
        },
        {
            'name': 'Streak_4_1_12',
            'func': streak_breakout_strategy,
            'params': {'consecutive_candles': 4, 'atr_window_min': 1.0, 'atr_window_max': 12.0, 'risk_reward_ratio_sl': 0.5, 'risk_reward_ratio_tp': 1.5}
        },

        # MA Streak variations
        {
            'name': 'MAStreak_4_50',
            'func': ma_streak_strategy,
            'params': {'consecutive_candles': 4, 'ma_period': 50}
        },
        {
            'name': 'MAStreak_4_200',
            'func': ma_streak_strategy,
            'params': {'consecutive_candles': 4, 'ma_period': 200}
        },
    ]

    # TP/SL configurations to test
    tp_sl_configs = [
        {'tp': 0.02, 'sl': 0.01, 'name': 'TP2_SL1'},
        {'tp': 0.03, 'sl': 0.015, 'name': 'TP3_SL1.5'},
        {'tp': 0.05, 'sl': 0.025, 'name': 'TP5_SL2.5'},
        {'tp': 0.06, 'sl': 0.03, 'name': 'TP6_SL3'},
        {'tp': 0.10, 'sl': 0.05, 'name': 'TP10_SL5'},
    ]

    results = []
    total_tests = len(strategy_configs) * len(tp_sl_configs)
    test_num = 0

    print(f"\nRunning {total_tests} strategy configurations...")
    print("=" * 70)

    start_time = time.time()

    for strat_cfg in strategy_configs:
        for tp_sl in tp_sl_configs:
            test_num += 1
            elapsed = time.time() - start_time
            eta = (elapsed / test_num) * (total_tests - test_num) if test_num > 0 else 0

            print(f"[{test_num}/{total_tests}] {strat_cfg['name']} + {tp_sl['name']} | ETA: {eta/60:.1f}min", end="")

            result = run_strategy_backtest(
                symbols=symbols,
                strategy_func=strat_cfg['func'],
                strategy_name=f"{strat_cfg['name']}_{tp_sl['name']}",
                config_params={**strat_cfg['params'], **tp_sl},
                data_dir=DATA_DIR,
                capital=100_000,
                max_positions=20,
                position_size_pct=0.1,
                tp_pct=tp_sl['tp'],
                sl_pct=tp_sl['sl'],
                leverage=1.0,
                verbose=False,
            )

            if result['success']:
                result['score'] = calculate_score(result['metrics'])
                print(f" | Score: {result['score']:.1f} | Sharpe: {result['metrics']['sharpe_ratio']:.2f} | Return: {result['metrics']['return_pct']:.1f}% | Win%: {result['metrics']['win_rate']:.1f}% | DD: {result['metrics']['max_drawdown']:.1f}%")
            else:
                result['score'] = -999999
                print(f" | FAILED: {result['error'][:50]}")

            results.append(result)

    total_time = (time.time() - start_time) / 60

    # Sort by score
    results_df = pd.DataFrame([{
        'strategy': r['strategy'],
        'score': r['score'],
        'final_capital': r['final_capital'],
        'total_trades': r['metrics']['total_trades'] if r['metrics'] else 0,
        'win_rate': r['metrics']['win_rate'] if r['metrics'] else 0,
        'return_pct': r['metrics']['return_pct'] if r['metrics'] else 0,
        'sharpe_ratio': r['metrics']['sharpe_ratio'] if r['metrics'] else 0,
        'max_drawdown': r['metrics']['max_drawdown'] if r['metrics'] else 0,
        'profit_factor': r['metrics']['profit_factor'] if r['metrics'] else 0,
        'sortino_ratio': r['metrics']['sortino_ratio'] if r['metrics'] else 0,
        'calmar_ratio': r['metrics']['calmar_ratio'] if r['metrics'] else 0,
        'total_fees': r['metrics']['total_fees'] if r['metrics'] else 0,
    } for r in results])

    results_df = results_df.sort_values('score', ascending=False)

    print("\n" + "=" * 70)
    print("TOP 10 STRATEGIES")
    print("=" * 70)
    print(results_df.head(10).to_string(index=False))

    # Best strategy details
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST STRATEGY DETAILS")
    print("=" * 70)
    print(f"Strategy: {best['strategy']}")
    print(f"Score: {best['score']:.2f}")
    print(f"Final Capital: ${best['final_capital']:,.2f}")
    print(f"Total Return: {best['return_pct']:.2f}%")
    print(f"Total Trades: {int(best['total_trades'])}")
    print(f"Win Rate: {best['win_rate']:.2f}%")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {best['sortino_ratio']:.3f}")
    print(f"Calmar Ratio: {best['calmar_ratio']:.3f}")
    print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
    print(f"Profit Factor: {best['profit_factor']:.3f}")
    print(f"Total Fees: ${best['total_fees']:,.2f}")

    # Save results
    output_file = os.path.join(DATA_DIR, "strategy_comparison.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save summary
    summary_file = os.path.join(DATA_DIR, "best_strategy_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("STRATEGY ANALYSIS RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbols tested: {len(symbols)}\n")
        f.write(f"Configurations tested: {total_tests}\n")
        f.write(f"Total runtime: {total_time:.1f} minutes\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("TOP 5 STRATEGIES:\n")
        f.write("=" * 50 + "\n")
        for i, row in results_df.head(5).iterrows():
            f.write(f"\n{i+1}. {row['strategy']}\n")
            f.write(f"   Score: {row['score']:.2f} | Return: {row['return_pct']:.2f}%\n")
            f.write(f"   Sharpe: {row['sharpe_ratio']:.3f} | Win Rate: {row['win_rate']:.2f}%\n")
            f.write(f"   Max DD: {row['max_drawdown']:.2f}% | Profit Factor: {row['profit_factor']:.3f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("BEST STRATEGY:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Strategy: {best['strategy']}\n")
        f.write(f"Score: {best['score']:.2f}\n")
        f.write(f"Final Capital: ${best['final_capital']:,.2f}\n")
        f.write(f"Total Return: {best['return_pct']:.2f}%\n")
        f.write(f"Sharpe Ratio: {best['sharpe_ratio']:.3f}\n")
        f.write(f"Max Drawdown: {best['max_drawdown']:.2f}%\n")

    print(f"Summary saved to: {summary_file}")

    return results_df


if __name__ == "__main__":
    main()