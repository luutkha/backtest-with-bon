"""
Batch Backtest Runner
Run backtest on multiple symbols at once
"""

import os
import pandas as pd
from typing import List, Callable
from backtest_engine import BacktestEngine, BacktestConfig, ExitPriority
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest(
    symbol: str,
    data_dir: str,
    strategy: Callable,
    config: BacktestConfig
) -> dict:
    """Run backtest for a single symbol"""
    try:
        engine = BacktestEngine(
            config=config,
            data_dir=data_dir,
            symbol=symbol,
            strategy=strategy
        )
        engine.run()
        return engine.get_results()
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


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
    strategy: Callable,
    config: BacktestConfig,
    parallel: bool = True,
    max_workers: int = 4
) -> List[dict]:
    """Run backtest on multiple symbols"""
    results = []

    if parallel:
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
                    logger.info(f"Completed: {symbol}")
                except Exception as e:
                    logger.error(f"Failed: {symbol} - {e}")
    else:
        for symbol in symbols:
            result = run_backtest(symbol, data_dir, strategy, config)
            results.append(result)

    return results


def print_batch_summary(results: List[dict]):
    """Print aggregated batch summary"""
    # Filter out errors
    valid_results = [r for r in results if "error" not in r and r.get("total_trades", 0) > 0]

    if not valid_results:
        logger.info("No valid results to summarize")
        return

    total_trades = sum(r["total_trades"] for r in valid_results)
    total_pnl = sum(r["total_pnl"] for r in valid_results)
    total_fees = sum(r["total_fees"] for r in valid_results)
    profitable_symbols = sum(1 for r in valid_results if r["total_pnl"] > 0)
    losing_symbols = sum(1 for r in valid_results if r["total_pnl"] <= 0)

    # Calculate overall metrics
    all_pnls = [r["total_pnl"] for r in valid_results]
    all_max_dd = [r["max_drawdown"] for r in valid_results]
    all_pf = [r["profit_factor"] for r in valid_results if r["profit_factor"] > 0]
    all_rr = [r["risk_reward"] for r in valid_results if r["risk_reward"] > 0]

    # Win rates
    long_wins = sum(r["long_wins"] for r in valid_results)
    long_trades = sum(r["long_trades"] for r in valid_results)
    short_wins = sum(r["short_wins"] for r in valid_results)
    short_trades = sum(r["short_trades"] for r in valid_results)

    long_win_rate = long_wins/long_trades*100 if long_trades > 0 else 0
    short_win_rate = short_wins/short_trades*100 if short_trades > 0 else 0

    # Long/Short RR
    long_rr_values = [r["long_rr"] for r in valid_results if r["long_rr"] > 0]
    short_rr_values = [r["short_rr"] for r in valid_results if r["short_rr"] > 0]

    # Best and worst symbols
    best_symbol = max(valid_results, key=lambda x: x['total_pnl']) if valid_results else None
    worst_symbol = min(valid_results, key=lambda x: x['total_pnl']) if valid_results else None

    logger.info("=" * 80)
    logger.info("BATCH BACKTEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Symbols Tested: {len(valid_results)}")
    logger.info(f"Profitable Symbols: {profitable_symbols} ({profitable_symbols/len(valid_results)*100:.1f}%)")
    logger.info(f"Losing Symbols: {losing_symbols} ({losing_symbols/len(valid_results)*100:.1f}%)")
    logger.info("-" * 80)
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Total PnL: {total_pnl:.2f}")
    logger.info(f"Total Fees: {total_fees:.2f}")
    logger.info("-" * 80)
    if all_pnls:
        logger.info(f"Avg PnL per Symbol: {total_pnl/len(valid_results):.2f}")
        if best_symbol:
            logger.info(f"Best PnL: {best_symbol['total_pnl']:.2f} ({best_symbol['symbol']})")
        if worst_symbol:
            logger.info(f"Worst PnL: {worst_symbol['total_pnl']:.2f} ({worst_symbol['symbol']})")
    if all_max_dd:
        logger.info(f"Avg Max DD: {sum(all_max_dd)/len(all_max_dd):.2f}%")
    if all_pf:
        logger.info(f"Avg Profit Factor: {sum(all_pf)/len(all_pf):.2f}")
    if all_rr:
        logger.info(f"Avg Risk Reward: {sum(all_rr)/len(all_rr):.2f}")
    logger.info("-" * 80)
    logger.info(f"Long Win Rate: {long_win_rate:.2f}% ({long_wins}/{long_trades})")
    logger.info(f"Short Win Rate: {short_win_rate:.2f}% ({short_wins}/{short_trades})")
    if long_rr_values:
        logger.info(f"Long RR: {sum(long_rr_values)/len(long_rr_values):.2f}")
    if short_rr_values:
        logger.info(f"Short RR: {sum(short_rr_values)/len(short_rr_values):.2f}")
    logger.info("=" * 80)

    # Print top 5 best and worst
    logger.info("\n=== TOP 5 BEST ===")
    sorted_by_pnl = sorted(valid_results, key=lambda x: x['total_pnl'], reverse=True)[:5]
    for i, r in enumerate(sorted_by_pnl, 1):
        logger.info(f"{i}. {r['symbol']}: PnL={r['total_pnl']:.2f}, Trades={r['total_trades']}, WR={r['win_rate']:.1f}%")

    logger.info("\n=== TOP 5 WORST ===")
    sorted_by_pnl_worst = sorted(valid_results, key=lambda x: x['total_pnl'])[:5]
    for i, r in enumerate(sorted_by_pnl_worst, 1):
        logger.info(f"{i}. {r['symbol']}: PnL={r['total_pnl']:.2f}, Trades={r['total_trades']}, WR={r['win_rate']:.1f}%")

    return {
        "total_symbols": len(valid_results),
        "profitable_symbols": profitable_symbols,
        "losing_symbols": losing_symbols,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "avg_pnl_per_symbol": total_pnl / len(valid_results),
    }


def save_results_to_csv(results: List[dict], filename: str = "batch_results.csv"):
    """Save batch results to CSV"""
    # Filter out errors and trades
    clean_results = []
    for r in results:
        if "error" not in r:
            clean_r = {k: v for k, v in r.items() if k != "trades"}
            clean_results.append(clean_r)

    if clean_results:
        df = pd.DataFrame(clean_results)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")


if __name__ == "__main__":
    from sample_strategy import sample_strategy

    DATA_DIR = "C:\\Personals\\Code\\backtest-with-bon"

    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.0004,
        slippage=0.0001,
        exit_priority=ExitPriority.CONSERVATIVE,
        tp_pct=0.02,
        sl_pct=0.01,
        position_size_pct=0.95,
        leverage=1.0,
    )

    # Get all available symbols
    all_symbols = get_available_symbols(DATA_DIR)
    print(f"Found {len(all_symbols)} symbols")

    # Test on just 3 symbols for quick test
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    # Or test specific symbols
    # symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print(f"Testing {len(symbols)} symbols: {symbols}")

    # Run batch backtest
    results = run_batch(
        symbols=symbols,
        data_dir=DATA_DIR,
        strategy=sample_strategy,
        config=config,
        parallel=True,
        max_workers=4
    )

    # Print summary
    print_batch_summary(results)

    # Save to CSV
    save_results_to_csv(results, "batch_results.csv")
