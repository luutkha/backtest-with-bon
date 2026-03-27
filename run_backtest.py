"""
Backtest Runner - Run streak breakout strategy on all symbols
Usage: python run_backtest.py
"""

import os
import glob
from datetime import datetime

from backtest import UnifiedPortfolioBacktest, UnifiedPortfolioConfig
from backtest.signals import streak_breakout_strategy


def get_all_symbols(data_dir: str) -> list:
    """Get all symbols that have both 1h and 5m data"""
    h1_dir = os.path.join(data_dir, "1h")
    m1_dir = os.path.join(data_dir, "5m")

    h1_files = glob.glob(os.path.join(h1_dir, "*.csv"))
    m1_files = glob.glob(os.path.join(m1_dir, "*.csv"))

    h1_symbols = {os.path.splitext(os.path.basename(f))[0] for f in h1_files}
    m1_symbols = {os.path.splitext(os.path.basename(f))[0] for f in m1_files}

    common_symbols = sorted(h1_symbols & m1_symbols)
    return common_symbols


def run_backtest(
    symbols: list,
    initial_capital: float = 100_000,
    max_positions: int = 20,
    position_size_pct: float = 0.05,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    leverage: float = 1.0,
    output_file: str = None,
    data_dir: str = None,
    verbose: bool = False,
) -> dict:
    """
    Run backtest with given parameters

    Args:
        symbols: List of trading symbols
        initial_capital: Starting capital
        max_positions: Maximum simultaneous positions
        position_size_pct: Position size as % of capital
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage
        leverage: Leverage multiplier
        output_file: Path to save trades CSV (optional)
        data_dir: Data directory path
        verbose: Print progress

    Returns:
        Backtest results dict
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    config = UnifiedPortfolioConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        leverage=leverage,
        verbose=verbose,
    )

    engine = UnifiedPortfolioBacktest(
        config=config,
        strategy=streak_breakout_strategy,
        data_dir=data_dir,
    )

    if verbose:
        print(f"Loading {len(symbols)} symbols...")

    engine.load_data(symbols)

    if verbose:
        print("Running backtest...")

    results = engine.run_backtest()

    if output_file:
        engine.export_trades_csv(output_file)
        if verbose:
            print(f"Trades saved to: {output_file}")

    return results


def print_results(results: dict, symbols_count: int, runtime: str = ""):
    """Print backtest results"""
    metrics = results["metrics"]
    trades_df = results["trades_df"]

    print()
    print("=" * 70)
    print("STREAK BREAKOUT STRATEGY - BACKTEST RESULTS")
    print("=" * 70)
    print(f"Symbols: {symbols_count}")
    print(f"Runtime: {runtime}")
    print("-" * 70)

    # Main metrics
    print("METRICS:")
    print(f"  Initial Capital:   ${results.get('config', {}).get('initial_capital', 0):>15,.2f}")
    print(f"  Final Capital:    ${results['final_capital']:>15,.2f}")
    print(f"  Total Return:     {results['metrics']['return_pct']:>15.2f}%")
    print(f"  Total Trades:     {results['metrics']['total_trades']:>15}")
    print(f"  Win Rate:         {results['metrics']['win_rate']:>15.2f}%")
    print(f"  Profit Factor:    {results['metrics']['profit_factor']:>15.3f}")
    print()

    # Risk metrics
    print("RISK METRICS:")
    print(f"  Sharpe Ratio:     {results['metrics']['sharpe_ratio']:>15.3f}")
    print(f"  Sortino Ratio:   {results['metrics']['sortino_ratio']:>15.3f}")
    print(f"  Calmar Ratio:    {results['metrics']['calmar_ratio']:>15.3f}")
    print(f"  Max Drawdown:    {results['metrics']['max_drawdown']:>15.2f}%")
    print(f"  Volatility:      {results['metrics']['volatility']:>15.2f}%")
    print()

    # Trade breakdown
    print("TRADE BREAKDOWN:")
    status_counts = trades_df["status"].value_counts()
    for status, count in status_counts.items():
        pct = count / results["metrics"]["total_trades"] * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()

    # Holdings
    print("HOLD TIME DISTRIBUTION:")
    print(f"  Min hold bars:   {trades_df['hold_bars'].min()}")
    print(f"  Max hold bars:   {trades_df['hold_bars'].max()}")
    print(f"  Avg hold bars:   {trades_df['hold_bars'].mean():.1f}")
    print()
    print("=" * 70)


def main():
    """Main entry point"""
    import time

    # Configuration
    DATA_DIR = r"C:\Personals\Code\backtest-with-bon"
    OUTPUT_FILE = os.path.join(DATA_DIR, "all_trades.csv")

    # Backtest parameters
    INITIAL_CAPITAL = 10000
    MAX_POSITIONS = 20000000
    POSITION_SIZE_PCT = 0.01  # 5% per trade
    TP_PCT = 0.06  # 2% take profit
    SL_PCT = 0.06  # 1% stop loss
    LEVERAGE = 10

    print("=" * 70)
    print("BACKTEST RUNNER")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,}")
    print(f"Max positions: {MAX_POSITIONS}")
    print(f"Position size: {POSITION_SIZE_PCT * 100}%")
    print(f"TP/SL: {TP_PCT * 100}% / {SL_PCT * 100}%")
    print()

    # Get all symbols
    print("Scanning for symbols...")
    symbols = get_all_symbols(DATA_DIR)
    print(f"Found {len(symbols)} symbols with 1h + 5m data")
    print()

    # Run backtest
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = run_backtest(
        symbols=symbols,
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        position_size_pct=POSITION_SIZE_PCT,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        leverage=LEVERAGE,
        output_file=OUTPUT_FILE,
        data_dir=DATA_DIR,
        verbose=True,
    )

    end_time = time.time()
    runtime = f"{(end_time - start_time) / 60:.1f} min"
    print()
    print(f"Backtest completed in {runtime}")

    # Print results
    print_results(results, len(symbols), runtime)

    # Save summary to file
    summary_file = os.path.join(DATA_DIR, "backtest_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Backtest Summary - {start_dt}\n")
        f.write(f"Symbols: {len(symbols)}\n")
        f.write(f"Runtime: {runtime}\n")
        f.write(f"Final Capital: ${results['final_capital']:,.2f}\n")
        f.write(f"Return: {results['metrics']['return_pct']:.2f}%\n")
        f.write(f"Trades: {results['metrics']['total_trades']}\n")
        f.write(f"Win Rate: {results['metrics']['win_rate']:.2f}%\n")
        f.write(f"Sharpe: {results['metrics']['sharpe_ratio']:.3f}\n")
        f.write(f"Max DD: {results['metrics']['max_drawdown']:.2f}%\n")
        f.write(f"Trades CSV: {OUTPUT_FILE}\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
