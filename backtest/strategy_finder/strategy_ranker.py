"""
Strategy Ranking and Output System.

Ranks strategies by multiple criteria and formats output
in various formats (CSV, JSON, TXT).
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Container for a single strategy result"""
    symbol: str
    strategy_name: str
    params: Dict[str, Any]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    expectancy: float
    long_trades: int
    short_trades: int
    consistency_score: float = 0.0
    walk_forward_return: float = 0.0
    statistical_passed: bool = False
    rank_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyRanker:
    """
    Rank and filter strategy results.

    Provides multiple ranking criteria and filtering options.
    """

    def __init__(
        self,
        min_trades: int = 30,
        min_sharpe: float = 1.0,
        min_profit_factor: float = 1.2,
        max_drawdown: float = 30.0
    ):
        """
        Initialize strategy ranker.

        Args:
            min_trades: Minimum trades required
            min_sharpe: Minimum Sharpe ratio
            min_profit_factor: Minimum profit factor
            max_drawdown: Maximum drawdown %
        """
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown

    def filter_significant(
        self,
        results: List[StrategyResult]
    ) -> List[StrategyResult]:
        """Filter strategies that meet minimum criteria"""
        filtered = []

        for r in results:
            if (r.total_trades >= self.min_trades and
                r.sharpe_ratio >= self.min_sharpe and
                r.profit_factor >= self.min_profit_factor and
                r.max_drawdown <= self.max_drawdown):
                filtered.append(r)

        logger.info(f"Filtered {len(results)} -> {len(filtered)} strategies")
        return filtered

    def rank_by_sharpe(
        self,
        results: List[StrategyResult],
        ascending: bool = False
    ) -> List[StrategyResult]:
        """Rank by Sharpe ratio (primary metric)"""
        return sorted(results, key=lambda r: r.sharpe_ratio, reverse=not ascending)

    def rank_by_profit_factor(
        self,
        results: List[StrategyResult],
        ascending: bool = False
    ) -> List[StrategyResult]:
        """Rank by profit factor (secondary metric)"""
        return sorted(results, key=lambda r: r.profit_factor, reverse=not ascending)

    def rank_by_return(
        self,
        results: List[StrategyResult],
        ascending: bool = False
    ) -> List[StrategyResult]:
        """Rank by total return"""
        return sorted(results, key=lambda r: r.return_pct, reverse=not ascending)

    def rank_by_consistency(
        self,
        results: List[StrategyResult],
        ascending: bool = False
    ) -> List[StrategyResult]:
        """Rank by walk-forward consistency score"""
        return sorted(results, key=lambda r: r.consistency_score, reverse=not ascending)

    def rank_by_drawdown(
        self,
        results: List[StrategyResult],
        ascending: bool = True
    ) -> List[StrategyResult]:
        """Rank by drawdown (lower is better)"""
        return sorted(results, key=lambda r: r.max_drawdown, reverse=ascending)

    def composite_rank(
        self,
        results: List[StrategyResult],
        weights: Optional[Dict[str, float]] = None
    ) -> List[StrategyResult]:
        """
        Composite ranking using multiple criteria.

        Args:
            results: List of strategy results
            weights: Dict of metric -> weight (default: equal weights)

        Returns:
            Sorted list by composite score
        """
        if weights is None:
            weights = {
                'sharpe_ratio': 0.4,
                'profit_factor': 0.3,
                'consistency_score': 0.2,
                'return_pct': 0.1,
            }

        for r in results:
            score = 0.0
            score += weights.get('sharpe_ratio', 0) * max(0, r.sharpe_ratio)
            score += weights.get('profit_factor', 0) * max(0, r.profit_factor - 1)  # Normalize PF
            score += weights.get('consistency_score', 0) * r.consistency_score
            score += weights.get('return_pct', 0) * max(0, r.return_pct / 100)  # Normalize return
            r.rank_score = score

        return sorted(results, key=lambda r: r.rank_score, reverse=True)

    def get_top_n(
        self,
        results: List[StrategyResult],
        n: int = 5,
        by: str = 'composite'
    ) -> List[StrategyResult]:
        """
        Get top N strategies.

        Args:
            results: List of strategy results
            n: Number to return
            by: Ranking method ('sharpe', 'pf', 'return', 'composite')

        Returns:
            Top N strategies
        """
        if by == 'sharpe':
            ranked = self.rank_by_sharpe(results)
        elif by == 'pf':
            ranked = self.rank_by_profit_factor(results)
        elif by == 'return':
            ranked = self.rank_by_return(results)
        elif by == 'consistency':
            ranked = self.rank_by_consistency(results)
        elif by == 'drawdown':
            ranked = self.rank_by_drawdown(results)
        else:
            ranked = self.composite_rank(results)

        return ranked[:n]

    def get_top_per_symbol(
        self,
        results: List[StrategyResult],
        n: int = 5
    ) -> Dict[str, List[StrategyResult]]:
        """
        Get top N strategies per symbol.

        Args:
            results: List of strategy results
            n: Number per symbol

        Returns:
            Dict of symbol -> list of top strategies
        """
        # Group by symbol
        by_symbol: Dict[str, List[StrategyResult]] = {}
        for r in results:
            if r.symbol not in by_symbol:
                by_symbol[r.symbol] = []
            by_symbol[r.symbol].append(r)

        # Get top N per symbol
        top_per_symbol = {}
        for symbol, symbol_results in by_symbol.items():
            top_per_symbol[symbol] = self.get_top_n(symbol_results, n=n)

        return top_per_symbol


class OutputFormatter:
    """Format strategy results for output"""

    def __init__(self, ranker: StrategyRanker):
        self.ranker = ranker

    def to_dataframe(
        self,
        results: List[StrategyResult]
    ) -> pd.DataFrame:
        """Convert results to DataFrame"""
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            row = {
                'symbol': r.symbol,
                'strategy': r.strategy_name,
                'params': json.dumps(r.params),
                'trades': r.total_trades,
                'win_rate': f"{r.win_rate:.2f}%",
                'pnl': f"${r.total_pnl:.2f}",
                'return': f"{r.return_pct:.2f}%",
                'sharpe': f"{r.sharpe_ratio:.3f}",
                'pf': f"{r.profit_factor:.3f}",
                'max_dd': f"{r.max_drawdown:.2f}%",
                'consistency': f"{r.consistency_score:.2f}",
                'wf_return': f"{r.walk_forward_return:.2f}%",
                'passed': 'Y' if r.statistical_passed else 'N',
                'rank_score': f"{r.rank_score:.4f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def to_csv(
        self,
        results: List[StrategyResult],
        filepath: str,
        include_params: bool = False
    ) -> None:
        """Save results to CSV"""
        if not results:
            pd.DataFrame().to_csv(filepath, index=False)
            return

        rows = []
        for r in results:
            row = {
                'symbol': r.symbol,
                'strategy': r.strategy_name,
                'total_trades': r.total_trades,
                'winning_trades': r.winning_trades,
                'losing_trades': r.losing_trades,
                'win_rate': r.win_rate,
                'total_pnl': r.total_pnl,
                'return_pct': r.return_pct,
                'max_drawdown': r.max_drawdown,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'profit_factor': r.profit_factor,
                'expectancy': r.expectancy,
                'long_trades': r.long_trades,
                'short_trades': r.short_trades,
                'consistency_score': r.consistency_score,
                'walk_forward_return': r.walk_forward_return,
                'statistical_passed': r.statistical_passed,
                'rank_score': r.rank_score,
            }

            if include_params:
                row['params'] = json.dumps(r.params)

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} results to {filepath}")

    def to_json(
        self,
        results: List[StrategyResult],
        filepath: str
    ) -> None:
        """Save results to JSON"""
        data = [r.to_dict() for r in results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(data)} results to {filepath}")

    def to_summary_txt(
        self,
        results: List[StrategyResult],
        filepath: str,
        top_n: int = 10
    ) -> None:
        """Create human-readable summary file"""
        lines = []
        lines.append("=" * 80)
        lines.append("STRATEGY FINDER RESULTS")
        lines.append("=" * 80)
        lines.append(f"Total strategies found: {len(results)}")
        lines.append("")

        if not results:
            lines.append("No profitable strategies found.")
        else:
            # Top by composite score
            lines.append(f"TOP {top_n} STRATEGIES (by composite score):")
            lines.append("-" * 80)

            top = self.ranker.get_top_n(results, n=top_n, by='composite')

            for i, r in enumerate(top, 1):
                lines.append(f"\n{i}. {r.symbol} | {r.strategy_name}")
                lines.append(f"   Params: {json.dumps(r.params)}")
                lines.append(f"   Trades: {r.total_trades} | Win Rate: {r.win_rate:.2f}%")
                lines.append(f"   Sharpe: {r.sharpe_ratio:.3f} | PF: {r.profit_factor:.3f}")
                lines.append(f"   Return: {r.return_pct:.2f}% | Max DD: {r.max_drawdown:.2f}%")
                lines.append(f"   Consistency: {r.consistency_score:.2f} | WF Return: {r.walk_forward_return:.2f}%")
                lines.append(f"   Rank Score: {r.rank_score:.4f}")

            # Summary by symbol
            lines.append("\n" + "=" * 80)
            lines.append("TOP STRATEGY PER SYMBOL:")
            lines.append("-" * 80)

            top_per_symbol = self.ranker.get_top_per_symbol(results, n=1)
            for symbol, strategies in sorted(top_per_symbol.items()):
                if strategies:
                    r = strategies[0]
                    lines.append(f"{symbol}: {r.strategy_name} (Sharpe: {r.sharpe_ratio:.3f}, PF: {r.profit_factor:.3f})")

        lines.append("\n" + "=" * 80)

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Saved summary to {filepath}")

    def print_summary(
        self,
        results: List[StrategyResult],
        top_n: int = 10
    ) -> None:
        """Print summary to console"""
        if not results:
            print("No profitable strategies found.")
            return

        print(f"\n{'='*80}")
        print(f"STRATEGY FINDER - {len(results)} strategies found")
        print(f"{'='*80}")

        top = self.ranker.get_top_n(results, n=top_n, by='composite')

        print(f"\nTOP {top_n} STRATEGIES:")
        print(f"{'-'*80}")

        for i, r in enumerate(top, 1):
            print(f"{i}. {r.symbol} | {r.strategy_name}")
            print(f"   Trades: {r.total_trades} | Win Rate: {r.win_rate:.2f}% | Return: {r.return_pct:.2f}%")
            print(f"   Sharpe: {r.sharpe_ratio:.3f} | PF: {r.profit_factor:.3f} | Max DD: {r.max_drawdown:.2f}%")
            print(f"   Consistency: {r.consistency_score:.2f} | Rank Score: {r.rank_score:.4f}")
            print()
