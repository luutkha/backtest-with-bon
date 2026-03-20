"""
Abstract base class for backtest engines.

Provides common interface and shared functionality for all backtest implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseBacktestEngine(ABC):
    """
    Abstract base class for all backtest engines.

    Subclasses must implement:
    - load_data(symbols)
    - run_backtest()
    - print_summary()

    Provides common methods:
    - validate_config()
    - get_results()
    """

    def __init__(self, config: Any, strategy: Any, data_dir: str):
        """
        Initialize the base backtest engine.

        Args:
            config: Configuration object
            strategy: Strategy function
            data_dir: Data directory path
        """
        self.config = config
        self.strategy = strategy
        self.data_dir = data_dir

        # Results placeholders
        self.trades: List = []
        self.equity_df: Optional[Any] = None
        self.metrics: Dict[str, Any] = {}
        self.final_capital: float = 0.0

    @abstractmethod
    def load_data(self, symbols: List[str]) -> None:
        """
        Load market data for given symbols.

        Args:
            symbols: List of trading symbols
        """
        pass

    @abstractmethod
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest.

        Returns:
            Dictionary containing backtest results
        """
        pass

    @abstractmethod
    def print_summary(self) -> None:
        """
        Print backtest summary to console.
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the backtest configuration.

        Returns:
            True if config is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check if config has validate method (BacktestConfig or UnifiedPortfolioConfig)
        if hasattr(self.config, 'validate'):
            return self.config.validate()

        # Fallback: basic validation
        if hasattr(self.config, 'initial_capital'):
            if self.config.initial_capital <= 0:
                raise ValueError("initial_capital must be positive")

        if hasattr(self.config, 'tp_pct'):
            if self.config.tp_pct <= 0:
                raise ValueError("tp_pct must be positive")

        if hasattr(self.config, 'sl_pct'):
            if self.config.sl_pct <= 0:
                raise ValueError("sl_pct must be positive")

        if hasattr(self.config, 'leverage'):
            if self.config.leverage < 1:
                raise ValueError("leverage must be >= 1")

        if hasattr(self.config, 'position_size_pct'):
            if self.config.position_size_pct <= 0 or self.config.position_size_pct > 1.0:
                raise ValueError("position_size_pct must be between 0 and 1")

        return True

    def get_results(self) -> Dict[str, Any]:
        """
        Get backtest results as dictionary.

        Returns:
            Dictionary containing all results
        """
        return {
            'config': self._config_to_dict(),
            'metrics': self.metrics,
            'trades': self.trades,
            'final_capital': self.final_capital,
        }

    def _config_to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration as dict
        """
        if hasattr(self.config, 'to_dict'):
            return self.config.to_dict()

        # Fallback: extract common fields
        result = {}
        for attr in ['initial_capital', 'max_positions', 'position_size_pct',
                     'tp_pct', 'sl_pct', 'leverage']:
            if hasattr(self.config, attr):
                result[attr] = getattr(self.config, attr)

        return result
