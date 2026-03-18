"""
Data Layer - Market data preparation and validation.
Responsibilities:
- Load OHLCV data from CSV files
- Validate data schema
- Align timeframes
- Handle timezone normalization
- Detect missing candles

No trading logic allowed here.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data layer configuration"""
    base_path: str
    h1_folder: str = "1h"
    m1_folder: str = "5m"  # Using 5m as provided in the data
    timestamp_col: str = "opentime"  # lowercase after normalization
    required_columns: Tuple[str, ...] = ("opentime", "open", "high", "low", "close", "volume")  # lowercase
    default_parse_dates: bool = False


@dataclass
class DataValidationResult:
    """Result of data validation"""
    is_valid: bool
    missing_candles: int = 0
    duplicates: int = 0
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = None
    issues: List[str] = None

    def __post_init__(self):
        if self.gaps is None:
            self.gaps = []
        if self.issues is None:
            self.issues = []


class DataLoader:
    """
    Handles loading and preparation of OHLCV market data.
    Stateless and deterministic - no trading logic.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig(base_path="")

    def load_ohlcv(self, symbol: str, timeframe: str, base_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe folder name (e.g., "1h", "5m")
            base_path: Base directory path

        Returns:
            DataFrame with OHLCV data sorted by timestamp
        """
        base = base_path or self.config.base_path
        folder = self.config.h1_folder if timeframe == "1h" else self.config.m1_folder
        file_path = os.path.join(base, folder, f"{symbol}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Ensure proper column names (lowercase)
        df = self._normalize_columns(df)

        # Validate required columns exist (after normalization)
        self._validate_columns(df, self.config.required_columns)

        # Sort by timestamp ascending
        df = df.sort_values(self.config.timestamp_col).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} {timeframe} candles for {symbol} from {file_path}")

        return df

    def load_pair_data(
        self,
        symbol: str,
        base_path: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both 1h and intrabar (5m) data for a symbol.

        Args:
            symbol: Trading symbol
            base_path: Base directory path
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)

        Returns:
            Tuple of (h1_dataframe, m1_dataframe)
        """
        base = base_path or self.config.base_path

        h1_data = self.load_ohlcv(symbol, "1h", base)
        m1_data = self.load_ohlcv(symbol, "5m", base)

        # Apply time filters
        if start_time is not None:
            h1_data = h1_data[h1_data[self.config.timestamp_col] >= start_time]
            m1_data = m1_data[m1_data[self.config.timestamp_col] >= start_time]

        if end_time is not None:
            h1_data = h1_data[h1_data[self.config.timestamp_col] <= end_time]
            m1_data = m1_data[m1_data[self.config.timestamp_col] <= end_time]

        # Reset indices after filtering
        h1_data = h1_data.reset_index(drop=True)
        m1_data = m1_data.reset_index(drop=True)

        return h1_data, m1_data

    def _validate_columns(self, df: pd.DataFrame, required: Tuple[str, ...]) -> None:
        """Validate that required columns exist"""
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase"""
        df.columns = df.columns.str.lower()
        return df

    def validate_schema(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Validate OHLCV data schema.

        Checks:
        - Required columns present
        - No missing values in OHLCV
        - High >= Low, High >= Open, High >= Close
        - Low <= Open, Low <= Close
        - Positive volumes
        - No duplicate timestamps
        """
        issues = []
        missing_candles = 0
        duplicates = 0
        gaps = []

        # Check for missing values
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if df[ohlcv_cols].isnull().any().any():
            issues.append("Found missing values in OHLCV data")

        # Check OHLCV relationships
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"Found {invalid_hl} rows where high < low")

        invalid_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if invalid_high > 0:
            issues.append(f"Found {invalid_high} rows where high < open/close")

        invalid_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
        if invalid_low > 0:
            issues.append(f"Found {invalid_low} rows where low > open/close")

        # Check for negative/zero values
        if (df['volume'] <= 0).any():
            issues.append("Found non-positive volume values")

        # Check for duplicates
        duplicates = df['opentime'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")

        # Detect gaps (for 1h data)
        if len(df) > 1:
            df_sorted = df.sort_values('opentime')
            time_diffs = df_sorted['opentime'].diff()

            # Expected: 1h = 3600000ms, 5m = 300000ms
            # Allow some tolerance
            h1_expected = 3600000
            tolerance = 0.1  # 10% tolerance

            for i in range(1, len(time_diffs)):
                diff = time_diffs.iloc[i]
                if abs(diff - h1_expected) / h1_expected > tolerance:
                    start = df_sorted.iloc[i-1]['opentime']
                    end = df_sorted.iloc[i]['opentime']
                    gaps.append((pd.to_datetime(start, unit='ms'), pd.to_datetime(end, unit='ms')))

        is_valid = len(issues) == 0

        return DataValidationResult(
            is_valid=is_valid,
            missing_candles=missing_candles,
            duplicates=duplicates,
            gaps=gaps,
            issues=issues
        )

    def align_timeframes(
        self,
        h1_data: pd.DataFrame,
        m1_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align 1h and intrabar dataframes to common time range.

        Ensures both datasets cover the same period.
        """
        h1_start = h1_data['opentime'].min()
        h1_end = h1_data['opentime'].max()
        m1_start = m1_data['opentime'].min()
        m1_end = m1_data['opentime'].max()

        # Find overlapping range
        overlap_start = max(h1_start, m1_start)
        overlap_end = min(h1_end, m1_end)

        if overlap_start >= overlap_end:
            raise ValueError("No overlapping time range between 1h and 5m data")

        h1_aligned = h1_data[
            (h1_data['opentime'] >= overlap_start) &
            (h1_data['opentime'] <= overlap_end)
        ].reset_index(drop=True)

        m1_aligned = m1_data[
            (m1_data['opentime'] >= overlap_start) &
            (m1_data['opentime'] <= overlap_end)
        ].reset_index(drop=True)

        logger.info(f"Aligned data: {len(h1_aligned)} 1h candles, {len(m1_aligned)} 5m candles")

        return h1_aligned, m1_aligned

    def resample_if_needed(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        timestamp_col: str = 'opentime'
    ) -> pd.DataFrame:
        """
        Resample data to target timeframe if needed.

        Args:
            df: Source dataframe
            target_timeframe: Target timeframe (e.g., '1h', '4h', '1d')
            timestamp_col: Name of timestamp column

        Returns:
            Resampled dataframe
        """
        # This is a placeholder - in practice, you'd implement resampling logic
        # For now, just return the original dataframe
        return df

    def detect_missing_candles(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int,
        timestamp_col: str = 'opentime'
    ) -> List[Tuple[int, int]]:
        """
        Detect missing candles in the data.

        Args:
            df: OHLCV dataframe
            timeframe_minutes: Expected timeframe in minutes
            timestamp_col: Timestamp column name

        Returns:
            List of (expected_time, actual_time) tuples for missing candles
        """
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        expected_interval = timeframe_minutes * 60 * 1000  # Convert to milliseconds

        missing = []

        for i in range(1, len(df_sorted)):
            prev_time = df_sorted.iloc[i-1][timestamp_col]
            curr_time = df_sorted.iloc[i][timestamp_col]
            diff = curr_time - prev_time

            if diff > expected_interval * 1.1:  # 10% tolerance
                # Calculate how many candles are missing
                num_missing = int(round((diff - expected_interval) / expected_interval))
                for j in range(num_missing):
                    expected = prev_time + (j + 1) * expected_interval
                    missing.append((expected, curr_time if j == num_missing - 1 else None))

        if missing:
            logger.warning(f"Detected {len(missing)} missing candles")

        return missing
