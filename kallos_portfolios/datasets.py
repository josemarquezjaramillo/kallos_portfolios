"""
Price data loading and date generation utilities for portfolio backtesting.
Handles monthly rebalancing dates and price data alignment.
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sqlalchemy.ext.asyncio import AsyncSession

from .storage import load_prices_for_period, fetch_monthly_universe

logger = logging.getLogger(__name__)


def generate_weekly_rebalance_dates(start_date: date, end_date: date) -> List[date]:
    """
    Generate weekly rebalancing dates (first business day of each week, preferably Monday).
    
    Args:
        start_date: Portfolio start date
        end_date: Portfolio end date
        
    Returns:
        List of weekly rebalancing dates
    """
    rebalance_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Find the Monday of the current week (Monday = 0)
        days_to_monday = current_date.weekday()  # 0=Monday, 1=Tuesday, etc.
        monday_of_week = current_date - timedelta(days=days_to_monday)
        
        # If the Monday is before our start date, move to next Monday
        if monday_of_week < start_date:
            monday_of_week += timedelta(days=7)
        
        # Find first business day of that week (Monday or next business day)
        week_start = monday_of_week
        week_end = monday_of_week + timedelta(days=6)  # Sunday
        
        # Get business days for that week
        business_days = pd.bdate_range(start=week_start, end=week_end, freq='B')
        
        if len(business_days) > 0:
            first_business_day = business_days[0].date()
            
            if first_business_day <= end_date and first_business_day not in rebalance_dates:
                rebalance_dates.append(first_business_day)
        
        # Move to next week
        current_date = monday_of_week + timedelta(days=7)
    
    logger.info(f"Generated {len(rebalance_dates)} weekly rebalance dates from {start_date} to {end_date}")
    return rebalance_dates

def generate_monthly_rebalance_dates(start_date: date, end_date: date) -> List[date]:
    """
    Generate monthly rebalancing dates (first business day of each month).
    
    Args:
        start_date: Portfolio start date
        end_date: Portfolio end date
        
    Returns:
        List of monthly rebalancing dates
    """
    rebalance_dates = []
    current_date = start_date.replace(day=1)  # Start from first of month
    
    while current_date <= end_date:
        # Find first business day of the month
        first_business_day = pd.bdate_range(
            start=current_date,
            end=current_date + timedelta(days=6),
            freq='B'
        )[0].date()
        
        if first_business_day <= end_date:
            rebalance_dates.append(first_business_day)
        
        # Move to next month
        current_date += relativedelta(months=1)
    
    logger.info(f"Generated {len(rebalance_dates)} monthly rebalance dates from {start_date} to {end_date}")
    return rebalance_dates


def generate_business_date_range(start_date: date, end_date: date) -> List[date]:
    """
    Generate business day range for daily return calculations.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of business dates
    """
    business_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    date_list = [d.date() for d in business_dates]
    
    logger.info(f"Generated {len(date_list)} business dates from {start_date} to {end_date}")
    return date_list


async def load_universe_and_prices_for_rebalance(
    session: AsyncSession,
    rebalance_date: date,
    lookback_days: int
) -> Tuple[List[str], pd.DataFrame]:
    """
    Load monthly universe and historical prices for a rebalancing date.
    
    Args:
        session: Database session
        rebalance_date: Monthly rebalancing date
        lookback_days: Number of lookback days for price history
        
    Returns:
        Tuple of (universe symbols, price DataFrame)
    """
    # Get universe for this month
    universe = await fetch_monthly_universe(session, rebalance_date)
    
    if not universe:
        logger.error(f"No universe found for rebalance date {rebalance_date}")
        return [], pd.DataFrame()
    
    # Calculate lookback start date
    lookback_start = rebalance_date - timedelta(days=lookback_days + 30)  # Extra buffer
    
    # Load price data
    prices = await load_prices_for_period(
        session, 
        universe, 
        lookback_start, 
        rebalance_date
    )
    
    if prices.empty:
        logger.error(f"No price data loaded for rebalance date {rebalance_date}")
        return universe, pd.DataFrame()
    
    # Ensure we have enough history
    if len(prices) < lookback_days:
        logger.warning(f"Insufficient price history: {len(prices)} days < {lookback_days} required")
    
    # Filter to exact lookback period
    if len(prices) > lookback_days:
        prices = prices.tail(lookback_days)
    
    # Check data quality
    coverage = validate_price_data_quality(prices, universe, rebalance_date)
    
    logger.info(f"Loaded prices for {len(prices.columns)} symbols, {len(prices)} days for {rebalance_date}")
    return universe, prices


def validate_price_data_quality(
    prices: pd.DataFrame, 
    expected_universe: List[str], 
    rebalance_date: date,
    min_coverage: float = 0.8
) -> float:
    """
    Validate price data quality and coverage.
    
    Args:
        prices: Price DataFrame
        expected_universe: Expected symbols
        rebalance_date: Rebalancing date for context
        min_coverage: Minimum acceptable data coverage
        
    Returns:
        float: Data coverage ratio (0-1)
    """
    if prices.empty or not expected_universe:
        return 0.0
    
    # Check symbol coverage
    available_symbols = set(prices.columns)
    expected_symbols = set(expected_universe)
    missing_symbols = expected_symbols - available_symbols
    
    symbol_coverage = len(available_symbols) / len(expected_symbols)
    
    if missing_symbols:
        logger.warning(f"Missing symbols for {rebalance_date}: {missing_symbols}")
    
    # Check data completeness (non-NaN values)
    data_completeness = 1 - prices.isnull().sum().sum() / (len(prices) * len(prices.columns))
    
    # Check for extreme returns (likely data errors)
    daily_returns = prices.pct_change().dropna()
    extreme_returns = (daily_returns.abs() > 0.9).sum().sum()
    
    if extreme_returns > 0:
        logger.warning(f"Found {extreme_returns} extreme daily returns (>90%) for {rebalance_date}")
    
    # Overall quality score
    quality_score = min(symbol_coverage, data_completeness)
    
    if quality_score < min_coverage:
        logger.warning(f"Low data quality for {rebalance_date}: {quality_score:.1%} < {min_coverage:.1%}")
    
    logger.info(f"Data quality for {rebalance_date}: {quality_score:.1%} (symbols: {symbol_coverage:.1%}, completeness: {data_completeness:.1%})")
    return quality_score


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data by removing outliers and handling missing values.
    
    Args:
        prices: Raw price DataFrame
        
    Returns:
        Cleaned price DataFrame
    """
    if prices.empty:
        return prices
    
    cleaned_prices = prices.copy()
    
    # Remove negative or zero prices
    cleaned_prices = cleaned_prices.where(cleaned_prices > 0)
    
    # Calculate daily returns
    daily_returns = cleaned_prices.pct_change()
    
    # Remove extreme returns (likely data errors)
    extreme_threshold = 0.9  # 90% daily return threshold
    extreme_mask = daily_returns.abs() > extreme_threshold
    
    if extreme_mask.sum().sum() > 0:
        logger.warning(f"Removing {extreme_mask.sum().sum()} extreme return observations")
        # Set extreme returns to NaN
        cleaned_prices = cleaned_prices.mask(extreme_mask.shift(1, fill_value=False))
    
    # Forward fill missing values (max 3 days)
    cleaned_prices = cleaned_prices.fillna(method='ffill', limit=3)
    
    # Backward fill remaining missing values (max 3 days)
    cleaned_prices = cleaned_prices.fillna(method='bfill', limit=3)
    
    # Drop columns with too much missing data (>50%)
    missing_pct = cleaned_prices.isnull().sum() / len(cleaned_prices)
    high_missing_cols = missing_pct[missing_pct > 0.5].index
    
    if len(high_missing_cols) > 0:
        logger.warning(f"Dropping symbols with >50% missing data: {list(high_missing_cols)}")
        cleaned_prices = cleaned_prices.drop(columns=high_missing_cols)
    
    # Final check for remaining missing values
    remaining_missing = cleaned_prices.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Still have {remaining_missing} missing values after cleaning")
    
    logger.info(f"Cleaned price data: {len(cleaned_prices.columns)} symbols, {len(cleaned_prices)} days")
    return cleaned_prices


def align_prices_to_date_range(
    prices: pd.DataFrame, 
    start_date: date, 
    end_date: date
) -> pd.DataFrame:
    """
    Align price data to specific date range for backtesting.
    
    Args:
        prices: Price DataFrame
        start_date: Desired start date
        end_date: Desired end date
        
    Returns:
        Aligned price DataFrame
    """
    if prices.empty:
        return prices
    
    # Convert date objects to datetime for comparison
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    # Filter to date range
    mask = (prices.index >= start_dt) & (prices.index <= end_dt)
    aligned_prices = prices.loc[mask]
    
    # Generate business day range for the period
    business_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    
    # Reindex to include all business days (forward fill missing dates)
    aligned_prices = aligned_prices.reindex(business_dates, method='ffill')
    
    logger.info(f"Aligned prices to {len(aligned_prices)} business days from {start_date} to {end_date}")
    return aligned_prices


def calculate_returns_matrix(prices: pd.DataFrame, return_type: str = 'simple') -> pd.DataFrame:
    """
    Calculate returns matrix from price data.
    
    Args:
        prices: Price DataFrame
        return_type: Type of returns ('simple', 'log')
        
    Returns:
        Returns DataFrame
    """
    if prices.empty:
        return pd.DataFrame()
    
    # Ensure all prices are float64 before calculating returns
    prices_float = prices.copy()
    for col in prices_float.columns:
        prices_float[col] = pd.to_numeric(prices_float[col], errors='coerce').astype('float64')
    
    if return_type == 'simple':
        returns = prices_float.pct_change()
    elif return_type == 'log':
        returns = np.log(prices_float / prices_float.shift(1))
    else:
        raise ValueError(f"Unknown return type: {return_type}")
    
    # Drop first row (NaN from pct_change) and ensure float64 dtype
    returns = returns.dropna(how='all')
    returns = returns.astype('float64')
    
    logger.info(f"Calculated {return_type} returns: {len(returns.columns)} symbols, {len(returns)} periods")
    return returns


def create_price_data_summary(prices: pd.DataFrame) -> Dict[str, any]:
    """
    Create summary statistics for price data quality assessment.
    
    Args:
        prices: Price DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    if prices.empty:
        return {'error': 'Empty price data'}
    
    daily_returns = prices.pct_change().dropna()
    
    summary = {
        'n_symbols': len(prices.columns),
        'n_periods': len(prices),
        'start_date': prices.index.min().date(),
        'end_date': prices.index.max().date(),
        'missing_values': prices.isnull().sum().sum(),
        'missing_percentage': prices.isnull().sum().sum() / (len(prices) * len(prices.columns)),
        'mean_daily_return': daily_returns.mean().mean(),
        'mean_daily_volatility': daily_returns.std().mean(),
        'extreme_returns_count': (daily_returns.abs() > 0.5).sum().sum(),
        'symbols': list(prices.columns)
    }
    
    return summary


async def prepare_full_dataset_for_backtest(
    session: AsyncSession,
    start_date: date,
    end_date: date,
    rebalance_dates: List[date],
    lookback_days: int = 252
) -> Dict[date, Tuple[List[str], pd.DataFrame]]:
    """
    Prepare complete dataset for backtesting with monthly universe rotation.
    
    Args:
        session: Database session
        start_date: Portfolio start date
        end_date: Portfolio end date
        rebalance_dates: List of monthly rebalancing dates
        lookback_days: Lookback period for each rebalance
        
    Returns:
        Dictionary mapping rebalance dates to (universe, prices) tuples
    """
    dataset = {}
    
    for i, rebalance_date in enumerate(rebalance_dates):
        logger.info(f"Preparing data for rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")
        
        try:
            universe, prices = await load_universe_and_prices_for_rebalance(
                session, rebalance_date, lookback_days
            )
            
            if universe and not prices.empty:
                # Clean and validate the data
                cleaned_prices = clean_price_data(prices)
                
                if not cleaned_prices.empty:
                    dataset[rebalance_date] = (universe, cleaned_prices)
                    logger.info(f"Successfully prepared data for {rebalance_date}")
                else:
                    logger.error(f"No clean price data available for {rebalance_date}")
            else:
                logger.error(f"Failed to load data for {rebalance_date}")
                
        except Exception as e:
            logger.error(f"Error preparing data for {rebalance_date}: {e}")
            continue
    
    logger.info(f"Prepared dataset for {len(dataset)}/{len(rebalance_dates)} rebalance dates")
    return dataset


def get_overlapping_universe(dataset: Dict[date, Tuple[List[str], pd.DataFrame]]) -> List[str]:
    """
    Get symbols that appear in all rebalancing periods (for analysis purposes).
    
    Args:
        dataset: Dataset dictionary from prepare_full_dataset_for_backtest
        
    Returns:
        List of symbols present in all periods
    """
    if not dataset:
        return []
    
    universes = [universe for universe, _ in dataset.values()]
    
    if not universes:
        return []
    
    # Find intersection of all universes
    overlapping = set(universes[0])
    for universe in universes[1:]:
        overlapping = overlapping.intersection(set(universe))
    
    overlapping_list = sorted(list(overlapping))
    
    total_unique_symbols = len(set().union(*universes))
    logger.info(f"Overlapping universe: {len(overlapping_list)}/{total_unique_symbols} symbols appear in all periods")
    
    return overlapping_list
