"""
Database operations for Kallos Portfolios using async SQLAlchemy.
Handles all CRUD operations with proper connection management and error handling.
"""

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, String, Date, DateTime, Numeric, Integer, 
    UniqueConstraint, Index, text, func, event
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

# Import simple config pattern
from json import loads
from .exceptions import DataValidationError, InsufficientDataError, DatabaseError

# Simple parameter class to replace config dependency
class OptimizationParams:
    def __init__(self, **kwargs):
        self.objective = kwargs.get('objective', 'max_sharpe')
        self.max_weight = kwargs.get('max_weight', 0.35)
        self.min_names = kwargs.get('min_names', 3)
        self.lookback_days = kwargs.get('lookback_days', 252)
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.0)
        self.gamma = kwargs.get('gamma', 0.01)
        self.l2_reg = kwargs.get('l2_reg', 0.01)
        self.target_volatility = kwargs.get('target_volatility', None)
        self.target_return = kwargs.get('target_return', None)

logger = logging.getLogger(__name__)

# Simple function to load database config using your existing pattern
def load_db_config():
    """Load database configuration using the existing pattern"""
    try:
        db_kwargs = loads(open('/home/jlmarquez11/kallos/kallos_runner/.env').read())
        return db_kwargs
    except FileNotFoundError:
        logger.warning("Database config file not found, using defaults")
        return {
            'postgres_user': 'postgres',
            'postgres_password': 'password', 
            'postgres_host': 'localhost',
            'postgres_port': 5432,
            'postgres_db': 'kallos'
        }

# SQLAlchemy models
Base = declarative_base()


class OptimizationParamsDB(Base):
    """Database model for optimization parameters."""
    __tablename__ = 'optimization_params'
    __table_args__ = {'schema': 'portfolio_simulations'}
    __table_args__ = {'schema': 'portfolio_simulations'}
    
    run_id = Column(String, primary_key=True, nullable=False)
    objective = Column(String, nullable=False)
    max_weight = Column(Numeric, nullable=False)
    min_names = Column(Integer, nullable=False)
    lookback_days = Column(Integer, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    risk_free_rate = Column(Numeric, nullable=False, default=0.0)
    gamma = Column(Numeric, nullable=False, default=0.01)
    l2_reg = Column(Numeric, nullable=False, default=0.01)
    target_volatility = Column(Numeric, nullable=True)
    target_return = Column(Numeric, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WeightsWeekly(Base):
    """Database model for weekly portfolio weights by strategy."""
    __tablename__ = 'weights_weekly'
    __table_args__ = {'schema': 'portfolio_simulations'}
    __table_args__ = (
        UniqueConstraint('run_id', 'date', 'symbol', 'strategy'),
        {'schema': 'portfolio_simulations'}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    symbol = Column(String, nullable=False)
    weight = Column(Numeric, nullable=False)
    strategy = Column(String, nullable=False)  # 'gru', 'historical', 'market_cap'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReturnsDaily(Base):
    """Database model for daily portfolio returns by strategy."""
    __tablename__ = 'returns_daily'
    __table_args__ = {'schema': 'portfolio_simulations'}
    __table_args__ = (
        UniqueConstraint('run_id', 'date', 'strategy'),
        {'schema': 'portfolio_simulations'}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    portfolio_return = Column(Numeric, nullable=False)
    strategy = Column(String, nullable=False)  # 'gru', 'historical', 'market_cap'
    benchmark_return = Column(Numeric, nullable=True)
    active_return = Column(Numeric, nullable=True)
    turnover = Column(Numeric, nullable=True)
    transaction_costs = Column(Numeric, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyMarketData(Base):
    """Daily market data model matching existing database schema."""
    __tablename__ = 'daily_market_data'
    __table_args__ = {'schema': 'public'}
    
    id = Column(String, primary_key=True, nullable=False)  # coin_id
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    price = Column(Numeric, nullable=False)
    market_cap = Column(Numeric, nullable=True)
    volume = Column(Numeric, nullable=True)
    open = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('id', 'timestamp', name='daily_market_data_id_timestamp_key'),
        Index('idx_daily_market_data_id_timestamp', 'id', 'timestamp'),
        Index('idx_daily_market_data_timestamp', 'timestamp'),
    )


class DailyIndexCoinContributions(Base):
    """Database model for market cap benchmark data."""
    __tablename__ = 'daily_index_coin_contributions'
    __table_args__ = {'schema': 'public'}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False)
    coin_id = Column(String, nullable=False)
    period_start_date_for_constituents = Column(Date, nullable=False)
    previous_day_price = Column(Numeric, nullable=True)
    current_day_price = Column(Numeric, nullable=True)
    weight_on_previous_day = Column(Numeric, nullable=True)
    end_of_day_weight = Column(Numeric, nullable=False)
    individual_coin_return = Column(Numeric, nullable=True)
    return_contribution_to_index = Column(Numeric, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('trade_date', 'coin_id'),
    )


class IndexMonthlyConstituents(Base):
    """Database model for monthly universe constituents."""
    __tablename__ = 'index_monthly_constituents'
    __table_args__ = {'schema': 'public'}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    period_start_date = Column(Date, nullable=False)
    coin_id = Column(String, nullable=False)
    initial_market_cap_at_rebalance = Column(Numeric, nullable=True)
    initial_weight_at_rebalance = Column(Numeric, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('period_start_date', 'coin_id'),
    )


# Async engine and session factory
async_engine = None
AsyncSessionLocal = None


def init_async_db(database_url: Optional[str] = None):
    """
    Initialize async database engine and session factory with production-ready configuration.
    
    Args:
        database_url: PostgreSQL connection string (optional, uses config if None)
    """
    global async_engine, AsyncSessionLocal
    
    # Use simple config loading
    if not database_url:
        db_config = load_db_config()
        database_url = f"postgresql+asyncpg://{db_config['postgres_user']}:{db_config['postgres_password']}@{db_config['postgres_host']}:{db_config['postgres_port']}/{db_config['postgres_db']}"
    
    async_engine = create_async_engine(
        database_url,
        # Connection pooling configuration with simple defaults
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,           # Validate connections before use
        pool_recycle=3600,            # Recycle connections every hour
        pool_timeout=30,              # Timeout for getting connection from pool
        # Note: QueuePool is not compatible with async engines, using default async pool
        echo=False,                   # Simple default
        # Additional connection args
        connect_args={
            "server_settings": {
                "application_name": "kallos_portfolios",
                "jit": "off"  # Disable JIT for predictable performance
            }
        }
    )
    
    # Add event listeners for connection monitoring
    @event.listens_for(async_engine.sync_engine, "connect")
    def receive_connect(dbapi_connection, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(async_engine.sync_engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")
    
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    logger.info(f"Initialized async database engine with connection pooling: {database_url}")


async def check_database_health(database_url: Optional[str] = None) -> bool:
    """
    Check database connectivity and basic functionality.
    
    Args:
        database_url: PostgreSQL connection string (optional, uses config if None)
    
    Returns:
        True if database is healthy, False otherwise
    """
    try:
        async with get_async_session(database_url) as session:
            # Test basic connectivity
            result = await session.execute(text("SELECT 1"))
            test_value = result.scalar()
            
            if test_value != 1:
                logger.error("Database health check failed: unexpected result")
                return False
            
            # Test table existence
            result = await session.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'portfolio_simulations' 
                AND table_name IN ('optimization_params', 'weights_weekly', 'returns_daily')
            """))
            table_count = result.scalar()
            
            if table_count < 3:
                logger.error(f"Missing required tables. Found {table_count}/3")
                return False
            
            logger.info("Database health check passed")
            return True
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


@asynccontextmanager
async def get_async_session(database_url: Optional[str] = None):
    """
    Enhanced async context manager for database sessions with comprehensive error handling.
    
    Args:
        database_url: Optional database URL override
        
    Yields:
        AsyncSession: Database session with proper transaction management
    """
    if async_engine is None:
        init_async_db(database_url)
    
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
        logger.debug("Database session committed successfully")
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session rolled back due to error: {e}")
        raise
    finally:
        await session.close()
        logger.debug("Database session closed")


async def create_tables():
    """Create all database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Created all database tables")


# Optimization parameters operations
async def store_optimization_params(session: AsyncSession, run_id: str, params: OptimizationParams) -> bool:
    """
    Store optimization parameters in database.
    
    Args:
        session: Database session
        run_id: Unique run identifier
        params: Optimization parameters
        
    Returns:
        bool: Success status
    """
    try:
        # Convert string dates to date objects
        start_date = datetime.strptime(params.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(params.end_date, '%Y-%m-%d').date()
        
        db_params = OptimizationParamsDB(
            run_id=run_id,
            objective=params.objective,
            max_weight=float(params.max_weight),
            min_names=params.min_names,
            lookback_days=params.lookback_days,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=float(params.risk_free_rate),
            gamma=float(params.gamma),
            l2_reg=float(params.l2_reg),
            target_volatility=float(params.target_volatility) if params.target_volatility else None,
            target_return=float(params.target_return) if params.target_return else None
        )
        
        session.add(db_params)
        await session.flush()
        
        logger.info(f"Stored optimization parameters for run_id: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing optimization parameters: {e}")
        return False


async def load_optimization_params(session: AsyncSession, run_id: str) -> Optional[OptimizationParams]:
    """
    Load optimization parameters from database.
    
    Args:
        session: Database session
        run_id: Unique run identifier
        
    Returns:
        OptimizationParams or None if not found
    """
    try:
        result = await session.get(OptimizationParamsDB, run_id)
        if result is None:
            logger.warning(f"No optimization parameters found for run_id: {run_id}")
            return None
        
        return OptimizationParams(
            objective=result.objective,
            max_weight=float(result.max_weight),
            min_names=result.min_names,
            lookback_days=result.lookback_days,
            start_date=result.start_date.strftime('%Y-%m-%d'),
            end_date=result.end_date.strftime('%Y-%m-%d'),
            risk_free_rate=float(result.risk_free_rate),
            gamma=float(result.gamma),
            l2_reg=float(result.l2_reg),
            target_volatility=float(result.target_volatility) if result.target_volatility else None,
            target_return=float(result.target_return) if result.target_return else None
        )
        
    except Exception as e:
        logger.error(f"Error loading optimization parameters: {e}")
        return None


# Monthly universe operations
async def fetch_monthly_universe(session: AsyncSession, target_date: date) -> List[str]:
    """
    Retrieve investable universe for month containing target_date.
    Uses most recent period_start_date <= target_date.
    
    Args:
        session: Database session
        target_date: Date to find universe for
        
    Returns:
        List of coin_ids for the universe
    """
    try:
        query = text("""
            SELECT coin_id 
            FROM public.index_monthly_constituents 
            WHERE period_start_date = (
                SELECT MAX(period_start_date) 
                FROM public.index_monthly_constituents 
                WHERE period_start_date <= :target_date
            )
            ORDER BY coin_id
        """)
        
        result = await session.execute(query, {"target_date": target_date})
        universe = [row[0] for row in result.fetchall()]
        
        logger.info(f"Fetched universe of {len(universe)} assets for {target_date}")
        return universe
        
    except Exception as e:
        logger.error(f"Error fetching monthly universe for {target_date}: {e}")
        return []


# Price data operations
async def load_prices_for_period(
    session: AsyncSession, 
    coin_ids: List[str],  # Changed from symbols to coin_ids
    start_date: date, 
    end_date: date,
    min_data_coverage: float = 0.8
) -> pd.DataFrame:
    """
    Load price data from daily_market_data table.
    
    Args:
        session: Database session
        coin_ids: List of coin identifiers (matches id column)
        start_date: Start date for price data
        end_date: End date for price data
        min_data_coverage: Minimum required data coverage (0.0-1.0)
        
    Returns:
        DataFrame with coin_ids as columns, dates as index, price as values
        
    Raises:
        DataValidationError: If data quality is insufficient
        InsufficientDataError: If not enough data is available
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not coin_ids:
        raise ValueError("Coin ID list cannot be empty")
    
    if len(coin_ids) != len(set(coin_ids)):
        duplicate_ids = [s for s in coin_ids if coin_ids.count(s) > 1]
        raise ValueError(f"Duplicate coin IDs found: {duplicate_ids}")
    
    if start_date >= end_date:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
    
    if not (0.0 <= min_data_coverage <= 1.0):
        raise ValueError(f"min_data_coverage must be between 0.0 and 1.0, got {min_data_coverage}")
    
    try:
        # Query the daily_market_data table with proper column names
        query = text("""
            SELECT id, timestamp, price, volume, market_cap, close
            FROM public.daily_market_data 
            WHERE id = ANY(:coin_ids) 
            AND timestamp::date BETWEEN :start_date AND :end_date
            ORDER BY id, timestamp
        """)
        
        result = await session.execute(query, {
            'coin_ids': coin_ids,
            'start_date': start_date,
            'end_date': end_date
        })
        
        data = result.fetchall()
        
        if not data:
            raise InsufficientDataError(
                f"No price data found for coin IDs {coin_ids} "
                f"between {start_date} and {end_date}"
            )
        
        # Convert to DataFrame
        prices_df = pd.DataFrame(data, columns=['id', 'timestamp', 'price', 'volume', 'market_cap', 'close'])
        prices_df['date'] = pd.to_datetime(prices_df['timestamp']).dt.date
        prices_df.set_index('date', inplace=True)
        
        # Ensure all numeric columns are float64 (not Decimal)
        numeric_columns = ['price', 'volume', 'market_cap', 'close']
        for col in numeric_columns:
            if col in prices_df.columns:
                # Convert to numeric, but preserve original values if conversion fails
                original_values = prices_df[col].copy()
                prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce').astype('float64')
                
                # Check if any valid data was lost in conversion
                converted_nans = prices_df[col].isna() & original_values.notna()
                if converted_nans.any():
                    problematic_values = original_values[converted_nans].unique()
                    logger.warning(f"Non-numeric values converted to NaN in {col}: {problematic_values}")
        
        # Use close price if available, otherwise use price
        price_column = 'close' if 'close' in prices_df.columns and not prices_df['close'].isna().all() else 'price'
        
        # Check for problematic price values before pivoting
        price_data = prices_df[price_column]
        zero_prices = (price_data == 0).sum()
        negative_prices = (price_data < 0).sum()
        nan_prices = price_data.isna().sum()
        
        if zero_prices > 0:
            logger.warning(f"Found {zero_prices} zero prices in {price_column} - these will cause issues in return calculations")
        if negative_prices > 0:
            logger.warning(f"Found {negative_prices} negative prices in {price_column}")
        if nan_prices > 0:
            logger.warning(f"Found {nan_prices} NaN prices in {price_column} - these will be excluded")
        
        # Pivot to get coin_ids as columns
        prices_pivot = prices_df.pivot_table(
            index='date', 
            columns='id', 
            values=price_column,
            aggfunc='first'
        )
        
        # Data quality validation
        total_expected_points = len(coin_ids) * len(prices_pivot)
        actual_points = prices_pivot.count().sum()
        data_coverage = actual_points / total_expected_points if total_expected_points > 0 else 0
        
        if data_coverage < min_data_coverage:
            raise DataValidationError(
                f"Insufficient data coverage: {data_coverage:.2%} < {min_data_coverage:.2%}. "
                f"Missing {total_expected_points - actual_points} data points"
            )
        
        # Check for extreme price movements (potential data errors)
        daily_returns = prices_pivot.pct_change().dropna()
        extreme_returns = (daily_returns.abs() > 0.9).any(axis=1)
        
        if extreme_returns.any():
            extreme_dates = extreme_returns[extreme_returns].index.tolist()
            logger.warning(f"Extreme price movements detected on dates: {extreme_dates}")
        
        # Check for zero or negative prices
        invalid_prices = (prices_pivot <= 0).any(axis=1)
        if invalid_prices.any():
            invalid_dates = invalid_prices[invalid_prices].index.tolist()
            raise DataValidationError(f"Invalid (zero/negative) prices found on dates: {invalid_dates}")
        
        # Filter to requested coin_ids only
        available_ids = [s for s in coin_ids if s in prices_pivot.columns]
        missing_ids = [s for s in coin_ids if s not in prices_pivot.columns]
        
        if missing_ids:
            logger.warning(f"Coin IDs not found in data: {missing_ids}")
        
        if not available_ids:
            raise InsufficientDataError(f"None of the requested coin IDs {coin_ids} were found in database")
        
        prices_final = prices_pivot[available_ids]
        
        logger.info(
            f"Loaded price data: {len(available_ids)} coin IDs, "
            f"{len(prices_final)} dates, {data_coverage:.2%} coverage"
        )
        
        return prices_final
        
    except Exception as e:
        if isinstance(e, (DataValidationError, InsufficientDataError, ValueError)):
            raise
        
    except Exception as e:
        if isinstance(e, (DataValidationError, InsufficientDataError, ValueError)):
            raise
        logger.error(f"Unexpected error loading price data: {e}")
        raise DatabaseError(f"Failed to load price data: {str(e)}") from e


async def save_portfolio_run(
    session: AsyncSession,
    run_id: str,
    start_date: date,
    end_date: date,
    strategy_name: str,
    assets: List[str],
    final_value: float,
    total_return: float,
    sharpe_ratio: float,
    volatility: float,
    max_drawdown: float,
    metadata: dict = None
) -> int:
    """
    Save portfolio run results to database.
    
    Args:
        session: Database session
        run_id: Unique identifier for this portfolio run
        start_date: Portfolio start date
        end_date: Portfolio end date
        strategy_name: Name of the strategy used
        assets: List of assets in portfolio
        final_value: Final portfolio value
        total_return: Total return percentage
        sharpe_ratio: Portfolio Sharpe ratio
        volatility: Portfolio volatility
        max_drawdown: Maximum drawdown
        metadata: Additional metadata as dict
        
    Returns:
        Portfolio run ID from database
    """
    try:
        query = text("""
            INSERT INTO portfolio_runs (
                run_id, start_date, end_date, strategy_name, assets,
                final_value, total_return, sharpe_ratio, volatility, max_drawdown,
                metadata, created_at
            ) VALUES (
                :run_id, :start_date, :end_date, :strategy_name, :assets,
                :final_value, :total_return, :sharpe_ratio, :volatility, :max_drawdown,
                :metadata, NOW()
            ) 
            ON CONFLICT (run_id, strategy_name) 
            DO UPDATE SET 
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                assets = EXCLUDED.assets,
                final_value = EXCLUDED.final_value,
                total_return = EXCLUDED.total_return,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                volatility = EXCLUDED.volatility,
                max_drawdown = EXCLUDED.max_drawdown,
                metadata = EXCLUDED.metadata,
                created_at = NOW()
            RETURNING id
        """)
        
        # Convert metadata to JSON string for PostgreSQL JSONB
        import json
        metadata_json = json.dumps(metadata) if metadata is not None else None
        
        result = await session.execute(query, {
            'run_id': run_id,
            'start_date': start_date,
            'end_date': end_date,
            'strategy_name': strategy_name,
            'assets': assets,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'metadata': metadata_json
        })
        
        portfolio_run_id = result.scalar()
        await session.commit()
        
        logger.info(f"Saved portfolio run {run_id} to database with ID {portfolio_run_id}")
        return portfolio_run_id
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to save portfolio run {run_id}: {e}")
        raise DatabaseError(f"Failed to save portfolio run: {str(e)}") from e


async def save_portfolio_weights(
    session: AsyncSession,
    portfolio_run_id: int,
    rebalance_date: date,
    weights: pd.Series
) -> None:
    """
    Save portfolio weights for a specific rebalancing date.
    
    Args:
        session: Database session
        portfolio_run_id: Portfolio run ID from database
        rebalance_date: Date of rebalancing
        weights: Portfolio weights as pandas Series
    """
    try:
        # Insert weights for each asset
        for asset, weight in weights.items():
            query = text("""
                INSERT INTO portfolio_weights (
                    portfolio_run_id, rebalance_date, asset, weight
                ) VALUES (
                    :portfolio_run_id, :rebalance_date, :asset, :weight
                )
                ON CONFLICT (portfolio_run_id, rebalance_date, asset) 
                DO UPDATE SET weight = EXCLUDED.weight
            """)
            
            await session.execute(query, {
                'portfolio_run_id': portfolio_run_id,
                'rebalance_date': rebalance_date,
                'asset': asset,
                'weight': float(weight)
            })
        
        await session.commit()
        logger.info(f"Saved {len(weights)} weights for portfolio {portfolio_run_id} on {rebalance_date}")
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to save portfolio weights: {e}")
        raise DatabaseError(f"Failed to save portfolio weights: {str(e)}") from e


async def save_portfolio_returns(
    session: AsyncSession,
    portfolio_run_id: int,
    returns_series: pd.Series
) -> None:
    """
    Save daily portfolio returns.
    
    Args:
        session: Database session
        portfolio_run_id: Portfolio run ID from database
        returns_series: Daily returns as pandas Series with date index
    """
    try:
        # Convert returns to list of dicts for bulk insert
        returns_data = []
        for return_date, return_value in returns_series.items():
            # Convert date index to proper date object
            if hasattr(return_date, 'date'):
                date_obj = return_date.date()
            else:
                date_obj = return_date
                
            returns_data.append({
                'portfolio_run_id': portfolio_run_id,
                'date': date_obj,
                'return_value': float(return_value)
            })
        
        if returns_data:
            # Use executemany for bulk insert
            query = text("""
                INSERT INTO portfolio_returns (portfolio_run_id, date, return_value)
                VALUES (:portfolio_run_id, :date, :return_value)
                ON CONFLICT (portfolio_run_id, date) 
                DO UPDATE SET return_value = EXCLUDED.return_value
            """)
            
            await session.execute(query, returns_data)
            await session.commit()
            
            logger.info(f"Saved {len(returns_data)} daily returns for portfolio {portfolio_run_id}")
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to save portfolio returns: {e}")
        raise DatabaseError(f"Failed to save portfolio returns: {str(e)}") from e


async def fetch_monthly_universes(
    session: AsyncSession, 
    start_date: date,
    end_date: date
) -> List[str]:
    
    query = text("""
        SELECT period_start_date as date, coin_id
        FROM public.index_monthly_constituents 
        WHERE period_start_date >= :start_date
        AND period_start_date <= :end_date
        ORDER BY period_start_date ASC, initial_market_cap_at_rebalance DESC
    """)

    result = await session.execute(query, {'start_date': start_date, 'end_date': end_date})
    data = result.fetchall()
    if not data:
        raise InsufficientDataError(f"No universe data found for date {start_date} and {end_date}")
    else:
        df = pd.DataFrame(data, columns=['date', 'coin_id'])
        return df
    

async def fetch_model_definitions(
    session: AsyncSession, 
    universes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fetch model training data for coins in universe DataFrame where universe dates
    fall within the model's test period.
    
    Args:
        session: Database session
        universes_df: DataFrame with 'date' and 'coin_id' columns
        
    Returns:
        DataFrame with model training data
    """
    if universes_df.empty:
        raise ValueError("Universe DataFrame cannot be empty")    
    
    query = text("""
        SELECT DISTINCT
            mtv.model,
            mtv.coin_id,
            mtv.year_end,
            mtv.quarter_end,
            mtv.train_date_end,
            mtv.test_start_date,
            mtv.test_end_date,
            mtv.study_name,
            mtv.num_trials,
            mtv.best_rmse,
            mtv.best_da,
            mtv.avg_rmse,
            mtv.avg_da
        FROM public.model_train_view mtv
        INNER JOIN (VALUES {}) AS u(universe_date, universe_coin_id) 
            ON mtv.coin_id = u.universe_coin_id
            AND u.universe_date::date BETWEEN mtv.test_start_date AND mtv.test_end_date
        ORDER BY mtv.coin_id, mtv.test_start_date
    """.format(','.join([f"('{row.date}', '{row.coin_id}')" for _, row in universes_df.iterrows()])))
    
    result = await session.execute(query)
    data = result.fetchall()
    
    if not data:
        raise InsufficientDataError(f"No model training data found matching universe criteria")
    
    df = pd.DataFrame(data, columns=[
        'model', 'coin_id', 'year_end', 'quarter_end', 
        'train_date_end', 'test_start_date', 'test_end_date',
        'study_name', 'num_trials', 'best_rmse', 'best_da', 
        'avg_rmse', 'avg_da'
    ])
    df['study_name'] = df['study_name'].str.strip()
    return df


# Weight storage operations
async def upsert_weights(
    session: AsyncSession, 
    run_id: str, 
    date: date, 
    weights: pd.Series, 
    strategy: str = 'gru'
) -> bool:
    """
    Efficient batch upsert for portfolio weights.
    
    Args:
        session: Database session
        run_id: Unique run identifier
        date: Rebalancing date
        weights: Portfolio weights (symbol -> weight)
        strategy: Strategy name ('gru', 'historical', 'market_cap')
        
    Returns:
        bool: Success status
    """
    try:
        # Prepare weight data for batch insert
        weight_data = []
        for symbol, weight in weights.items():
            if pd.notna(weight) and weight > 1e-6:  # Filter out tiny weights
                weight_data.append({
                    'run_id': run_id,
                    'date': date,
                    'symbol': symbol,
                    'weight': float(weight),
                    'strategy': strategy
                })
        
        if not weight_data:
            logger.warning(f"No valid weights to store for {run_id}, {date}, {strategy}")
            return False
        
        # Use PostgreSQL ON CONFLICT for efficient upsert
        stmt = insert(WeightsWeekly).values(weight_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['run_id', 'date', 'symbol', 'strategy'],
            set_=dict(
                weight=stmt.excluded.weight,
                updated_at=datetime.utcnow()
            )
        )
        
        await session.execute(stmt)
        
        logger.info(f"Stored {len(weight_data)} weights for {strategy} strategy on {date}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing weights: {e}")
        return False


# Market cap weight operations
async def load_market_cap_weights(
    session: AsyncSession, 
    start_date: date, 
    end_date: date
) -> Dict[date, pd.Series]:
    """
    Load market cap weights from daily_index_coin_contributions.
    
    Args:
        session: Database session
        start_date: Start date for weight data
        end_date: End date for weight data
        
    Returns:
        Dictionary mapping date to weight series
    """
    try:
        query = text("""
            SELECT trade_date, coin_id, end_of_day_weight
            FROM public.daily_index_coin_contributions
            WHERE trade_date BETWEEN :start_date AND :end_date
            AND end_of_day_weight IS NOT NULL
            ORDER BY trade_date, coin_id
        """)
        
        result = await session.execute(query, {
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Group by date
        weights_by_date = {}
        for row in result.fetchall():
            trade_date = row[0]
            coin_id = row[1]
            weight = float(row[2])
            
            if trade_date not in weights_by_date:
                weights_by_date[trade_date] = {}
            
            weights_by_date[trade_date][coin_id] = weight
        
        # Convert to pandas Series for each date
        market_cap_weights = {}
        for date, weights_dict in weights_by_date.items():
            weights_series = pd.Series(weights_dict)
            # Normalize to sum to 1
            if weights_series.sum() > 0:
                weights_series = weights_series / weights_series.sum()
            market_cap_weights[date] = weights_series
        
        logger.info(f"Loaded market cap weights for {len(market_cap_weights)} dates")
        return market_cap_weights
        
    except Exception as e:
        logger.error(f"Error loading market cap weights: {e}")
        return {}


# Crypto symbol mapping utilities
def get_crypto_symbol_variations(symbol: str) -> List[str]:
    """
    Maps between different crypto identifier formats.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC')
        
    Returns:
        List of possible variations
    """
    symbol_map = {
        'BTC': ['bitcoin', 'btc', 'Bitcoin', 'BTC'],
        'ETH': ['ethereum', 'eth', 'Ethereum', 'ETH'],
        'ADA': ['cardano', 'ada', 'Cardano', 'ADA'],
        'DOT': ['polkadot', 'dot', 'Polkadot', 'DOT'],
        'SOL': ['solana', 'sol', 'Solana', 'SOL'],
        'AVAX': ['avalanche-2', 'avax', 'Avalanche', 'AVAX'],
        'MATIC': ['matic-network', 'matic', 'Polygon', 'MATIC'],
        'LINK': ['chainlink', 'link', 'Chainlink', 'LINK'],
        'UNI': ['uniswap', 'uni', 'Uniswap', 'UNI'],
        'LTC': ['litecoin', 'ltc', 'Litecoin', 'LTC']
    }
    
    # Return variations for known symbols, otherwise return the symbol itself
    return symbol_map.get(symbol.upper(), [symbol.lower(), symbol.upper()])


def align_market_cap_weights_to_rebalance_dates(
    market_cap_weights: Dict[date, pd.Series], 
    rebalance_dates: List[date], 
    universe: List[str]
) -> Dict[date, pd.Series]:
    """
    Align daily market cap weights to monthly rebalance schedule.
    
    Args:
        market_cap_weights: Daily market cap weights
        rebalance_dates: Monthly rebalancing dates
        universe: Current optimization universe
        
    Returns:
        Dictionary mapping rebalance dates to aligned weights
    """
    try:
        aligned_weights = {}
        
        for rebalance_date in rebalance_dates:
            # Find closest date with available weights (forward-fill)
            available_dates = [d for d in market_cap_weights.keys() if d <= rebalance_date]
            
            if not available_dates:
                logger.warning(f"No market cap weights available for {rebalance_date}")
                continue
            
            closest_date = max(available_dates)
            weights = market_cap_weights[closest_date].copy()
            
            # Map coin_ids to symbols and filter to universe
            universe_weights = pd.Series(0.0, index=universe)
            
            for coin_id, weight in weights.items():
                # Try to find matching symbol in universe
                for symbol in universe:
                    variations = get_crypto_symbol_variations(symbol)
                    if coin_id in variations:
                        universe_weights[symbol] = weight
                        break
            
            # Normalize to sum to 1 within universe
            if universe_weights.sum() > 0:
                universe_weights = universe_weights / universe_weights.sum()
            else:
                # Equal weight fallback
                universe_weights = pd.Series(1.0 / len(universe), index=universe)
                logger.warning(f"No market cap weights found for universe on {rebalance_date}, using equal weights")
            
            aligned_weights[rebalance_date] = universe_weights
        
        logger.info(f"Aligned market cap weights to {len(aligned_weights)} rebalance dates")
        return aligned_weights
        
    except Exception as e:
        logger.error(f"Error aligning market cap weights: {e}")
        return {}


# Returns storage operations
async def store_daily_returns(
    session: AsyncSession,
    run_id: str,
    returns_dict: Dict[str, Dict[date, float]]
) -> bool:
    """
    Store daily portfolio returns for all strategies.
    
    Args:
        session: Database session
        run_id: Unique run identifier
        returns_dict: Nested dict {strategy: {date: return}}
        
    Returns:
        bool: Success status
    """
    try:
        return_data = []
        
        for strategy, daily_returns in returns_dict.items():
            for date, portfolio_return in daily_returns.items():
                if pd.notna(portfolio_return):
                    return_data.append({
                        'run_id': run_id,
                        'date': date,
                        'portfolio_return': float(portfolio_return),
                        'strategy': strategy
                    })
        
        if not return_data:
            logger.warning(f"No valid returns to store for run_id: {run_id}")
            return False
        
        # Batch insert with conflict handling
        stmt = insert(ReturnsDaily).values(return_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['run_id', 'date', 'strategy'],
            set_=dict(
                portfolio_return=stmt.excluded.portfolio_return,
                updated_at=datetime.utcnow()
            )
        )
        
        await session.execute(stmt)
        
        logger.info(f"Stored {len(return_data)} daily return records")
        return True
        
    except Exception as e:
        logger.error(f"Error storing daily returns: {e}")
        return False


async def load_daily_returns(
    session: AsyncSession,
    run_id: str
) -> Dict[str, pd.Series]:
    """
    Load daily portfolio returns for all strategies.
    
    Args:
        session: Database session
        run_id: Unique run identifier
        
    Returns:
        Dictionary mapping strategy to return series
    """
    try:
        query = text("""
            SELECT date, strategy, portfolio_return
            FROM portfolio_simulations.returns_daily
            WHERE run_id = :run_id
            ORDER BY date, strategy
        """)
        
        result = await session.execute(query, {"run_id": run_id})
        
        returns_by_strategy = {}
        for row in result.fetchall():
            date = row[0]
            strategy = row[1]
            portfolio_return = float(row[2])
            
            if strategy not in returns_by_strategy:
                returns_by_strategy[strategy] = {}
            
            returns_by_strategy[strategy][date] = portfolio_return
        
        # Convert to pandas Series
        strategy_returns = {}
        for strategy, returns_dict in returns_by_strategy.items():
            returns_series = pd.Series(returns_dict)
            returns_series.index = pd.to_datetime(returns_series.index)
            returns_series = returns_series.sort_index()
            strategy_returns[strategy] = returns_series
        
        logger.info(f"Loaded returns for {len(strategy_returns)} strategies")
        return strategy_returns
        
    except Exception as e:
        logger.error(f"Error loading daily returns: {e}")
        return {}
