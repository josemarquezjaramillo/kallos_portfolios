"""
Historical Portfolio Simulator

Portfolio simulator that uses naive historical returns as benchmark.
Inherits shared functionality from BasePortfolioSimulator.
"""

import logging
from datetime import date, timedelta
from typing import List, Optional
import pandas as pd

from .base_simulator import BasePortfolioSimulator
from ..storage import get_async_session, load_prices_for_period

logger = logging.getLogger(__name__)


class HistoricalPortfolioSimulator(BasePortfolioSimulator):
    """
    Portfolio simulator using naive historical expected returns as benchmark.
    
    Calculates expected returns using historical mean returns over a lookback period.
    Provides a direct comparison benchmark to evaluate the value-add of ML forecasts.
    """
    
    def __init__(self, universe_df: pd.DataFrame, rebalancing_dates: List[date], 
                 objective: str = 'min_volatility', lookback_days: int = 252):
        """
        Initialize historical portfolio simulator.
        
        Args:
            universe_df: DataFrame with columns ['date', 'coin_id'] defining investable universe
            rebalancing_dates: List of dates when portfolio should be rebalanced
            objective: Portfolio optimization objective ('min_volatility', 'max_sharpe', 'max_expected_returns')
            lookback_days: Number of days to look back for historical return calculation (default: 252 = 1 year)
        """
        super().__init__(universe_df, rebalancing_dates, objective)
        
        self.lookback_days = lookback_days
        
        logger.info(f"📊 Historical simulator initialized with {lookback_days}-day lookback period")
    
    async def calculate_historical_expected_returns(self, assets: List[str], target_date: date) -> Optional[pd.Series]:
        """
        Calculate naive historical expected returns using pypfopt.
        
        Args:
            assets: List of assets to calculate returns for
            target_date: Date for which to calculate expected returns
            
        Returns:
            Series of expected returns or None if calculation fails
        """
        try:
            from pypfopt import expected_returns
            
            # Calculate lookback period
            end_date = target_date - timedelta(days=1)  # Day before rebalancing
            start_date = end_date - timedelta(days=self.lookback_days)
            
            logger.debug(f"📈 Loading price data for {len(assets)} assets from {start_date} to {end_date}")
            
            # Load historical price data
            async with get_async_session() as session:
                price_data = await load_prices_for_period(
                    session=session,
                    coin_ids=assets,
                    start_date=start_date,
                    end_date=end_date,
                    min_data_coverage=0.5  # Allow some missing data
                )
            
            if price_data.empty:
                logger.warning(f"⚠️  No price data available for {target_date}")
                return None
            
            logger.debug(f"✅ Loaded price data: {price_data.shape} for expected returns calculation")
            
            # Calculate historical mean returns using pypfopt
            annual_expected_returns = expected_returns.mean_historical_return(
                price_data,
                frequency=252,  # Daily data, 252 trading days per year
                compounding=True
            )
            
            # Convert annual to weekly returns for consistency with GRU predictions
            weekly_expected_returns = (1 + annual_expected_returns) ** (1/52) - 1
            
            # Filter to only assets that have data
            valid_returns = weekly_expected_returns.dropna()
            
            if valid_returns.empty:
                logger.warning(f"⚠️  No valid expected returns calculated for {target_date}")
                return None
            
            logger.debug(f"📊 Calculated historical expected returns for {len(valid_returns)} assets")
            logger.debug(f"   Mean return: {valid_returns.mean():.4f} weekly ({valid_returns.mean()*52:.2%} annualized)")
            logger.debug(f"   Return range: {valid_returns.min():.4f} to {valid_returns.max():.4f} weekly")
            
            return valid_returns
            
        except Exception as e:
            logger.error(f"❌ Historical expected returns calculation failed for {target_date}: {e}")
            return None
    
    async def get_expected_returns_for_date(self, assets: List[str], target_date: date) -> Optional[pd.Series]:
        """
        Implementation of base class method using historical returns.
        
        Args:
            assets: List of asset identifiers
            target_date: Date for which to get expected returns
            
        Returns:
            Series with historically-based expected returns indexed by asset, or None if calculation fails
        """
        expected_returns = await self.calculate_historical_expected_returns(assets, target_date)
        
        if expected_returns is None or expected_returns.empty:
            return None
            
        # Filter out any invalid returns (NaN, inf, etc.)
        expected_returns = expected_returns.dropna()
        expected_returns = expected_returns[expected_returns.notna()]
        expected_returns = expected_returns[expected_returns != 0]  # Remove zero returns
        
        if expected_returns.empty:
            logger.warning(f"⚠️  All historical expected returns were invalid for {target_date}")
            return None
            
        logger.debug(f"📊 Historical expected returns for {target_date}: "
                    f"{len(expected_returns)} assets, range: {expected_returns.min():.4f} to {expected_returns.max():.4f}")
        
        return expected_returns


def load_historical_data_from_csv(csv_dir: str = "/home/jlmarquez11/kallos") -> tuple:
    """
    Load data required for historical simulation from CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        
    Returns:
        Tuple of (universe_df, rebalancing_dates)
    """
    import pandas as pd
    from pathlib import Path
    
    logger.info(f"📂 Loading historical simulation data from {csv_dir}")
    
    try:
        # Load universe data
        universe_file = Path(csv_dir) / "universe_2022.csv"
        universe_df = pd.read_csv(universe_file)
        universe_df['date'] = pd.to_datetime(universe_df['date']).dt.date
        
        # Load rebalancing dates (same as GRU for fair comparison)
        rebalancing_file = Path(csv_dir) / "rebalancing_predictions.csv"
        rebalancing_df = pd.read_csv(rebalancing_file)
        rebalancing_dates = pd.to_datetime(rebalancing_df['date']).dt.date.unique().tolist()
        rebalancing_dates = sorted(rebalancing_dates)
        
        logger.info(f"✅ Loaded historical simulation data:")
        logger.info(f"   📊 Universe: {len(universe_df)} entries, {universe_df['coin_id'].nunique()} unique assets")
        logger.info(f"   📅 Rebalancing dates: {len(rebalancing_dates)} dates")
        
        return universe_df, rebalancing_dates
        
    except Exception as e:
        logger.error(f"❌ Error loading historical simulation data: {e}")
        raise


async def run_historical_simulation(objective: str = 'min_volatility', lookback_days: int = 252, 
                                   csv_dir: str = "/home/jlmarquez11/kallos"):
    """
    Run complete historical portfolio simulation.
    
    Args:
        objective: Optimization objective ('min_volatility', 'max_sharpe', 'max_expected_returns')
        lookback_days: Number of days to look back for historical return calculation
        csv_dir: Directory containing CSV data files
        
    Returns:
        Run ID of the completed simulation
    """
    logger.info(f"🚀 Starting historical portfolio simulation with {objective} objective")
    logger.info(f"📊 Using {lookback_days}-day lookback period")
    
    try:
        # Load data
        universe_df, rebalancing_dates = load_historical_data_from_csv(csv_dir)
        
        # Create and run simulator
        simulator = HistoricalPortfolioSimulator(
            universe_df=universe_df,
            rebalancing_dates=rebalancing_dates,
            objective=objective,
            lookback_days=lookback_days
        )
        
        # Run complete backtest
        run_id = await simulator.run_complete_backtest()
        
        logger.info(f"✅ Historical simulation completed: {run_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"❌ Historical simulation failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    import sys
    
    # Get parameters from command line or use defaults
    objective = sys.argv[1] if len(sys.argv) > 1 else 'min_volatility'
    lookback_days = int(sys.argv[2]) if len(sys.argv) > 2 else 252
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run simulation
    asyncio.run(run_historical_simulation(objective, lookback_days))
