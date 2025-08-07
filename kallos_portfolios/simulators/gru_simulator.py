"""
GRU Portfolio Simulator

Portfolio simulator that uses GRU neural network predictions for expected returns.
Inherits shared functionality from BasePortfolioSimulator.
"""

import logging
from datetime import date
from typing import List, Optional, Dict
import pandas as pd

from .base_simulator import BasePortfolioSimulator

logger = logging.getLogger(__name__)


class GRUPortfolioSimulator(BasePortfolioSimulator):
    """
    Portfolio simulator using GRU model return forecasts.
    
    Uses pre-computed GRU predictions for portfolio optimization with configurable objectives.
    """
    
    def __init__(self, universe_df: pd.DataFrame, rebalancing_dates: List[date], 
                 predictions_df: pd.DataFrame, objective: str = 'min_volatility'):
        """
        Initialize GRU portfolio simulator.
        
        Args:
            universe_df: DataFrame with columns ['date', 'coin_id'] defining investable universe
            rebalancing_dates: List of dates when portfolio should be rebalanced
            predictions_df: DataFrame with columns ['date', 'coin_id', 'estimate'] containing GRU predictions
            objective: Portfolio optimization objective ('min_volatility', 'max_sharpe', 'max_expected_returns')
        """
        super().__init__(universe_df, rebalancing_dates, objective)
        
        self.predictions_df = predictions_df.copy()
        
        # Ensure predictions dates are proper date objects
        if not pd.api.types.is_datetime64_any_dtype(self.predictions_df['date']):
            self.predictions_df['date'] = pd.to_datetime(self.predictions_df['date']).dt.date
            
        # Remove any duplicate predictions (keep first occurrence)
        initial_count = len(self.predictions_df)
        self.predictions_df = self.predictions_df.drop_duplicates(subset=['date', 'coin_id'], keep='first')
        removed_count = initial_count - len(self.predictions_df)
        
        if removed_count > 0:
            logger.warning(f"⚠️  Removed {removed_count} duplicate predictions from dataset")
            
        logger.info(f"📊 GRU predictions loaded: {len(self.predictions_df)} entries for "
                   f"{self.predictions_df['coin_id'].nunique()} assets")
        logger.info(f"📅 Prediction dates: {self.predictions_df['date'].min()} to {self.predictions_df['date'].max()}")
    
    def get_predictions_for_date(self, target_date: date, assets: List[str]) -> pd.Series:
        """
        Get GRU predictions for specific date and assets.
        
        Args:
            target_date: Date for which to get predictions
            assets: List of asset identifiers to get predictions for
            
        Returns:
            Series of expected returns indexed by asset identifier
        """
        # First try exact date match
        predictions = self.predictions_df[
            (self.predictions_df['date'] == target_date) & 
            (self.predictions_df['coin_id'].isin(assets))
        ]
        
        # If no exact match, find the closest date within +/- 7 days
        if predictions.empty:
            available_dates = sorted(self.predictions_df['date'].unique())
            closest_date = None
            min_diff = float('inf')
            
            for pred_date in available_dates:
                diff = abs((pred_date - target_date).days)
                if diff <= 7 and diff < min_diff:  # Within 7 days
                    min_diff = diff
                    closest_date = pred_date
            
            if closest_date is not None:
                predictions = self.predictions_df[
                    (self.predictions_df['date'] == closest_date) & 
                    (self.predictions_df['coin_id'].isin(assets))
                ]
                logger.debug(f"Using predictions from {closest_date} for target date {target_date}")
        
        if predictions.empty:
            logger.warning(f"⚠️  No GRU predictions found for {target_date} (checked {len(assets)} assets)")
            return pd.Series(dtype=float)
        
        # Remove any remaining duplicates (keep first occurrence)
        predictions = predictions.drop_duplicates(subset=['coin_id'], keep='first')
        logger.debug(f"Found GRU predictions for {len(predictions)} assets on {target_date}")
        
        # Convert to Series indexed by coin_id  
        expected_returns = predictions.set_index('coin_id')['estimate']
        
        return expected_returns
    
    async def get_expected_returns_for_date(self, assets: List[str], target_date: date) -> Optional[pd.Series]:
        """
        Implementation of base class method using GRU predictions.
        
        Args:
            assets: List of asset identifiers
            target_date: Date for which to get expected returns
            
        Returns:
            Series with GRU-based expected returns indexed by asset, or None if no predictions available
        """
        expected_returns = self.get_predictions_for_date(target_date, assets)
        
        if expected_returns.empty:
            return None
            
        # Filter out any invalid predictions (NaN, inf, etc.)
        expected_returns = expected_returns.dropna()
        expected_returns = expected_returns[expected_returns.notna()]
        expected_returns = expected_returns[expected_returns != 0]  # Remove zero predictions
        
        if expected_returns.empty:
            logger.warning(f"⚠️  All GRU predictions were invalid for {target_date}")
            return None
            
        logger.debug(f"📊 GRU expected returns for {target_date}: "
                    f"{len(expected_returns)} assets, range: {expected_returns.min():.4f} to {expected_returns.max():.4f}")
        
        return expected_returns


def load_gru_data_from_csv(csv_dir: str = "/home/jlmarquez11/kallos") -> tuple:
    """
    Load GRU-specific data from CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        
    Returns:
        Tuple of (universe_df, rebalancing_dates, predictions_df)
    """
    import pandas as pd
    from pathlib import Path
    
    logger.info(f"📂 Loading GRU data from {csv_dir}")
    
    try:
        # Load universe data
        universe_file = Path(csv_dir) / "universe_2022.csv"
        universe_df = pd.read_csv(universe_file)
        universe_df['date'] = pd.to_datetime(universe_df['date']).dt.date
        
        # Load rebalancing dates
        rebalancing_file = Path(csv_dir) / "rebalancing_predictions.csv"
        rebalancing_df = pd.read_csv(rebalancing_file)
        rebalancing_dates = pd.to_datetime(rebalancing_df['date']).dt.date.tolist()
        
        # Load GRU predictions
        predictions_file = Path(csv_dir) / "rebalancing_predictions.csv"
        predictions_df = pd.read_csv(predictions_file)
        predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
        
        logger.info(f"✅ Loaded GRU data:")
        logger.info(f"   📊 Universe: {len(universe_df)} entries, {universe_df['coin_id'].nunique()} unique assets")
        logger.info(f"   📅 Rebalancing dates: {len(rebalancing_dates)} dates")
        logger.info(f"   🧠 GRU predictions: {len(predictions_df)} entries")
        
        return universe_df, rebalancing_dates, predictions_df
        
    except Exception as e:
        logger.error(f"❌ Error loading GRU data: {e}")
        raise


async def run_gru_simulation(objective: str = 'min_volatility', csv_dir: str = "/home/jlmarquez11/kallos"):
    """
    Run complete GRU portfolio simulation.
    
    Args:
        objective: Optimization objective ('min_volatility', 'max_sharpe', 'max_expected_returns')
        csv_dir: Directory containing CSV data files
        
    Returns:
        Run ID of the completed simulation
    """
    logger.info(f"🚀 Starting GRU portfolio simulation with {objective} objective")
    
    try:
        # Load data
        universe_df, rebalancing_dates, predictions_df = load_gru_data_from_csv(csv_dir)
        
        # Create and run simulator
        simulator = GRUPortfolioSimulator(
            universe_df=universe_df,
            rebalancing_dates=rebalancing_dates,
            predictions_df=predictions_df,
            objective=objective
        )
        
        # Run complete backtest
        run_id = await simulator.run_complete_backtest()
        
        logger.info(f"✅ GRU simulation completed: {run_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"❌ GRU simulation failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    import sys
    
    # Get objective from command line or use default
    objective = sys.argv[1] if len(sys.argv) > 1 else 'min_volatility'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run simulation
    asyncio.run(run_gru_simulation(objective))
