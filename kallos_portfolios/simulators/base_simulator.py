"""
Base Portfolio Simulator

Contains shared functionality for all portfolio simulators to eliminate code duplication.
Subclasses only need to implement return forecasting logic.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt

from ..optimisers import PortfolioOptimizer, OptimizationParams
from ..storage import (
    get_async_session, save_portfolio_run, save_portfolio_weights,
    save_portfolio_returns, load_prices_for_period
)
from ..evaluation import generate_quantstats_reports

logger = logging.getLogger(__name__)


class BasePortfolioSimulator:
    """
    Base class for portfolio simulators with shared functionality.
    
    Subclasses must implement:
    - get_expected_returns_for_date(): Return forecasting logic
    """
    
    def __init__(self, universe_df: pd.DataFrame, rebalancing_dates: List[date], 
                 objective: str = 'min_volatility'):
        """
        Initialize simulator with universe and rebalancing schedule.
        
        Args:
            universe_df: DataFrame with columns ['date', 'coin_id'] defining investable universe
            rebalancing_dates: List of dates when portfolio should be rebalanced
            objective: Portfolio optimization objective ('min_volatility', 'max_sharpe', 'max_expected_returns')
        """
        self.universe_df = universe_df.copy()
        self.rebalancing_dates = sorted(rebalancing_dates)
        self.objective = objective
        
        # Ensure universe dates are datetime
        if not pd.api.types.is_datetime64_any_dtype(self.universe_df['date']):
            self.universe_df['date'] = pd.to_datetime(self.universe_df['date']).dt.date
            
        logger.info(f"🔧 Initialized {self.__class__.__name__}")
        logger.info(f"📅 Rebalancing dates: {len(self.rebalancing_dates)} ({self.rebalancing_dates[0]} to {self.rebalancing_dates[-1]})")
        logger.info(f"🎯 Optimization objective: {self.objective}")
        logger.info(f"🌐 Universe: {self.universe_df['coin_id'].nunique()} unique assets")
    
    def get_assets_for_date(self, target_date: date) -> List[str]:
        """
        Get available assets for a specific date from universe.
        Matches by year and month since universe is typically monthly.
        
        Args:
            target_date: Date for which to get available assets
            
        Returns:
            List of asset identifiers available for the given date
        """
        try:
            # Match by year and month 
            target_year = target_date.year
            target_month = target_date.month
            
            # Find universe entries matching the target year/month
            universe_subset = self.universe_df[
                (pd.to_datetime(self.universe_df['date']).dt.year == target_year) &
                (pd.to_datetime(self.universe_df['date']).dt.month == target_month)
            ]
            
            if universe_subset.empty:
                logger.warning(f"⚠️  No universe found for {target_date} ({target_year}-{target_month:02d})")
                return []
            
            assets = universe_subset['coin_id'].tolist()
            logger.debug(f"📋 {target_date}: {len(assets)} assets in universe")
            return assets
            
        except Exception as e:
            logger.error(f"❌ Error getting assets for {target_date}: {e}")
            return []
    
    async def get_expected_returns_for_date(self, assets: List[str], target_date: date) -> Optional[pd.Series]:
        """
        Get expected returns for assets on a given date.
        This is the main method that subclasses must implement.
        
        Args:
            assets: List of asset identifiers
            target_date: Date for which to get expected returns
            
        Returns:
            Series with expected returns indexed by asset, or None if unable to generate
        """
        raise NotImplementedError("Subclasses must implement get_expected_returns_for_date()")
    
    async def optimize_portfolio(self, expected_returns: pd.Series, target_date: date) -> Optional[pd.Series]:
        """
        Optimize portfolio weights given expected returns.
        
        Args:
            expected_returns: Series of expected returns indexed by asset
            target_date: Date for the optimization (used for logging)
            
        Returns:
            Series of optimal weights indexed by asset, or None if optimization fails
        """
        if expected_returns.empty or expected_returns.isna().all():
            logger.warning(f"⚠️  No valid expected returns for {target_date}")
            return None
        
        try:
            # Load price data for covariance estimation
            start_date = target_date - timedelta(days=365)  # 1 year lookback
            
            async with get_async_session() as session:
                price_data = await load_prices_for_period(
                    session=session,
                    coin_ids=expected_returns.index.tolist(),
                    start_date=start_date,
                    end_date=target_date,
                    min_data_coverage=0.3  # Require 30% data coverage
                )
            
            if price_data.empty:
                logger.warning(f"⚠️  No price data for covariance estimation on {target_date}")
                return None
            
            # Calculate returns for covariance matrix
            returns_data = price_data.pct_change().dropna()
            
            if returns_data.empty or len(returns_data) < 30:  # Need at least 30 observations
                logger.warning(f"⚠️  Insufficient return data for {target_date}")
                return None
            
            # Align expected returns with available price data
            common_assets = expected_returns.index.intersection(returns_data.columns)
            if len(common_assets) < 2:
                logger.warning(f"⚠️  Not enough common assets for optimization on {target_date}")
                return None
            
            expected_returns_aligned = expected_returns.loc[common_assets]
            returns_for_cov = returns_data[common_assets]
            
            # Create optimization parameters
            params = OptimizationParams(
                expected_returns=expected_returns_aligned,
                returns_data=returns_for_cov,
                objective=self.objective,
                max_weight=0.4,  # Maximum 40% allocation to any single asset
                min_weight=0.01   # Minimum 1% allocation if included
            )
            
            # Run optimization
            optimizer = PortfolioOptimizer(params)
            result = optimizer.optimize()
            
            if result is None or result.empty:
                logger.warning(f"⚠️  Portfolio optimization failed for {target_date}")
                return None
            
            # Normalize weights to sum to 1
            weights = result / result.sum()
            
            logger.debug(f"✅ Optimized portfolio for {target_date}: {len(weights)} assets, "
                        f"max weight: {weights.max():.1%}, min weight: {weights.min():.1%}")
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ Portfolio optimization error for {target_date}: {e}")
            return None
    
    async def generate_weekly_portfolios(self) -> Dict[date, pd.Series]:
        """
        Generate optimized portfolios for each rebalancing date.
        This method orchestrates the return forecasting and optimization process.
        
        Returns:
            Dictionary mapping rebalancing dates to portfolio weight series
        """
        logger.info(f"🔄 Generating portfolios for {len(self.rebalancing_dates)} rebalancing dates...")
        
        weekly_weights = {}
        successful_optimizations = 0
        
        for i, rebal_date in enumerate(self.rebalancing_dates, 1):
            logger.info(f"📊 Processing {rebal_date} ({i}/{len(self.rebalancing_dates)})...")
            
            try:
                # Get available assets for this date
                assets = self.get_assets_for_date(rebal_date)
                if not assets:
                    logger.warning(f"⚠️  No assets available for {rebal_date}")
                    continue
                
                # Get expected returns (this is method that subclasses implement)
                expected_returns = await self.get_expected_returns_for_date(assets, rebal_date)
                if expected_returns is None or expected_returns.empty:
                    logger.warning(f"⚠️  No expected returns available for {rebal_date}")
                    continue
                
                # Optimize portfolio
                weights = await self.optimize_portfolio(expected_returns, rebal_date)
                if weights is not None and not weights.empty:
                    weekly_weights[rebal_date] = weights
                    successful_optimizations += 1
                    logger.info(f"✅ Portfolio optimized for {rebal_date}: {len(weights)} assets")
                
            except Exception as e:
                logger.error(f"❌ Error processing {rebal_date}: {e}")
                continue
        
        logger.info(f"🎯 Generated {successful_optimizations}/{len(self.rebalancing_dates)} portfolios successfully")
        return weekly_weights
    
    async def simulate_performance(self, weekly_weights: Dict[date, pd.Series]) -> Dict:
        """
        Simulate portfolio performance using VectorBT with realistic rebalancing.
        
        Args:
            weekly_weights: Dictionary mapping rebalancing dates to portfolio weights
            
        Returns:
            Dictionary containing performance results and metrics
        """
        logger.info("🎯 Simulating portfolio performance with VectorBT...")
        
        if not weekly_weights:
            raise ValueError("No portfolio weights provided for simulation")
        
        # Get all assets used across all portfolios
        all_assets = set()
        for weights in weekly_weights.values():
            all_assets.update(weights.index)
        all_assets = sorted(list(all_assets))
        
        # Define simulation period with buffers
        start_date = min(self.rebalancing_dates) - timedelta(days=7)
        end_date = max(self.rebalancing_dates) + timedelta(days=7)
        
        logger.info(f"📈 Loading price data for {len(all_assets)} assets...")
        
        # Load price data for simulation
        async with get_async_session() as session:
            price_data = await load_prices_for_period(
                session=session,
                coin_ids=all_assets,
                start_date=start_date,
                end_date=end_date,
                min_data_coverage=0.7
            )
        
        if price_data.empty:
            raise ValueError("No price data available for simulation")
        
        logger.info(f"📊 Loaded price data: {price_data.shape} (dates × assets)")
        
        # Create rebalancing signals and weights DataFrames
        rebalance_signals = pd.DataFrame(False, index=price_data.index, columns=all_assets)
        weights_df = pd.DataFrame(0.0, index=price_data.index, columns=all_assets)
        
        # Set rebalancing signals and weights
        for rebal_date, weights in weekly_weights.items():
            if rebal_date in price_data.index:
                rebalance_signals.loc[rebal_date, :] = True
                for asset, weight in weights.items():
                    if asset in weights_df.columns:
                        weights_df.loc[rebal_date, asset] = weight
                
                logger.debug(f"📅 {rebal_date}: Set {len(weights)} weights")
        
        # Run VectorBT simulation
        logger.info("🚀 Running VectorBT portfolio simulation...")
        
        portfolio = vbt.Portfolio.from_orders(
            price_data,
            size=weights_df,
            size_type='targetpercent',  # Target percentage allocation
            group_by=True,              # Treat as single portfolio
            cash_sharing=True,          # Share cash across all assets
            call_seq='auto',            # Automatic call sequence
            freq='1D'                   # Daily frequency
        )
        
        # Calculate performance metrics
        portfolio_value = portfolio.value()
        portfolio_returns = portfolio.returns()
        
        # Remove any invalid returns (inf, -inf, NaN)
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if portfolio_returns.empty:
            raise ValueError("No valid portfolio returns generated")
        
        # Calculate key metrics
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'portfolio_returns': portfolio_returns,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'start_value': float(portfolio_value.iloc[0]),
            'final_value': float(portfolio_value.iloc[-1]),
            'total_periods': len(portfolio_returns),
            'rebalancing_dates': list(weekly_weights.keys())
        }
        
        logger.info(f"📈 Simulation completed: {total_return:.2%} total return, "
                   f"{sharpe_ratio:.2f} Sharpe ratio, {max_drawdown:.2%} max drawdown")
        
        return results
    
    async def save_results(self, weekly_weights: Dict[date, pd.Series], 
                          performance_results: Dict) -> str:
        """
        Save portfolio run results to database.
        
        Args:
            weekly_weights: Portfolio weights for each rebalancing date
            performance_results: Performance metrics from simulation
            
        Returns:
            Run ID of the saved portfolio run
        """
        logger.info("💾 Saving portfolio results to database...")
        
        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.__class__.__name__.replace('Simulator', '').replace('Portfolio', '')
        run_id = f"{strategy_name}_{self.objective.title().replace('_', '_')}_{timestamp}"
        
        try:
            async with get_async_session() as session:
                # Save main portfolio run
                portfolio_run_id = await save_portfolio_run(
                    session=session,
                    run_id=run_id,
                    start_date=min(self.rebalancing_dates),
                    end_date=max(self.rebalancing_dates),
                    strategy_name=f"{strategy_name} {self.objective.replace('_', ' ').title()}",
                    assets=list(set().union(*[weights.index for weights in weekly_weights.values()])),
                    final_value=performance_results['final_value'],
                    total_return=performance_results['total_return'],
                    sharpe_ratio=performance_results['sharpe_ratio'],
                    volatility=performance_results['volatility'],
                    max_drawdown=performance_results['max_drawdown']
                )
                
                # Save portfolio weights for each rebalancing date
                for rebal_date, weights in weekly_weights.items():
                    await save_portfolio_weights(
                        session=session,
                        portfolio_run_id=portfolio_run_id,
                        rebalance_date=rebal_date,
                        weights=weights.to_dict()
                    )
                
                # Save daily portfolio returns
                returns_dict = {}
                for date_idx, return_val in performance_results['portfolio_returns'].items():
                    if pd.notna(return_val) and np.isfinite(return_val):
                        returns_dict[date_idx.date() if hasattr(date_idx, 'date') else date_idx] = float(return_val)
                
                if returns_dict:
                    await save_portfolio_returns(
                        session=session,
                        portfolio_run_id=portfolio_run_id,
                        returns=returns_dict
                    )
                
                logger.info(f"✅ Saved portfolio run: {run_id} (ID: {portfolio_run_id})")
                logger.info(f"📊 Saved {len(weekly_weights)} weight sets and {len(returns_dict)} daily returns")
                
                return run_id
                
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            raise
    
    def generate_reports(self, performance_results: Dict, run_id: str):
        """
        Generate QuantStats tearsheet and performance reports.
        
        Args:
            performance_results: Performance metrics and returns from simulation
            run_id: Unique identifier for this portfolio run
        """
        logger.info("📊 Generating performance reports...")
        
        try:
            # Create reports directory
            reports_dir = Path("reports") / f"portfolio_reports_{datetime.now().strftime('%Y%m%d')}"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate QuantStats tearsheet
            returns_series = performance_results['portfolio_returns']
            strategy_name = self.__class__.__name__.replace('Simulator', '').replace('Portfolio', '')
            
            tearsheet_file = generate_quantstats_reports(
                returns=returns_series,
                strategy_name=f"{strategy_name}_{self.objective}",
                output_dir=reports_dir
            )
            
            if tearsheet_file and tearsheet_file.exists():
                logger.info(f"📈 Generated tearsheet: {tearsheet_file}")
            else:
                logger.warning("⚠️  Failed to generate tearsheet")
                
        except Exception as e:
            logger.error(f"❌ Error generating reports: {e}")
    
    async def run_complete_backtest(self) -> str:
        """
        Run the complete backtesting workflow.
        
        Returns:
            Run ID of the completed backtest
        """
        logger.info(f"🚀 Starting complete backtest with {self.__class__.__name__}")
        logger.info(f"🎯 Objective: {self.objective}")
        logger.info(f"📅 Period: {self.rebalancing_dates[0]} to {self.rebalancing_dates[-1]}")
        
        try:
            # Generate weekly portfolios
            weekly_weights = await self.generate_weekly_portfolios()
            
            if not weekly_weights:
                raise ValueError("No successful portfolio optimizations generated")
            
            # Simulate performance
            performance_results = await self.simulate_performance(weekly_weights)
            
            # Save results to database
            run_id = await self.save_results(weekly_weights, performance_results)
            
            # Generate reports
            self.generate_reports(performance_results, run_id)
            
            logger.info(f"✅ Backtest completed successfully: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise
