"""
Main workflow orchestration for Kallos Portfolios system.
Implements complete three-strategy portfolio analysis pipeline.
"""

import logging
import logging.config
import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import yaml

from .config.settings import settings, OptimizationParams, create_run_id
from .storage import (
    get_async_session, store_optimization_params, load_optimization_params,
    fetch_monthly_universe, load_prices_for_period, upsert_weights,
    load_market_cap_weights, align_market_cap_weights_to_rebalance_dates,
    store_daily_returns, load_daily_returns
)
from .datasets import (
    generate_monthly_rebalance_dates, prepare_full_dataset_for_backtest,
    clean_price_data
)
from .models import forecast_returns, validate_model_availability
from .optimisers import (
    optimize_portfolio_gru, optimize_portfolio_historical, 
    create_market_cap_weights, force_sell_non_universe,
    validate_portfolio_weights
)
from .backtest import run_three_strategy_backtest, validate_backtest_data
from .evaluation import run_complete_evaluation

# Setup logging
def setup_logging():
    """Setup logging configuration from YAML file."""
    logging_config_path = Path(__file__).parent / "config" / "logging.yaml"
    
    if logging_config_path.exists():
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Fallback logging configuration
        logging.basicConfig(
            level=getattr(logging, settings.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

setup_logging()
logger = logging.getLogger(__name__)


class KallosPortfolioRunner:
    """
    Main orchestrator for three-strategy portfolio analysis.
    
    Implements complete workflow:
    1. Setup & Data Loading
    2. Monthly Universe & Optimization Loop
    3. Backtesting & Performance Analysis
    4. Statistical Analysis & Reporting
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize portfolio runner.
        
        Args:
            database_url: Optional database URL override
        """
        self.database_url = database_url or settings.database_url
        self.model_dir = settings.model_dir
        self.report_path = settings.report_path
        
        # Ensure directories exist
        self.model_dir.mkdir(exist_ok=True)
        self.report_path.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Kallos Portfolio Runner")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Report directory: {self.report_path}")
    
    async def run_portfolio_analysis(
        self, 
        run_id: str,
        params: OptimizationParams
    ) -> Dict[str, Any]:
        """
        Execute complete three-strategy portfolio analysis.
        
        Args:
            run_id: Unique run identifier
            params: Optimization parameters
            
        Returns:
            Complete analysis results dictionary
        """
        logger.info(f"Starting portfolio analysis for run_id: {run_id}")
        
        try:
            # Phase 1: Setup & Data Loading
            results = await self._phase1_setup(run_id, params)
            if 'error' in results:
                return results
            
            rebalance_dates = results['rebalance_dates']
            
            # Phase 2: Monthly Universe & Optimization Loop
            strategy_weights = await self._phase2_optimization_loop(run_id, params, rebalance_dates)
            if not strategy_weights:
                return {'error': 'No strategy weights generated'}
            
            # Phase 3: Backtesting & Performance Analysis
            strategy_returns = await self._phase3_backtesting(run_id, params, strategy_weights, rebalance_dates)
            if not strategy_returns:
                return {'error': 'Backtesting failed'}
            
            # Phase 4: Statistical Analysis & Reporting
            evaluation_results = await self._phase4_evaluation(strategy_returns, run_id)
            
            # Combine all results
            final_results = {
                'run_id': run_id,
                'parameters': params.dict(),
                'rebalance_dates': rebalance_dates,
                'strategy_weights': strategy_weights,
                'strategy_returns': strategy_returns,
                'evaluation_results': evaluation_results,
                'status': 'completed'
            }
            
            logger.info(f"Portfolio analysis completed successfully for run_id: {run_id}")
            return final_results
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed for run_id {run_id}: {e}")
            return {'error': str(e), 'run_id': run_id}
    
    async def _phase1_setup(self, run_id: str, params: OptimizationParams) -> Dict[str, Any]:
        """
        Phase 1: Setup & Data Loading
        - Store optimization parameters
        - Generate monthly rebalancing dates
        - Validate model availability
        """
        logger.info("Phase 1: Setup & Data Loading")
        
        try:
            async with get_async_session(self.database_url) as session:
                # Store optimization parameters
                success = await store_optimization_params(session, run_id, params)
                if not success:
                    return {'error': 'Failed to store optimization parameters'}
                
                # Generate monthly rebalancing dates
                start_date = datetime.strptime(params.start_date, '%Y-%m-%d').date()
                end_date = datetime.strptime(params.end_date, '%Y-%m-%d').date()
                
                rebalance_dates = generate_monthly_rebalance_dates(start_date, end_date)
                
                if not rebalance_dates:
                    return {'error': 'No rebalancing dates generated'}
                
                logger.info(f"Generated {len(rebalance_dates)} rebalancing dates from {start_date} to {end_date}")
                
                # Validate model availability for first universe
                first_universe = await fetch_monthly_universe(session, rebalance_dates[0])
                if first_universe:
                    model_availability = validate_model_availability(self.model_dir, first_universe)
                    available_models = sum(model_availability.values())
                    logger.info(f"Model availability: {available_models}/{len(first_universe)} models available")
                
                return {
                    'rebalance_dates': rebalance_dates,
                    'start_date': start_date,
                    'end_date': end_date
                }
                
        except Exception as e:
            logger.error(f"Phase 1 setup failed: {e}")
            return {'error': f'Phase 1 setup failed: {e}'}
    
    async def _phase2_optimization_loop(
        self, 
        run_id: str, 
        params: OptimizationParams, 
        rebalance_dates: List[date]
    ) -> Dict[str, Dict[date, pd.Series]]:
        """
        Phase 2: Monthly Universe & Optimization Loop
        - Load monthly universe for each rebalancing date
        - Generate GRU forecasts and optimize all three strategies
        - Store portfolio weights
        """
        logger.info("Phase 2: Monthly Universe & Optimization Loop")
        
        strategy_weights = {
            'gru': {},
            'historical': {},
            'market_cap': {}
        }
        
        try:
            async with get_async_session(self.database_url) as session:
                for i, rebalance_date in enumerate(rebalance_dates):
                    logger.info(f"Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")
                    
                    # Load monthly universe
                    monthly_universe = await fetch_monthly_universe(session, rebalance_date)
                    if not monthly_universe:
                        logger.warning(f"No universe found for {rebalance_date}, skipping")
                        continue
                    
                    logger.info(f"Universe for {rebalance_date}: {len(monthly_universe)} assets")
                    
                    # Load price data for optimization
                    lookback_start = rebalance_date - pd.Timedelta(days=params.lookback_days + 30)
                    prices = await load_prices_for_period(
                        session, monthly_universe, lookback_start.date(), rebalance_date
                    )
                    
                    if prices.empty:
                        logger.warning(f"No price data for {rebalance_date}, skipping")
                        continue
                    
                    # Clean price data
                    cleaned_prices = clean_price_data(prices)
                    if cleaned_prices.empty:
                        logger.warning(f"No clean price data for {rebalance_date}, skipping")
                        continue
                    
                    # Strategy 1: GRU-optimized portfolio
                    logger.info(f"Optimizing GRU strategy for {rebalance_date}")
                    gru_forecasts = await forecast_returns(
                        cleaned_prices, 
                        self.model_dir,
                        lookback_days=30,
                        max_workers=settings.max_workers
                    )
                    
                    if not gru_forecasts.empty:
                        gru_weights = await optimize_portfolio_gru(gru_forecasts, cleaned_prices, params)
                        if not gru_weights.empty:
                            # Force sell non-universe assets
                            gru_weights = force_sell_non_universe(gru_weights, monthly_universe)
                            strategy_weights['gru'][rebalance_date] = gru_weights
                            
                            # Store in database
                            await upsert_weights(session, run_id, rebalance_date, gru_weights, 'gru')
                    
                    # Strategy 2: Historical mean optimized portfolio
                    logger.info(f"Optimizing historical strategy for {rebalance_date}")
                    historical_weights = await optimize_portfolio_historical(cleaned_prices, params)
                    if not historical_weights.empty:
                        # Force sell non-universe assets
                        historical_weights = force_sell_non_universe(historical_weights, monthly_universe)
                        strategy_weights['historical'][rebalance_date] = historical_weights
                        
                        # Store in database
                        await upsert_weights(session, run_id, rebalance_date, historical_weights, 'historical')
                    
                    # Strategy 3: Market cap weighted portfolio
                    logger.info(f"Creating market cap weights for {rebalance_date}")
                    
                    # Load market cap weights for this month
                    month_start = rebalance_date.replace(day=1)
                    month_end = rebalance_date
                    market_cap_data = await load_market_cap_weights(session, month_start, month_end)
                    
                    if market_cap_data:
                        # Get weights for rebalancing date (or closest available)
                        available_dates = [d for d in market_cap_data.keys() if d <= rebalance_date]
                        if available_dates:
                            closest_date = max(available_dates)
                            market_cap_weights_raw = market_cap_data[closest_date]
                            
                            # Align to universe
                            market_cap_weights = create_market_cap_weights(market_cap_weights_raw, monthly_universe)
                            strategy_weights['market_cap'][rebalance_date] = market_cap_weights
                            
                            # Store in database
                            await upsert_weights(session, run_id, rebalance_date, market_cap_weights, 'market_cap')
                    
                    # Validate weights
                    for strategy_name, weights in [
                        ('gru', strategy_weights['gru'].get(rebalance_date)),
                        ('historical', strategy_weights['historical'].get(rebalance_date)),
                        ('market_cap', strategy_weights['market_cap'].get(rebalance_date))
                    ]:
                        if weights is not None:
                            validation = validate_portfolio_weights(weights, params)
                            if not validation['valid']:
                                logger.warning(f"Invalid weights for {strategy_name} on {rebalance_date}: {validation}")
                
                logger.info(f"Optimization loop completed: {len(strategy_weights['gru'])} GRU, "
                          f"{len(strategy_weights['historical'])} historical, "
                          f"{len(strategy_weights['market_cap'])} market cap portfolios")
                
                return strategy_weights
                
        except Exception as e:
            logger.error(f"Phase 2 optimization loop failed: {e}")
            return {}
    
    async def _phase3_backtesting(
        self,
        run_id: str,
        params: OptimizationParams,
        strategy_weights: Dict[str, Dict[date, pd.Series]],
        rebalance_dates: List[date]
    ) -> Dict[str, pd.Series]:
        """
        Phase 3: Backtesting & Performance Analysis
        - Load complete price dataset for backtesting period
        - Run vectorized backtests for all strategies
        - Store daily returns
        """
        logger.info("Phase 3: Backtesting & Performance Analysis")
        
        try:
            async with get_async_session(self.database_url) as session:
                # Get all symbols from strategy weights
                all_symbols = set()
                for strategy_dict in strategy_weights.values():
                    for weights in strategy_dict.values():
                        all_symbols.update(weights.index)
                
                all_symbols = sorted(list(all_symbols))
                logger.info(f"Loading prices for {len(all_symbols)} symbols for backtesting")
                
                # Load complete price dataset for backtesting
                start_date = datetime.strptime(params.start_date, '%Y-%m-%d').date()
                end_date = datetime.strptime(params.end_date, '%Y-%m-%d').date()
                
                # Add buffer for initial lookback
                extended_start = start_date - pd.Timedelta(days=60)
                
                prices = await load_prices_for_period(
                    session, all_symbols, extended_start.date(), end_date
                )
                
                if prices.empty:
                    logger.error("No price data loaded for backtesting")
                    return {}
                
                # Clean price data
                cleaned_prices = clean_price_data(prices)
                
                # Validate backtest data
                validation = validate_backtest_data(cleaned_prices, strategy_weights, rebalance_dates)
                if not validation['valid']:
                    logger.error(f"Backtest validation failed: {validation['errors']}")
                    if validation['warnings']:
                        logger.warning(f"Backtest warnings: {validation['warnings']}")
                
                # Run three-strategy backtest
                logger.info("Running vectorized backtests for all strategies")
                strategy_returns = run_three_strategy_backtest(
                    strategy_weights, cleaned_prices, rebalance_dates
                )
                
                if not strategy_returns:
                    logger.error("No strategy returns generated from backtesting")
                    return {}
                
                # Convert to dictionary format for database storage
                returns_dict = {}
                for strategy_name, returns_series in strategy_returns.items():
                    returns_dict[strategy_name] = {}
                    for timestamp, return_value in returns_series.items():
                        if pd.notna(return_value):
                            date_key = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                            returns_dict[strategy_name][date_key] = float(return_value)
                
                # Store daily returns in database
                logger.info("Storing daily returns in database")
                await store_daily_returns(session, run_id, returns_dict)
                
                logger.info(f"Backtesting completed for {len(strategy_returns)} strategies")
                return strategy_returns
                
        except Exception as e:
            logger.error(f"Phase 3 backtesting failed: {e}")
            return {}
    
    async def _phase4_evaluation(
        self,
        strategy_returns: Dict[str, pd.Series],
        run_id: str
    ) -> Dict[str, Any]:
        """
        Phase 4: Statistical Analysis & Reporting
        - Generate comprehensive performance metrics
        - Execute pairwise hypothesis tests
        - Create QuantStats tearsheets and comparison reports
        """
        logger.info("Phase 4: Statistical Analysis & Reporting")
        
        try:
            # Create run-specific report directory
            run_report_dir = self.report_path / run_id
            run_report_dir.mkdir(exist_ok=True)
            
            # Run complete evaluation
            evaluation_results = await run_complete_evaluation(
                strategy_returns,
                output_dir=run_report_dir,
                benchmark_strategy='market_cap'
            )
            
            if 'error' in evaluation_results:
                logger.error(f"Evaluation failed: {evaluation_results['error']}")
                return evaluation_results
            
            logger.info(f"Evaluation completed successfully")
            logger.info(f"Reports generated in: {run_report_dir}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Phase 4 evaluation failed: {e}")
            return {'error': str(e)}


# Convenience functions for external use
async def run_portfolio(
    run_id: Optional[str] = None,
    params: Optional[OptimizationParams] = None,
    database_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete portfolio analysis.
    
    Args:
        run_id: Optional run identifier (generated if not provided)
        params: Optional optimization parameters (loaded from DB if not provided)
        database_url: Optional database URL override
        
    Returns:
        Complete analysis results
    """
    if run_id is None:
        run_id = create_run_id("portfolio_analysis")
    
    runner = KallosPortfolioRunner(database_url)
    
    if params is None:
        # Try to load from database
        async with get_async_session(database_url) as session:
            params = await load_optimization_params(session, run_id)
            if params is None:
                raise ValueError(f"No optimization parameters found for run_id: {run_id}")
    
    return await runner.run_portfolio_analysis(run_id, params)


async def run_portfolio_with_config(
    config_dict: Dict[str, Any],
    run_id: Optional[str] = None,
    database_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run portfolio analysis with configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary with optimization parameters
        run_id: Optional run identifier
        database_url: Optional database URL override
        
    Returns:
        Complete analysis results
    """
    if run_id is None:
        run_id = create_run_id("portfolio_config")
    
    # Create OptimizationParams from config
    params = OptimizationParams(**config_dict)
    
    runner = KallosPortfolioRunner(database_url)
    return await runner.run_portfolio_analysis(run_id, params)


def create_example_config() -> Dict[str, Any]:
    """
    Create example configuration for portfolio optimization.
    
    Returns:
        Example configuration dictionary
    """
    return {
        'objective': 'max_sharpe',
        'max_weight': 0.35,
        'min_names': 3,
        'l2_reg': 0.01,
        'risk_free_rate': 0.0,
        'gamma': 1.0,
        'lookback_days': 252,
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    }


def get_system_info() -> Dict[str, Any]:
    """
    Get system information and configuration.
    
    Returns:
        Dictionary with system information
    """
    return {
        'version': '1.0.0',
        'strategies': ['gru', 'historical', 'market_cap'],
        'model_dir': settings.model_dir,
        'database_url': settings.database_url,
        'log_level': settings.log_level
    }


# Export main functions
__all__ = [
    'KallosPortfolioRunner',
    'run_portfolio',
    'run_portfolio_with_config',
    'create_example_config',
    'get_system_info',
    'setup_logging'
]
