"""
Main Python interface for Kallos Portfolios system.
Provides high-level functions for portfolio optimization, backtesting, and analysis.
"""

import logging
import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from .config.settings import OptimizationParams, create_run_id, validate_run_id, settings
from .exceptions import (
    KallosPortfoliosError, DataValidationError, InsufficientDataError,
    ModelLoadingError, OptimizationError, BacktestError
)
from .storage import (
    get_async_session, check_database_health, store_optimization_params,
    load_prices_for_period, fetch_monthly_universe
)
from .models import forecast_returns, ModelManager
from .optimisers import PortfolioOptimizer
from .backtest import PortfolioBacktester
from .evaluation import PortfolioEvaluator
from .datasets import generate_monthly_rebalance_dates

logger = logging.getLogger(__name__)


class KallosPortfolios:
    """
    Main interface for the Kallos Portfolios system.
    
    This class provides a high-level Python API for:
    - Portfolio optimization using three strategies (GRU, Historical, Market Cap)
    - Backtesting with transaction costs and rebalancing
    - Performance evaluation and comparison
    - Model management and data validation
    """
    
    def __init__(self, 
                 model_dir: Optional[Union[str, Path]] = None,
                 database_url: Optional[str] = None,
                 max_workers: int = 4):
        """
        Initialize Kallos Portfolios system.
        
        Args:
            model_dir: Directory containing trained GRU models (uses settings default if None)
            database_url: PostgreSQL connection string (uses settings default if None)
            max_workers: Maximum worker threads for parallel processing
        """
        self.model_dir = Path(model_dir) if model_dir else settings.model_dir
        self.database_url = database_url or settings.database_url
        self.max_workers = max_workers
        
        # Initialize components
        self.model_manager = ModelManager(self.model_dir, cache_models=True)
        self._optimizer = None
        self._backtester = None
        self._evaluator = None
        
        logger.info(f"Initialized Kallos Portfolios system")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Max workers: {self.max_workers}")
    
    async def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Dictionary with health status of all components
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "database": False,
            "models": {},
            "configuration": True,
            "overall": False
        }
        
        try:
            # Check database connectivity
            health_status["database"] = await check_database_health(self.database_url)
            
            # Check model availability
            available_symbols = self.model_manager.get_available_symbols()
            health_status["models"] = {
                "available_symbols": available_symbols,
                "count": len(available_symbols),
                "model_dir_exists": self.model_dir.exists()
            }
            
            # Overall health
            health_status["overall"] = (
                health_status["database"] and 
                health_status["configuration"] and 
                len(available_symbols) > 0
            )
            
            logger.info(f"System health check: {'✅ HEALTHY' if health_status['overall'] else '❌ ISSUES'}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)
            return health_status
    
    def create_optimization_params(self,
                                   start_date: Union[str, date],
                                   end_date: Union[str, date],
                                   objective: str = 'max_sharpe',
                                   max_weight: float = 0.35,
                                   min_names: int = 3,
                                   lookback_days: int = 252,
                                   **kwargs) -> OptimizationParams:
        """
        Create and validate optimization parameters.
        
        Args:
            start_date: Portfolio start date (YYYY-MM-DD or date object)
            end_date: Portfolio end date (YYYY-MM-DD or date object)
            objective: Optimization objective ('max_sharpe', 'min_volatility', etc.)
            max_weight: Maximum weight per asset (0.0-1.0)
            min_names: Minimum number of assets in portfolio
            lookback_days: Historical lookback period for covariance estimation
            **kwargs: Additional optimization parameters
        
        Returns:
            Validated OptimizationParams object
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Convert dates to strings if needed
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')
        
        params = OptimizationParams(
            start_date=start_date,
            end_date=end_date,
            objective=objective,
            max_weight=max_weight,
            min_names=min_names,
            lookback_days=lookback_days,
            **kwargs
        )
        
        logger.info(f"Created optimization parameters: {objective} from {start_date} to {end_date}")
        return params
    
    async def get_available_universe(self, target_date: Union[str, date]) -> List[str]:
        """
        Get the investable universe for a specific date.
        
        Args:
            target_date: Date to get universe for
            
        Returns:
            List of available coin IDs
        """
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        async with get_async_session(self.database_url) as session:
            universe = await fetch_monthly_universe(session, target_date)
            
        logger.info(f"Retrieved universe of {len(universe)} coin IDs for {target_date}")
        return universe
    
    async def load_price_data(self,
                              coin_ids: List[str],
                              start_date: Union[str, date],
                              end_date: Union[str, date],
                              min_data_coverage: float = 0.8) -> pd.DataFrame:
        """
        Load historical price data for given coin IDs and date range.
        
        Args:
            coin_ids: List of coin identifiers
            start_date: Start date for price data
            end_date: End date for price data
            min_data_coverage: Minimum required data coverage (0.0-1.0)
            
        Returns:
            DataFrame with prices indexed by date, coin_ids as columns
            
        Raises:
            DataValidationError: If data quality is insufficient
            InsufficientDataError: If not enough data is available
        """
        # Convert string dates to date objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        async with get_async_session(self.database_url) as session:
            prices = await load_prices_for_period(
                session, coin_ids, start_date, end_date, min_data_coverage
            )
        
        logger.info(f"Loaded price data: {len(prices.columns)} coin IDs, {len(prices)} dates")
        return prices
    
    async def generate_forecasts(self,
                                 coin_ids: List[str],
                                 end_date: Union[str, date],
                                 lookback_days: int = 252) -> pd.Series:
        """
        Generate GRU-based return forecasts for given coin IDs using temporal model selection.
        
        Args:
            coin_ids: List of coin identifiers
            end_date: End date for historical data (forecast point)
            lookback_days: Number of days of historical data to use
            
        Returns:
            Series of predicted weekly returns by coin_id
            
        Raises:
            ModelInferenceError: If forecast generation fails
        """
        # Calculate start date based on lookback
        if isinstance(end_date, str):
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date_obj = end_date
        
        start_date_obj = end_date_obj - pd.Timedelta(days=lookback_days + 30)  # Add buffer
        
        # Load price data
        prices = await self.load_price_data(coin_ids, start_date_obj, end_date_obj)
        
        # Generate forecasts with temporal model selection
        async with get_async_session(self.database_url) as session:
            forecasts = await forecast_returns(
                prices, self.model_dir, target_date=end_date_obj, session=session,
                lookback_days=lookback_days, max_workers=self.max_workers
            )
        
        logger.info(f"Generated forecasts for {len(forecasts)} coin IDs")
        return forecasts
    
    async def optimize_portfolio(self,
                                 params: OptimizationParams,
                                 coin_ids: Optional[List[str]] = None,
                                 expected_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using three strategies: GRU, Historical, and Market Cap.
        
        Args:
            params: Optimization parameters
            coin_ids: List of coin IDs to optimize (auto-detected if None)
            expected_returns: Pre-computed expected returns (generated if None)
            
        Returns:
            Dictionary with optimization results for all three strategies
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            # Auto-detect coin_ids if not provided
            if coin_ids is None:
                end_date_obj = datetime.strptime(params.end_date, '%Y-%m-%d').date()
                coin_ids = await self.get_available_universe(end_date_obj)
            
            # Load price data
            prices = await self.load_price_data(
                coin_ids, params.start_date, params.end_date
            )
            
            # Generate GRU forecasts if not provided
            if expected_returns is None:
                expected_returns = await self.generate_forecasts(
                    list(prices.columns), params.end_date, params.lookback_days
                )
            
            # Initialize optimizer
            optimizer = PortfolioOptimizer(params)
            
            # Optimize using all three strategies
            results = {}
            
            # 1. GRU Strategy
            try:
                gru_weights = optimizer.optimize_gru_strategy(expected_returns, prices)
                results['gru'] = {
                    'weights': gru_weights,
                    'expected_returns': expected_returns[gru_weights.index],
                    'strategy': 'gru'
                }
                logger.info(f"GRU optimization: {len(gru_weights)} assets, max weight: {gru_weights.max():.3f}")
            except Exception as e:
                logger.warning(f"GRU optimization failed: {e}")
                results['gru'] = {'error': str(e)}
            
            # 2. Historical Strategy
            try:
                hist_weights = optimizer.optimize_historical_strategy(prices)
                results['historical'] = {
                    'weights': hist_weights,
                    'strategy': 'historical'
                }
                logger.info(f"Historical optimization: {len(hist_weights)} assets, max weight: {hist_weights.max():.3f}")
            except Exception as e:
                logger.warning(f"Historical optimization failed: {e}")
                results['historical'] = {'error': str(e)}
            
            # 3. Market Cap Strategy
            try:
                # Get market cap data for weighting
                end_date_obj = datetime.strptime(params.end_date, '%Y-%m-%d').date()
                async with get_async_session(self.database_url) as session:
                    universe_data = await fetch_monthly_universe(session, end_date_obj)
                
                mcap_weights = optimizer.optimize_market_cap_strategy(universe_data)
                results['market_cap'] = {
                    'weights': mcap_weights,
                    'strategy': 'market_cap'
                }
                logger.info(f"Market cap optimization: {len(mcap_weights)} assets, max weight: {mcap_weights.max():.3f}")
            except Exception as e:
                logger.warning(f"Market cap optimization failed: {e}")
                results['market_cap'] = {'error': str(e)}
            
            # Check if any strategy succeeded
            successful_strategies = [k for k, v in results.items() if 'error' not in v]
            if not successful_strategies:
                raise OptimizationError("All optimization strategies failed")
            
            logger.info(f"Portfolio optimization completed: {len(successful_strategies)}/3 strategies successful")
            return results
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise OptimizationError(f"Portfolio optimization failed: {str(e)}") from e
    
    async def run_backtest(self,
                           optimization_results: Dict[str, Any],
                           params: OptimizationParams,
                           transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        Run backtesting for all optimized strategies.
        
        Args:
            optimization_results: Results from optimize_portfolio()
            params: Optimization parameters
            transaction_cost: Transaction cost rate (default 0.1%)
            
        Returns:
            Dictionary with backtest results for all strategies
            
        Raises:
            BacktestError: If backtesting fails
        """
        try:
            # Initialize backtester
            backtester = PortfolioBacktester(
                transaction_cost=transaction_cost,
                rebalance_frequency='monthly'
            )
            
            # Load price data for backtesting
            prices = await self.load_price_data(
                [], params.start_date, params.end_date  # Will auto-detect symbols
            )
            
            backtest_results = {}
            
            for strategy_name, strategy_result in optimization_results.items():
                if 'error' in strategy_result:
                    backtest_results[strategy_name] = strategy_result
                    continue
                
                try:
                    # Run backtest for this strategy
                    weights = strategy_result['weights']
                    result = await backtester.run_backtest(
                        weights, prices, params.start_date, params.end_date
                    )
                    
                    backtest_results[strategy_name] = {
                        **strategy_result,
                        'backtest': result,
                        'performance': result.get('performance_metrics', {})
                    }
                    
                    logger.info(f"Backtest completed for {strategy_name} strategy")
                    
                except Exception as e:
                    logger.warning(f"Backtest failed for {strategy_name}: {e}")
                    backtest_results[strategy_name] = {
                        **strategy_result,
                        'backtest_error': str(e)
                    }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            raise BacktestError(f"Backtesting failed: {str(e)}") from e
    
    async def evaluate_performance(self,
                                   backtest_results: Dict[str, Any],
                                   benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Evaluate and compare performance of all strategies.
        
        Args:
            backtest_results: Results from run_backtest()
            benchmark: Benchmark returns series (optional)
            
        Returns:
            Dictionary with performance evaluation and comparison
        """
        try:
            evaluator = PortfolioEvaluator()
            
            # Extract returns series for each strategy
            strategy_returns = {}
            for strategy_name, result in backtest_results.items():
                if 'backtest' in result and 'returns' in result['backtest']:
                    strategy_returns[strategy_name] = result['backtest']['returns']
            
            if not strategy_returns:
                raise ValueError("No valid backtest results found for evaluation")
            
            # Perform comprehensive evaluation
            evaluation_results = await evaluator.compare_strategies(
                strategy_returns, benchmark
            )
            
            logger.info(f"Performance evaluation completed for {len(strategy_returns)} strategies")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            raise ValueError(f"Performance evaluation failed: {str(e)}") from e
    
    async def run_complete_analysis(self,
                                    start_date: Union[str, date],
                                    end_date: Union[str, date],
                                    run_id: Optional[str] = None,
                                    symbols: Optional[List[str]] = None,
                                    **optimization_kwargs) -> Dict[str, Any]:
        """
        Run complete portfolio analysis: optimization + backtesting + evaluation.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            run_id: Unique identifier (auto-generated if None)
            symbols: Symbols to analyze (auto-detected if None)
            **optimization_kwargs: Additional optimization parameters
            
        Returns:
            Complete analysis results including all steps
        """
        if run_id is None:
            run_id = create_run_id("analysis")
        else:
            run_id = validate_run_id(run_id)
        
        logger.info(f"Starting complete analysis: {run_id}")
        
        try:
            # 1. Create optimization parameters
            params = self.create_optimization_params(
                start_date=start_date,
                end_date=end_date,
                **optimization_kwargs
            )
            
            # 2. Store parameters in database
            async with get_async_session(self.database_url) as session:
                await store_optimization_params(session, run_id, params)
            
            # 3. Portfolio optimization
            logger.info("Step 1/4: Portfolio optimization")
            optimization_results = await self.optimize_portfolio(params, symbols)
            
            # 4. Backtesting
            logger.info("Step 2/4: Backtesting")
            backtest_results = await self.run_backtest(optimization_results, params)
            
            # 5. Performance evaluation
            logger.info("Step 3/4: Performance evaluation")
            evaluation_results = await self.evaluate_performance(backtest_results)
            
            # 6. Compile final results
            logger.info("Step 4/4: Compiling results")
            final_results = {
                'run_id': run_id,
                'parameters': params.model_dump(),
                'optimization': optimization_results,
                'backtest': backtest_results,
                'evaluation': evaluation_results,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            logger.info(f"Complete analysis finished: {run_id}")
            return final_results
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            error_result = {
                'run_id': run_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
            return error_result


# Convenience functions for direct usage
async def run_portfolio_optimization(start_date: Union[str, date],
                                     end_date: Union[str, date],
                                     objective: str = 'max_sharpe',
                                     max_weight: float = 0.35,
                                     **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run portfolio optimization with default settings.
    
    Args:
        start_date: Portfolio start date
        end_date: Portfolio end date
        objective: Optimization objective
        max_weight: Maximum weight per asset
        **kwargs: Additional parameters
        
    Returns:
        Optimization results
    """
    kallos = KallosPortfolios()
    params = kallos.create_optimization_params(
        start_date=start_date,
        end_date=end_date,
        objective=objective,
        max_weight=max_weight,
        **kwargs
    )
    return await kallos.optimize_portfolio(params)


async def run_complete_analysis(start_date: Union[str, date],
                                end_date: Union[str, date],
                                run_id: Optional[str] = None,
                                **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run complete portfolio analysis.
    
    Args:
        start_date: Analysis start date
        end_date: Analysis end date
        run_id: Unique identifier (auto-generated if None)
        **kwargs: Additional parameters
        
    Returns:
        Complete analysis results
    """
    kallos = KallosPortfolios()
    return await kallos.run_complete_analysis(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        **kwargs
    )


def get_system_info() -> Dict[str, Any]:
    """
    Get information about the Kallos Portfolios system configuration.
    
    Returns:
        System configuration information
    """
    return {
        'version': '1.0.0',
        'model_dir': str(settings.model_dir),
        'database_url': settings.database_url.split('@')[0] + '@***',  # Hide credentials
        'max_workers': settings.max_workers,
        'log_level': settings.log_level,
        'supported_objectives': ['max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return'],
        'supported_strategies': ['gru', 'historical', 'market_cap']
    }
