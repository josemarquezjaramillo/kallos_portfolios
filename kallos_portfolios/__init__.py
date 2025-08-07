"""
Kallos Portfolios - Cryptocurrency Portfolio Optimization Package.

Simplified package for portfolio optimization with multiple strategies:
- GRU-based forecasting
- Historical returns benchmark  
- Market-weighted benchmark
"""

import logging

# Version
__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import main functionality
try:
    from .storage import get_async_session, fetch_monthly_universe, load_prices_for_period
    from .simulators import GRUPortfolioSimulator, HistoricalPortfolioSimulator
    from .analysis import run_three_strategy_comparison
    from .datasets import generate_monthly_rebalance_dates, generate_weekly_rebalance_dates, calculate_returns_matrix
    from .optimisers import optimize_portfolio_historical, validate_portfolio_weights, OptimizationParams
    from .backtest import run_backtest, run_three_strategy_backtest
    from .evaluation import generate_quantstats_reports, create_strategy_comparison_report
    
    # Main workflow function
    from typing import List
    from datetime import date
    import pandas as pd

    async def run_portfolio(
        simulation_name: str,
        assets: List[str],
        start_date: date,
        end_date: date,
        strategies: List[str],
        initial_investment: float,
        rebalance_frequency: str
    ):
        """
        Runs a full portfolio backtest and evaluation for multiple strategies.
        
        Args:
            simulation_name: A unique name for this simulation run.
            assets: A list of asset IDs (e.g., ['bitcoin', 'ethereum']).
            start_date: The start date for the backtest.
            end_date: The end date for the backtest.
            strategies: A list of strategies to run (e.g., ['equal_weight', 'historical_max_sharpe']).
            initial_investment: The starting capital for the portfolio.
            rebalance_frequency: How often to rebalance ('weekly', 'monthly').
            
        Returns:
            A dictionary containing the results, including performance metrics and portfolio value over time.
        """
        logging.info(f"Starting portfolio simulation: {simulation_name}")
        
        # Step 1: Global Initialization
        db_session_factory = get_async_session()
        async with db_session_factory() as session:
            try:
                # 1.1 Generate rebalance schedule
                if rebalance_frequency == 'weekly':
                    rebalance_dates = generate_weekly_rebalance_dates(start_date, end_date)
                else:
                    rebalance_dates = generate_monthly_rebalance_dates(start_date, end_date)
                
                logging.info(f"Generated {len(rebalance_dates)} rebalance dates")
                
                # 1.2 Load full price history for ALL potential assets
                all_prices = await load_prices_for_period(
                    session=session,
                    start_date=start_date,
                    end_date=end_date,
                    assets=None  # Load all assets, we'll filter later
                )
                
                # 1.3 Calculate full returns matrix
                full_returns = calculate_returns_matrix(all_prices)
                logging.info(f"Loaded price data for {len(full_returns.columns)} assets")
                
                # Step 2: Main Rebalancing Loop (Weekly/Monthly)
                all_strategy_weights = {strategy: {} for strategy in strategies}
                current_monthly_universe = []
                last_processed_month = None
                
                for rebalance_date in rebalance_dates:
                    logging.info(f"Processing rebalance date: {rebalance_date}")
                    
                    # 2.1 Check if month has changed - fetch new universe
                    current_month = rebalance_date.replace(day=1)  # First day of month
                    if current_month != last_processed_month:
                        logging.info(f"Fetching new universe for month: {current_month}")
                        current_monthly_universe = await fetch_monthly_universe(
                            session=session, 
                            rebalance_date=current_month
                        )
                        logging.info(f"Universe contains {len(current_monthly_universe)} assets")
                        last_processed_month = current_month
                    
                    # 2.2 Filter returns to current universe only
                    universe_assets = [asset for asset in current_monthly_universe if asset in full_returns.columns]
                    if not universe_assets:
                        logging.warning(f"No assets found in universe for {rebalance_date}")
                        continue
                        
                    returns_for_universe = full_returns[universe_assets]
                    
                    # 2.3 Slice historical data up to rebalance date
                    historical_returns = returns_for_universe.loc[:rebalance_date]
                    
                    if historical_returns.empty or len(historical_returns) < 30:  # Need minimum history
                        logging.warning(f"Insufficient history for {rebalance_date}")
                        continue
                    
                    # 2.4 Optimize for each strategy
                    for strategy in strategies:
                        if strategy == 'historical_max_sharpe':
                            # Call the optimizer with filtered universe data
                            params = OptimizationParams(objective='max_sharpe')
                            weights = await optimize_portfolio_historical(
                                prices=historical_returns,
                                params=params
                            )
                            # Validate weights
                            validated_weights = validate_portfolio_weights(weights)
                            all_strategy_weights[strategy][rebalance_date] = validated_weights
                            
                        elif strategy == 'equal_weight':
                            # Simple equal weighting
                            n_assets = len(universe_assets)
                            equal_weights = pd.Series(
                                data=[1.0/n_assets] * n_assets,
                                index=universe_assets
                            )
                            all_strategy_weights[strategy][rebalance_date] = equal_weights
                            
                        elif strategy == 'market_cap_weight':
                            # Market cap weighted (placeholder - would need market cap data)
                            # For now, use equal weights
                            n_assets = len(universe_assets)
                            cap_weights = pd.Series(
                                data=[1.0/n_assets] * n_assets,
                                index=universe_assets
                            )
                            all_strategy_weights[strategy][rebalance_date] = cap_weights
                            
                        else:
                            raise ValueError(f"Unknown strategy: {strategy}")
                
                # Step 3: Vectorized Backtesting
                logging.info("Running backtests for all strategies")
                backtest_results = run_three_strategy_backtest(
                    strategy_weights=all_strategy_weights,
                    prices=all_prices,
                    rebalance_dates=rebalance_dates
                )
                
                # Step 4: Performance Evaluation
                logging.info("Generating performance reports")
                
                # Generate individual quantstats reports
                quantstats_reports = {}
                for strategy, returns_series in backtest_results.items():
                    if not returns_series.empty:
                        report = await generate_quantstats_reports(
                            portfolio_returns=returns_series,
                            strategy_name=strategy,
                            output_path=f"reports/{simulation_name}_{strategy}.html"
                        )
                        quantstats_reports[strategy] = report
                
                # Create comparison report
                comparison_report = await create_strategy_comparison_report(
                    backtest_results=backtest_results,
                    simulation_name=simulation_name
                )
                
                # Step 5: Final Output
                final_results = {
                    'simulation_name': simulation_name,
                    'strategies': strategies,
                    'rebalance_dates_count': len(rebalance_dates),
                    'backtest_results': backtest_results,
                    'performance_comparison': comparison_report,
                    'quantstats_reports': quantstats_reports,
                    'final_portfolio_values': {
                        strategy: (1 + returns_series).prod() * initial_investment
                        for strategy, returns_series in backtest_results.items()
                        if not returns_series.empty
                    },
                    'status': 'completed'
                }
                
                logging.info(f"Successfully completed portfolio simulation: {simulation_name}")
                return final_results
                
            except Exception as e:
                logging.error(f"Error during portfolio simulation '{simulation_name}': {e}", exc_info=True)
                await session.rollback()
                raise


    __all__ = [
        'run_portfolio',
        'get_async_session',
        'fetch_monthly_universe', 
        'load_prices_for_period',
        'generate_monthly_rebalance_dates',
        'calculate_returns_matrix',
        'optimize_portfolio_historical',
        'validate_portfolio_weights',
        'run_backtest',
        'calculate_performance_metrics',
        'calculate_strategy_metrics',
        'create_comparison_report'
    ]

except ImportError as e:
    logging.warning(f"Some imports failed: {e}")
    
    # Fallback minimal version
    async def run_portfolio(run_id: str, start_date: str, end_date: str):
        """Fallback version when imports fail."""
        return {
            'run_id': run_id,
            'start_date': start_date,
            'end_date': end_date,
            'status': 'import_error',
            'error': 'Some module imports failed'
        }
    
    __all__ = ['run_portfolio']
