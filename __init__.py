"""
Kallos Portfolios - Cryptocurrency Portfolio Optimization Package

This package provides multiple portfolio optimization strategies:
- GRU-based neural network forecasting
- Historical returns benchmark
- Market-weighted benchmark
"""

# Import main functionality from the package
try:
    from .kallos_portfolios.simulators import GRUPortfolioSimulator, HistoricalPortfolioSimulator
    from .kallos_portfolios.analysis import run_three_strategy_comparison
    from .kallos_portfolios.storage import get_async_session, load_prices_for_period
    from .kallos_portfolios.evaluation import create_three_strategy_comparison
    
    __all__ = [
        'GRUPortfolioSimulator',
        'HistoricalPortfolioSimulator', 
        'run_three_strategy_comparison',
        'create_three_strategy_comparison',
        'get_async_session',
        'load_prices_for_period'
    ]
    
except ImportError as e:
    # Fallback for development/testing
    print(f"Warning: Could not import all modules: {e}")
    __all__ = []

__version__ = "1.0.0"
