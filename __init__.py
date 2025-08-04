# Top-level kallos_portfolios package
from .kallos_portfolios import (
    KallosPortfolioRunner,
    run_portfolio,
    run_portfolio_with_config,
    create_example_config,
    get_system_info,
    setup_logging
)

__version__ = "1.0.0"
__all__ = [
    'KallosPortfolioRunner',
    'run_portfolio',
    'run_portfolio_with_config', 
    'create_example_config',
    'get_system_info',
    'setup_logging'
]
