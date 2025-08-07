#!/usr/bin/env python3
"""
Three Strategy Comparison Compatibility Wrapper

DEPRECATED: This script is maintained for backward compatibility.
New code should use: 
    from kallos_portfolios.analysis import run_three_strategy_comparison
"""

import warnings
import asyncio
import sys
from datetime import date

def main():
    warnings.warn(
        "This script is deprecated. Use 'from kallos_portfolios.analysis import run_three_strategy_comparison' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("🔄 Redirecting to new analysis module...")
    
    # Import the new implementation
    from kallos_portfolios.analysis import run_three_strategy_comparison
    
    # Check for command line arguments
    if len(sys.argv) == 5:
        try:
            gru_id = int(sys.argv[1])
            hist_id = int(sys.argv[2])
            start_str = sys.argv[3]  # Format: YYYY-MM-DD
            end_str = sys.argv[4]    # Format: YYYY-MM-DD
            
            start_date = date.fromisoformat(start_str)
            end_date = date.fromisoformat(end_str)
            
            asyncio.run(run_three_strategy_comparison(
                gru_portfolio_id=gru_id,
                historical_portfolio_id=hist_id,
                start_date=start_date,
                end_date=end_date
            ))
            
        except ValueError as e:
            print(f"❌ Invalid command line arguments: {e}")
            print("\nUsage: python three_strategy_comparison.py <gru_id> <hist_id> <start_date> <end_date>")
            print("Example: python three_strategy_comparison.py 13 16 2022-01-03 2022-12-26")
    else:
        print("Usage: python three_strategy_comparison.py <gru_id> <hist_id> <start_date> <end_date>")
        print("Example: python three_strategy_comparison.py 13 16 2022-01-03 2022-12-26")
        print("\nOr use programmatically:")
        print("from kallos_portfolios.analysis import run_three_strategy_comparison")
        print("await run_three_strategy_comparison(13, 16, date(2022,1,3), date(2022,12,26))")

if __name__ == "__main__":
    main()
