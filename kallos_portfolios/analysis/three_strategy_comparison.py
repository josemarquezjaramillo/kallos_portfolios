#!/usr/bin/env python3
"""
Three-Strategy Portfolio Comparison Runner

Compares GRU vs Historical vs Market-Weighted portfolios with comprehensive statistics.
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

from ..evaluation import create_three_strategy_comparison


async def run_three_strategy_comparison(
    gru_portfolio_id: int,
    historical_portfolio_id: int,
    start_date: date,
    end_date: date
):
    """
    Run the complete three-strategy comparison with explicit parameters.
    
    Args:
        gru_portfolio_id: portfolio_runs.id for GRU strategy
        historical_portfolio_id: portfolio_runs.id for Historical strategy
        start_date: Start date for comparison
        end_date: End date for comparison
    """
    
    print("🎯 Kallos Portfolios - Three-Strategy Comparison")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   GRU Portfolio ID: {gru_portfolio_id}")
    print(f"   Historical Portfolio ID: {historical_portfolio_id}")
    print(f"   Analysis Period: {start_date} to {end_date}")
    print("=" * 60)
    
    try:
        results = await create_three_strategy_comparison(
            gru_portfolio_id=gru_portfolio_id,
            historical_portfolio_id=historical_portfolio_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in results:
            print(f"❌ Comparison failed: {results['error']}")
            return
        
        print("\n✅ Three-Strategy Comparison Completed!")
        print(f"📁 Reports generated in: {results['output_directory']}")
        
        print("\n📊 Individual Strategy Tearsheets:")
        if 'quantstats_reports' in results:
            for strategy, tearsheet_path in results['quantstats_reports'].items():
                tearsheet_name = tearsheet_path.name if hasattr(tearsheet_path, 'name') else str(tearsheet_path).split('/')[-1]
                if strategy == 'GRU_Portfolio':
                    print(f"   • 🤖 GRU Portfolio: {tearsheet_name}")
                elif strategy == 'Historical_Portfolio':
                    print(f"   • 📈 Naive Portfolio: {tearsheet_name}")
                elif strategy == 'Market_Weighted':
                    print(f"   • 🏛️  Market Weighted: {tearsheet_name}")
                else:
                    print(f"   • 📊 {strategy}: {tearsheet_name}")
        
        print("\n� Comprehensive Analysis:")
        print(f"   • 🎯 Strategy Comparison Report: strategy_comparison_report.html")
        print(f"   • 📋 Performance Metrics: performance_metrics.csv") 
        print(f"   • 📈 Combined Returns: combined_returns.csv")
        print(f"   • � Statistical Tests: hypothesis_tests.json")
        
        # Print summary of performance metrics
        if 'performance_metrics' in results and not results['performance_metrics'].empty:
            metrics = results['performance_metrics']
            print(f"\n🎯 Performance Summary:")
            print(f"{'Strategy':25} | {'Total Return':>12} | {'Sharpe Ratio':>12} | {'Max Drawdown':>12}")
            print("-" * 70)
            
            for strategy in metrics.index:
                total_ret = metrics.loc[strategy, 'Total_Return']
                sharpe = metrics.loc[strategy, 'Sharpe_Ratio'] 
                max_dd = metrics.loc[strategy, 'Max_Drawdown']
                print(f"{strategy:25} | {total_ret:11.2%} | {sharpe:11.2f} | {max_dd:11.2%}")
        
        # Print hypothesis test summary
        if 'hypothesis_test_results' in results:
            test_results = results['hypothesis_test_results']
            if test_results:
                print(f"\n🔬 Statistical Test Summary:")
                for pair_name, tests in test_results.items():
                    pair_display = pair_name.replace('_vs_', ' vs ').replace('_', ' ')
                    print(f"\n   {pair_display}:")
                    
                    if 'mean_difference_test' in tests:
                        mean_test = tests['mean_difference_test']
                        significance = "✅ SIGNIFICANT" if mean_test['p_value'] < 0.05 else "❌ Not significant"
                        print(f"      Mean Return Difference: {significance} (p={mean_test['p_value']:.4f})")
                    
                    if 'variance_equality_test' in tests:
                        var_test = tests['variance_equality_test']
                        significance = "✅ SIGNIFICANT" if var_test['p_value'] < 0.05 else "❌ Not significant"
                        print(f"      Risk Difference: {significance} (p={var_test['p_value']:.4f})")
        
        print(f"\n🎉 Analysis complete! Check {results['output_directory']} for detailed reports.")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


if __name__ == "__main__":
    import sys
    
    print("🎯 Kallos Portfolios - Three-Strategy Comparison")
    print("=" * 50)
    
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
        print("await run_three_strategy_comparison(13, 16, date(2022,1,3), date(2022,12,26))")
