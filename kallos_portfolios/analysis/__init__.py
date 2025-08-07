"""
Kallos Portfolios Analysis Package

This package contains analysis tools for comparing different portfolio strategies:
- three_strategy_comparison: Compare GRU, Historical, and Market-Weighted strategies
"""

from .three_strategy_comparison import run_three_strategy_comparison

__all__ = ['run_three_strategy_comparison']
