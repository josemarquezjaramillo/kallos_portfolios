"""
Vectorized backtesting engine using vectorbt for efficient portfolio simulation.
Implements simple, clean backtesting focused on core performance measurement.
"""

import logging
from datetime import date
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import vectorbt as vbt

logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Vectorized portfolio backtesting with focus on clean performance measurement.
    
    Key features:
    - Simple rebalancing logic
    - No transaction costs (can be added later)
    - Clean performance measurement for strategy comparison
    - Handles missing data gracefully
    """
    
    def __init__(self, prices: pd.DataFrame):
        """
        Initialize backtester with price data.
        
        Args:
            prices: Price data matrix (dates x symbols)
        """
        self.prices = prices
        self.returns = self._calculate_returns()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns from price data."""
        if self.prices.empty:
            return pd.DataFrame()
        
        returns = self.prices.pct_change().fillna(0)
        return returns
    
    def _align_weights_to_returns(
        self, 
        weights_dict: Dict[date, pd.Series], 
        rebalance_dates: List[date]
    ) -> pd.DataFrame:
        """
        Align portfolio weights to daily return dates with forward-fill.
        
        Args:
            weights_dict: Dictionary mapping rebalance dates to weight series
            rebalance_dates: List of rebalancing dates
            
        Returns:
            DataFrame with daily weights (dates x symbols)
        """
        if not weights_dict or self.returns.empty:
            return pd.DataFrame()
        
        # Get all symbols from weights
        all_symbols = set()
        for weights in weights_dict.values():
            all_symbols.update(weights.index)
        all_symbols = sorted(list(all_symbols))
        
        # Create weights DataFrame
        weights_df = pd.DataFrame(
            index=self.returns.index,
            columns=all_symbols,
            dtype=float
        ).fillna(0.0)
        
        # Fill in rebalancing weights
        for rebalance_date, weights in weights_dict.items():
            # Convert date to pandas timestamp for indexing
            rebalance_ts = pd.Timestamp(rebalance_date)
            
            # Find the closest date in returns index
            if rebalance_ts in weights_df.index:
                for symbol, weight in weights.items():
                    if symbol in weights_df.columns:
                        weights_df.loc[rebalance_ts, symbol] = weight
            else:
                # Find closest date after rebalance date
                future_dates = weights_df.index[weights_df.index >= rebalance_ts]
                if len(future_dates) > 0:
                    closest_date = future_dates[0]
                    for symbol, weight in weights.items():
                        if symbol in weights_df.columns:
                            weights_df.loc[closest_date, symbol] = weight
        
        # Forward fill weights between rebalancing dates
        weights_df = weights_df.fillna(method='ffill')
        
        # Handle initial period before first rebalance
        first_rebalance_idx = None
        for rebalance_date in sorted(rebalance_dates):
            rebalance_ts = pd.Timestamp(rebalance_date)
            if rebalance_ts in weights_df.index:
                first_rebalance_idx = weights_df.index.get_loc(rebalance_ts)
                break
        
        if first_rebalance_idx is not None and first_rebalance_idx > 0:
            # Fill initial period with zeros
            weights_df.iloc[:first_rebalance_idx] = 0.0
        
        return weights_df
    
    def run_backtest(
        self, 
        weights_dict: Dict[date, pd.Series],
        rebalance_dates: List[date]
    ) -> pd.Series:
        """
        Run vectorized backtest for a single strategy.
        
        Args:
            weights_dict: Portfolio weights by rebalancing date
            rebalance_dates: List of rebalancing dates
            
        Returns:
            Series of daily portfolio returns
        """
        try:
            if not weights_dict:
                self.logger.warning("No weights provided for backtesting")
                return pd.Series(dtype=float)
            
            # Align weights to return dates
            weights_df = self._align_weights_to_returns(weights_dict, rebalance_dates)
            
            if weights_df.empty:
                self.logger.warning("No aligned weights for backtesting")
                return pd.Series(dtype=float)
            
            # Ensure returns and weights have same columns
            common_symbols = self.returns.columns.intersection(weights_df.columns)
            
            if len(common_symbols) == 0:
                self.logger.error("No common symbols between returns and weights")
                return pd.Series(dtype=float)
            
            returns_aligned = self.returns[common_symbols]
            weights_aligned = weights_df[common_symbols]
            
            # Calculate daily portfolio returns
            portfolio_returns = (weights_aligned * returns_aligned).sum(axis=1)
            
            # Handle rebalancing dates - weights update at close of rebalancing day
            # This is realistic: decide weights during day, implement at close
            
            self.logger.info(f"Backtest completed: {len(portfolio_returns)} daily returns")
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return pd.Series(dtype=float)


async def run_backtest(
    weights_dict: Dict[date, pd.Series],
    prices: pd.DataFrame,
    rebalance_dates: List[date]
) -> Dict[date, float]:
    """
    Simple vectorized portfolio backtesting function.
    
    Args:
        weights_dict: Rebalancing weights by date
        prices: Price data matrix (dates x symbols)
        rebalance_dates: Rebalancing schedule
        
    Returns:
        Dictionary mapping dates to daily portfolio returns
    """
    try:
        backtester = PortfolioBacktester(prices)
        portfolio_returns = backtester.run_backtest(weights_dict, rebalance_dates)
        
        # Convert to dictionary format
        returns_dict = portfolio_returns.to_dict()
        
        # Convert timestamp keys to date keys
        date_returns = {}
        for timestamp, return_value in returns_dict.items():
            if pd.notna(return_value):
                date_key = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                date_returns[date_key] = float(return_value)
        
        logger.info(f"Backtest completed: {len(date_returns)} daily returns")
        return date_returns
        
    except Exception as e:
        logger.error(f"Error in run_backtest: {e}")
        return {}


def run_three_strategy_backtest(
    strategy_weights: Dict[str, Dict[date, pd.Series]],
    prices: pd.DataFrame,
    rebalance_dates: List[date]
) -> Dict[str, pd.Series]:
    """
    Run backtests for all three strategies in parallel.
    
    Args:
        strategy_weights: Nested dict {strategy: {date: weights}}
        prices: Price data matrix
        rebalance_dates: Rebalancing schedule
        
    Returns:
        Dictionary mapping strategy names to return series
    """
    strategy_returns = {}
    
    for strategy_name, weights_dict in strategy_weights.items():
        try:
            logger.info(f"Running backtest for {strategy_name} strategy")
            
            backtester = PortfolioBacktester(prices)
            returns_series = backtester.run_backtest(weights_dict, rebalance_dates)
            
            if not returns_series.empty:
                strategy_returns[strategy_name] = returns_series
                logger.info(f"{strategy_name} backtest: {len(returns_series)} returns, "
                          f"mean: {returns_series.mean():.4f}, std: {returns_series.std():.4f}")
            else:
                logger.warning(f"No returns generated for {strategy_name} strategy")
                
        except Exception as e:
            logger.error(f"Error in {strategy_name} backtest: {e}")
    
    return strategy_returns


def calculate_portfolio_drift(
    initial_weights: pd.Series,
    returns_series: pd.Series,
    rebalance_date: date
) -> pd.Series:
    """
    Calculate how portfolio weights drift due to asset performance.
    
    Args:
        initial_weights: Portfolio weights at rebalancing
        returns_series: Asset returns since rebalancing
        rebalance_date: Date of last rebalancing
        
    Returns:
        Current portfolio weights after drift
    """
    try:
        # Calculate cumulative returns since rebalancing
        cumulative_returns = (1 + returns_series).cumprod()
        
        # Calculate current values
        current_values = initial_weights * cumulative_returns
        
        # Calculate current weights
        total_value = current_values.sum()
        current_weights = current_values / total_value if total_value > 0 else initial_weights
        
        return current_weights
        
    except Exception as e:
        logger.error(f"Error calculating portfolio drift: {e}")
        return initial_weights


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series
) -> float:
    """
    Calculate portfolio turnover between two weight allocations.
    
    Args:
        old_weights: Previous portfolio weights
        new_weights: New portfolio weights
        
    Returns:
        Portfolio turnover (0-2, where 2 = complete portfolio replacement)
    """
    try:
        # Align weights
        all_assets = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_assets, fill_value=0.0)
        new_aligned = new_weights.reindex(all_assets, fill_value=0.0)
        
        # Calculate turnover as sum of absolute weight changes
        turnover = (new_aligned - old_aligned).abs().sum()
        
        return float(turnover)
        
    except Exception as e:
        logger.error(f"Error calculating turnover: {e}")
        return 0.0


def analyze_rebalancing_impact(
    strategy_returns: Dict[str, pd.Series],
    rebalance_dates: List[date]
) -> pd.DataFrame:
    """
    Analyze the impact of rebalancing on portfolio performance.
    
    Args:
        strategy_returns: Dictionary of strategy return series
        rebalance_dates: List of rebalancing dates
        
    Returns:
        DataFrame with rebalancing impact analysis
    """
    analysis_results = []
    
    for strategy_name, returns_series in strategy_returns.items():
        try:
            rebalance_timestamps = [pd.Timestamp(d) for d in rebalance_dates]
            
            # Calculate returns around rebalancing dates
            rebalance_impacts = []
            
            for rebalance_ts in rebalance_timestamps:
                if rebalance_ts in returns_series.index:
                    # Get returns before and after rebalancing
                    idx = returns_series.index.get_loc(rebalance_ts)
                    
                    pre_return = returns_series.iloc[idx-1] if idx > 0 else np.nan
                    post_return = returns_series.iloc[idx+1] if idx < len(returns_series)-1 else np.nan
                    rebalance_return = returns_series.iloc[idx]
                    
                    rebalance_impacts.append({
                        'date': rebalance_ts.date(),
                        'pre_return': pre_return,
                        'rebalance_return': rebalance_return,
                        'post_return': post_return
                    })
            
            if rebalance_impacts:
                impact_df = pd.DataFrame(rebalance_impacts)
                
                analysis_results.append({
                    'strategy': strategy_name,
                    'n_rebalances': len(rebalance_impacts),
                    'avg_rebalance_return': impact_df['rebalance_return'].mean(),
                    'avg_pre_return': impact_df['pre_return'].mean(),
                    'avg_post_return': impact_df['post_return'].mean(),
                    'rebalance_volatility': impact_df['rebalance_return'].std()
                })
                
        except Exception as e:
            logger.error(f"Error analyzing rebalancing impact for {strategy_name}: {e}")
    
    if analysis_results:
        return pd.DataFrame(analysis_results)
    else:
        return pd.DataFrame()


def create_backtest_summary(
    strategy_returns: Dict[str, pd.Series],
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Create comprehensive backtest summary statistics.
    
    Args:
        strategy_returns: Dictionary of strategy return series
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        DataFrame with summary statistics for each strategy
    """
    summary_data = []
    
    for strategy_name, returns_series in strategy_returns.items():
        try:
            if returns_series.empty:
                continue
            
            # Filter returns to date range
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            mask = (returns_series.index >= start_ts) & (returns_series.index <= end_ts)
            filtered_returns = returns_series.loc[mask]
            
            if filtered_returns.empty:
                continue
            
            # Calculate summary statistics
            n_periods = len(filtered_returns)
            total_return = (1 + filtered_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / n_periods) - 1
            
            daily_volatility = filtered_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Drawdown calculation
            cumulative = (1 + filtered_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1)
            max_drawdown = drawdown.min()
            
            # Win rate
            win_rate = (filtered_returns > 0).mean()
            
            # Risk metrics
            var_95 = filtered_returns.quantile(0.05)
            cvar_95 = filtered_returns[filtered_returns <= var_95].mean()
            
            summary_data.append({
                'Strategy': strategy_name,
                'Total_Return': total_return,
                'Annualized_Return': annualized_return,
                'Annualized_Volatility': annualized_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'Win_Rate': win_rate,
                'VaR_95': var_95,
                'CVaR_95': cvar_95,
                'Skewness': filtered_returns.skew(),
                'Kurtosis': filtered_returns.kurtosis(),
                'N_Observations': n_periods
            })
            
        except Exception as e:
            logger.error(f"Error creating summary for {strategy_name}: {e}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.set_index('Strategy', inplace=True)
        return summary_df
    else:
        return pd.DataFrame()


def validate_backtest_data(
    prices: pd.DataFrame,
    strategy_weights: Dict[str, Dict[date, pd.Series]],
    rebalance_dates: List[date]
) -> Dict[str, any]:
    """
    Validate input data for backtesting.
    
    Args:
        prices: Price data matrix
        strategy_weights: Strategy weights dictionary
        rebalance_dates: Rebalancing dates
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check prices
    if prices.empty:
        validation['errors'].append("Empty price data")
        validation['valid'] = False
    else:
        # Check for missing data
        missing_pct = prices.isnull().sum().sum() / (len(prices) * len(prices.columns))
        if missing_pct > 0.1:
            validation['warnings'].append(f"High missing data percentage: {missing_pct:.1%}")
        
        # Check for extreme returns
        returns = prices.pct_change().dropna()
        extreme_returns = (returns.abs() > 0.5).sum().sum()
        if extreme_returns > 0:
            validation['warnings'].append(f"Found {extreme_returns} extreme returns (>50%)")
    
    # Check strategy weights
    if not strategy_weights:
        validation['errors'].append("No strategy weights provided")
        validation['valid'] = False
    else:
        for strategy, weights_dict in strategy_weights.items():
            if not weights_dict:
                validation['warnings'].append(f"No weights for {strategy} strategy")
            else:
                # Check weight sum
                for date, weights in weights_dict.items():
                    weight_sum = weights.sum()
                    if abs(weight_sum - 1.0) > 0.01:
                        validation['warnings'].append(
                            f"{strategy} weights on {date} sum to {weight_sum:.3f}, not 1.0"
                        )
    
    # Check rebalance dates
    if not rebalance_dates:
        validation['errors'].append("No rebalancing dates provided")
        validation['valid'] = False
    
    return validation
