"""
Portfolio optimization engine using PyPortfolioOpt and CVXPY.
Implements three-strategy optimization framework with advanced constraints.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.exceptions import OptimizationError

from .config.settings import OptimizationParams

logger = logging.getLogger(__name__)

# Suppress PyPortfolioOpt warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pypfopt")


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with cardinality constraints and multiple objectives.
    
    Supports:
    - GRU-based expected returns
    - Historical mean returns
    - Market cap weighted portfolios
    - Cardinality constraints via CVXPY
    - Multiple optimization objectives
    """
    
    def __init__(self, params: OptimizationParams):
        """
        Initialize portfolio optimizer with configuration parameters.
        
        Args:
            params: Optimization parameters and constraints
        """
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _prepare_covariance_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate covariance matrix using Ledoit-Wolf shrinkage.
        
        Args:
            prices: Historical price data
            
        Returns:
            Weekly covariance matrix
        """
        try:
            # Use Ledoit-Wolf shrinkage for robust covariance estimation
            cov_estimator = risk_models.CovarianceShrinkage(prices, frequency=252)
            daily_cov = cov_estimator.ledoit_wolf()
            
            # Convert daily to weekly covariance (assuming 5 trading days per week)
            weekly_cov = daily_cov * 5
            
            # Ensure positive semi-definite
            eigenvals = np.linalg.eigvals(weekly_cov)
            if np.min(eigenvals) < -1e-8:
                self.logger.warning("Covariance matrix not positive semi-definite, applying regularization")
                # Add small value to diagonal for numerical stability
                weekly_cov += np.eye(len(weekly_cov)) * 1e-6
            
            return weekly_cov
            
        except Exception as e:
            self.logger.error(f"Error computing covariance matrix: {e}")
            # Fallback: diagonal covariance matrix
            n_assets = len(prices.columns)
            return pd.DataFrame(
                np.eye(n_assets) * 0.01,  # 1% weekly variance
                index=prices.columns,
                columns=prices.columns
            )
    
    def _add_cardinality_constraints(self, ef: EfficientFrontier) -> EfficientFrontier:
        """
        Add cardinality constraints using CVXPY binary variables.
        
        Args:
            ef: EfficientFrontier instance
            
        Returns:
            EfficientFrontier with cardinality constraints
        """
        n_assets = len(ef.expected_returns)
        
        # Binary variables for asset selection
        binary_vars = cp.Variable(n_assets, boolean=True)
        
        # Minimum number of assets constraint
        ef.add_constraint(cp.sum(binary_vars) >= self.params.min_names)
        
        # Link binary variables to weights
        for i in range(n_assets):
            # If binary_var[i] = 0, then weight[i] must be 0
            # If binary_var[i] = 1, then weight[i] can be > 0
            ef.add_constraint(ef._w[i] >= 1e-6 * binary_vars[i])
            ef.add_constraint(ef._w[i] <= self.params.max_weight * binary_vars[i])
        
        return ef
    
    def _optimize_portfolio_base(
        self, 
        expected_returns: pd.Series, 
        prices: pd.DataFrame
    ) -> Optional[pd.Series]:
        """
        Base portfolio optimization method with common constraints.
        
        Args:
            expected_returns: Expected weekly returns
            prices: Historical price data for covariance estimation
            
        Returns:
            Optimal weights or None if optimization fails
        """
        try:
            # Align expected returns and prices
            common_assets = expected_returns.index.intersection(prices.columns)
            if len(common_assets) == 0:
                self.logger.error("No common assets between expected returns and prices")
                return None
            
            mu = expected_returns.loc[common_assets]
            price_data = prices[common_assets]
            
            # Check minimum number of assets
            if len(common_assets) < self.params.min_names:
                self.logger.error(f"Insufficient assets: {len(common_assets)} < {self.params.min_names}")
                return None
            
            # Estimate covariance matrix
            sigma = self._prepare_covariance_matrix(price_data)
            sigma = sigma.loc[common_assets, common_assets]
            
            # Initialize EfficientFrontier
            ef = EfficientFrontier(mu, sigma)
            
            # Add L2 regularization
            ef.add_objective(objective_functions.L2_reg, gamma=self.params.l2_reg)
            
            # Add weight constraints
            for asset in mu.index:
                ef.add_constraint(lambda w, asset=asset: w[asset] <= self.params.max_weight)
            
            # Add cardinality constraints
            ef = self._add_cardinality_constraints(ef)
            
            # Optimize based on objective
            if self.params.objective == 'max_sharpe':
                weekly_rf = self.params.risk_free_rate / 52  # Convert annual to weekly
                weights = ef.max_sharpe(risk_free_rate=weekly_rf)
                
            elif self.params.objective == 'min_volatility':
                weights = ef.min_volatility()
                
            elif self.params.objective == 'efficient_risk':
                if self.params.target_volatility is None:
                    raise ValueError("target_volatility required for efficient_risk objective")
                # Convert annual to weekly volatility
                weekly_vol = self.params.target_volatility / np.sqrt(52)
                weights = ef.efficient_risk(weekly_vol)
                
            elif self.params.objective == 'efficient_return':
                if self.params.target_return is None:
                    raise ValueError("target_return required for efficient_return objective")
                # Convert annual to weekly return
                weekly_ret = (1 + self.params.target_return) ** (1/52) - 1
                weights = ef.efficient_return(weekly_ret)
                
            else:
                raise ValueError(f"Unknown objective: {self.params.objective}")
            
            # Clean weights (remove tiny positions)
            cleaned_weights = ef.clean_weights(cutoff=1e-4)
            
            # Convert to pandas Series with all original assets (zeros for non-selected)
            full_weights = pd.Series(0.0, index=expected_returns.index)
            for asset, weight in cleaned_weights.items():
                if asset in full_weights.index:
                    full_weights[asset] = weight
            
            # Normalize to ensure sum = 1
            total_weight = full_weights.sum()
            if total_weight > 0:
                full_weights = full_weights / total_weight
            
            # Validate constraints
            n_holdings = (full_weights > 1e-4).sum()
            max_weight = full_weights.max()
            
            self.logger.info(f"Optimization successful: {n_holdings} holdings, max weight: {max_weight:.1%}")
            
            return full_weights
            
        except OptimizationError as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in optimization: {e}")
            return None


async def optimize_portfolio_gru(
    gru_forecasts: pd.Series, 
    prices: pd.DataFrame, 
    params: OptimizationParams
) -> pd.Series:
    """
    Optimize portfolio using GRU return forecasts.
    
    Args:
        gru_forecasts: GRU-predicted weekly returns
        prices: Historical price data for covariance estimation
        params: Optimization parameters
        
    Returns:
        Optimal portfolio weights
    """
    if gru_forecasts.empty:
        logger.warning("No GRU forecasts provided")
        return pd.Series(dtype=float)
    
    optimizer = PortfolioOptimizer(params)
    weights = optimizer._optimize_portfolio_base(gru_forecasts, prices)
    
    if weights is not None:
        logger.info(f"GRU optimization: {(weights > 1e-4).sum()} holdings from {len(gru_forecasts)} forecasts")
    else:
        logger.error("GRU optimization failed")
        # Fallback: equal weight portfolio
        weights = pd.Series(1.0 / len(gru_forecasts), index=gru_forecasts.index)
    
    return weights


async def optimize_portfolio_historical(
    prices: pd.DataFrame, 
    params: OptimizationParams
) -> pd.Series:
    """
    Optimize portfolio using historical mean returns (benchmark strategy).
    
    Uses identical optimization procedure as GRU strategy but with different return estimates.
    
    Args:
        prices: Historical price data
        params: Optimization parameters
        
    Returns:
        Optimal portfolio weights
    """
    try:
        if prices.empty:
            logger.warning("No price data provided for historical optimization")
            return pd.Series(dtype=float)
        
        # Calculate historical mean returns
        annual_returns = expected_returns.mean_historical_return(
            prices, 
            frequency=252,  # Daily data
            compounding=True
        )
        
        # Convert annual to weekly returns
        weekly_returns = (1 + annual_returns) ** (1/52) - 1
        
        optimizer = PortfolioOptimizer(params)
        weights = optimizer._optimize_portfolio_base(weekly_returns, prices)
        
        if weights is not None:
            logger.info(f"Historical optimization: {(weights > 1e-4).sum()} holdings from {len(weekly_returns)} assets")
        else:
            logger.error("Historical optimization failed")
            # Fallback: equal weight portfolio
            weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        
        return weights
        
    except Exception as e:
        logger.error(f"Error in historical optimization: {e}")
        return pd.Series(dtype=float)


def create_market_cap_weights(
    market_cap_weights: pd.Series, 
    universe: List[str]
) -> pd.Series:
    """
    Create market cap weighted portfolio for given universe.
    
    Args:
        market_cap_weights: Market cap weights from index data
        universe: Current optimization universe
        
    Returns:
        Market cap weights aligned to universe
    """
    try:
        # Initialize universe weights to zero
        universe_weights = pd.Series(0.0, index=universe)
        
        # Map market cap weights to universe symbols
        for coin_id, weight in market_cap_weights.items():
            # Find matching symbol in universe
            for symbol in universe:
                # Check for direct match or common variations
                if (coin_id.lower() == symbol.lower() or 
                    coin_id.lower() in symbol.lower() or 
                    symbol.lower() in coin_id.lower()):
                    universe_weights[symbol] = weight
                    break
        
        # Normalize to sum to 1 within universe
        total_weight = universe_weights.sum()
        if total_weight > 0:
            universe_weights = universe_weights / total_weight
        else:
            # Equal weight fallback if no market cap data
            universe_weights = pd.Series(1.0 / len(universe), index=universe)
            logger.warning("No market cap weights found, using equal weights")
        
        n_holdings = (universe_weights > 1e-4).sum()
        logger.info(f"Market cap weights: {n_holdings} holdings, max weight: {universe_weights.max():.1%}")
        
        return universe_weights
        
    except Exception as e:
        logger.error(f"Error creating market cap weights: {e}")
        return pd.Series(1.0 / len(universe), index=universe)


def force_sell_non_universe(weights: pd.Series, current_universe: List[str]) -> pd.Series:
    """
    Force sell assets that are no longer in the current universe.
    
    Args:
        weights: Portfolio weights
        current_universe: Current investable universe
        
    Returns:
        Cleaned weights with non-universe assets set to zero
    """
    cleaned_weights = weights.copy()
    
    # Set weights to zero for assets not in current universe
    non_universe_assets = []
    for asset in cleaned_weights.index:
        if asset not in current_universe:
            cleaned_weights[asset] = 0.0
            non_universe_assets.append(asset)
    
    if non_universe_assets:
        logger.info(f"Force selling {len(non_universe_assets)} non-universe assets: {non_universe_assets}")
    
    # Renormalize remaining weights
    total_weight = cleaned_weights.sum()
    if total_weight > 0:
        cleaned_weights = cleaned_weights / total_weight
    
    return cleaned_weights


def validate_portfolio_weights(weights: pd.Series, params: OptimizationParams) -> Dict[str, bool]:
    """
    Validate portfolio weights against constraints.
    
    Args:
        weights: Portfolio weights
        params: Optimization parameters
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'sum_to_one': False,
        'non_negative': False,
        'max_weight_constraint': False,
        'min_holdings_constraint': False,
        'valid': False
    }
    
    if weights.empty:
        return validation
    
    # Check sum to one (within tolerance)
    total_weight = weights.sum()
    validation['sum_to_one'] = abs(total_weight - 1.0) < 1e-6
    
    # Check non-negative weights
    validation['non_negative'] = (weights >= -1e-6).all()
    
    # Check maximum weight constraint
    max_weight = weights.max()
    validation['max_weight_constraint'] = max_weight <= params.max_weight + 1e-6
    
    # Check minimum holdings constraint
    n_holdings = (weights > 1e-4).sum()
    validation['min_holdings_constraint'] = n_holdings >= params.min_names
    
    # Overall validity
    validation['valid'] = all([
        validation['sum_to_one'],
        validation['non_negative'],
        validation['max_weight_constraint'],
        validation['min_holdings_constraint']
    ])
    
    if not validation['valid']:
        logger.warning(f"Portfolio validation failed: {validation}")
    
    return validation


def calculate_portfolio_statistics(
    weights: pd.Series, 
    expected_returns: pd.Series, 
    covariance_matrix: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate portfolio statistics for a given weight allocation.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns
        covariance_matrix: Covariance matrix
        
    Returns:
        Dictionary with portfolio statistics
    """
    try:
        # Align data
        common_assets = weights.index.intersection(expected_returns.index).intersection(covariance_matrix.index)
        w = weights.loc[common_assets]
        mu = expected_returns.loc[common_assets]
        sigma = covariance_matrix.loc[common_assets, common_assets]
        
        # Portfolio expected return
        portfolio_return = (w * mu).sum()
        
        # Portfolio volatility
        portfolio_variance = np.dot(w, np.dot(sigma, w))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Number of holdings
        n_holdings = (w > 1e-4).sum()
        
        # Concentration measures
        max_weight = w.max()
        herfindahl_index = (w ** 2).sum()  # Portfolio concentration
        
        stats = {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'n_holdings': int(n_holdings),
            'max_weight': float(max_weight),
            'herfindahl_index': float(herfindahl_index),
            'effective_holdings': float(1 / herfindahl_index) if herfindahl_index > 0 else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating portfolio statistics: {e}")
        return {'error': str(e)}


def compare_optimization_strategies(
    gru_weights: pd.Series,
    historical_weights: pd.Series,
    market_cap_weights: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare portfolio characteristics across all three strategies.
    
    Args:
        gru_weights: GRU-optimized weights
        historical_weights: Historical mean optimized weights
        market_cap_weights: Market cap weighted portfolio
        expected_returns: Expected returns
        covariance_matrix: Covariance matrix
        
    Returns:
        DataFrame comparing portfolio characteristics
    """
    strategies = {
        'GRU': gru_weights,
        'Historical': historical_weights,
        'Market_Cap': market_cap_weights
    }
    
    comparison_data = []
    
    for strategy_name, weights in strategies.items():
        if not weights.empty:
            stats = calculate_portfolio_statistics(weights, expected_returns, covariance_matrix)
            stats['Strategy'] = strategy_name
            comparison_data.append(stats)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Strategy', inplace=True)
        return comparison_df
    else:
        return pd.DataFrame()


def rebalance_portfolio(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rebalance_threshold: float = 0.05
) -> Tuple[pd.Series, float]:
    """
    Calculate rebalancing trades and turnover.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        rebalance_threshold: Minimum weight change to trigger rebalancing
        
    Returns:
        Tuple of (trades, turnover)
    """
    # Align weights
    all_assets = current_weights.index.union(target_weights.index)
    current_aligned = current_weights.reindex(all_assets, fill_value=0.0)
    target_aligned = target_weights.reindex(all_assets, fill_value=0.0)
    
    # Calculate trades
    trades = target_aligned - current_aligned
    
    # Apply rebalancing threshold
    small_trades = trades.abs() < rebalance_threshold
    trades[small_trades] = 0.0
    
    # Adjust target weights to account for threshold
    adjusted_target = current_aligned + trades
    
    # Calculate turnover (sum of absolute trades)
    turnover = trades.abs().sum()
    
    return adjusted_target, turnover
