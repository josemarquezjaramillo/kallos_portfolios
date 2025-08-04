"""
Model loading and inference for Kallos Portfolios with temporal model selection.

"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Import Darts models and custom loss function
try:
    from darts.models import BlockRNNModel, TransformerModel
    from darts.models.forecasting.forecasting_model import ForecastingModel
    from kallos_models.loss import DirectionSelectiveMSELoss
    from kallos_models.architectures import load_model_with_custom_loss
    DARTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Darts or kallos_models not available: {e}. Falling back to basic model loading.")
    DARTS_AVAILABLE = False

from .exceptions import ModelLoadingError, ModelInferenceError

logger = logging.getLogger(__name__)

# Constants for feature engineering and inference
DEFAULT_LOOKBACK_DAYS = 30
VOLATILITY_WINDOW = 5
MOMENTUM_WINDOW = 5
MAX_WEEKLY_RETURN = 0.5
MIN_WEEKLY_RETURN = -0.5
DEFAULT_MAX_WORKERS = 4


def get_quarter_info(target_date: date) -> Tuple[int, str]:
    """
    Get year and quarter for a given date.
    
    Args:
        target_date: Date to get quarter info for
        
    Returns:
        Tuple of (year, quarter_string) e.g., (2023, "Q2")
    """
    year = target_date.year
    month = target_date.month
    
    if month <= 3:
        quarter = "Q1"
    elif month <= 6:
        quarter = "Q2"
    elif month <= 9:
        quarter = "Q3"
    else:
        quarter = "Q4"
    
    return year, quarter


def get_previous_quarter(year: int, quarter: str) -> Tuple[int, str]:
    """
    Get the previous quarter for model selection.
    
    Args:
        year: Current year
        quarter: Current quarter (Q1, Q2, Q3, Q4)
        
    Returns:
        Tuple of (previous_year, previous_quarter)
    """
    quarter_map = {"Q1": "Q4", "Q2": "Q1", "Q3": "Q2", "Q4": "Q3"}
    prev_quarter = quarter_map[quarter]
    prev_year = year - 1 if quarter == "Q1" else year
    
    return prev_year, prev_quarter


async def get_available_models_for_date(
    session: AsyncSession, 
    target_date: date,
    coin_ids: List[str],
    model_type: str = "gru"
) -> Dict[str, str]:
    """
    Get available models for the given date from the database.
    
    Args:
        session: Database session
        target_date: Date for which to find models
        coin_ids: List of coin IDs to get models for
        model_type: Model type (default: 'gru')
        
    Returns:
        Dictionary mapping coin_id to model_name for available models
    """
    try:
        # Find the appropriate quarter for the target date
        target_year, target_quarter = get_quarter_info(target_date)
        
        # Get the previous quarter (models are trained on previous quarter)
        model_year, model_quarter = get_previous_quarter(target_year, target_quarter)
        
        # Query available models from the model_train_view (public schema)
        query = text("""
            SELECT coin_id, study_name, test_start_date, test_end_date
            FROM public.model_train_view
            WHERE model = :model_type
            AND coin_id = ANY(:coin_ids)
            AND year_end = :model_year
            AND quarter_end = :model_quarter
            AND :target_date BETWEEN test_start_date AND test_end_date
            ORDER BY coin_id
        """)
        
        result = await session.execute(query, {
            'model_type': model_type,
            'coin_ids': coin_ids,
            'model_year': model_year,
            'model_quarter': model_quarter,
            'target_date': target_date
        })
        
        data = result.fetchall()
        
        # Create mapping from coin_id to model_name
        available_models = {}
        for row in data:
            coin_id = row[0]
            study_name = row[1]
            available_models[coin_id] = study_name
        
        logger.info(
            f"Found {len(available_models)} models for {target_date} "
            f"(looking for {model_type} models from {model_year} {model_quarter})"
        )
        
        return available_models
        
    except Exception as e:
        logger.error(f"Error querying available models: {e}")
        return {}


class ModelManager:
    """
    Enhanced model manager with temporal model selection and caching.
    """
    
    def __init__(self, model_dir: Path, cache_models: bool = True):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory containing trained model files
            cache_models: Whether to cache loaded models in memory
        """
        self.model_dir = Path(model_dir)
        self.cache_models = cache_models
        self._model_cache: Dict[str, object] = {}
        self._scaler_cache: Dict[str, StandardScaler] = {}
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")
    
    def _get_model_files(self, model_name: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Get model and scaler file paths for a model name.
        
        Args:
            model_name: Full model name (e.g., 'gru_bitcoin_2023_Q2_7D_customloss')
            
        Returns:
            Tuple of (model_path, scaler_path)
        """
        # Look for different possible model file extensions
        possible_extensions = ['.pt', '.pkl', '.joblib', '.darts']
        model_path = None
        
        for ext in possible_extensions:
            candidate = self.model_dir / f"{model_name}{ext}"
            if candidate.exists():
                model_path = candidate
                break
        
        # Scaler should have the _scaler.pkl suffix
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        
        return model_path, scaler_path if scaler_path.exists() else None
    
    def load_model(self, model_name: str) -> object:
        """
        Load a model by its full name with enhanced compatibility.
        
        Args:
            model_name: Full model name (e.g., 'gru_bitcoin_2023_Q2_7D_customloss')
            
        Returns:
            Loaded model (Darts ForecastingModel or PyTorch model)
            
        Raises:
            ModelLoadingError: If model cannot be loaded
        """
        if self.cache_models and model_name in self._model_cache:
            return self._model_cache[model_name]
        
        model_path, _ = self._get_model_files(model_name)
        
        if model_path is None:
            raise ModelLoadingError(f"Model file not found for: {model_name}")
        
        try:
            if DARTS_AVAILABLE:
                # Try to load with kallos_models compatibility first
                try:
                    model = load_model_with_custom_loss(str(model_path))
                    logger.debug(f"Loaded Darts model with custom loss: {model_name}")
                except Exception as darts_error:
                    logger.warning(f"Darts custom loader failed for {model_name}, trying fallback: {darts_error}")
                    # Fallback to basic PyTorch loading
                    model = torch.load(model_path, map_location='cpu')
                    if hasattr(model, 'eval'):
                        model.eval()
                    logger.debug(f"Loaded PyTorch model (fallback): {model_name}")
            else:
                # Basic PyTorch loading
                model = torch.load(model_path, map_location='cpu')
                if hasattr(model, 'eval'):
                    model.eval()
                logger.debug(f"Loaded PyTorch model: {model_name}")
            
            if self.cache_models:
                self._model_cache[model_name] = model
            
            return model
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model {model_name}: {str(e)}") from e
    
    def load_scaler(self, model_name: str) -> StandardScaler:
        """
        Load a scaler by its model name.
        
        Args:
            model_name: Full model name
            
        Returns:
            Fitted StandardScaler
            
        Raises:
            ModelLoadingError: If scaler cannot be loaded
        """
        if self.cache_models and model_name in self._scaler_cache:
            return self._scaler_cache[model_name]
        
        _, scaler_path = self._get_model_files(model_name)
        
        if scaler_path is None:
            raise ModelLoadingError(f"Scaler file not found for: {model_name}")
        
        try:
            scaler = joblib.load(scaler_path)
            
            if not isinstance(scaler, StandardScaler):
                logger.warning(f"Expected StandardScaler, got {type(scaler)} for {model_name}")
                # Try to create a compatible scaler if possible
                if hasattr(scaler, 'transform') and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    logger.info(f"Using compatible scaler for {model_name}")
                else:
                    raise TypeError(f"Incompatible scaler type: {type(scaler)}")
            
            if self.cache_models:
                self._scaler_cache[model_name] = scaler
            
            logger.debug(f"Loaded scaler for: {model_name}")
            return scaler
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load scaler for {model_name}: {str(e)}") from e
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available model symbols in the model directory.
        
        Returns:
            List of available symbols
        """
        if not self.model_dir.exists():
            return []
        
        symbols = set()
        for file_path in self.model_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith('_scaler.pkl'):
                # Extract coin_id from full model name
                file_stem = file_path.stem
                if '_model' in file_stem:
                    symbol = file_stem.split('_model')[0]
                else:
                    # For quarterly models like 'gru_bitcoin_2023_Q2_7D_customloss'
                    parts = file_stem.split('_')
                    if len(parts) >= 2:
                        symbol = parts[1]  # Extract coin_id
                
                if symbol:
                    symbols.add(symbol)
        
        return sorted(list(symbols))


def prepare_features(prices: pd.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, np.ndarray]:
    """
    Prepare features for model inference.
    
    Args:
        prices: Price DataFrame with coin_ids as columns
        lookback_days: Number of days for feature engineering
        
    Returns:
        Dictionary mapping coin_id to feature arrays
    """
    features_dict = {}
    
    for coin_id in prices.columns:
        coin_prices = prices[coin_id].dropna()
        
        if len(coin_prices) < lookback_days:
            logger.warning(f"Insufficient data for {coin_id}: {len(coin_prices)} < {lookback_days}")
            continue
        
        # Calculate features
        recent_prices = coin_prices.iloc[-lookback_days:]
        
        # Price-based features
        price_pct_change = recent_prices.pct_change().fillna(0)
        log_returns = np.log(recent_prices / recent_prices.shift(1)).fillna(0)
        
        # Volatility features
        volatility = price_pct_change.rolling(window=min(VOLATILITY_WINDOW, len(price_pct_change))).std().fillna(0)
        
        # Momentum features  
        momentum = recent_prices.rolling(window=min(MOMENTUM_WINDOW, len(recent_prices))).mean().pct_change().fillna(0)
        
        # Combine features
        features = np.column_stack([
            recent_prices.values,
            price_pct_change.values,
            log_returns.values,
            volatility.values,
            momentum.values
        ])
        
        features_dict[coin_id] = features
    
    return features_dict


def predict_single_symbol_darts(coin_id: str, features: np.ndarray, model: object, scaler: StandardScaler) -> float:
    """
    Generate prediction for a single coin using Darts model.
    
    Args:
        coin_id: Coin identifier
        features: Prepared features array
        model: Loaded Darts model
        scaler: Fitted scaler
        
    Returns:
        Predicted weekly return
    """
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        # For Darts models, we need to create TimeSeries
        # This is a simplified version - you may need to adjust based on your exact setup
        if hasattr(model, 'predict'):
            # Create a simple time series from the features
            dates = pd.date_range('2023-01-01', periods=len(features_scaled), freq='D')
            ts = pd.Series(features_scaled[:, 0], index=dates)
            
            # Generate prediction
            forecast = model.predict(n=7)  # 7-day forecast
            weekly_return = float(forecast.values()[-1])
        else:
            # Fallback to PyTorch-style prediction
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            with torch.no_grad():
                prediction = model(features_tensor)
                weekly_return = float(prediction.squeeze().item())
        
        # Apply bounds
        weekly_return = max(MIN_WEEKLY_RETURN, min(MAX_WEEKLY_RETURN, weekly_return))
        
        return weekly_return
        
    except Exception as e:
        logger.error(f"Darts prediction failed for {coin_id}: {e}")
        raise


def predict_single_symbol_pytorch(coin_id: str, features: np.ndarray, model: object, scaler: StandardScaler) -> float:
    """
    Generate prediction for a single coin using PyTorch model.
    
    Args:
        coin_id: Coin identifier
        features: Prepared features array
        model: Loaded PyTorch model
        scaler: Fitted scaler
        
    Returns:
        Predicted weekly return
    """
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            if hasattr(model, '__call__'):
                prediction = model(features_tensor)
            else:
                # Handle different model types
                prediction = model.forward(features_tensor)
            
            weekly_return = float(prediction.squeeze().item())
        
        # Apply bounds
        weekly_return = max(MIN_WEEKLY_RETURN, min(MAX_WEEKLY_RETURN, weekly_return))
        
        return weekly_return
        
    except Exception as e:
        logger.error(f"PyTorch prediction failed for {coin_id}: {e}")
        raise


async def forecast_returns(
    prices: pd.DataFrame,
    model_dir: Path,
    target_date: date = None,
    session: AsyncSession = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    model_type: str = "gru"
) -> pd.Series:
    """
    Generate return forecasts using temporal model selection.
    
    Args:
        prices: Price data DataFrame with coin_ids as columns
        model_dir: Directory containing trained models
        target_date: Date for which to generate forecasts
        session: Database session for model lookup
        lookback_days: Number of days for feature engineering
        max_workers: Maximum threads for parallel processing
        model_type: Model type (default: 'gru')
        
    Returns:
        Series of predicted weekly returns indexed by coin_id
        
    Raises:
        ModelInferenceError: If inference fails critically
        ValueError: If input data is invalid
    """
    if prices.empty:
        raise ValueError("Price data is empty")
    
    if len(prices) < lookback_days:
        raise ValueError(f"Insufficient price history: {len(prices)} < {lookback_days} required")
    
    try:
        # Get coin_ids from price data
        coin_ids = list(prices.columns)
        
        # If session is provided, use temporal model selection
        if session is not None:
            if target_date is None:
                target_date = prices.index[-1].date() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            
            # Find available models for the target date
            available_models = await get_available_models_for_date(
                session, target_date, coin_ids, model_type
            )
            
            if not available_models:
                raise ModelInferenceError(
                    f"No {model_type} models available for {target_date}. "
                    f"Requested coin_ids: {coin_ids}"
                )
            
            missing_models = [coin_id for coin_id in coin_ids if coin_id not in available_models]
            if missing_models:
                logger.warning(f"No models found for coin_ids: {missing_models}")
        
        else:
            # Fallback to simple model names for backward compatibility
            available_models = {coin_id: f"{model_type}_{coin_id}_model" for coin_id in coin_ids}
        
        # Initialize model manager
        model_manager = ModelManager(model_dir, cache_models=True)
        
        # Prepare features for coins with available models
        available_coin_ids = list(available_models.keys())
        features_dict = prepare_features(prices[available_coin_ids], lookback_days)
        
        # Load models and scalers
        models = {}
        scalers = {}
        loading_errors = {}
        
        for coin_id, model_name in available_models.items():
            try:
                model = model_manager.load_model(model_name)
                scaler = model_manager.load_scaler(model_name)
                models[coin_id] = model
                scalers[coin_id] = scaler
                logger.debug(f"Loaded model for {coin_id}: {model_name}")
            except Exception as e:
                loading_errors[coin_id] = str(e)
                logger.warning(f"Failed to load model for {coin_id} ({model_name}): {e}")
        
        if not models:
            raise ModelInferenceError(f"No models could be loaded. Errors: {loading_errors}")
        
        # Parallel prediction
        predictions = {}
        inference_errors = {}
        
        def predict_coin(coin_id: str) -> Optional[float]:
            """Predict returns for a single coin_id."""
            try:
                if coin_id not in features_dict:
                    logger.warning(f"No features available for {coin_id}")
                    return None
                
                features = features_dict[coin_id]
                model = models[coin_id]
                scaler = scalers[coin_id]
                
                # Try Darts prediction first, fallback to PyTorch
                if DARTS_AVAILABLE and hasattr(model, 'predict'):
                    result = predict_single_symbol_darts(coin_id, features, model, scaler)
                else:
                    result = predict_single_symbol_pytorch(coin_id, features, model, scaler)
                
                return result
                
            except Exception as e:
                inference_errors[coin_id] = str(e)
                logger.warning(f"Inference failed for {coin_id}: {e}")
                return None
        
        # Execute parallel inference
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_coin = {
                executor.submit(predict_coin, coin_id): coin_id 
                for coin_id in models.keys()
            }
            
            for future in future_to_coin:
                coin_id = future_to_coin[future]
                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        predictions[coin_id] = result
                except Exception as e:
                    inference_errors[coin_id] = str(e)
                    logger.warning(f"Future execution failed for {coin_id}: {e}")
        
        # Validate results
        if not predictions:
            raise ModelInferenceError(
                f"No successful predictions generated. "
                f"Loading errors: {loading_errors}, "
                f"Inference errors: {inference_errors}"
            )
        
        success_rate = len(predictions) / len(available_coin_ids)
        logger.info(
            f"Generated predictions for {len(predictions)}/{len(available_coin_ids)} coin_ids "
            f"({success_rate:.1%} success rate)"
        )
        
        return pd.Series(predictions)
        
    except Exception as e:
        if isinstance(e, (ModelInferenceError, ValueError)):
            raise
        logger.error(f"Unexpected error in forecast_returns: {e}")
        raise ModelInferenceError(f"Forecast generation failed: {str(e)}") from e


def validate_model_availability(model_dir: Path, coin_ids: List[str]) -> Dict[str, bool]:
    """
    Check which models are available for the given coin IDs.
    
    Args:
        model_dir: Directory containing model files
        coin_ids: List of coin IDs to check
        
    Returns:
        Dictionary mapping coin_id to availability boolean
    """
    model_manager = ModelManager(model_dir, cache_models=False)
    availability = {}
    
    for coin_id in coin_ids:
        try:
            # Try to find model files for this coin_id
            model_path, scaler_path = model_manager._get_model_files(f"gru_{coin_id}_model")
            availability[coin_id] = model_path is not None and scaler_path is not None
        except Exception:
            availability[coin_id] = False
    
    return availability
