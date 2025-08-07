"""
Model prediction generation for portfolio rebalancing.

This module handles loading trained models and generating predictions for specific
rebalancing dates using the same infrastructure as the model evaluator.
"""

import logging
import os
import pickle
from datetime import date, timedelta
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from darts import TimeSeries
from kallos_models.architectures import load_model_with_custom_loss
from kallos_models import datasets
from kallos_models.preprocessing import transform_features_to_dataframe

from .exceptions import InsufficientDataError, DatabaseError
from .storage import get_async_session

logger = logging.getLogger(__name__)

def match_rebalancing_to_models(rebalancing_dates: List[date], model_definitions: pd.DataFrame) -> pd.DataFrame:
    """
    Match each rebalancing date to its corresponding model definition entries.
    
    Args:
        rebalancing_dates: List of rebalancing dates
        model_definitions: DataFrame with model data including test date ranges
        
    Returns:
        DataFrame with rebalancing_date and corresponding model data
    """
    matched_data = []
    
    for rebal_date in rebalancing_dates:
        # Find model definitions where rebalancing date falls within test period
        matching_models = model_definitions[
            (model_definitions['test_start_date'] <= rebal_date) & 
            (model_definitions['test_end_date'] >= rebal_date)
        ].copy()
        
        if not matching_models.empty:
            # Add the rebalancing date to each matching model
            matching_models['rebalancing_date'] = rebal_date
            matched_data.append(matching_models)
    
    if matched_data:
        result_df = pd.concat(matched_data, ignore_index=True)
        logger.info(f"Matched {len(result_df)} model-rebalancing combinations")
        return result_df
    else:
        logger.warning("No matching model-rebalancing combinations found")
        return pd.DataFrame()


def generate_predictions(rebalancing_model_map, feature_groups, db_kwargs):
    """
    Generate predictions for each model in the rebalancing model map.
    
    Args:
        rebalancing_model_map (pd.DataFrame): DataFrame containing model definitions and rebalancing dates.
        feature_groups (dict): Dictionary of feature groups for normalization.
        db_kwargs (dict): Database connection parameters.
        
    Returns:
        pd.DataFrame: DataFrame containing forecasts for each model.
    """
    forecasts = []
    for index, model in rebalancing_model_map.iterrows(): 
        asset_id = model['coin_id']   
        target_col = 'pct_return_30d'
        model_path = os.path.join('trained_models', model['coin_id'], model['study_name'] + '.pt')
        scaler_path = os.path.join('trained_models', model['coin_id'], model['study_name'] + '_scaler.pkl')
        test_start_date = model['test_start_date']
        test_end_date = model['test_end_date']

        # load model
        model = load_model_with_custom_loss(model_path)
        # load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # 2. Load data for the required period (test set + lookback window)
        # The model needs `input_chunk_length` of data prior to the test start date
        required_start_date = pd.to_datetime(test_start_date) - pd.DateOffset(days=model.input_chunk_length)

        # Use the same database loading approach as other modules
        full_df = datasets.load_features_from_db(asset_id, test_end_date, db_kwargs)
        eval_df = full_df.loc[required_start_date:]

        # 3. Prepare data
        all_feature_cols = [col for group in feature_groups.values() for col in group]
        target_series = TimeSeries.from_dataframe(eval_df[[target_col]], freq=eval_df.index.freq)

        features_df = eval_df[all_feature_cols]

        # Use the new function to handle transformation and DataFrame creation
        features_norm_df = transform_features_to_dataframe(
            scaler, features_df, feature_groups
        )

        covariates_series = TimeSeries.from_dataframe(features_norm_df, freq=eval_df.index.freq)

        # 4. Generate forecast - use fixed 90 days for test period
        
        forecast_series = model.predict(
            n=90,  # Fixed 90-day prediction window
            past_covariates=covariates_series  # Match tuner.py approach (no target series)
        )

        # 5. Store forecast
        forecast_df = forecast_series.to_series().reset_index()
        forecast_df['coin_id'] = asset_id    
        forecasts.append(forecast_df)

    # Combine all forecasts into a single DataFrame
    forecasts = pd.concat(forecasts, ignore_index=True)
    forecasts.rename(columns={'timestamp':'date','pct_return_30d':'estimate'}, inplace=True)

    return forecasts