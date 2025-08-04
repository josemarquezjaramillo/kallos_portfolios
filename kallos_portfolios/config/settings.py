"""
Configuration management for Kallos Portfolios system using Pydantic V2.
Handles environment-driven configuration with validation and type safety.
"""

from pathlib import Path
from typing import Optional, Literal
from datetime import date, datetime
import re
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class OptimizationParams(BaseModel):
    """Portfolio optimization parameters with comprehensive validation."""
    
    # Core optimization settings
    objective: Literal['max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return'] = Field(
        default='max_sharpe',
        description="Portfolio optimization objective function"
    )
    max_weight: float = Field(
        default=0.35,
        gt=0.01,
        le=1.0,
        description="Maximum allocation per asset (0.01-1.0)"
    )
    min_names: int = Field(
        default=3,
        ge=2,
        le=50,
        description="Minimum number of holdings for diversification (2-50)"
    )
    l2_reg: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="L2 regularization parameter for weight smoothing (0.0-1.0)"
    )
    
    # Risk and return parameters
    risk_free_rate: float = Field(
        default=0.0,
        ge=-0.1,
        le=0.5,
        description="Annual risk-free rate for Sharpe ratio calculation (-0.1 to 0.5)"
    )
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Risk aversion parameter for mean-variance optimization (0.0-10.0)"
    )
    target_volatility: Optional[float] = Field(
        default=None,
        gt=0.01,
        le=2.0,
        description="Target portfolio volatility for efficient risk optimization (0.01-2.0)"
    )
    target_return: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=5.0,
        description="Target portfolio return for efficient return optimization (-1.0 to 5.0)"
    )
    
    # Data parameters
    lookback_days: int = Field(
        default=252,
        ge=30,
        le=1000,
        description="Historical lookback period for covariance estimation (30-1000 days)"
    )
    start_date: str = Field(
        description="Portfolio start date (YYYY-MM-DD)"
    )
    end_date: str = Field(
        description="Portfolio end date (YYYY-MM-DD)"
    )
    
    @field_validator('objective')
    @classmethod
    def validate_objective(cls, v):
        """Validates portfolio optimization objectives."""
        allowed = ['max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return']
        if v not in allowed:
            raise ValueError(f'Objective must be one of {allowed}')
        return v
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        """Validates date format is YYYY-MM-DD."""
        from datetime import datetime
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v

    @model_validator(mode='after')
    def validate_date_range(self) -> 'OptimizationParams':
        """Validate date range and combinations."""
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d').date()
        
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        # Check minimum backtest period
        days_diff = (end_date - start_date).days
        if days_diff < 30:
            raise ValueError(f"Backtest period too short: {days_diff} days (minimum 30)")
        
        # Check lookback vs backtest period
        if self.lookback_days >= days_diff:
            raise ValueError(
                f"lookback_days ({self.lookback_days}) must be less than "
                f"backtest period ({days_diff} days)"
            )
        
        return self

    @model_validator(mode='after')
    def validate_objective_parameters(self) -> 'OptimizationParams':
        """Validate objective-specific parameters."""
        if self.objective == 'efficient_risk' and self.target_volatility is None:
            raise ValueError("target_volatility required for efficient_risk objective")
        
        if self.objective == 'efficient_return' and self.target_return is None:
            raise ValueError("target_return required for efficient_return objective")
        
        # Validate max_weight vs min_names consistency
        theoretical_max_weight = 1.0 / self.min_names
        if self.max_weight < theoretical_max_weight * 0.8:  # Allow some tolerance
            raise ValueError(
                f"max_weight ({self.max_weight:.3f}) too low for min_names ({self.min_names}). "
                f"Theoretical minimum: {theoretical_max_weight:.3f}"
            )
        
        return self


class Settings(BaseSettings):
    """
    Main application settings with environment variable support.
    
    Environment variables can be prefixed with KALLOS_ for namespace isolation.
    Example: KALLOS_DATABASE_URL, KALLOS_MODEL_DIR, etc.
    """
    
    # Database configuration
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/kallos_portfolios",
        description="PostgreSQL connection string with asyncpg driver"
    )
    
    # Model and data paths
    model_dir: Path = Field(
        default=Path("/home/jlmarquez11/kallos/trained_models"),
        description="Directory containing GRU models and scalers"
    )
    report_path: Path = Field(
        default=Path("/home/jlmarquez11/kallos/kallos_portfolios/reports"),
        description="Output directory for analysis reports"
    )
    
    # Logging configuration
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = Field(
        default='INFO',
        description="Logging verbosity level"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Optional log file path (defaults to console only)"
    )
    
    # Performance settings
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum worker threads for parallel model inference"
    )
    db_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    db_max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum database connection pool overflow"
    )
    
    @field_validator('model_dir', 'report_path')
    @classmethod
    def validate_and_create_paths(cls, v):
        """Ensures directories exist and are accessible."""
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        """Validates PostgreSQL URL format with asyncpg driver."""
        if not v.startswith(('postgresql+asyncpg://', 'postgresql://')):
            raise ValueError('Database URL must use postgresql+asyncpg:// or postgresql:// scheme')
        return v
    
    class Config:
        env_prefix = 'KALLOS_'
        case_sensitive = False
        env_file = '.env'
        env_file_encoding = 'utf-8'


# Global settings instance
settings = Settings()


def get_optimization_params_from_db(run_id: str) -> OptimizationParams:
    """
    Load optimization parameters from database for a specific run.
    This will be implemented in storage.py to fetch from optimization_params table.
    """
    # Placeholder - will be implemented with actual database query
    raise NotImplementedError("Will be implemented in storage.py")


def create_run_id(prefix: str = "portfolio", timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique run ID for portfolio optimization runs with validation.
    
    Args:
        prefix: String prefix for the run ID (alphanumeric, hyphens, underscores only)
        timestamp: Custom timestamp (defaults to now)
        
    Returns:
        Unique run ID string with timestamp
        
    Raises:
        ValueError: If prefix contains invalid characters
    """
    # Validate prefix
    if not prefix:
        raise ValueError("Prefix cannot be empty")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', prefix):
        raise ValueError("Prefix can only contain alphanumeric characters, hyphens, and underscores")
    
    if len(prefix) > 50:
        raise ValueError("Prefix too long (max 50 characters)")
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Format: prefix_YYYYMMDD_HHMMSS
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}_{timestamp_str}"
    
    # Final validation of generated run_id length
    if len(run_id) > 100:
        raise ValueError(f"Generated run_id too long: {len(run_id)} > 100 characters")
    
    return run_id


def validate_run_id(run_id: str) -> str:
    """
    Validate run_id format for SQL safety and consistency.
    
    Args:
        run_id: The run ID to validate
        
    Returns:
        The validated run_id
        
    Raises:
        ValueError: If run_id format is invalid
    """
    if not run_id:
        raise ValueError("run_id cannot be empty")
    
    if len(run_id) > 100:
        raise ValueError("run_id too long (max 100 characters)")
    
    # Allow alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', run_id):
        raise ValueError("run_id can only contain alphanumeric characters, hyphens, and underscores")
    
    return run_id
