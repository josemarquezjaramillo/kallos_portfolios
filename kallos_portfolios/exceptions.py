"""
Custom exception classes for Kallos Portfolios system.
Provides specific error types for better error handling and debugging.
"""


class KallosPortfoliosError(Exception):
    """Base exception class for all Kallos Portfolios errors."""
    pass


class DataValidationError(KallosPortfoliosError):
    """Exception raised when input data fails validation checks."""
    pass


class InsufficientDataError(KallosPortfoliosError):
    """Exception raised when there is insufficient data for analysis."""
    pass


class ModelLoadingError(KallosPortfoliosError):
    """Exception raised when model files cannot be loaded."""
    pass


class ModelInferenceError(KallosPortfoliosError):
    """Exception raised when model inference fails."""
    pass


class OptimizationError(KallosPortfoliosError):
    """Exception raised when portfolio optimization fails."""
    pass


class BacktestError(KallosPortfoliosError):
    """Exception raised when backtesting encounters errors."""
    pass


class DatabaseError(KallosPortfoliosError):
    """Exception raised for database-related errors."""
    pass


class ConfigurationError(KallosPortfoliosError):
    """Exception raised for configuration-related errors."""
    pass
