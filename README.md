# Kallos Portfolios

**Cryptocurrency Portfolio Construction and Backtesting Framework**

A comprehensive portfolio optimization system that implements multiple return forecasting strategies for cryptocurrency investments. The framework provides a modular architecture for comparing GRU neural network predictions against traditional historical methods and market-weighted benchmarks.

## Overview

Kallos Portfolios is designed as a modular framework that enables systematic comparison of different portfolio construction approaches. The system implements three distinct strategies:

1. **GRU-based Portfolio Optimization**: Utilizes neural network return predictions for forward-looking portfolio construction
2. **Historical Returns Benchmark**: Implements traditional mean-variance optimization using historical return estimates
3. **Market-weighted Benchmark**: Provides passive index comparison using market capitalization weights

The framework emphasizes code reusability through inheritance-based architecture, eliminating duplication while maintaining strategy-specific customization capabilities.

## Architecture

### Core Package Structure

The system is organized into specialized modules, each handling distinct aspects of the portfolio optimization workflow:

```
kallos_portfolios/
├── simulators/           # Portfolio simulation engines
├── analysis/            # Cross-strategy analysis tools
├── evaluation.py        # Performance measurement and reporting
├── storage.py          # Database operations and data persistence
├── optimisers.py       # Portfolio optimization algorithms
├── backtest.py         # Backtesting infrastructure
├── models.py           # Machine learning model management
├── datasets.py         # Data loading and preprocessing
├── predictors.py       # Return forecasting implementations
└── exceptions.py       # Custom exception definitions
```

### Module Interactions

The system follows a layered architecture where modules interact through well-defined interfaces:

**Data Layer**: `storage.py` provides unified data access, supporting both historical price retrieval and portfolio result persistence. It abstracts database operations and ensures consistent data formatting across all modules.

**Model Layer**: `models.py` handles machine learning model lifecycle management, including model discovery, loading, and prediction generation. `predictors.py` extends this with strategy-specific forecasting implementations.

**Optimization Layer**: `optimisers.py` implements portfolio optimization algorithms, accepting expected return forecasts and producing optimal weight allocations using modern portfolio theory principles.

**Simulation Layer**: The `simulators/` package contains the core simulation engines that orchestrate the complete workflow from data loading through performance evaluation.

**Analysis Layer**: `evaluation.py` and the `analysis/` package provide comprehensive performance measurement, statistical testing, and comparative analysis capabilities.

## Simulators Package

### Base Simulator Architecture

The simulators package implements an inheritance-based architecture that eliminates code duplication while maintaining flexibility for strategy-specific implementations.

#### BasePortfolioSimulator

The base simulator class provides shared functionality used by all portfolio strategies:

**Universe Management**: Handles investable asset universe definition and temporal asset availability. The `get_assets_for_date()` method manages monthly universe updates and ensures consistent asset selection across strategies.

**Portfolio Optimization**: Implements the complete optimization workflow through `optimize_portfolio()`, including covariance matrix estimation, constraint definition, and numerical optimization execution.

**Performance Simulation**: The `simulate_performance()` method uses VectorBT for realistic backtesting, including transaction costs, rebalancing mechanics, and portfolio drift simulation.

**Result Persistence**: Manages database storage of portfolio weights, daily returns, and performance metrics through the `save_results()` method.

**Report Generation**: Coordinates with the evaluation module to produce QuantStats tearsheets and comprehensive performance reports.

The base class defines the template method `get_expected_returns_for_date()` which subclasses must implement to provide strategy-specific return forecasts.

#### GRUPortfolioSimulator

Extends BasePortfolioSimulator to implement neural network-based return forecasting:

**Prediction Management**: Loads and manages GRU model predictions from pre-computed datasets. Implements temporal matching to align predictions with rebalancing dates.

**Data Validation**: Performs comprehensive validation of neural network outputs, including duplicate removal, outlier detection, and missing value handling.

**Temporal Alignment**: Manages the mapping between rebalancing dates and available prediction dates, implementing fallback strategies for missing predictions.

#### HistoricalPortfolioSimulator

Extends BasePortfolioSimulator to implement traditional mean-reversion strategies:

**Historical Return Calculation**: Uses PyPortfolioOpt to compute historical mean returns over configurable lookback periods. Supports multiple frequency conversions and compounding methodologies.

**Data Quality Management**: Implements robust handling of missing historical data, including coverage requirements and data sufficiency checks.

**Benchmark Implementation**: Provides a direct comparison baseline that represents traditional quantitative portfolio management approaches.

### Simulator Workflow Integration

The simulators coordinate with other system modules through standardized interfaces:

1. **Data Acquisition**: Simulators request price data from `storage.py` with specific date ranges and coverage requirements
2. **Return Forecasting**: Each simulator implements strategy-specific forecasting through the `get_expected_returns_for_date()` method
3. **Optimization Execution**: Simulators pass expected returns to `optimisers.py` for constraint-based optimization
4. **Performance Evaluation**: Completed simulations integrate with `evaluation.py` for comprehensive performance analysis
5. **Result Storage**: All results are persisted through `storage.py` using standardized schemas

## Analysis Package

### Three-Strategy Comparison Framework

The analysis package provides sophisticated tools for systematic strategy comparison:

**Statistical Testing**: Implements hypothesis testing for return differences, volatility comparisons, and risk-adjusted performance evaluation. Uses appropriate statistical tests for time series data.

**Performance Attribution**: Decomposes portfolio performance into return forecasting quality, optimization efficiency, and execution effectiveness components.

**Risk Analysis**: Provides comprehensive risk measurement including Value at Risk, Expected Shortfall, and drawdown analysis across all strategies.

**Visualization**: Generates comparative visualizations including performance charts, risk-return scatter plots, and rolling performance analysis.

### Report Generation

The analysis framework produces multiple output formats:

**Individual Strategy Tearsheets**: Detailed QuantStats reports for each strategy providing comprehensive performance metrics, return distributions, and risk analysis.

**Comparative Analysis Reports**: Side-by-side comparison documents highlighting relative performance, statistical significance of differences, and risk-adjusted metrics.

**Statistical Summary**: Hypothesis testing results with appropriate statistical confidence levels and practical significance assessments.

## Data Management

### Storage Architecture

The storage module provides a unified interface for all data operations:

**Async Database Operations**: All database interactions use asyncio for non-blocking operations, enabling efficient concurrent data access.

**Connection Management**: Implements connection pooling and automatic retry logic for robust database operations.

**Schema Management**: Defines and maintains database schemas for portfolio runs, weights, returns, and performance metrics.

**Data Validation**: Ensures data integrity through comprehensive validation before database persistence.

### Price Data Management

**Multi-Asset Support**: Handles price data for hundreds of cryptocurrency assets with different inception dates and data availability.

**Coverage Requirements**: Implements configurable minimum data coverage requirements to ensure statistical validity.

**Missing Data Handling**: Provides multiple strategies for handling missing price data including forward filling, interpolation, and exclusion.

## Optimization Framework

### Portfolio Optimization

The optimization module implements modern portfolio theory with practical constraints:

**Objective Functions**: Supports multiple optimization objectives including minimum variance, maximum Sharpe ratio, and maximum expected return.

**Constraint Management**: Implements position limits, sector constraints, and turnover limitations for realistic portfolio construction.

**Numerical Stability**: Uses robust optimization algorithms with fallback strategies for numerical edge cases.

**Risk Model Integration**: Incorporates sophisticated risk models including factor-based and sample-based covariance estimation.

### Parameter Configuration

**OptimizationParams Class**: Provides structured configuration management for all optimization parameters including objectives, constraints, and numerical settings.

**Validation**: Comprehensive parameter validation ensures optimization feasibility before execution.

**Default Management**: Implements sensible defaults while allowing full customization for advanced users.

## Performance Evaluation

### Metrics Calculation

The evaluation module provides comprehensive performance measurement:

**Return-Based Metrics**: Total return, annualized return, volatility, and Sharpe ratio calculation with proper handling of compounding and frequency conversion.

**Risk Metrics**: Maximum drawdown, Value at Risk, Expected Shortfall, and downside deviation measurement.

**Advanced Analytics**: Skewness, kurtosis, tail ratio, and other higher-moment statistics for complete return distribution analysis.

### Statistical Testing

**Hypothesis Testing Framework**: Implements appropriate statistical tests for portfolio comparison including t-tests for return differences and F-tests for volatility comparison.

**Multiple Comparison Correction**: Applies appropriate corrections for multiple hypothesis testing to maintain statistical validity.

**Effect Size Analysis**: Measures practical significance in addition to statistical significance for meaningful strategy comparison.

## Installation and Setup

### Requirements

The system requires Python 3.8+ with the following core dependencies:

- pandas and numpy for data manipulation
- asyncpg and sqlalchemy for database operations
- vectorbt for backtesting infrastructure
- quantstats for performance analysis
- pypfopt for portfolio optimization
- scikit-learn for statistical utilities

### Database Configuration

The system requires PostgreSQL database access with the following schema:

**Portfolio Runs**: Stores high-level information about each portfolio simulation including strategy, date range, and summary performance metrics.

**Portfolio Weights**: Maintains detailed weight allocations for each rebalancing date and asset combination.

**Portfolio Returns**: Stores daily portfolio returns for detailed performance analysis and reporting.

**Price Data**: Requires historical price data for all investable assets with consistent date formatting and proper handling of missing values.

## Usage Examples

### Basic Simulation Execution

```python
from kallos_portfolios.simulators import GRUPortfolioSimulator, HistoricalPortfolioSimulator
from datetime import date

# Load required data
universe_df, rebalancing_dates, predictions_df = load_data()

# Initialize GRU simulator
gru_simulator = GRUPortfolioSimulator(
    universe_df=universe_df,
    rebalancing_dates=rebalancing_dates,
    predictions_df=predictions_df,
    objective='min_volatility'
)

# Run complete backtest
run_id = await gru_simulator.run_complete_backtest()
```

### Strategy Comparison

```python
from kallos_portfolios.analysis import run_three_strategy_comparison
from datetime import date

# Compare all three strategies
results = await run_three_strategy_comparison(
    gru_portfolio_id=15,
    historical_portfolio_id=16,
    start_date=date(2022, 1, 3),
    end_date=date(2022, 12, 26)
)
```

### Custom Optimization Parameters

```python
from kallos_portfolios.optimisers import OptimizationParams

# Define custom optimization parameters
params = OptimizationParams(
    objective='max_sharpe',
    max_weight=0.25,        # Maximum 25% allocation per asset
    min_weight=0.02,        # Minimum 2% allocation if included
    turnover_limit=0.50     # Maximum 50% turnover per rebalancing
)
```

## Development and Extension

### Adding New Strategies

The framework supports easy extension through the BasePortfolioSimulator interface:

1. Inherit from BasePortfolioSimulator
2. Implement the `get_expected_returns_for_date()` method
3. Add any strategy-specific initialization in `__init__()`
4. Register the new simulator in the simulators package `__init__.py`

### Custom Optimization Objectives

New optimization objectives can be added by extending the OptimizationParams class and implementing the corresponding optimization logic in the PortfolioOptimizer class.

### Additional Analysis Tools

The analysis package can be extended with additional comparison metrics, statistical tests, or visualization tools by adding modules to the analysis package.

## Performance Considerations

### Computational Efficiency

The system implements several optimizations for large-scale portfolio analysis:

**Vectorized Operations**: All numerical computations use pandas and numpy vectorization for optimal performance.

**Database Optimization**: Implements connection pooling, batch operations, and optimized queries for efficient data access.

**Memory Management**: Uses efficient data structures and implements proper cleanup to handle large datasets.

**Parallel Processing**: Supports concurrent execution of independent operations including multiple strategy evaluation.

### Scalability

The architecture supports scaling to larger asset universes and longer backtesting periods:

**Modular Design**: Independent modules can be optimized or replaced without affecting other system components.

**Database Abstraction**: Storage layer can be adapted to different database systems or data sources.

**Configuration Management**: Comprehensive parameter configuration enables tuning for different computational environments.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome through GitHub pull requests. Please ensure all code follows the established architecture patterns and includes appropriate documentation and testing.

## Support

For questions, issues, or feature requests, please use the GitHub issue tracking system or contact the development team directly.

### Key Innovations

- **🔄 Temporal Model Selection**: Automatic quarterly model rotation
- **🗄️ Database Schema Alignment**: Works with existing `daily_market_data` table
- **🪙 Coin ID Consistency**: Uses `coin_id` throughout (no symbol mapping needed)
- **⚡ Parallel Model Inference**: Multi-threaded prediction pipeline
- **📊 Advanced Constraints**: Cardinality constraints via CVXPY integration
- **🧪 Statistical Testing**: Comprehensive hypothesis testing framework

## 📊 Three-Strategy Framework

### Strategy 1: GRU-Optimized Portfolio
- **Forecasting**: **Quarterly-trained GRU models** with temporal selection
- **Model Selection**: Automatic model lookup from `model_train_view`
- **Features**: 30-day lookback with returns, volatility, and momentum
- **Optimization**: Mean-variance with cardinality and weight constraints
- **Innovation**: **Quarterly model rotation** for optimal performance

### Strategy 2: Historical Mean Optimized
- **Forecasting**: 1-year historical mean returns (252 trading days)
- **Optimization**: **Identical procedure** as GRU strategy for fair comparison
- **Purpose**: Isolates value of ML forecasting vs traditional methods

### Strategy 3: Market Cap Weighted
- **Weights**: **Actual market cap data** from `daily_index_coin_contributions`
- **Rebalancing**: Monthly alignment to optimization schedule
- **Purpose**: Passive benchmark representing market consensus

## 🗄️ Database Schema Integration

### **Existing Tables Used (No Changes Required)**

```sql
-- ✅ Price data from existing table
daily_market_data (
    id,                    -- coin_id (e.g., 'bitcoin', 'ethereum')
    timestamp,             -- timezone-aware timestamp
    price,                 -- close price for analysis
    market_cap, volume,    -- additional market data
    open, high, low, close -- OHLC data
)

-- ✅ Market cap weights from existing table  
daily_index_coin_contributions (
    trade_date,            -- date for weight lookup
    coin_id,               -- matches daily_market_data.id
    end_of_day_weight      -- market cap weight (0.0-1.0)
)

-- ✅ Universe from existing table
index_monthly_constituents (
    period_start_date,     -- rebalancing period
    coin_id,               -- investable universe
    initial_weight_at_rebalance  -- starting weights
)

-- ✅ Model discovery from existing view
model_train_view (
    model,                 -- 'gru'
    coin_id,               -- 'bitcoin', 'ethereum', etc.
    year_end,              -- training year
    quarter_end,           -- training quarter
    test_start_date,       -- model usage start
    test_end_date          -- model usage end
)
```

### **New Tables Created by System**

```sql
-- Portfolio optimization parameters
optimization_params (run_id, objective, constraints, dates, ...)

-- Weekly portfolio weights by strategy  
weights_weekly (run_id, date, coin_id, weight, strategy)

-- Daily portfolio returns by strategy
returns_daily (run_id, date, portfolio_return, strategy)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/josemarquezjaramillo/kallos-models.git
cd kallos-models/kallos_portfolios

# Install package
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 2. Database Configuration

**✅ No schema changes required!** Works with existing database:

```bash
# Configure database connection (existing database)
export KALLOS_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/your_existing_db"
```

### 3. Python API Usage

```python
from kallos_portfolios.main import KallosPortfolios
from datetime import date

# Initialize system
kallos = KallosPortfolios(
    model_dir="/path/to/trained_models",  # Quarterly model directory
    database_url="postgresql+asyncpg://localhost:5432/your_db"
)

# Create optimization parameters (uses coin_ids)
params = kallos.create_optimization_params(
    strategy='max_sharpe',
    start_date='2023-01-01',
    end_date='2023-03-31',
    symbols=['bitcoin', 'ethereum', 'cardano'],  # coin_ids
    max_weight=0.35,
    min_names=3
)

# Run complete analysis
results = await kallos.run_complete_analysis('2023-01-01', '2023-03-31')

# Access results by strategy
gru_returns = results['returns']['gru']
historical_returns = results['returns']['historical'] 
market_cap_returns = results['returns']['market_cap']
```

### 4. Legacy Configuration Usage

```python
import asyncio
from kallos_portfolios import run_portfolio_with_config

# Execute three-strategy analysis
config = {
    'objective': 'max_sharpe',
    'max_weight': 0.35,
    'min_names': 3,
    'start_date': '2023-01-01',
    'end_date': '2023-12-31'
}

results = asyncio.run(run_portfolio_with_config(config))
```

## � Temporal Model Management

### **Quarterly Model Selection**

The system automatically selects the appropriate model based on the target date:

```python
from kallos_portfolios.models import get_quarter_info, get_previous_quarter
from datetime import date

# For July 15, 2023 (Q3)
target_date = date(2023, 7, 15)
year, quarter = get_quarter_info(target_date)        # (2023, "Q3")
model_year, model_quarter = get_previous_quarter(year, quarter)  # (2023, "Q2")

# System automatically uses: gru_bitcoin_2023_Q2_7D_customloss.pt
```

### **Model Usage Periods**

Each model is valid for a specific 3-month period defined by `model_train_view`:

| Training Quarter | Usage Period | Model File Example |
|-----------------|--------------|-------------------|
| Q4 2022 | Jan-Mar 2023 | `gru_bitcoin_2022_Q4_7D_customloss.pt` |
| Q1 2023 | Apr-Jun 2023 | `gru_bitcoin_2023_Q1_7D_customloss.pt` |  
| Q2 2023 | Jul-Sep 2023 | `gru_bitcoin_2023_Q2_7D_customloss.pt` |
| Q3 2023 | Oct-Dec 2023 | `gru_bitcoin_2023_Q3_7D_customloss.pt` |

### **Database Model Discovery**

```sql
-- Models automatically discovered from existing view
SELECT coin_id, study_name, test_start_date, test_end_date
FROM model_train_view
WHERE model = 'gru' 
  AND year_end = 2023 
  AND quarter_end = 'Q2'
  AND '2023-07-15' BETWEEN test_start_date AND test_end_date;
```

## �📈 Analysis Workflow

### **Phase 1: Temporal Setup & Data Loading**
- **Automatic model discovery** from `model_train_view`
- **Universe loading** from `index_monthly_constituents`
- **Price data loading** from `daily_market_data`  
- **Market cap weights** from `daily_index_coin_contributions`

### **Phase 2: Three-Strategy Optimization**
```python
# 1. GRU Strategy (with temporal model selection)
gru_forecasts = await forecast_returns(
    prices, model_dir, target_date=rebalance_date, session=session
)
gru_weights = optimizer.optimize_gru_strategy(gru_forecasts, prices)

# 2. Historical Strategy  
historical_weights = optimizer.optimize_historical_strategy(prices)

# 3. Market Cap Strategy (from database)
market_cap_weights = await load_market_cap_weights(session, date)
```

### **Phase 3: Backtesting & Evaluation**
- **Vectorized backtesting** with transaction costs
- **Statistical hypothesis testing** between strategies
- **Performance analysis** with QuantStats integration
- Generate monthly rebalancing dates
- Validate GRU model availability

### Phase 2: Monthly Optimization Loop
```python
for rebalance_date in monthly_dates:
    # Load monthly universe
    universe = await fetch_monthly_universe(session, rebalance_date)
    
    # Load price data with lookback
    prices = await load_prices_for_period(universe, lookback_period)
    
    # Strategy 1: GRU optimization
    gru_forecasts = await forecast_returns(prices, model_dir)
    gru_weights = await optimize_portfolio_gru(gru_forecasts, prices, params)
    
    # Strategy 2: Historical optimization
    historical_weights = await optimize_portfolio_historical(prices, params)
    
    # Strategy 3: Market cap weights
    market_cap_weights = await load_market_cap_weights(rebalance_date)
    
    # Store all weights
    await store_weights(session, run_id, rebalance_date, all_weights)
```

### Phase 3: Backtesting
- Load complete price dataset for backtesting period
- Run vectorized backtests for all three strategies
- Calculate daily portfolio returns with realistic rebalancing

### Phase 4: Statistical Analysis
- Generate comprehensive performance metrics
- Execute pairwise hypothesis tests between strategies
- Create QuantStats tearsheets and comparison reports

## 🧪 Statistical Testing Framework

### Hypothesis Tests Performed

1. **Mean Return Difference (Paired t-test)**
   - H₀: Mean excess return = 0 vs H₁: Mean excess return ≠ 0
   - Tests if one strategy has significantly higher returns

2. **Variance Equality (F-test)**
   - H₀: Variances equal vs H₁: Variances differ
   - Tests if strategies have different risk levels

3. **Distribution Similarity (Kolmogorov-Smirnov)**
   - H₀: Distributions identical vs H₁: Distributions differ
   - Tests if return distributions are fundamentally different

### Performance Metrics

- **Return Metrics**: Total return, annualized return, Sharpe ratio
- **Risk Metrics**: Volatility, max drawdown, VaR, CVaR
- **Distribution**: Skewness, kurtosis, win rate
- **Concentration**: Herfindahl index, effective holdings

## 🔧 Advanced Usage

### Custom Optimization Objectives

```python
from kallos_portfolios.config import OptimizationParams

# Maximum Sharpe ratio
params = OptimizationParams(objective='max_sharpe', risk_free_rate=0.02)

# Minimum volatility
params = OptimizationParams(objective='min_volatility')

# Target volatility
params = OptimizationParams(objective='efficient_risk', target_volatility=0.15)

# Target return
params = OptimizationParams(objective='efficient_return', target_return=0.10)
```

### Constraint Customization

```python
# Conservative portfolio
conservative_params = OptimizationParams(
    max_weight=0.25,     # Max 25% per asset
    min_names=5,         # Min 5 holdings
    l2_reg=0.1          # High regularization
)

# Concentrated portfolio
concentrated_params = OptimizationParams(
    max_weight=0.50,     # Max 50% per asset
    min_names=2,         # Min 2 holdings
    l2_reg=0.001        # Low regularization
)
```

### Parallel Execution

```python
from kallos_portfolios.config import settings

# Configure parallel processing
settings.max_workers = 8  # Increase worker threads
settings.db_pool_size = 20  # Larger connection pool

# Model inference parallelization automatically handled
```

## 📊 Output Reports

### Generated Reports

1. **QuantStats Tearsheets**: Professional performance analysis
   - `{strategy}_tearsheet.html`
   - Detailed metrics, charts, and risk analysis

2. **Strategy Comparison Report**: Statistical test results
   - `strategy_comparison_report.html`
   - Hypothesis test results with significance highlighting

3. **Performance Metrics**: Comprehensive statistics
   - DataFrame with all strategies compared
   - Exportable to CSV/Excel

### Sample Report Structure

```
reports/
├── portfolio_analysis_20240803_143022/
│   ├── gru_tearsheet.html
│   ├── historical_tearsheet.html
│   ├── market_cap_tearsheet.html
│   └── strategy_comparison_report.html
```

## 🛠️ Configuration

### Environment Variables

```bash
# Database connection
export KALLOS_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/kallos_portfolios"

# Model directory
export KALLOS_MODEL_DIR="/path/to/trained_models"

# Report output
export KALLOS_REPORT_PATH="/path/to/reports"

# Logging level
export KALLOS_LOG_LEVEL="INFO"
```

### **Configuration File (`.env`)**

```env
# Database connection (existing database)
KALLOS_DATABASE_URL=postgresql+asyncpg://localhost:5432/your_existing_db

# Quarterly model directory
KALLOS_MODEL_DIR=/home/user/kallos/trained_models

# Report output directory  
KALLOS_REPORT_PATH=/home/user/kallos/reports

# Logging configuration
KALLOS_LOG_LEVEL=INFO
KALLOS_MAX_WORKERS=4
```

## 🪙 Coin ID Integration

### **Consistent Identifier Usage**

The system uses **coin_ids** throughout, matching your database schema:

```python
# ✅ Correct: Use coin_ids (matches database)
universe = ['bitcoin', 'ethereum', 'cardano', 'polkadot', 'chainlink']

# ❌ Incorrect: Don't use symbols  
universe = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
```

### **Database Alignment**

| Table | Column | Example Values |
|-------|--------|---------------|
| `daily_market_data` | `id` | `bitcoin`, `ethereum` |
| `daily_index_coin_contributions` | `coin_id` | `bitcoin`, `ethereum` |
| `index_monthly_constituents` | `coin_id` | `bitcoin`, `ethereum` |
| `model_train_view` | `coin_id` | `bitcoin`, `ethereum` |

### **No Mapping Required**

- **Model files**: `gru_bitcoin_2023_Q2_7D_customloss.pt`
- **Database queries**: `WHERE coin_id = 'bitcoin'`
- **Portfolio weights**: Indexed by `coin_id`
- **Complete consistency** throughout the pipeline

## 🔍 Model Requirements

### **Quarterly Model Directory Structure**

```
trained_models/
├── gru_bitcoin_2023_Q1_7D_customloss.pt      # Q1 model file
├── gru_bitcoin_2023_Q1_7D_customloss_scaler.pkl  # Q1 scaler
├── gru_bitcoin_2023_Q2_7D_customloss.pt      # Q2 model file  
├── gru_bitcoin_2023_Q2_7D_customloss_scaler.pkl  # Q2 scaler
├── gru_ethereum_2023_Q1_7D_customloss.pt     # Ethereum Q1
├── gru_ethereum_2023_Q1_7D_customloss_scaler.pkl
└── ... (for each coin_id and quarter)
```

### **Model Naming Convention**

**Format**: `{model}_{coin_id}_{year}_{quarter}_7D_customloss`

- **model**: Model type (`gru`)
- **coin_id**: Database coin identifier (`bitcoin`, `ethereum`, `cardano`)
- **year**: Training year (`2023`)
- **quarter**: Training quarter (`Q1`, `Q2`, `Q3`, `Q4`)
- **suffix**: Fixed (`7D_customloss`)

### **Model Specifications**

- **Input Features**: Returns, log-returns, volatility, momentum (4 features)
- **Sequence Length**: 30 days (configurable)
- **Output**: Single predicted weekly return
- **Scaling**: StandardScaler preprocessing required
- **Framework**: Compatible with Darts and PyTorch models

## 🚨 Dependencies

### Core Dependencies

```
pandas>=2.2.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
PyPortfolioOpt>=1.5.4   # Portfolio optimization
cvxpy>=1.4.0           # Convex optimization
vectorbt>=0.25.0        # Vectorized backtesting
quantstats>=0.0.62      # Performance analysis
sqlalchemy[asyncio]>=2.0.0  # Async database
asyncpg>=0.29.0         # PostgreSQL async driver
torch>=2.0.0            # PyTorch for models
pydantic>=2.0.0         # Configuration management
```

### Explicitly Excluded

- **TA-Lib**: Installation complexity, not needed
- **python-dotenv**: Pydantic handles .env files
- **click**: No CLI interface needed
- **tqdm**: Progress bars not required

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=kallos_portfolios tests/

# Async tests
pytest -v tests/test_async_operations.py
```

## 📈 Performance Considerations

### Database Optimization

- **Connection Pooling**: Configurable pool size and overflow
- **Batch Operations**: Efficient bulk inserts with conflict handling
- **Indexing**: Optimized indexes for common queries
- **Async Operations**: Non-blocking I/O throughout

### Computational Efficiency

- **Vectorized Operations**: pandas/numpy vectorization
- **Parallel Model Inference**: ThreadPoolExecutor for GRU predictions
- **Memory Management**: Efficient data structures and cleanup
- **Caching**: Model and scaler caching to avoid reloading

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Code formatting
black kallos_portfolios/

# Linting
flake8 kallos_portfolios/

# Type checking
mypy kallos_portfolios/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyPortfolioOpt**: Excellent portfolio optimization library
- **vectorbt**: High-performance backtesting framework
- **QuantStats**: Professional performance analysis
- **Pydantic**: Robust configuration management
- **SQLAlchemy**: Powerful async database toolkit

## 🆕 Latest Updates

### **Database Schema Alignment (v1.0.0)**
- ✅ **Zero database changes required** - works with existing tables
- ✅ **daily_market_data integration** - uses existing price data
- ✅ **coin_id consistency** - no symbol mapping needed
- ✅ **Timezone-aware timestamps** - proper datetime handling

### **Temporal Model Management (v1.0.0)**  
- ✅ **Quarterly model selection** - automatic model rotation
- ✅ **Database model discovery** - integrates with `model_train_view`
- ✅ **3-month usage periods** - aligned with training cycle
- ✅ **Graceful fallbacks** - handles missing models

### **Production-Ready Features (v1.0.0)**
- ✅ **Python API** - `KallosPortfolios` class for easy integration
- ✅ **Enhanced error handling** - comprehensive exception management
- ✅ **Model caching** - efficient quarterly model loading
- ✅ **Parallel processing** - optimized for performance

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/josemarquezjaramillo/kallos-models/issues)
- **Documentation**: [Project Wiki](https://github.com/josemarquezjaramillo/kallos-models/wiki)
- **Email**: josemarquezjaramillo@gmail.com

---

**Kallos Portfolios v1.0.0** - *Production-ready cryptocurrency portfolio optimization with quarterly model management and database integration*
