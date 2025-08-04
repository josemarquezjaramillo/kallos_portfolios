-- Kallos Portfolios Database Schema
-- PostgreSQL schema for cryptocurrency portfolio optimization system

-- Create database (run this separately if needed)
-- CREATE DATABASE kallos_portfolios;

-- Create schema for portfolio simulation tables
CREATE SCHEMA IF NOT EXISTS portfolio_simulations;

-- Set search path to include the new schema
-- SET search_path = portfolio_simulations, public;

-- IMPORTANT: After running this schema, update your application code to:
-- 1. Use schema-qualified table names (e.g., portfolio_simulations.weights_weekly)
-- 2. Or set the search_path in your database connections
-- 3. Update SQLAlchemy models to include __table_args__ = {'schema': 'portfolio_simulations'}

-- Optimization run parameters table
CREATE TABLE IF NOT EXISTS portfolio_simulations.optimization_params (
    run_id VARCHAR PRIMARY KEY,
    objective VARCHAR NOT NULL CHECK (objective IN ('max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return')),
    max_weight NUMERIC NOT NULL CHECK (max_weight > 0 AND max_weight <= 1),
    min_names INTEGER NOT NULL CHECK (min_names >= 2),
    lookback_days INTEGER NOT NULL CHECK (lookback_days >= 30),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    risk_free_rate NUMERIC NOT NULL DEFAULT 0.0 CHECK (risk_free_rate >= 0),
    gamma NUMERIC NOT NULL DEFAULT 1.0 CHECK (gamma >= 0),
    l2_reg NUMERIC NOT NULL DEFAULT 0.01 CHECK (l2_reg >= 0),
    target_volatility NUMERIC NULL CHECK (target_volatility > 0),
    target_return NUMERIC NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure valid date range
    CONSTRAINT valid_date_range CHECK (end_date > start_date)
);

-- Weekly portfolio weights by strategy
CREATE TABLE IF NOT EXISTS portfolio_simulations.weights_weekly (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    weight NUMERIC NOT NULL CHECK (weight >= 0 AND weight <= 1),
    strategy VARCHAR NOT NULL CHECK (strategy IN ('gru', 'historical', 'market_cap')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint ensures one weight per symbol per date per strategy per run
    CONSTRAINT weights_weekly_unique UNIQUE (run_id, date, symbol, strategy)
);

-- Daily portfolio returns by strategy
CREATE TABLE IF NOT EXISTS portfolio_simulations.returns_daily (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    portfolio_return NUMERIC NOT NULL,
    strategy VARCHAR NOT NULL CHECK (strategy IN ('gru', 'historical', 'market_cap')),
    benchmark_return NUMERIC NULL,
    active_return NUMERIC NULL,
    turnover NUMERIC NULL CHECK (turnover >= 0),
    transaction_costs NUMERIC NULL CHECK (transaction_costs >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint ensures one return per date per strategy per run
    CONSTRAINT returns_daily_unique UNIQUE (run_id, date, strategy)
);


-- Indexes for performance optimization

-- Optimization params indexes
CREATE INDEX IF NOT EXISTS idx_optimization_params_dates ON portfolio_simulations.optimization_params (start_date, end_date);

-- Weights indexes
CREATE INDEX IF NOT EXISTS idx_weights_weekly_run_strategy ON portfolio_simulations.weights_weekly (run_id, strategy);
CREATE INDEX IF NOT EXISTS idx_weights_weekly_date ON portfolio_simulations.weights_weekly (date);
CREATE INDEX IF NOT EXISTS idx_weights_weekly_symbol ON portfolio_simulations.weights_weekly (symbol);

-- Returns indexes
CREATE INDEX IF NOT EXISTS idx_returns_daily_run_strategy ON portfolio_simulations.returns_daily (run_id, strategy);
CREATE INDEX IF NOT EXISTS idx_returns_daily_date ON portfolio_simulations.returns_daily (date);

-- Create functions for automated timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic updated_at timestamp updates
CREATE TRIGGER update_optimization_params_updated_at BEFORE UPDATE ON portfolio_simulations.optimization_params
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_weights_weekly_updated_at BEFORE UPDATE ON portfolio_simulations.weights_weekly
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_returns_daily_updated_at BEFORE UPDATE ON portfolio_simulations.returns_daily
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for easier data access and analysis

-- View for latest optimization runs
CREATE OR REPLACE VIEW portfolio_simulations.latest_optimization_runs AS
SELECT 
    run_id,
    objective,
    max_weight,
    min_names,
    start_date,
    end_date,
    created_at,
    (end_date - start_date) as duration_days
FROM portfolio_simulations.optimization_params
ORDER BY created_at DESC;

-- View for portfolio performance summary
CREATE OR REPLACE VIEW portfolio_simulations.portfolio_performance_summary AS
WITH strategy_stats AS (
    SELECT 
        run_id,
        strategy,
        COUNT(*) as n_observations,
        AVG(portfolio_return) as avg_daily_return,
        stddev_samp(portfolio_return) as daily_volatility,
        MIN(portfolio_return) as min_return,
        MAX(portfolio_return) as max_return,
        AVG(portfolio_return) * 252 as annualized_return,
        stddev_samp(portfolio_return) * sqrt(252) as annualized_volatility
    FROM portfolio_simulations.returns_daily
    GROUP BY run_id, strategy
)
SELECT 
    *,
    CASE 
        WHEN annualized_volatility > 0 
        THEN annualized_return / annualized_volatility 
        ELSE NULL 
    END as sharpe_ratio
FROM strategy_stats
ORDER BY run_id, strategy;

-- View for weight concentration analysis
CREATE OR REPLACE VIEW portfolio_simulations.weight_concentration_analysis AS
SELECT 
    run_id,
    strategy,
    date,
    COUNT(*) as n_holdings,
    MAX(weight) as max_weight,
    SUM(weight * weight) as herfindahl_index,
    CASE 
        WHEN SUM(weight * weight) > 0 
        THEN 1.0 / SUM(weight * weight) 
        ELSE NULL 
    END as effective_holdings
FROM portfolio_simulations.weights_weekly
WHERE weight > 0.001  -- Only count meaningful positions
GROUP BY run_id, strategy, date
ORDER BY run_id, strategy, date;

-- View for universe evolution tracking
CREATE OR REPLACE VIEW portfolio_simulations.universe_evolution AS
SELECT 
    period_start_date,
    COUNT(*) as n_constituents,
    SUM(initial_market_cap_at_rebalance) as total_market_cap,
    AVG(initial_weight_at_rebalance) as avg_weight,
    MAX(initial_weight_at_rebalance) as max_weight,
    STRING_AGG(coin_id, ', ' ORDER BY initial_weight_at_rebalance DESC) as constituents
FROM public.index_monthly_constituents
GROUP BY period_start_date
ORDER BY period_start_date;

-- Comments for documentation
COMMENT ON TABLE portfolio_simulations.optimization_params IS 'Portfolio optimization run parameters and constraints';
COMMENT ON TABLE portfolio_simulations.weights_weekly IS 'Portfolio weights by strategy for each rebalancing date';
COMMENT ON TABLE portfolio_simulations.returns_daily IS 'Daily portfolio returns by strategy with performance metrics';

COMMENT ON VIEW portfolio_simulations.latest_optimization_runs IS 'Most recent portfolio optimization runs with basic info';
COMMENT ON VIEW portfolio_simulations.portfolio_performance_summary IS 'Aggregated performance statistics by run and strategy';
COMMENT ON VIEW portfolio_simulations.weight_concentration_analysis IS 'Portfolio concentration metrics over time';
COMMENT ON VIEW portfolio_simulations.universe_evolution IS 'How the investable universe changes over time';

-- Example data validation queries (uncomment to run)

-- Check weight sum validation
-- SELECT run_id, strategy, date, SUM(weight) as total_weight
-- FROM portfolio_simulations.weights_weekly 
-- GROUP BY run_id, strategy, date 
-- HAVING ABS(SUM(weight) - 1.0) > 0.01;

-- Check for missing return data
-- SELECT DISTINCT run_id, strategy 
-- FROM portfolio_simulations.weights_weekly w
-- WHERE NOT EXISTS (
--     SELECT 1 FROM portfolio_simulations.returns_daily r 
--     WHERE r.run_id = w.run_id AND r.strategy = w.strategy
-- );

-- Check daily market data quality
-- SELECT id, 
--        COUNT(*) as n_records,
--        MIN(timestamp::date) as first_date,
--        MAX(timestamp::date) as last_date,
--        COUNT(DISTINCT timestamp::date) as unique_dates
-- FROM public.daily_market_data 
-- GROUP BY id 
-- ORDER BY n_records DESC;
