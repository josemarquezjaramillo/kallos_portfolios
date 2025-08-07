-- Portfolio Storage Schema for kallos_portfolios
-- Add these tables to your PostgreSQL database

-- Main portfolio runs table
CREATE TABLE IF NOT EXISTS portfolio_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    assets TEXT[] NOT NULL,
    final_value DECIMAL(15,2),
    total_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    volatility DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(run_id, strategy_name)
);

-- Portfolio weights for each rebalancing date
CREATE TABLE IF NOT EXISTS portfolio_weights (
    id SERIAL PRIMARY KEY,
    portfolio_run_id INTEGER REFERENCES portfolio_runs(id) ON DELETE CASCADE,
    rebalance_date DATE NOT NULL,
    asset VARCHAR(50) NOT NULL,
    weight DECIMAL(10,6) NOT NULL,
    
    UNIQUE(portfolio_run_id, rebalance_date, asset)
);

-- Daily portfolio returns
CREATE TABLE IF NOT EXISTS portfolio_returns (
    id SERIAL PRIMARY KEY,
    portfolio_run_id INTEGER REFERENCES portfolio_runs(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    return_value DECIMAL(12,8) NOT NULL,
    
    UNIQUE(portfolio_run_id, date)
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_portfolio_runs_run_id ON portfolio_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_runs_dates ON portfolio_runs(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_runs_strategy ON portfolio_runs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_portfolio_weights_date ON portfolio_weights(rebalance_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_weights_asset ON portfolio_weights(asset);
CREATE INDEX IF NOT EXISTS idx_portfolio_returns_date ON portfolio_returns(date);

-- Add comments for documentation
COMMENT ON TABLE portfolio_runs IS 'Main table storing portfolio backtesting run results';
COMMENT ON TABLE portfolio_weights IS 'Portfolio weights for each rebalancing date';
COMMENT ON TABLE portfolio_returns IS 'Daily portfolio returns for each run';

COMMENT ON COLUMN portfolio_runs.run_id IS 'Unique identifier for the portfolio run';
COMMENT ON COLUMN portfolio_runs.strategy_name IS 'Strategy used (e.g., GRU_Model, Max_Sharpe, Market_Cap)';
COMMENT ON COLUMN portfolio_runs.assets IS 'Array of asset identifiers in the portfolio';
COMMENT ON COLUMN portfolio_runs.metadata IS 'Additional run parameters and configuration';

COMMENT ON COLUMN portfolio_weights.weight IS 'Portfolio weight (0.0 to 1.0) for the asset';
COMMENT ON COLUMN portfolio_returns.return_value IS 'Daily return as decimal (e.g., 0.01 = 1%)';
