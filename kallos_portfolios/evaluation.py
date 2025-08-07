"""
Statistical analysis and performance evaluation for portfolio strategies.
Implements comprehensive hypothesis testing and performance comparison framework.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import date, datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
import quantstats as qs

logger = logging.getLogger(__name__)


class PortfolioEvaluator:
    """
    Comprehensive portfolio performance evaluation and statistical testing.
    
    Implements rigorous three-strategy comparison framework with hypothesis testing.
    """
    
    def __init__(self, strategy_returns: Dict[str, pd.Series]):
        """
        Initialize evaluator with strategy return series.
        
        Args:
            strategy_returns: Dictionary mapping strategy names to return series
        """
        self.strategy_returns = strategy_returns
        self.strategy_names = list(strategy_returns.keys())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_performance_metrics(self) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics for all strategies.
        
        Returns:
            DataFrame with performance metrics by strategy
        """
        metrics_data = []
        
        for strategy_name, returns in self.strategy_returns.items():
            if returns.empty:
                continue
            
            try:
                # Basic return statistics
                daily_mean = returns.mean()
                daily_std = returns.std()
                total_return = (1 + returns).prod() - 1
                
                # Annualized metrics
                n_periods = len(returns)
                trading_days_per_year = 252
                annualized_return = (1 + total_return) ** (trading_days_per_year / n_periods) - 1
                annualized_volatility = daily_std * np.sqrt(trading_days_per_year)
                
                # Risk-adjusted returns
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
                
                # Drawdown analysis
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns / running_max - 1)
                max_drawdown = drawdown.min()
                
                # Win rate and distribution
                win_rate = (returns > 0).mean()
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                # Risk metrics
                var_95 = returns.quantile(0.05)
                cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                
                # Calmar ratio (annual return / max drawdown)
                calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
                
                metrics_data.append({
                    'Strategy': strategy_name,
                    'Daily_Mean_Return': daily_mean,
                    'Daily_Std_Deviation': daily_std,
                    'Annualized_Return': annualized_return,
                    'Annualized_Volatility': annualized_volatility,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Max_Drawdown': max_drawdown,
                    'Calmar_Ratio': calmar_ratio,
                    'Total_Return': total_return,
                    'Win_Rate': win_rate,
                    'Skewness': skewness,
                    'Kurtosis': kurtosis,
                    'VaR_95': var_95,
                    'CVaR_95': cvar_95,
                    'Observations': n_periods
                })
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {strategy_name}: {e}")
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.set_index('Strategy', inplace=True)
            return metrics_df
        else:
            return pd.DataFrame()
    
    def perform_pairwise_hypothesis_tests(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Perform comprehensive pairwise hypothesis tests between strategies.
        
        Returns:
            Nested dictionary with test results for all strategy pairs
        """
        test_results = {}
        
        for i, strategy_a in enumerate(self.strategy_names):
            for j, strategy_b in enumerate(self.strategy_names):
                if i >= j:  # Only test unique pairs
                    continue
                
                pair_name = f"{strategy_a}_vs_{strategy_b}"
                test_results[pair_name] = self._test_strategy_pair(
                    self.strategy_returns[strategy_a],
                    self.strategy_returns[strategy_b],
                    strategy_a,
                    strategy_b
                )
        
        return test_results
    
    def _test_strategy_pair(
        self, 
        returns_a: pd.Series, 
        returns_b: pd.Series,
        name_a: str,
        name_b: str
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical testing between two strategies.
        
        Args:
            returns_a: Return series for strategy A
            returns_b: Return series for strategy B
            name_a: Name of strategy A
            name_b: Name of strategy B
            
        Returns:
            Dictionary with all test results
        """
        # Align return series by date
        aligned_data = pd.DataFrame({name_a: returns_a, name_b: returns_b}).dropna()
        
        if len(aligned_data) < 30:
            self.logger.warning(f"Insufficient data for testing {name_a} vs {name_b}: {len(aligned_data)} observations")
            return {'error': 'Insufficient data for testing'}
        
        returns_a_aligned = aligned_data[name_a]
        returns_b_aligned = aligned_data[name_b]
        excess_returns = returns_a_aligned - returns_b_aligned
        
        results = {
            'sample_size': len(aligned_data),
            'strategy_a': name_a,
            'strategy_b': name_b
        }
        
        # Test 1: Mean return difference (Paired t-test)
        results['mean_difference_test'] = self._test_mean_difference(excess_returns, name_a, name_b)
        
        # Test 2: Variance equality (F-test)
        results['variance_equality_test'] = self._test_variance_equality(returns_a_aligned, returns_b_aligned)
        
        # Test 3: Distribution similarity (Kolmogorov-Smirnov test)
        results['distribution_test'] = self._test_distribution_similarity(returns_a_aligned, returns_b_aligned)
        
        # Test 4: Stochastic dominance
        results['stochastic_dominance'] = self._test_stochastic_dominance(returns_a_aligned, returns_b_aligned)
        
        return results
    
    def _test_mean_difference(self, excess_returns: pd.Series, name_a: str, name_b: str) -> Dict[str, Any]:
        """
        Test if mean excess return is significantly different from zero.
        H₀: Mean excess return = 0 vs H₁: Mean excess return ≠ 0
        """
        try:
            mean_excess = excess_returns.mean()
            t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
            
            # Calculate confidence interval
            se = excess_returns.std() / np.sqrt(len(excess_returns))
            ci_lower = mean_excess - 1.96 * se
            ci_upper = mean_excess + 1.96 * se
            
            # Effect size (Cohen's d)
            cohens_d = mean_excess / excess_returns.std() if excess_returns.std() > 0 else 0
            
            return {
                'test_name': 'Paired t-test for mean return difference',
                'null_hypothesis': f'Mean return of {name_a} = Mean return of {name_b}',
                'mean_excess_return': mean_excess,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_5pct': p_value < 0.05,
                'confidence_interval_95': (ci_lower, ci_upper),
                'cohens_d': cohens_d,
                'interpretation': self._interpret_mean_test(p_value, mean_excess, name_a, name_b)
            }
            
        except Exception as e:
            return {'error': f'Mean difference test failed: {e}'}
    
    def _test_variance_equality(self, returns_a: pd.Series, returns_b: pd.Series) -> Dict[str, Any]:
        """
        Test if variances are significantly different (F-test).
        H₀: Variances are equal vs H₁: Variances differ
        """
        try:
            var_a = returns_a.var()
            var_b = returns_b.var()
            
            # F-statistic (larger variance in numerator)
            if var_a >= var_b:
                f_stat = var_a / var_b
                df1, df2 = len(returns_a) - 1, len(returns_b) - 1
            else:
                f_stat = var_b / var_a
                df1, df2 = len(returns_b) - 1, len(returns_a) - 1
            
            # Two-tailed p-value
            p_value = 2 * min(
                stats.f.cdf(f_stat, df1, df2),
                1 - stats.f.cdf(f_stat, df1, df2)
            )
            
            return {
                'test_name': 'F-test for variance equality',
                'null_hypothesis': 'Variances are equal',
                'variance_a': var_a,
                'variance_b': var_b,
                'f_statistic': f_stat,
                'degrees_of_freedom': (df1, df2),
                'p_value': p_value,
                'significant_at_5pct': p_value < 0.05,
                'interpretation': self._interpret_variance_test(p_value, var_a, var_b)
            }
            
        except Exception as e:
            return {'error': f'Variance equality test failed: {e}'}
    
    def _test_distribution_similarity(self, returns_a: pd.Series, returns_b: pd.Series) -> Dict[str, Any]:
        """
        Test if return distributions are significantly different (Kolmogorov-Smirnov test).
        H₀: Distributions are identical vs H₁: Distributions differ
        """
        try:
            ks_stat, ks_p = stats.ks_2samp(returns_a, returns_b)
            
            return {
                'test_name': 'Kolmogorov-Smirnov test for distribution equality',
                'null_hypothesis': 'Return distributions are identical',
                'ks_statistic': ks_stat,
                'p_value': ks_p,
                'significant_at_5pct': ks_p < 0.05,
                'interpretation': self._interpret_ks_test(ks_p, ks_stat)
            }
            
        except Exception as e:
            return {'error': f'Distribution test failed: {e}'}
    
    def _test_stochastic_dominance(self, returns_a: pd.Series, returns_b: pd.Series) -> Dict[str, Any]:
        """
        Test for first-order stochastic dominance.
        Strategy A dominates B if CDF_A(x) <= CDF_B(x) for all x.
        """
        try:
            # Create common grid
            combined = pd.concat([returns_a, returns_b])
            grid = np.linspace(combined.min(), combined.max(), 100)
            
            # Calculate empirical CDFs
            cdf_a = np.array([np.mean(returns_a <= x) for x in grid])
            cdf_b = np.array([np.mean(returns_b <= x) for x in grid])
            
            # Test dominance
            a_dominates_b = np.all(cdf_a <= cdf_b)
            b_dominates_a = np.all(cdf_b <= cdf_a)
            
            # Maximum CDF difference
            max_diff = np.max(np.abs(cdf_a - cdf_b))
            
            return {
                'test_name': 'First-order stochastic dominance test',
                'a_dominates_b': a_dominates_b,
                'b_dominates_a': b_dominates_a,
                'no_dominance': not (a_dominates_b or b_dominates_a),
                'max_cdf_difference': max_diff,
                'interpretation': self._interpret_dominance_test(a_dominates_b, b_dominates_a)
            }
            
        except Exception as e:
            return {'error': f'Stochastic dominance test failed: {e}'}
    
    def _interpret_mean_test(self, p_value: float, mean_excess: float, name_a: str, name_b: str) -> str:
        """Interpret mean difference test results."""
        if p_value < 0.01:
            significance = "highly significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not significant (p >= 0.10)"
        
        direction = "higher" if mean_excess > 0 else "lower"
        
        return f"{name_a} has {direction} mean returns than {name_b} ({significance})"
    
    def _interpret_variance_test(self, p_value: float, var_a: float, var_b: float) -> str:
        """Interpret variance equality test results."""
        if p_value < 0.05:
            higher_var = "A" if var_a > var_b else "B"
            return f"Variances are significantly different (p < 0.05), Strategy {higher_var} has higher variance"
        else:
            return "Variances are not significantly different (p >= 0.05)"
    
    def _interpret_ks_test(self, p_value: float, ks_stat: float) -> str:
        """Interpret Kolmogorov-Smirnov test results."""
        if p_value < 0.05:
            return f"Return distributions are significantly different (p < 0.05, KS statistic = {ks_stat:.3f})"
        else:
            return f"Return distributions are not significantly different (p >= 0.05, KS statistic = {ks_stat:.3f})"
    
    def _interpret_dominance_test(self, a_dominates_b: bool, b_dominates_a: bool) -> str:
        """Interpret stochastic dominance test results."""
        if a_dominates_b:
            return "Strategy A first-order stochastically dominates Strategy B"
        elif b_dominates_a:
            return "Strategy B first-order stochastically dominates Strategy A"
        else:
            return "No clear first-order stochastic dominance relationship"


def generate_quantstats_reports(
    strategy_returns: Dict[str, pd.Series],
    benchmark_strategy: str = 'market_cap',
    output_dir: Path = None
) -> Dict[str, Path]:
    """
    Generate QuantStats tearsheet reports for all strategies with proper naming.
    
    Args:
        strategy_returns: Dictionary of strategy return series
        benchmark_strategy: Strategy to use as benchmark
        output_dir: Output directory for reports
        
    Returns:
        Dictionary mapping strategy names to report file paths
    """
    if output_dir is None:
        output_dir = Path.cwd() / "reports"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    report_paths = {}
    
    # Get benchmark returns
    benchmark_returns = strategy_returns.get(benchmark_strategy)
    
    for strategy_name, returns in strategy_returns.items():
        if returns.empty:
            continue
        
        try:
            # Create proper naming for tearsheets
            if strategy_name == 'GRU_Portfolio':
                tearsheet_name = 'GRU_Portfolio_tearsheet.html'
                display_title = 'GRU Portfolio Strategy Performance Analysis'
            elif strategy_name == 'Historical_Portfolio':
                tearsheet_name = 'Naive_Portfolio_tearsheet.html'  # Renamed as requested
                display_title = 'Naive Portfolio Strategy Performance Analysis'
            elif strategy_name == 'Market_Weighted':
                tearsheet_name = 'Market_Weighted_tearsheet.html'
                display_title = 'Market Weighted Portfolio Performance Analysis'
            else:
                tearsheet_name = f'{strategy_name}_tearsheet.html'
                display_title = f"{strategy_name.replace('_', ' ')} Strategy Performance Analysis"
            
            report_path = output_dir / tearsheet_name
            
            # Generate QuantStats report
            qs.reports.html(
                returns,
                benchmark=benchmark_returns if strategy_name != benchmark_strategy else None,
                output=str(report_path),
                title=display_title,
                download_filename=tearsheet_name
            )
            
            report_paths[strategy_name] = report_path
            logger.info(f"Generated QuantStats report for {strategy_name}: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating QuantStats report for {strategy_name}: {e}")
    
    return report_paths


def create_strategy_comparison_report(
    evaluator: PortfolioEvaluator,
    output_dir: Path = None
) -> Path:
    """
    Create comprehensive HTML report comparing all strategies.
    
    Args:
        evaluator: PortfolioEvaluator instance with test results
        output_dir: Output directory for report
        
    Returns:
        Path to generated HTML report
    """
    if output_dir is None:
        output_dir = Path.cwd() / "reports"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Calculate performance metrics
    performance_metrics = evaluator.calculate_performance_metrics()
    
    # Perform hypothesis tests
    test_results = evaluator.perform_pairwise_hypothesis_tests()
    
    # Generate HTML report
    html_content = _generate_comparison_html(performance_metrics, test_results)
    
    report_path = output_dir / "strategy_comparison_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated strategy comparison report: {report_path}")
    return report_path


def _generate_comparison_html(
    performance_metrics: pd.DataFrame,
    test_results: Dict[str, Dict[str, Dict[str, Any]]]
) -> str:
    """Generate HTML content for strategy comparison report."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kallos Portfolios - Strategy Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #f2f2f2; font-weight: bold; }
            .test-section { margin: 20px 0; padding: 15px; border: 1px solid #ccc; }
            .significant { background-color: #ffeb3b; }
            .not-significant { background-color: #f5f5f5; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Kallos Portfolios - Strategy Comparison Report</h1>
            <p>Three-Strategy Portfolio Analysis with Statistical Testing</p>
        </div>
    """
    
    # Performance metrics table
    if not performance_metrics.empty:
        html += """
        <div class="section">
            <h2>Performance Metrics Comparison</h2>
            <table>
        """
        
        # Table header
        html += "<tr><th>Strategy</th>"
        for col in performance_metrics.columns:
            html += f"<th>{col.replace('_', ' ')}</th>"
        html += "</tr>"
        
        # Table rows
        for strategy in performance_metrics.index:
            html += f"<tr><td style='text-align: left;'><strong>{strategy}</strong></td>"
            for col in performance_metrics.columns:
                value = performance_metrics.loc[strategy, col]
                if isinstance(value, (int, float)):
                    if 'Return' in col or 'Ratio' in col:
                        html += f"<td>{value:.4f}</td>"
                    elif 'Rate' in col:
                        html += f"<td>{value:.2%}</td>"
                    else:
                        html += f"<td>{value:.4f}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"
        
        html += "</table></div>"
    
    # Statistical test results
    html += """
    <div class="section">
        <h2>Statistical Hypothesis Tests</h2>
        <p>Comprehensive pairwise testing between strategies using multiple statistical tests.</p>
    """
    
    for pair_name, test_data in test_results.items():
        if 'error' in test_data:
            continue
        
        html += f"""
        <div class="test-section">
            <h3>{pair_name.replace('_', ' vs ')}</h3>
            <p><strong>Sample Size:</strong> {test_data['sample_size']} observations</p>
        """
        
        # Mean difference test
        if 'mean_difference_test' in test_data:
            mean_test = test_data['mean_difference_test']
            significance_class = "significant" if mean_test.get('significant_at_5pct', False) else "not-significant"
            
            html += f"""
            <div class="{significance_class}">
                <h4>1. Mean Return Difference Test (Paired t-test)</h4>
                <p><strong>Null Hypothesis:</strong> {mean_test.get('null_hypothesis', 'N/A')}</p>
                <p><strong>Mean Excess Return:</strong> {mean_test.get('mean_excess_return', 0):.6f}</p>
                <p><strong>t-statistic:</strong> {mean_test.get('t_statistic', 0):.3f}</p>
                <p><strong>p-value:</strong> {mean_test.get('p_value', 1):.4f}</p>
                <p><strong>Result:</strong> {mean_test.get('interpretation', 'N/A')}</p>
            </div>
            """
        
        # Variance equality test
        if 'variance_equality_test' in test_data:
            var_test = test_data['variance_equality_test']
            significance_class = "significant" if var_test.get('significant_at_5pct', False) else "not-significant"
            
            html += f"""
            <div class="{significance_class}">
                <h4>2. Variance Equality Test (F-test)</h4>
                <p><strong>F-statistic:</strong> {var_test.get('f_statistic', 0):.3f}</p>
                <p><strong>p-value:</strong> {var_test.get('p_value', 1):.4f}</p>
                <p><strong>Result:</strong> {var_test.get('interpretation', 'N/A')}</p>
            </div>
            """
        
        # Distribution test
        if 'distribution_test' in test_data:
            dist_test = test_data['distribution_test']
            significance_class = "significant" if dist_test.get('significant_at_5pct', False) else "not-significant"
            
            html += f"""
            <div class="{significance_class}">
                <h4>3. Distribution Similarity Test (Kolmogorov-Smirnov)</h4>
                <p><strong>KS statistic:</strong> {dist_test.get('ks_statistic', 0):.4f}</p>
                <p><strong>p-value:</strong> {dist_test.get('p_value', 1):.4f}</p>
                <p><strong>Result:</strong> {dist_test.get('interpretation', 'N/A')}</p>
            </div>
            """
        
        html += "</div>"
    
    html += """
        </div>
        
        <div class="section">
            <h2>Interpretation Guide</h2>
            <ul>
                <li><strong>Highlighted (Yellow):</strong> Statistically significant results (p < 0.05)</li>
                <li><strong>Mean Difference Test:</strong> Tests if one strategy has significantly higher/lower returns</li>
                <li><strong>Variance Equality Test:</strong> Tests if strategies have significantly different risk levels</li>
                <li><strong>Distribution Test:</strong> Tests if return distributions are fundamentally different</li>
                <li><strong>Statistical Significance:</strong> p < 0.05 indicates results are unlikely due to chance</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html


async def run_complete_evaluation(
    strategy_returns: Dict[str, pd.Series],
    output_dir: Path = None,
    benchmark_strategy: str = 'market_cap'
) -> Dict[str, Any]:
    """
    Run complete portfolio evaluation including metrics, tests, and reports.
    
    Args:
        strategy_returns: Dictionary of strategy return series
        output_dir: Output directory for reports
        benchmark_strategy: Strategy to use as benchmark
        
    Returns:
        Dictionary with all evaluation results
    """
    try:
        if output_dir is None:
            output_dir = Path.cwd() / "reports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize evaluator
        evaluator = PortfolioEvaluator(strategy_returns)
        
        # Calculate performance metrics
        performance_metrics = evaluator.calculate_performance_metrics()
        
        # Perform hypothesis tests
        test_results = evaluator.perform_pairwise_hypothesis_tests()
        
        # Generate QuantStats reports for individual strategies
        quantstats_reports = generate_quantstats_reports(
            strategy_returns, 
            benchmark_strategy, 
            output_dir
        )
        
        # Generate comprehensive comparison report
        comparison_report = create_strategy_comparison_report(evaluator, output_dir)
        
        # Save combined returns data
        combined_returns = pd.DataFrame(strategy_returns)
        combined_returns.to_csv(output_dir / "combined_returns.csv")
        
        # Save performance metrics
        if not performance_metrics.empty:
            performance_metrics.to_csv(output_dir / "performance_metrics.csv")
        
        # Save hypothesis test results
        if test_results:
            import json
            with open(output_dir / "hypothesis_tests.json", 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
        
        logger.info("Complete portfolio evaluation finished successfully")
        
        return {
            'performance_metrics': performance_metrics,
            'hypothesis_test_results': test_results,
            'quantstats_reports': quantstats_reports,
            'comparison_report': comparison_report,
            'output_directory': output_dir
        }
        
    except Exception as e:
        logger.error(f"Error in complete evaluation: {e}")
        return {'error': str(e)}


async def load_market_weighted_returns(
    start_date: date, 
    end_date: date,
    db_kwargs: Dict = None
) -> pd.Series:
    """
    Load market-weighted portfolio returns from daily_crypto_index table.
    
    Args:
        start_date: Start date for returns
        end_date: End date for returns
        db_kwargs: Database connection parameters
        
    Returns:
        Series of daily market-weighted returns
    """
    from kallos_portfolios.storage import get_async_session
    import sqlalchemy as sa
    
    try:
        async with get_async_session() as session:
            query = sa.text("""
            SELECT 
                trade_date,
                index_value,
                LAG(index_value) OVER (ORDER BY trade_date) AS prev_value
            FROM daily_crypto_index 
            WHERE trade_date BETWEEN :start_date AND :end_date
            ORDER BY trade_date
            """)
            
            result = await session.execute(query, {
                'start_date': start_date,
                'end_date': end_date
            })
            data = result.fetchall()
            
            if not data:
                logger.warning(f"No market index data found between {start_date} and {end_date}")
                return pd.Series(dtype=float, name='market_weighted_returns')
            
            # Convert to DataFrame with proper data type handling
            rows = []
            for row in data:
                trade_date = row[0]
                index_value = float(row[1]) if row[1] is not None else None  # Convert Decimal to float
                prev_value = float(row[2]) if row[2] is not None else None   # Convert Decimal to float
                rows.append({
                    'trade_date': trade_date,
                    'index_value': index_value,
                    'prev_value': prev_value
                })
            
            df = pd.DataFrame(rows)
            df.set_index('trade_date', inplace=True)
            
            # Calculate daily returns, ensuring all operations are on float values
            df['daily_return'] = ((df['index_value'] / df['prev_value']) - 1.0).fillna(0.0)
            
            # Create returns series with datetime index
            returns = df['daily_return'].dropna()
            returns.index = pd.to_datetime(returns.index)
            returns.name = 'market_weighted_returns'
            
            # Ensure the series contains only float values
            returns = returns.astype(float)
            
            logger.info(f"Loaded {len(returns)} market-weighted returns from {returns.index[0].date()} to {returns.index[-1].date()}")
            
            return returns
            
    except Exception as e:
        logger.error(f"Error loading market-weighted returns: {e}")
        return pd.Series(dtype=float, name='market_weighted_returns')


async def load_portfolio_returns_by_id(portfolio_run_id: int) -> pd.Series:
    """
    Load portfolio returns using portfolio_runs.id.
    
    Args:
        portfolio_run_id: The id from portfolio_runs table
        
    Returns:
        Series of portfolio returns indexed by date
    """
    from kallos_portfolios.storage import get_async_session
    import sqlalchemy as sa
    
    try:
        async with get_async_session() as session:
            query = sa.text("""
            SELECT date, return_value
            FROM portfolio_returns 
            WHERE portfolio_run_id = :portfolio_id
            ORDER BY date
            """)
            
            result = await session.execute(query, {"portfolio_id": portfolio_run_id})
            data = result.fetchall()
            
            if not data:
                logger.warning(f"No portfolio returns found for portfolio_run_id: {portfolio_run_id}")
                return pd.Series(dtype=float)
            
            # Convert to pandas Series
            dates = [row[0] for row in data]
            returns = [float(row[1]) for row in data]
            
            series = pd.Series(returns, index=dates)
            series.index = pd.to_datetime(series.index)
            series.name = f'portfolio_{portfolio_run_id}'
            
            logger.info(f"Loaded {len(series)} returns for portfolio_run_id: {portfolio_run_id}")
            return series
            
    except Exception as e:
        logger.error(f"Error loading portfolio returns for ID {portfolio_run_id}: {e}")
        return pd.Series(dtype=float)


async def load_portfolio_returns_by_run_id(run_id: str) -> pd.Series:
    """
    Load portfolio returns for a specific run ID.
    
    Args:
        run_id: Portfolio run identifier
        
    Returns:
        Series of daily portfolio returns
    """
    from kallos_portfolios.storage import get_async_session
    import sqlalchemy as sa
    
    try:
        async with get_async_session() as session:
            # First get the portfolio_run_id
            run_query = sa.text("""
            SELECT id FROM portfolio_runs WHERE run_id = :run_id
            """)
            
            run_result = await session.execute(run_query, {'run_id': run_id})
            run_data = run_result.fetchone()
            
            if not run_data:
                logger.warning(f"No portfolio run found for run_id: {run_id}")
                return pd.Series(dtype=float, name=run_id)
            
            portfolio_run_id = run_data[0]
            
            # Load the returns
            returns_query = sa.text("""
            SELECT date, return_value
            FROM portfolio_returns 
            WHERE portfolio_run_id = :portfolio_run_id
            ORDER BY date
            """)
            
            returns_result = await session.execute(returns_query, {
                'portfolio_run_id': portfolio_run_id
            })
            returns_data = returns_result.fetchall()
            
            if not returns_data:
                logger.warning(f"No returns data found for run_id: {run_id}")
                return pd.Series(dtype=float, name=run_id)
            
            # Convert to Series
            returns_df = pd.DataFrame(returns_data, columns=['date', 'return_value'])
            returns_df.set_index('date', inplace=True)
            
            # Convert Decimal to float and create series
            returns_series = returns_df['return_value'].astype(float)
            returns_series.name = run_id
            
            logger.info(f"Loaded {len(returns_series)} returns for {run_id} from {returns_series.index[0]} to {returns_series.index[-1]}")
            
            return returns_series
            
    except Exception as e:
        logger.error(f"Error loading portfolio returns for {run_id}: {e}")
        return pd.Series(dtype=float, name=run_id)


async def create_three_strategy_comparison(
    gru_portfolio_id: int,
    historical_portfolio_id: int, 
    start_date: date,
    end_date: date,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Create comprehensive 3-strategy comparison report using portfolio_runs.id values:
    1. GRU Generated Portfolio (by portfolio_runs.id)
    2. Naive Historical Portfolio (by portfolio_runs.id)
    3. Market Weighted Portfolio (from daily_crypto_index table)
    
    Args:
        gru_portfolio_id: portfolio_runs.id for GRU strategy
        historical_portfolio_id: portfolio_runs.id for Historical strategy
        start_date: Start date for comparison
        end_date: End date for comparison
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with complete evaluation results
    """
    try:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "reports" / f"three_strategy_comparison_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Starting Three-Strategy Portfolio Comparison")
        logger.info(f"📊 GRU Portfolio ID: {gru_portfolio_id}")
        logger.info(f"📊 Historical Portfolio ID: {historical_portfolio_id}")
        logger.info(f"📊 Market Weighted Portfolio (from daily_crypto_index)")
        logger.info(f"📅 Comparison Period: {start_date} to {end_date}")
        
        # Load all three strategy returns
        strategy_returns = {}
        
        # 1. Load GRU portfolio returns by ID
        gru_returns = await load_portfolio_returns_by_id(gru_portfolio_id)
        if not gru_returns.empty:
            strategy_returns['GRU_Portfolio'] = gru_returns
            logger.info(f"✅ Loaded GRU returns: {len(gru_returns)} observations")
        else:
            logger.warning(f"⚠️  No GRU returns loaded for portfolio_id: {gru_portfolio_id}")
        
        # 2. Load Historical portfolio returns by ID
        hist_returns = await load_portfolio_returns_by_id(historical_portfolio_id)
        if not hist_returns.empty:
            strategy_returns['Historical_Portfolio'] = hist_returns
            logger.info(f"✅ Loaded Historical returns: {len(hist_returns)} observations")
        else:
            logger.warning(f"⚠️  No Historical returns loaded for portfolio_id: {historical_portfolio_id}")
        
        # 3. Load Market-weighted portfolio returns from daily_crypto_index
        market_returns = await load_market_weighted_returns(start_date, end_date)
        if not market_returns.empty:
            strategy_returns['Market_Weighted'] = market_returns
            logger.info(f"✅ Loaded Market returns: {len(market_returns)} observations")
        else:
            logger.warning(f"⚠️  No Market returns loaded for period {start_date} to {end_date}")
        
        if len(strategy_returns) < 2:
            raise ValueError(f"Need at least 2 strategies for comparison, got {len(strategy_returns)}")
        
        logger.info(f"🔄 Running comprehensive evaluation with {len(strategy_returns)} strategies...")
        
        # Run complete evaluation
        evaluation_results = await run_complete_evaluation(
            strategy_returns=strategy_returns,
            output_dir=output_dir,
            benchmark_strategy='Market_Weighted'  # Use market as benchmark
        )
        
        # Add metadata
        evaluation_results['comparison_metadata'] = {
            'gru_portfolio_id': gru_portfolio_id,
            'historical_portfolio_id': historical_portfolio_id,
            'comparison_period': f"{start_date} to {end_date}",
            'strategies_compared': list(strategy_returns.keys()),
            'benchmark_strategy': 'Market_Weighted',
            'total_observations': {name: len(returns) for name, returns in strategy_returns.items()},
            'report_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Three-strategy comparison completed! Reports in: {output_dir}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"❌ Three-strategy comparison failed: {e}")
        return {'error': str(e)}
