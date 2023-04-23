import sys
import subprocess

try:
    import pyPortfolioOpt
except:
    # Execute installation of dask if required
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pyPortfolioOpt"])

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


def get_optimised_portfolio(portfolio_hist_prices, 
                            end_date, 
                            risk_free_df, 
                            method = "sharpe", 
                            ret_computation_method = 'statmean', 
                            cov_estimate = 'ledoit_wolf', 
                            weight_bounds = (0,1), 
                            opt_backup = 'min_vol', 
                            show_weights = False):
    """
    Function returns the optimized weights given the historical prices of the portfolio, given either Sharpe or Sortino optimisation 
    
    Parameters:
        portfolio_hist_prices (pd.DataFrame): Dataframe indexed by date, and identifier codes (either Company ID or Ticker Code) as column headers 
        end_date (pd.DateTime): The last date to be used in computing returns
        risk_free_df (pd.DataFrame): The dataframe containing the historical risk free weight to be used as the benchmark rate, indexed by date and has only 1 column - the risk free rate 
        method (str): Either "sharpe" or "sortino"
        ret_computation_method (str): Method of computing mean historical returns. Either 'statmean' for historical mean or 'lognorm' for log normal computation. See https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html
        cov_estimate (str): Method used to estimate covariance. Either 'ledoit_wolf' or 'sample' for ledoit_wolf shrinkage estimator and sample covariance method respectively
        weight_bounds (tuple): tuple containing the lower bound and upper bound allowed for individual equity weights (lower_bound(float), upper_bound(float))
        opt_backup (str): In the event of sharpe/sortino optimisation failure (due to too many negative returns), choose 1 alternative weight distribution method. Either 'min_vol' for minimum volatility portfolio, or 'equal_weight' for equal weighting the portfolio 
        show_weights (boolean): Toggle to print out the weights. True to print, False otherwise
    Returns:
        OrderedDict: the optimised weights given by Sharpe or Sortino optimisation
    """
    # Validate arguments 
    assert method == 'sharpe' or method == 'sortino', "method must be 'sharpe' or 'sortino'"
    assert opt_backup == 'min_vol' or opt_backup == 'equal_weight', "opt_backup must be 'min_vol' or 'equal_weight'"
    assert ret_computation_method == 'statmean' or ret_computation_method == 'lognorm', "ret_computation_method must be 'statmean' or 'lognorm'"
    assert cov_estimate == 'ledoit_wolf' or cov_estimate == 'sample', "cov_estimate must be 'ledoit_wolf' or 'sample'"

    # Calculate expected returns and sample covariance
    mean_ret = expected_returns.mean_historical_return(portfolio_hist_prices)
    if ret_computation_method == 'lognorm':
        mean_ret = expected_returns.mean_historical_return(portfolio_hist_prices, log_returns = True)
    
    # print(mean_ret)
    # print(type(mean_ret))
    # display(portfolio_hist_prices)
    try:
        if cov_estimate == 'ledoit_wolf':
            covariance = risk_models.CovarianceShrinkage(portfolio_hist_prices).ledoit_wolf()
        else:
            covariance = risk_models.sample_cov(portfolio_hist_prices)
    except:
#         display(portfolio_hist_prices)
        raise
    semi_cov = risk_models.semicovariance(portfolio_hist_prices)
    cov_map = {'sharpe':covariance,'sortino':semi_cov}
    cov = cov_map[method]

#     print("START: ", portfolio_hist_prices.index.min())
#     print("END: ", end_date)
    # Optimize for maximal ratio
    #if the bounds is not applicable, distribute equal weight to each stock
    if weight_bounds[0] * portfolio_hist_prices.shape[1]>1.001 or weight_bounds[1] * portfolio_hist_prices.shape[1]<0.999: 
      cur_weight_bounds = (1/portfolio_hist_prices.shape[1], 1/portfolio_hist_prices.shape[1])
    else:
      cur_weight_bounds = weight_bounds 
      if show_weights:
        print('Weight Bounds:', cur_weight_bounds) 
    ef = EfficientFrontier(mean_ret, cov, weight_bounds = cur_weight_bounds)
    rf = risk_free_df[:end_date].iloc[-1].values[0] # Note this line is different from source code
    # print('riskfree:', rf) # For debugging, trust me you will need it some day 
    try:
        raw_weights = ef.max_sharpe(risk_free_rate = rf)
    except:
        if opt_backup == 'equal_weight':
            cur_weight_bounds = (1/portfolio_hist_prices.shape[1], 1/portfolio_hist_prices.shape[1])
            print("oh no, optimization error. Perhaps this quarter's stocks has too many negative expected returns, giving them equal weight of:", cur_weight_bounds)
        else:
            print("oh no, optimization error. Perhaps this quarter's stocks has too many negative expected returns, getting min. volatility portfolio with weight_bounds:", cur_weight_bounds)
        print('Start Period: ' + str(portfolio_hist_prices.index[-1])+'\n')
        ef = EfficientFrontier(mean_ret, cov, weight_bounds = cur_weight_bounds)
        print(mean_ret)
        # raw_weights = ef.max_sharpe()
        raw_weights = ef.min_volatility()
        # cleaned_weights = {}
        # raise
    
    cleaned_weights = ef.clean_weights()
    if show_weights: 
      print(cleaned_weights)
      type(cleaned_weights)
    return cleaned_weights