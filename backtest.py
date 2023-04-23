import numpy as np 
import pandas as pd
import compute_utils as compute
import references as refs
import optimisation as opt
from datetime import date, datetime
from dateutil.relativedelta import relativedelta



def generate_buy_sell_signal(t1, t0):
    """
    Generates Buy/Sell by omparing the day's corecast value and the previous day's forecast value 

    Parameters:
        t1 (float): forecast value of the current day
        t0 (float): forecast value of 1 day before the current day

    Returns:
        int: 1 for "Buy", 0 for "Sell" 
    """
    ## If it's the first in the series 
    if t1 > 0 and isinstance(t0, type(None)):
        print("Exe")
        return 1
    if t1 <= 0 and isinstance(t0, type(None)):
        return 0 
    ## Normal Checks
    # if today is negative when yesterday was positive, it's a sell
    if t1 <= 0 and t0 > 0:
        return 0
    # if today is positive when yesterday is negative, it's a buy 
    if t1 > 0 and t0 <= 0: 
        return 1
    
def generate_holding_profile(forecast_df): 
    """
    Function generates binary Buy/Hold/Sell profile based on the day's forecast value and the previous day's forecast value. 

    Parameters:
        forecast_df(pd.DataFrame): the dataframe containing the forecast values of a list of stocks 
    Returns:
        pd.DataFrame: contains data on whether the stock is being held on a certain date
    """
    forecast_df = forecast_df.copy()
    for col in forecast_df.columns: 
        forecast_df.rename(columns = {col: col + str("_t1_forecast")}, inplace = True)
        forecast_df[col + str("_t0_forecast")] = forecast_df[col + str("_t1_forecast")].shift(1)

        # first t0 will be NaN. Replac with default sell signal 
        forecast_df[col + str("_t0_forecast")].iloc[0] = 0

        forecast_df[col + str("_signal")] = forecast_df.apply(lambda x: generate_buy_sell_signal(x[col + str("_t1_forecast")], x[col + str("_t0_forecast")]), axis = 1)
        # Remove unrequired cols 
        forecast_df.drop([col + str("_t0_forecast"),col + str("_t1_forecast")], axis = 1, inplace = True)
        # Forward fill the NAs 
        # I.e if you were previously holding (i.e 1), you will continue holding unless otherwise
        forecast_df.ffill(inplace = True)
    return forecast_df 

def generate_holding_profile_dynamic(forecast_df, scale):
    # Rebalance to [-1, +1]
    output_df = forecast_df.div(scale).copy()
    output_df = output_df.add_suffix("_signal")
    return output_df

def log_buy_and_sell_history(return_and_holding_df):
    """
    Infers the buy and sell point, and prices, given the returns and holdings profile
    Is a helper function for run_strat_backtest 

    Paramters:
        return_and_holding_df (pd.DataFrame): dataframe with dates as index, and has 2 columns -
                                                1. returns column, with ticker code as column title 
                                                2. holding profile (binary 1/0) which has <ticker_code>_signal as the column title  
    Returns:
        pd.DataFrame: a dataframe detailing the buy/sell action, and at the relevant date and price
    """
    ticker_code = return_and_holding_df.columns[0]
    forecast_history = return_and_holding_df.copy() 

    forecast_history['signal-1'] = forecast_history[[ticker_code + "_signal"]].shift(1)
    log_buys = forecast_history[(forecast_history[ticker_code + "_signal"] == 1.0) & (forecast_history['signal-1'] == 0.0)].copy()
    # initial buy (if applicable)
    if forecast_history.iloc[0][ticker_code + '_signal'] > 0:
        log_buys = log_buys.append(forecast_history.iloc[0])
    log_buys = log_buys.drop(columns = log_buys.columns[1:])
    log_buys['Action'] = 'Buy'


    log_sells = forecast_history[(forecast_history[ticker_code + "_signal"] == 0.0) & (forecast_history['signal-1'] == 1.0)].copy()
    log_sells =  log_sells.drop(columns = log_sells.columns[1:])
    log_sells['Action'] = 'Sell'

    logs = pd.concat([log_buys, log_sells], axis = 0).sort_index()
    return logs

def run_strat_backtest(hist_price_df, forecast_df, start_date = None, end_date = None, init_capital = 1., display_logs = True, return_logs = False):
    """
    Function runs a single backtest on 1 security only. Given the historical prices, and the forecast values from forecasts.py, simulates the gain/loss from holding the security based on the given forecast strategy

    Parameters:
        hist_price_df (pd.DataFrame): historical prices of a single stock
        forecast_df(pd.DataFrame): the dataframe containing the forecast values of a stock 
        start_date (str): When to start looking at this strategy, in the form "YYYY-MM-DD" 
        end_date (str): When to stop looking at this strategy, in the form "YYYY-MM-DD"
        init_capital (float): the simulated amount of cash put into the subsystem 
        display_logs (bool): True to print logs of when buy/sell is issued. False otherwise
        return_logs (bool): True to return the logs as a dataframe, after the simulated return profile. False otherwise 
    Returns:
        pd.DataFrame: returns the simulated holding value executing this strategy
    
    Note:
        We assume start-of-day holding. Thus, the first value may not be exactly the initial capital
    """

    ## Data Validation 
    if len(hist_price_df.columns) != 1: raise Exception("hist_price_df can only have 1 column")
    if (len(forecast_df.columns)) != 1: raise Exception("forecast_df can only have 1 column")

    return_df = compute.get_returns_from_prices(hist_price_df)
    holding_profile = generate_holding_profile(forecast_df)
    # holding_profile = generate_holding_profile_dynamic(forecast_df, 20) 

    ticker_code = return_df.columns[0]
    holding_col_name = holding_profile.columns[0]

    return_and_holding_df = return_df.join(holding_profile, how = "inner")

    if isinstance(start_date, type(None)): start_date = return_and_holding_df.index.min()
    if isinstance(end_date, type(None)): end_date = return_and_holding_df.index.max()

    return_and_holding_df = return_and_holding_df.loc[start_date: end_date].copy()

    # Create buy_sell log 
    buy_sell_log = log_buy_and_sell_history(return_and_holding_df)
    buy_sell_log.rename(columns = {ticker_code: ticker_code + "_ret"}, inplace = True)
    # Note: Buy at today's opening price = ytd's closing price, Sell at today's closing price
    buy_sell_log = buy_sell_log.join(hist_price_df.shift(1), how = "inner") 
    # display only if required
    if display_logs: display(buy_sell_log)

    return_and_holding_df['sim_returns'] = return_and_holding_df[ticker_code].multiply(return_and_holding_df[holding_col_name]).add(1)
    # display(return_and_holding_df) #Let's check this
    # Get first nonNaN value 
    first_valid_idx = return_and_holding_df['sim_returns'].index.get_loc(return_and_holding_df['sim_returns'].first_valid_index())
    if first_valid_idx != 0: return_and_holding_df['sim_returns'].iloc[first_valid_idx - 1] = init_capital
    else: return_and_holding_df['sim_returns'].iloc[first_valid_idx] = init_capital

    sim_holding_value = pd.DataFrame(np.cumprod(return_and_holding_df['sim_returns']))
    sim_holding_value.rename(columns = {"sim_returns": ticker_code + "_holding value"}, inplace = True)

    del return_df, holding_profile, ticker_code, holding_col_name, return_and_holding_df 
    if return_logs: return sim_holding_value, buy_sell_log 
    else: return sim_holding_value

def run_strat_backtest_multi(hist_prices_df, forecasts_df, start_date = None, end_date = None, init_capitals = None, display_logs = True, return_logs = False):
    """
    Function applies run_strat_backtest() on multiple tickers and returns the holding value return on a list of tickers
    """

    # Get tickers 
    ticker_codes = list(hist_prices_df.columns) 
    output = None 
    buy_sell_log = None

    # Loop
    for ticker in ticker_codes:
        init_cap = init_capitals[ticker] if not isinstance(init_capitals, type(None)) else 1.0
        if isinstance(output, type(None)):
            output, buy_sell_log = run_strat_backtest(hist_prices_df[[ticker]], forecasts_df[[ticker]], start_date = start_date, end_date = end_date, init_capital = init_cap, display_logs = display_logs, return_logs = True) # set return logs to true here for consistency. 
        else:
            next_output, next_buy_sell_log = run_strat_backtest(hist_prices_df[[ticker]], forecasts_df[[ticker]], start_date = start_date, end_date = end_date,  init_capital = init_cap, display_logs = display_logs, return_logs = True)
            output = output.join(next_output, how = "outer")
            buy_sell_log = buy_sell_log.join(next_buy_sell_log, how = "outer", rsuffix = "_" + str(ticker))
    if return_logs: return output, buy_sell_log
    else: return output

def get_backtest_performance(holding_value_df, benchmark_df):
    """
    Generates key statistics of backtest, given the time series data of the strategy's holding value 

    Parameters:
        holding_value_df (pd.DataFrame): contains a single column with ticker code as column title, and the holding value at each period 
        benchmark_df (pd.DataFrame): contains a single column representing the risk free rate 
    Returns:
        pd.DataFrame: DataFrame containing the below statistics as rows, and a single column with ticker code as it's column title
    """

    strat_returns = compute.get_returns_from_prices(holding_value_df)
    strat_returns = strat_returns.join(benchmark_df, how = "outer")

    ticker_code = holding_value_df.columns[0] ## In the form of "TickerCode_holding returns"
    ticker_code = ticker_code[:ticker_code.find("_")] ## remove appended underscore onwards
    
    # Period holding returns 
    abs_holding_ret = holding_value_df.loc[holding_value_df.last_valid_index()].values[0] - holding_value_df.loc[holding_value_df.first_valid_index()].values[0] 

    # Percentage holding returns 
    perc_holding_ret = abs_holding_ret / holding_value_df.loc[holding_value_df.last_valid_index()].values[0]
    
    # Mean Returns
    mean_ret = strat_returns.mean().values[0]

    # Standard Deviation (Daily)
    std = strat_returns.std().values[0]

    # Min/Max 
    min = holding_value_df.min().values[0]
    max = holding_value_df.max().values[0]

    # Skew 
    skew = compute.calc_skew(strat_returns)

    # display(strat_returns[strat_returns.columns[0]].plot.hist())

    # Max Drawdown 
    max_drawdown = strat_returns.min().values[0]

    # Sharpe and Sortino Ratio 
    strat_returns['Excess Returns'] = strat_returns[strat_returns.columns[0]] - strat_returns['rf_rate']
    if std == 0. :
        sharpe = 0.
    else:
        sharpe = strat_returns.mean().values[0] / std

    return pd.DataFrame([sharpe, abs_holding_ret, perc_holding_ret, mean_ret, std, skew, max_drawdown, min, max],
                        index = ['sharpe', 'abs_holding_returns', 'perc_holding_returns', 'mean_ret', 'std', 'skew', 'max_drawdown', 'min', 'max'],
                        columns = [ticker_code])

def repeat_strat_backtest(hist_price_df, forecast_df, bench_df, start_and_end_dates, init_capital = 1., display_logs = True):
    """
    Given 1 strategy's forecast, conducts the backtest on 1 security over multiple periods (see notes for example)

    Parameters:
        hist_price_df (pd.DataFrame): historical prices of a single stock
        forecast_df(pd.DataFrame): the dataframe containing the forecast values of a stock 
        bench_df (pd.DataFrame): datafrmae containing the benchmark rate as its only column. Dates should match hist_price_df
        start_and_end_dates (list): a list of lists, each containing 2 elements - the start and end date. See notes 
        
    Returns 
        pd.DataFrame: dataframe containing each period as a column, and all statistics specified in get_backtest_performance() 

    Notes:
        E.g Suppose a monthly strat test for the year of 2021. Then the function will test for Jan - Feb, and generate the results from that period, then repeat Feb - Mar, Mar - Apr etc. 
        start_and_end_dates should have the structure [[<START_DATE_1>, <END_DATE_1>], [<START_DATE_2>, <END_DATE_2>],...]
    """
    results_df = None 
    for period in start_and_end_dates:
        if isinstance(results_df, type(None)):
            results_df = get_backtest_performance(run_strat_backtest(hist_price_df = hist_price_df, 
                                                                     forecast_df = forecast_df,
                                                                    start_date = period[0],
                                                                    end_date =  period[1], 
                                                                    init_capital = init_capital, 
                                                                    display_logs = display_logs), bench_df).copy() 
            # results_df.rename(columns = {results_df.columns[0]: results_df.columns[0] + "_strat_ending_" + str(period[1])}, inplace = True)
            results_df.rename(columns = {results_df.columns[0]: period[1]}, inplace = True)
        else:
            output = get_backtest_performance(run_strat_backtest(hist_price_df = hist_price_df, 
                                                                    forecast_df = forecast_df,
                                                                    start_date = period[0],
                                                                    end_date =  period[1], 
                                                                    init_capital = init_capital, 
                                                                    display_logs = display_logs), bench_df).copy() 
            # output.rename(columns = {output.columns[0]: output.columns[0] + "_strat_ending_" + str(period[1])}, inplace = True)
            output.rename(columns = {output.columns[0]: period[1]}, inplace = True)
            results_df = results_df.join(output, how = 'outer')
    return results_df

def gen_strat_stat_from_multi_instr(ticker_list, hist_prices_df, forecasts_df, bench_df, start_and_end_dates, init_capitals = None, stat_to_use = "sharpe", display_logs = False):
    """
    Generates a single strategy's (expressed in it's forecasts) performance over multiple instruments (tickers). Performance can be measured 
    by any of the metrics given in get_backtest_performance()

    Parameters:
        ticker_list (list): a list of strings representing the ticker code. Ticker code must be a column header in hist_prices_df and forcasts_df 
        hist_prices_df (pd.DataFrame): historical prices, with ticker codes as column headers, indexed by date. Can be multi-columned 
        forecasts_df (pd.DataFrame): forecast values of multiple instruments, indexed by date. Can be multi-columned. The output of any method from the Forecast module (e.g get_ewmac_forecasts)
        bench_df (pd.DataFrame): datafrmae containing the benchmark rate as its only column. Dates should match hist_price_df
        start_and_end_dates (list): a list of lists, each containing 2 elements - the start and end date. See notes 
        [Optional] stat_to_use (str): the type of statistics to aggregate the data on. Must be a row index of the output from get_backtest_performance. Defaults to Sharpe 
        display_logs (bool): True to print logs of when buy/sell is issued. False otherwise 

    Returns:
        pd.DataFrame: indexed by date, column with <STAT>_<TICKER_CODE> as column titles, and the stat_to_use values as data 
    Notes:
        start_and_end_dates should have the structure [[<START_DATE_1>, <END_DATE_1>], [<START_DATE_2>, <END_DATE_2>],...]
    """
    results_df = None
    for ticker in ticker_list:
        if isinstance(init_capitals, type(None)): init_capital = 1. 
        else: init_capital = init_capitals[ticker]
        # Get backtest results 
        backtest_result = repeat_strat_backtest(hist_price_df = hist_prices_df[[ticker]], 
                                                forecast_df = forecasts_df[[ticker]],
                                                bench_df =  bench_df, 
                                                start_and_end_dates = start_and_end_dates, 
                                                init_capital = init_capital, 
                                                display_logs = display_logs).T[[stat_to_use]]
        backtest_result.rename(columns = {stat_to_use: stat_to_use + "_" + str(ticker)}, inplace = True)
        if isinstance(results_df, type(None)):
            results_df = backtest_result.copy()
        else:
            results_df = results_df.join(backtest_result, how = "outer")
    return results_df


def get_corr_between_strats(strat_1_returns_df, strat_2_returns_df):
    """
    Compares between the time series of returns from 2 strategies to compute it's average correlation. See notes for method 

    Parameters:
        strat_1_returns_df (pd.DataFrame): time series of returns of a strategy across multiple assets. The output of gen_strat_stat_from_multi_instr(..., stat_to_use = "perc_hodling_returns")
        strat_2_returns_df (pd.DataFrame): as above, for a different strat. Column headers must exactly match strat_1_returns_df.columns

    Returns:
        float: the average correlation between the 2 strategies 

    Notes:
        Given 2 dataframes, each with structure: 

        -----| <TICKER1> | <TICKER2> | ... -
        date1|    r1,1   |   r2,1    | ... - 
          .  |     .     |     .     | ... -
          .  |     .     |     .     | ... -
        
        Algo will merge <TICKER1> from both dataframes and get the correlation of returns. Then, repeat for all tickers, before getting an average correlation

    For Future:
        Consider cross sectional data multiple regression panel data to get correlation across strategies

    """
    if not list(strat_1_returns_df.columns) == list(strat_2_returns_df.columns): #auto checks for order
        print("Columns not exactly the same")
    
    def get_corr_between_strats_single(strat_1_returns_df, strat_2_returns_df):
        combined_df = strat_1_returns_df.join(strat_2_returns_df, how = "inner", rsuffix = "_2")
        col_1_header = combined_df.columns[0]
        col_2_header = combined_df.columns[1]
        return combined_df[col_1_header].corr(combined_df[col_2_header])
    
    correl_list = [] 
    
    for col in strat_1_returns_df.columns:
        correl_list.append(get_corr_between_strats_single(strat_1_returns_df[[col]], strat_2_returns_df[[col]]))
    
    return np.array(correl_list).mean()

def get_weighted_forecasts(weight_vector, forecasts):
    """
    Returns the scaled forecasts for a list of tickers 

    Parameters:
        weight_vector (list): list of scalars containing the weight for each forecast 
        forecasts (list): a list of pd.DataFrames, each containing the forecast for a list of tickers (the output from any function in forecasts.py)
    
    Returns:
        pd.DataFrame: scaled forecasts 
    
    Notes:
        1. The length of the vector must equal the number of forecasts 
        2. Every forecast in forecasts should have the same number and order of tickers 
    """

    # Validation 
    if not  len(weight_vector) == len(forecasts): raise Exception("Vector has len " + str(len(weight_vector)) + ", but there are " + str(len(forecasts)) + "forecasts")
    # [Note]: not checking each forecast df

    weighted_forecasts = [] 
    for w, f in zip(weight_vector, forecasts):
        weighted_forecasts.append(w * f)

    final_weighted_forecast = weighted_forecasts[0].copy()
    for wf in weighted_forecasts[1:]:
        final_weighted_forecast = final_weighted_forecast.add(wf, fill_value = 0)

    return final_weighted_forecast

def get_scaled_forecasts(weighted_forecast, scaling_factor):
    """
    Returns the final scaled forecasts after diversification multiplier

    Parameters:
        weighted_forecast (pd.DataFrame): the output of get_weighted_forecasts() 
        scaling_factor (float): the scaling factor as defined. Use references.diversification_multplier_table
    """
    final_scaled_forecast = weighted_forecast * scaling_factor
      # Max Min Function 
    final_scaled_forecast.where(final_scaled_forecast < refs.forecast_max_min_parameters['max'], refs.forecast_max_min_parameters['max'], inplace = True)
    final_scaled_forecast.where(final_scaled_forecast > refs.forecast_max_min_parameters['min'], refs.forecast_max_min_parameters['min'], inplace = True)
    return final_scaled_forecast

def get_standardized_forecasts(scaled_forecasts, std_scalar):
    """
    Returns the proportion of vol scalars to position 

    Parameters:
        scaled_forecasts(pd.DataFrame): the output from get_scaled_forecasts() 
        std_scalar (int): the max positive value of the forecasts 
    
    Returns:
        pd.DataFrame: Standardized forecasts with the same structure as scaled_forecasts
    """
    # Validation 
    if not isinstance(std_scalar, int): raise Exception ("std_scalar is not an integer")
    return scaled_forecasts / std_scalar

def get_subsystem_position(std_forecast, vol_scalar): 
    """
    Returns the subsystem position to be taken (i.e number of units to be bought)

    Parameters:
        std_forecast (pd.Series): a row from the output of get_standardized_forecasts() 
        vol_scalar (pd.Series): the volatility scalar
    Returns:
        pd.Series: subsystem positions for each security
    """
    assert len(std_forecast) == len(vol_scalar), "Length mismatch for std_forecast and vol_scalar"
    return std_forecast * vol_scalar

def get_averaged_optimized_weights(hist_prices_df, 
                                    bench_df, 
                                    starting_observation_period,
                                    observation_rate,
                                    observation_freq = "m",  
                                    method = "sharpe",
                                    ret_computation_method = 'statmean', 
                                    cov_estimate = 'ledoit_wolf', 
                                    weight_bounds = (0,1), 
                                    opt_backup = 'min_vol', 
                                    show_weights = False):
    """
    Function bootstraps to find the average optimal weights that maximises either Sharpe or Sortino ratio 
    This is done via an expanding window observation, with the initial set by the "starting_observation_period", then expanded by the observation_freq (months) per step 

    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, and identifier codes (either Company ID or Ticker Code) as column headers 
        bench_df (pd.DataFrame): The dataframe containing the historical risk free weight to be used as the benchmark rate, indexed by date and has only 1 column - the risk free rate 
        starting_observation_period (int): the number of months to use for the first calculation of sharpe/sortino ratio 
        observation_rate (int): the number of months/days (depending on observation_freq) to increase the window by per iteration
        observation_freq (str): How much to expand the expanding window by. Either "m" or "d" for months or days respectively. Defaults to monthly
        method (str): Either "sharpe" or "sortino"
        ret_computation_method (str): Method of computing mean historical returns. Either 'statmean' for historical mean or 'lognorm' for log normal computation. See https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html
        cov_estimate (str): Method used to estimate covariance. Either 'ledoit_wolf' or 'sample' for ledoit_wolf shrinkage estimator and sample covariance method respectively
        weight_bounds (tuple): tuple containing the lower bound and upper bound allowed for individual equity weights (lower_bound(float), upper_bound(float))
        opt_backup (str): In the event of sharpe/sortino optimisation failure (due to too many negative returns), choose 1 alternative weight distribution method. Either 'min_vol' for minimum volatility portfolio, or 'equal_weight' for equal weighting the portfolio 
        show_weights (boolean): Toggle to print out the weights. True to print, False otherwise
    Returns:
        pd.Series: the mean weight computed from the observation of optimised weights per period. 
    """

    ## to compute, we first need all prices to be available 
    hist_prices_df = hist_prices_df.dropna().copy()

    # get first observation end date 
    obs_end_date = hist_prices_df.index[0] + relativedelta(months = starting_observation_period)
    
    output = None
    num_of_obs = 0 
    while obs_end_date < hist_prices_df.index[-1]: 
        ow = dict(opt.get_optimised_portfolio(
            portfolio_hist_prices = hist_prices_df[:obs_end_date],
            end_date = obs_end_date, 
            risk_free_df = bench_df, 
            method = method, 
            ret_computation_method = ret_computation_method, 
            cov_estimate = cov_estimate, 
            weight_bounds = weight_bounds, 
            opt_backup = opt_backup, 
            show_weights = show_weights
        ))
        ow = pd.DataFrame().from_dict(ow, orient = "index").T
        ow.index = [obs_end_date]
        if isinstance(output, type(None)):
            output = ow.copy() 
        else:
            output = output.append(ow)
        if observation_freq == "m": obs_end_date = obs_end_date + relativedelta(months = observation_rate)
        elif observation_freq == "d": obs_end_date = obs_end_date + relativedelta(days = observation_rate)
        else:
            print("observation_rate not 'm' or 'd'. Defaulted to m")
            obs_end_date + relativedelta(months = observation_rate)
        num_of_obs += 1
    print("Number of Samples: ", str(num_of_obs))
    # display(output.plot())
    return output.mean(axis = 0)