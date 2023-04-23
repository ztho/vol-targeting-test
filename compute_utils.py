import pandas as pd 
import numpy as np 


def get_returns_from_prices(hist_prices_df, log_prices = True, dropna = False):
    """
    Function computes the log_prices returns from a DataFrame containing historical prices 
    
    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
        log_prices (Boolean): True default to use log prices calculation, else use simple pct_change()
        dropna (Boolean): True if dropna, False otherwise
    Returns:
        pd.DataFrame: Dataframe indexed by date, with ticker codes as column headers and returns as rows
    """
    if log_prices:
        return np.log(1 + hist_prices_df.pct_change()).dropna(how="all") if dropna else np.log(1 + hist_prices_df.pct_change())
    else:
        return hist_prices_df.pct_change().dropna() if dropna else hist_prices_df.pct_change()

def calc_annualized_returns(hist_returns_df, periods_per_year):
    """
    Function returns array containing annualized returns of individual stocks
    
    :param hist_returns_df: pd.DataFrame containing historical prices
    :param periods_per_year: int - number of periods per year to compound. (e.g if data is monthly data, then periods per year = 12)
    
    :returns: np.array containing annualized returns of all stocks
    """
    cg = (1 + hist_returns_df).prod() 
    return cg ** (periods_per_year/hist_returns_df.count()) - 1

def calc_recent_vols(hist_returns_df, lookback_period = 90):
    """
    Calculates the recent volatility given a time series and a lookback period 

    Parameters:
        hist_returns_df (pd.DataFrame): pd.DataFrame containing historical prices
        lookback_period (int): number of periods used to calculate the volatility. E,g If hist_returns_df is in days, then lookback_period = 90 means 90 days vol
    Returns:
        pd.Series: Series containing tickers as keys, vols as values
    
    """
    ## Validation 
    if not isinstance(lookback_period, int): raise Exception("lookback_period is not integer")
    if hist_returns_df.shape[0] < lookback_period:
        lookback_period = hist_returns_df.shape[0]
        print("hist_returns_df has fewer periods than the lookback. Using {} periods".format(lookback_period))
    
    return hist_returns_df[-1 * lookback_period:].std()

def calc_skew(hist_returns_df):
    """
    Function calculates skew of returns given historical returns 
    
    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
    Returns:
        Series: skew of returns
    """

    return hist_returns_df.skew().values[0]

def calc_ewma(hist_prices_df, deg_of_mixing = None, span = None):
    """
    Function returns the exponentially weighted moving average 

    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
        deg_of_mixing (float): see df.ewma() for documentation
        span (int): the lookback period (eg. 24 for 24 days)
    Returns:
        pd.DataFrame: ewma-smoothed prices
    """

    ## Error Handling - Ensure not both None, or both not None
    if (isinstance(deg_of_mixing, type(None)) and isinstance(span, type(None))): raise Exception("Either deg_of_mixing or span span must be specified.")
    if not (isinstance(deg_of_mixing, type(None)) or isinstance(span, type(None))): raise Exception("Only 1 of deg_of_mixing or span can be specified")

    if not isinstance(deg_of_mixing, type(None)): return hist_prices_df.ewm(alpha = deg_of_mixing).mean()
    if not isinstance(span, type(None)): return hist_prices_df.ewm(span = span).mean()

def calc_ewma_vol(hist_prices_df, vol_lookback_period = 25):
    """
    Calculates and returns the calculated EWMA price standard deviation calculated over a certain lookback window 

    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
        vol_lookback_period (int): The number of periods to lookback to calculate the EWMA standard deviation (eg. 25 would be 25 days for daily prices)
    Returns:
        pd.DataFrame: the standard deviation of the EWMA price for each ticker in hist_prices_df, for each day 
    """
    absolute_price_returns = hist_prices_df - hist_prices_df.shift(1) # Today's Price - Yesterday's Price 
    return absolute_price_returns.ewm(span = vol_lookback_period).std() # Calculates the std of the ewm 

def calc_ewmac(hist_prices_df, ewma_cal_method = "span", ewma_fast_parameter = 16, ewma_slow_parameter = 64,  vol_standardize = True, vol_lookback_period = 25):
    """
    Function calculates the raw EWMA crossover, taking EWMA_fast - EWMA_slow to find crossover points. Option available for volatility standardising (reccomended)

    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
        ewma_cal_method (str): Either "span" or "deg_of_mixing". Method for calculating EWMA. 
        ewma_fast_parameter (int or float): the value of the parameter for "span" or "deg_of_mixing" to calculate the fast EWMA. 
        ewma_slow_parameter (int or float): the value of the parameter for "span" or "deg_of_mixing" to calculate the slow EWMA.
        vol_standardize (bool): Option to standardize EWMAC by its price volatility. True to standardize, False otherwise 
        vol_lookback_period (int): The number of periods to lookback to calculate the EWMA standard deviation (eg. 25 would be 25 days for daily prices)
    Returns:
        pd.DataFrame: the raw calculation of ewmac
    """
    if ewma_cal_method == "span":
        ewma_fast = calc_ewma(hist_prices_df, span = ewma_fast_parameter)
        ewma_slow = calc_ewma(hist_prices_df, span = ewma_slow_parameter)
    elif ewma_cal_method == "deg_of_mixing":
        ewma_fast = calc_ewma(hist_prices_df, deg_of_mixing = ewma_fast_parameter)
        ewma_slow = calc_ewma(hist_prices_df, deg_of_mixing = ewma_slow_parameter)
    else:
        raise Exception("ewma_cal_method must be 'span' or 'deg_of_mixing'")
    
    raw_ewmac = ewma_fast - ewma_slow 

    if vol_standardize:
        price_returns_ewma_vol = calc_ewma_vol(hist_prices_df, vol_lookback_period = vol_lookback_period)
        return raw_ewmac / price_returns_ewma_vol
    else:
        return raw_ewmac

def calc_covariance_matrix(hist_returns_df, cov_estimate = "ledoit_wolf"): 
    """
    
    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
        cov_estimate (str): Method used to estimate covariance. Either 'ledoit_wolf' or 'sample' for ledoit_wolf shrinkage estimator and sample covariance method respectively
    
    Returns:
        pd.DataFrame: covariance matrix
    Notes:
        See Shrinkage Estimator code from: https://github.com/WLM1ke/LedoitWolf
    
    """

    ret_df = hist_returns_df.dropna()
    
    if (cov_estimate == "ledoit_wolf"):
        t, n = ret_df.shape
        ticker_list = list(hist_returns_df.columns) 
        returns = ret_df.dropna().to_numpy()
        mean_returns = np.mean(returns, axis=0, keepdims=True)
        returns -= mean_returns
        sample_cov = returns.transpose() @ returns / t

        # sample average correlation
        var = np.diag(sample_cov).reshape(-1, 1)
        sqrt_var = var ** 0.5
        unit_cor_var = sqrt_var * sqrt_var.transpose()
        average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
        prior = average_cor * unit_cor_var
        np.fill_diagonal(prior, var)

        # pi-hat
        y = returns ** 2
        phi_mat = (y.transpose() @ y) / t - sample_cov ** 2
        phi = phi_mat.sum()

        # rho-hat
        theta_mat = ((returns ** 3).transpose() @ returns) / t - var * sample_cov
        np.fill_diagonal(theta_mat, 0)
        rho = (
            np.diag(phi_mat).sum()
            + average_cor * (1 / sqrt_var @ sqrt_var.transpose() * theta_mat).sum()
        )

        # gamma-hat
        gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

        # shrinkage constant
        kappa = (phi - rho) / gamma
        shrink = max(0, min(1, kappa / t))

        # estimator
        sigma = shrink * prior + (1 - shrink) * sample_cov
        return pd.DataFrame(sigma, columns = ticker_list, index = ticker_list) 
    return ret_df.cov()
