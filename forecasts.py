import numpy as np 
import pandas as pd
import references as refs 
import compute_utils as compute


def get_ewmac_forecasts(hist_prices_df,
                        ewma_cal_method = "span", 
                        ewma_fast_parameter = 16, 
                        ewma_slow_parameter = 64,  
                        vol_standardize = True, 
                        vol_lookback_period = 25, 
                        ewmac_forecast_scaler_dict = refs.ewmac_forecast_scaler):
    """
    Generates the forecast value time series 
    Parameters:
        hist_prices_df (pd.DataFrame): Dataframe indexed by date, with ticker codes as column headers and prices as rows. Output from pull_prices.get_multiple_hist_prices()
        ewma_cal_method (str): Either "span" or "deg_of_mixing". Method for calculating EWMA. 
        ewma_fast_parameter (int or float): the value of the parameter for "span" or "deg_of_mixing" to calculate the fast EWMA. 
        ewma_slow_parameter (int or float): the value of the parameter for "span" or "deg_of_mixing" to calculate the slow EWMA.
        vol_standardize (bool): Option to standardize EWMAC by its price volatility. True to standardize, False otherwise 
        vol_lookback_period (int): The number of periods to lookback to calculate the EWMA standard deviation (eg. 25 would be 25 days for daily prices)
        ewmac_forecast_scaler_dict (dict): dict containing key-value for forecast scalers
    Returns:
        pd.DataFrame: the scaled EWMAC forecasts
    """

    ewmac = compute.calc_ewmac(hist_prices_df, ewma_cal_method, ewma_fast_parameter, ewma_slow_parameter,  vol_standardize, vol_lookback_period)
    
    if str(ewma_fast_parameter) + "-" + str(ewma_slow_parameter) not in list(ewmac_forecast_scaler_dict.keys()): 
        # print(str(ewma_fast_parameter) + "-" + str(ewma_slow_parameter))
        # print(ewmac_forecast_scaler_dict.keys())
        print("Combination of fast and slow parameters not found dictionary. No scalar applied")
        forecast_scaler = 1.0 
    else:
        forecast_scaler = ewmac_forecast_scaler_dict[str(ewma_fast_parameter) + "-" + str(ewma_slow_parameter)]
    ewmac_forecast = ewmac.multiply(forecast_scaler)
    
    # Max Function 
    ewmac_forecast.where(ewmac_forecast < refs.forecast_max_min_parameters['max'], refs.forecast_max_min_parameters['max'], inplace = True) 
    # Min Function
    ewmac_forecast.where(ewmac_forecast > refs.forecast_max_min_parameters['min'], refs.forecast_max_min_parameters['min'], inplace = True)
    
    del ewmac 
    return ewmac_forecast

