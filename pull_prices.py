import pandas as pd 
import numpy as np 
import yfinance as yf

def get_hist_prices(ticker, start_date = None, end_date = None):
    """
    Get Open, High, Low, Close, Adjusted Close and Volume of a ticker 

    Parameters:
        ticker (str): ticker code of a company 
        [Optional] start_date (str): the target start date of drawing data, in the form "YYYY-MM-DD"
        [Optional] end_date (str): the target end date of drawing data, in the form "YYYY-MM-DD"
    Returns:
        pd.DataFrame: df with date as index, Open, High, Low, Close, Adj Close and Volume as columns
    """
    df = yf.download(ticker, start = start_date, end = end_date)
    return df

def get_multiple_hist_prices(ticker_list, target_col = "Adj Close", start_date = None, end_date = None, merge_how = "outer"):
    """
    Concatenates multiple output from get_hist_prices(), limited to only 1 column 

    Parameters:
        ticker_list (list): a list of strings of ticker codes 
        target_col (str): the column from the output of get_hist_prices() to preserve and concatenate on 
        [Optional] start_date (str): the target start date of drawing data, in the form "YYYY-MM-DD"
        [Optional] end_date (str): the target end date of drawing data, in the form "YYYY-MM-DD"
        [Optional] merge_how (str): method of merging multiple outputs. See documentation from pd.DataFrame.join() 

    Returns:
        pd.DataFrame: date as index, ticker code as columns, and data from target_col
    """
    output_df = None
    for ticker in ticker_list:
        price_df = get_hist_prices(ticker, start_date, end_date)[[target_col]].copy()
        price_df.rename(columns = {target_col: ticker}, inplace = True)
        if isinstance(output_df, type(None)): 
            output_df = price_df.copy() 
            del price_df 
        else:
            output_df = output_df.join(price_df, how = merge_how)
    return output_df
            
def fill_monthly_rf_rates(bench_df, hist_prices_df, column_name = "rf_rate"):
    """
    If bench_df is in monthly, convert it to daily data by merging it with daily hist_prices, and forward filling the data 
    """
    # ensure bench_df index is first a datetime 
    bench_df = bench_df.copy() 
    bench_df.index = pd.to_datetime(bench_df.index)
    combined = bench_df.join(hist_prices_df, how = "outer")
    combined = combined[[column_name]].copy() 
    combined = combined.ffill()
    return combined