import pandas as pd

def plot_time_series(time_series_list, start_date = None, end_date = None):
    data_to_plot = pd.concat(time_series_list, axis = 1)
    data_to_plot.plot()

def get_amount_to_buy(opt_weights_avg_scaled, activity_logs): 
    """
    Function takes in optimal scaled weights (given initial capital) and the activity logs, and displayes the most recent transaction that should be made 

    Parameters:
        opt_weights_avg_scaled (pd.DataFrame): optimized weights scaled by initial capital 
        activity_logs (pd.DataFrame): 
    
    Returns:
        pd.DataFrame: action for the latest trading day, including the ticker, price reference, action (Buy/Sell), Value to hold and units to buy
    """
    
    latest_log = activity_logs.iloc[-1].dropna().to_frame().T
    # drop returns columns
    latest_log = latest_log.loc[:, ~latest_log.columns.str.contains("_ret")]
    
    # Get list of actions
    list_of_cols_action = list(latest_log.loc[:,latest_log.columns.str.contains("Action")].columns)
    
    # Get tickers 
    list_of_cols = list(latest_log.loc[:, ~latest_log.columns.str.contains('Action')].columns)
    
    # reshape 
    t = latest_log.melt(value_vars = list_of_cols, var_name = "Ticker")
    t['Action'] = latest_log.loc[:, latest_log.columns.str.contains('Action')].T.values
    
    # Set up dataframe
    val_to_hold_df = opt_weights_avg_scaled.to_frame().reset_index().rename(columns = {'index': "Ticker", 0 : "Value To Hold"})
    
    # merge dataframes 
    t = t.merge(val_to_hold_df, how = "inner", on = ["Ticker"])
    t['Units To Buy'] = t['Value To Hold'].div(t['value'])
    t['Units To Buy'] = t['Units To Buy'].astype(float).round(2)
    return t