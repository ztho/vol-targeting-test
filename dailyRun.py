import pandas as pd 
import numpy as np 
import yfinance as yf
import importlib
import sys 

sys.path.insert(0, "../codebase/quantzt/")
import pull_prices
import compute_utils
import backtest
import forecasts as f
import optimisation as opt 
import graphing

ticker_list = ['NXPI', 'CHH', 'CRM', 'PLTR', 'TENB', 'HUBS', 'VRNS', 'AXNX', 'IRTC','TMCI']

# Draw Prices 
prices_df = pull_prices.get_multiple_hist_prices(ticker_list, target_col = "Open")
prices_df = prices_df.dropna(how = "any") 

# Get Benchmark returns 
bench_df = pd.read_csv("TB3MS.csv")
bench_df = bench_df.set_index(["DATE"])
bench_df.rename(columns = {"TB3MS": "rf_rate"}, inplace = True)
bench_df = bench_df.div(100)

bench_df = pull_prices.fill_monthly_rf_rates(bench_df, prices_df, column_name = "rf_rate") 

# Get forecast values 
forecasts_var1 = f.get_ewmac_forecasts(prices_df)
forecasts_var2 = f.get_ewmac_forecasts(prices_df, ewma_fast_parameter = 2, ewma_slow_parameter = 8)

#Note: Getting the weight vector shall be a manual process
weighted_forecast = backtest.get_weighted_forecasts([.5,.5], [forecasts_var1, forecasts_var2])
final_forecast = backtest.get_scaled_forecasts(weighted_forecast, 1.15)

# Get bootstrapped weights 
opt_weights_avg = backtest.get_averaged_optimized_weights(hist_prices_df = prices_df, 
                                                         bench_df = bench_df, 
                                                         starting_observation_period = 3, 
                                                         observation_rate = 1, 
                                                         observation_freq = "m",
                                                         method = "sharpe",
                                                         ret_computation_method = "statmean",
                                                         cov_estimate = 'ledoit_wolf', 
                                                         weight_bounds = (0,1), 
                                                         opt_backup = 'min_vol', 
                                                         show_weights = False)

# Get scaled values 
opt_weights_avg_sc = opt_weights_avg * 5000

# Get value of each holding and the logs
out_weight_opt, out_weight_opt_logs = backtest.run_strat_backtest_multi(prices_df.dropna(), 
                                                   final_forecast, 
                                                   start_date = "2022-09-01", 
#                                                    end_date = "2021-05-30",
                                                   init_capitals = opt_weights_avg_sc,
                                                  display_logs = False, 
                                                    return_logs = True)
# print(out_weight_opt_logs)
print(out_weight_opt_logs.iloc[-1].dropna())
print(graphing.get_amount_to_buy(opt_weights_avg_sc, out_weight_opt_logs))
