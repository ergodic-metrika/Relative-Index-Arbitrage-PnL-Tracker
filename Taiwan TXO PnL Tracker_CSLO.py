# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:37:15 2023

@author: sigma
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#Disclaimer: The relative arbitrage strategy was not fully implemented until October, 2022.
#Prior to October, 2022, it was a mixture of mostly put spread and a few ITM call as well as futures
#for quick delta adjustment.
#Since then, this relative arbitrage strategy has been consistenly implemented.


# Load the Excel file
excel_file = pd.ExcelFile('E:\Derivatives Trading\TAIEX derivatives trading record.xlsx')

# Get the sheet you want to read
sheet_name = 'ForPython' # Replace with the name of the sheet you want to read
df = excel_file.parse(sheet_name)

# Output data information
print(df.head())

#******Plotting setup*****#
# Generate some data
Date = df["Date"]
Date
y1 =df["PnL Index"]
y1
y2 = df["TAIEX"]
y2


# Get the maximum PnL value
max_pnl = df['PnL Index'].max()
max_pnl_date = df.loc[df['PnL Index']==max_pnl, 'Date'].values[0]


# Create the plot and set the first y-axis (left)
fig, ax1 = plt.subplots()
plt.xticks(rotation=90)
ax1.plot(Date, y1, 'b-')
ax1.scatter(max_pnl_date, max_pnl, color='red', marker='*')
ax1.set_xlabel('Date')
ax1.set_ylabel('PnL Index (Base = 100)', color='b')
ax1.tick_params('y', colors='b')

# Set the second y-axis (right)
ax2 = ax1.twinx()
ax2.plot(Date, y2, color='deepskyblue', marker=',')
ax2.set_ylabel('TAIEX', color='deepskyblue')
ax2.tick_params('y', colors='deepskyblue')

# Add message box
msg = "Red * : Highest PnL"
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax1.text(0.05, 0.95, msg, transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

# Show the plot
plt.title('PnL vs TAIEX')
plt.show()

#Pnl vs VIX
y3 = df["VIX"]
y3

# Create the plot and set the first y-axis (left)
fig, ax1 = plt.subplots()
plt.xticks(rotation=90)
ax1.plot(Date, y1, 'b-')
ax1.scatter(max_pnl_date, max_pnl, color='red', marker='*')
ax1.set_xlabel('Date')
ax1.set_ylabel('PnL Index (Base = 100)', color='b')
ax1.tick_params('y', colors='b')

# Set the second y-axis (right)
ax3 = ax1.twinx()
ax3.plot(Date, y3, 'fuchsia', marker='1', linestyle='-.')
ax3.set_ylabel('VIX', color='fuchsia')
ax3.tick_params('y', colors='fuchsia')


# Show the plot
plt.title('PnL vs VIX')
plt.show()

#Tracking error between PnL and TAIEX
PNL_returns = df['PnL Index'].pct_change()
TAIEX_returns = df['TAIEX'].pct_change()
diff_returns = PNL_returns  - TAIEX_returns 
tracking_error = diff_returns.std()


roll_te = diff_returns.rolling(20).std()

plt.figure(figsize=(10, 6))
plt.title('Rolling 20-Day Tracking Error btw PnL and TAIEX')
plt.plot(df['Date'], roll_te, color='mediumslateblue')
plt.xlabel('Date')
plt.ylabel('Tracking Error')
plt.show()

#Comment
#Apparently, when market is in turmoil, tracking error  will be widen, and vice versa.
#Due to the fact that my derivatives position is well hedged against the market shock


#Historical volatility
#GARCH model volatility

from arch import arch_model
from scipy.stats import mstats

# Calculate log returns
log_returns = np.log(y2/y2.shift(1)) 

# Remove NaN values  
log_returns = log_returns.dropna()  

# Winsorize outliers
log_returns = mstats.winsorize(log_returns, limits=0.1)

# Fit GARCH model
garch = arch_model(log_returns, p=1, q=1, dist='StudentsT')
garch_fit = garch.fit(update_freq=10)

# Extract volatility  
sigma = garch_fit.conditional_volatility
annual_vol = sigma.mean()*np.sqrt(250)*100

print(annual_vol)




#############################################Scientific experiment#################################
#Least Squares algo
from scipy.optimize import least_squares

# Set lower and upper bounds 
bounds =(10, 45)


# Objective function
def f(vix, PNL_returns , TAIEX_returns):
    diff =  (TAIEX_returns* annual_vol.std() )-(PNL_returns*vix.std()) 
    return diff.std()

# Set initial guess within bounds
x0 = [15.0] 

# By using Trust Region Reflective (bounded)
result1 = least_squares(f, x0,  bounds=bounds, method='trf', args=(TAIEX_returns, PNL_returns))
optimal_vix = result1.x[0]

print("Optimal VIX:", optimal_vix) 
print("Minimum Tracking Error:", f(optimal_vix, TAIEX_returns, PNL_returns))



# By using Levenberg-Marquardt algo (unbounded)
result2 = least_squares(f, x0, method='lm', args=(TAIEX_returns, PNL_returns))
optimal_vix = result2.x[0]

print("Optimal VIX:", optimal_vix) 
print("Minimum Tracking Error:", f(optimal_vix, TAIEX_returns, PNL_returns))


#Source: https://github.com/scipy/scipy/blob/v1.9.1/scipy/optimize/_lsq/least_squares.py

#* 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
            #  Doesn't handle bounds and sparse Jacobians. Usually the most
           #   efficient method for small unconstrained problems.
#* 'trf' : Trust Region Reflective algorithm, particularly suitable
            #  for large sparse problems with bounds. Generally robust method.

##########################################Performance###############################################
#Sharpe ratio
# Read in the portfolio returns data from a CSV file
R_first=df["PnL Index"].iloc[0,]
R_first
R_last=df["PnL Index"].iloc[273,]  #Always excel's actual row-2
R_last


portfolio_returns=(R_last-R_first)/R_first
portfolio_returns


daily_returns=df["Returns"]
daily_returns


# Max Drawdown Calculation for PnL Index
cumulative_returns = (1 + df["Returns"]).cumprod()
cumulative_max = cumulative_returns.cummax()
drawdown = (cumulative_returns / cumulative_max) - 1
max_drawdown = drawdown.min()

print("Max Drawdown:", max_drawdown)


# Calculate the excess returns and standard deviation
risk_free_rate = 0.0159  # Taiwan savings rate
excess_returns = portfolio_returns - risk_free_rate
std_dev = np.std(daily_returns)
print("Standard Deviation of Daily Return:", std_dev)



# Calculate the Sharpe ratio
Sharpe_Ratio = excess_returns / std_dev
print("Sharpe Ratio:", Sharpe_Ratio)


#Annualized Sharpe ratio
risk_free_rate_daily = (1 + risk_free_rate) ** (1/250) - 1
risk_free_rate_daily 
average_daily_returns = daily_returns.sum()/250
average_daily_returns
excess_daily_return=average_daily_returns-risk_free_rate_daily 
excess_daily_return

Annualized_Sharpe_Ratio=excess_daily_return/std_dev*np.sqrt(250)
print("Annualized Sharpe Ratio:", Annualized_Sharpe_Ratio)

# Calculate the Profit Factor
positive_returns = daily_returns[daily_returns > 0].sum()
negative_returns = daily_returns[daily_returns < 0].sum()

# Avoid division by zero
if negative_returns != 0:
    profit_factor = abs(positive_returns / negative_returns)
else:
    profit_factor = float('inf')

print("Profit Factor:", profit_factor)


#Portfolio ALpha
# Compute the mean returns
mean_PNL = PNL_returns.mean()
mean_TAIEX = TAIEX_returns.mean()

# Compute beta
covariance = PNL_returns.cov(TAIEX_returns)
variance = TAIEX_returns.var()
beta = covariance / variance
beta

# Compute alpha (assuming risk-free rate is 0)
alpha = (mean_PNL - (risk_free_rate_daily  +beta * mean_TAIEX))*np.sqrt(250)

# Print alpha
print("Alpha: ", alpha)

