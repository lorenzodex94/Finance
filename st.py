import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import ollama
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Get today's date
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Title of the Streamlit app
st.title(" AutoDex :telescope: - Oraculus  :crystal_ball:")
st.header("Timehistory", divider=True)
# Stock selection (user can choose the stock)
stock_symbol = st.selectbox(
    "Select a stock symbol",
    ('RACE.MI','LDO.MI','GOOGL', 'AAPL', 'MSFT', 'META', 'NVDA', 'SPY', 'TSLA', 'AMZN','VUSA.L','VUAA.L')  # You can add more symbols if needed
)


# Function for fetching and cleaning stock data
def get_clean_financial_data(stock_symbol, start_date, end_date):
    # Download data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Clean structure
    data.columns = data.columns.get_level_values(0)

    # Handle missing values
    data = data.ffill()

    # Standardize timezone
    data.index = data.index.tz_localize(None)

    return data

# Fetch historical stock data for DIA (Dow Jones Industrial Average ETF)
data = get_clean_financial_data(stock_symbol, '2020-01-01', yesterday)

# Use the 'Close' price as the target variable
data = data.reset_index()
data['Date_Ordinal'] = pd.to_numeric(data['Date'].map(pd.Timestamp.toordinal))

# Prepare features and target variable
X = data[['Date_Ordinal']].values
y = data['Close'].values

# Fit a Gaussian Mixture Model (GMM) to the data
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict the latent values using the GMM
latent_features = gmm.predict_proba(X)

# Combine latent features with original features
X_latent = np.hstack([X, latent_features])

# Fit a polynomial regression model on the combined features
poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_reg.fit(X_latent, y)

# Predict and evaluate the model
y_pred = poly_reg.predict(X_latent)
mse = mean_squared_error(y, y_pred)

# Calculate the residuals and their standard deviation
residuals = y - y_pred
std_dev = np.std(residuals)

# Create upper and lower standard deviation lines
upper_bound = y_pred + 2 * std_dev
lower_bound = y_pred - 2 * std_dev

# Create buy and sell signals
data['Buy_Signal'] = np.where(y < lower_bound, 1, 0)   # Buy when price is below lower bound
data['Sell_Signal'] = np.where(y > upper_bound, 1, 0)  # Sell when price is above upper bound

# Plotting
plt.figure(figsize=(12, 6))
plt.title(f'Polynomial Regression {stock_symbol} Data with Buy and Sell Signals - update {yesterday}')

# Plot price data
plt.plot(data['Date'], y, color='blue', label='Actual Closing Price')
plt.plot(data['Date'], y_pred, color='red', linestyle='--', label='Fitted Values')
plt.plot(data['Date'], upper_bound, color='green', linestyle=':', label='Upper Bound (±2 Std Dev)')
plt.plot(data['Date'], lower_bound, color='green', linestyle=':', label='Lower Bound (±2 Std Dev)')
plt.fill_between(data['Date'], lower_bound, upper_bound, color='green', alpha=0.1)

# Plot Buy Signals
buy_signals = data[data['Buy_Signal'] == 1]
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='magenta', label='Buy Signal', s=100)

# Plot Sell Signals
sell_signals = data[data['Sell_Signal'] == 1]
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='orange', label='Sell Signal', s=100)

plt.ylabel('Close Price')
plt.xlabel('Date')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
st.pyplot(plt)

######################## Prophet
st.header("Prophet Predictions", divider=True)

days_to_predict = st.slider("Giorni da Prevedere", 0, 360, 30)

df = data.reset_index()[['Date', 'Close']] 
df.columns = ['ds', 'y']  
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d') # Format date

# Create and fit the Prophet model
model = Prophet()
model.fit(df) 

# Create future dates for prediction
extended_time = model.make_future_dataframe(periods=days_to_predict)  # Predict for the next 30 days
future = extended_time
# Make predictions
forecast = model.predict(future)

# Extract last year of data
plt.figure(figsize=(12, 6))
plt.plot(forecast['ds'][-days_to_predict:], forecast['yhat'][-days_to_predict:], label=f'Predicted (Next {days_to_predict} Days)', linestyle='--')
plt.plot(forecast['ds'][:-days_to_predict], forecast['yhat'][:-days_to_predict], label='Modelled')
plt.plot(df['ds'], df['y'], label='Actual')
plt.legend()
plt.grid(True)
plt.title(f'Prophet Model - Actual vs Predicted for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45, ha='right')  
plt.show()

st.pyplot(plt)

#####################################################

st.header("S&P 500 HOT Stocks", divider=True)

# Function to perform analysis on multiple indices
def analyze_indices(tickers, start_date, end_date):
    signals = {}  # Initialize the dictionary
    for ticker in tickers:
        print(f'Analyzing {ticker}...')
        data = get_clean_financial_data(ticker, start_date, end_date)
        data = data.reset_index()
        data['Date_Ordinal'] = pd.to_numeric(data['Date'].map(pd.Timestamp.toordinal))

        # Prepare features and target variable
        X = data[['Date_Ordinal']].values
        y = data['Close'].values

        # Fit a Gaussian Mixture Model (GMM) to the data
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(X)

        # Predict the latent values using the GMM
        latent_features = gmm.predict_proba(X)

        # Combine latent features with original features
        X_latent = np.hstack([X, latent_features])

        # Fit a polynomial regression model on the combined features
        poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly_reg.fit(X_latent, y)

        # Predict and evaluate the model
        y_pred = poly_reg.predict(X_latent)
        mse = mean_squared_error(y, y_pred)

        # Calculate the residuals and their standard deviation
        residuals = y - y_pred
        std_dev = np.std(residuals)

        # Create upper and lower standard deviation lines
        upper_bound = y_pred + 2 * std_dev
        lower_bound = y_pred - 2 * std_dev

        # Create buy and sell signals
        data['Buy_Signal'] = np.where(y < lower_bound, 1, 0)
        data['Sell_Signal'] = np.where(y > upper_bound, 1, 0)

      # Check for signals before plotting
        if data['Buy_Signal'].iloc[-1] or data['Sell_Signal'].iloc[-1]:


        # Plotting
          plt.figure(figsize=(12, 6))
          plt.title(f'Polynomial Regression on {ticker} Data with Buy and Sell Signals - update {yesterday}')

        # Plot price data
          plt.plot(data['Date'], y, color='blue', label='Actual Closing Price')
          plt.plot(data['Date'], y_pred, color='red', linestyle='--', label='Fitted Values')
          plt.plot(data['Date'], upper_bound, color='green', linestyle=':', label='Upper Bound (±2 Std Dev)')
          plt.plot(data['Date'], lower_bound, color='green', linestyle=':', label='Lower Bound (±2 Std Dev)')
          plt.fill_between(data['Date'], lower_bound, upper_bound, color='green', alpha=0.1)

        # Plot Buy Signals
          buy_signals = data[data['Buy_Signal'] == 1]
          plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='magenta', label='Buy Signal', s=100)

        # Plot Sell Signals
          sell_signals = data[data['Sell_Signal'] == 1]
          plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='orange', label='Sell Signal', s=100)

          plt.ylabel('Close Price')
          plt.xlabel('Date')
          plt.xticks(rotation=0)
          plt.legend()
          plt.tight_layout()
          plt.grid(True)
          st.pyplot(plt)

        else:
          print("")

# Store the latest buy/sell signal
        signals[ticker] = {
            'buy': data['Buy_Signal'].iloc[-1],
            'sell': data['Sell_Signal'].iloc[-1]
        }

    return signals  # Return the dictionary

# Define a list of tickers to analyze
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JNJ', 'JPM', 'V',
    'PG', 'NVDA', 'UNH', 'HD', 'DIS', 'PYPL', 'VZ', 'NFLX', 'INTC', 'CMCSA',
    'PEP', 'T', 'CSCO', 'MRK', 'NKE', 'XOM', 'PFE', 'ABT', 'CVX', 'TMO', 'LLY',
    'MDT', 'IBM', 'QCOM', 'AVGO', 'TXN', 'AMGN', 'COST', 'NOW', 'SBUX', 'LMT',
    'HON', 'BA', 'CAT', 'GS', 'BLK', 'SYK', 'CHTR', 'AMT', 'ISRG', 'ADBE',
    'MDLZ', 'TGT', 'SPGI', 'DHR', 'LRCX', 'NEM', 'GILD', 'FISV',
    'FIS', 'ZTS', 'TROW', 'KMB', 'SYY', 'APD', 'C', 'NDAQ', 'MS', 'USB',
    'BKNG', 'ADP', 'LNT', 'DTE', 'ETR', 'DOV', 'NTRS', 'CARR', 'WBA',
    'KHC', 'MCO', 'VTRS', 'VFC', 'GWW', 'HIG', 'HWM', 'ICE', 'IP',
    'JCI', 'KMI', 'MSI', 'NWL', 'PGR', 'PH', 'PKG', 'RMD', 'SRE',
    'WAB', 'WDC', 'WMT', 'WST', 'XYL', 'ZBRA','RACE.MI','SPY','VUSA.L','VUAA.L','DIA','LDO.MI'
]
# Call the function to analyze the indices
#analyze_indices(tickers, '2020-01-01', yesterday)

# Call the function and store the signals
signals_dict = analyze_indices(tickers, '2020-01-01', yesterday)


# Print the signals for each index
#for ticker, signal in signals_dict.items():
#    print(f"{ticker}: Buy - {signal['buy']}, Sell - {signal['sell']}")

signals_df = pd.DataFrame(signals_dict).T  # Transpose to make tickers rows
filtered_df = signals_df[(signals_df['buy'] == 1) | (signals_df['sell'] == 1)]

print(filtered_df)
st.table(filtered_df)


##################

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import contextlib
import io

# List of top 50 stocks by market cap (you can modify this list as needed)
top_50_stocks = tickers

# Create lists to store results for daily, weekly, and monthly returns
daily_results = []
weekly_results = []
monthly_results = []

# Loop through each stock to fetch data and calculate probabilities
for stock in top_50_stocks:
    # Suppress the output of yfinance download
    with contextlib.redirect_stdout(io.StringIO()):
        # Fetch last 5 years of stock data
        data = yf.download(stock, period='5y', interval='1d')  # Fetch daily data for better granularity
    
    # Calculate daily returns and add it to the DataFrame
    data['Daily Return'] = data['Adj Close'].pct_change()  # Calculate daily returns

    # Drop the first row (which contains NA for daily return) instead of using dropna on the entire column
    daily_data = data[1:] 
    
    # Calculate mean and standard deviation of daily returns
    mean_daily_return = daily_data['Daily Return'].mean()
    std_daily_dev = daily_data['Daily Return'].std()

    # Calculate the probability that daily return is greater than 0%
    prob_daily_greater_than_0 = 1 - norm.cdf(0, mean_daily_return, std_daily_dev)

    # Append the daily results
    daily_results.append({
        'Stock': stock,
        'Mean Daily Return': mean_daily_return,
        'Standard Deviation (Daily)': std_daily_dev,
        'Probability (Daily Return > 0%)': prob_daily_greater_than_0
    })

    # Calculate weekly returns
    data['Weekly Return'] = data['Adj Close'].pct_change(periods=5)  # Calculate weekly returns (5 trading days)

    # Drop the first 5 rows (which contain NA for weekly return) instead of using dropna
    weekly_data = data[5:]  


    # Calculate mean and standard deviation of weekly returns
    mean_weekly_return = weekly_data['Weekly Return'].mean()
    std_weekly_dev = weekly_data['Weekly Return'].std()

    # Calculate the probability that weekly return is greater than 0%
    prob_weekly_greater_than_0 = 1 - norm.cdf(0, mean_weekly_return, std_weekly_dev)

    # Append the weekly results
    weekly_results.append({
        'Stock': stock,
        'Mean Weekly Return': mean_weekly_return,
        'Standard Deviation (Weekly)': std_weekly_dev,
        'Probability (Weekly Return > 0%)': prob_weekly_greater_than_0
    })

    # Calculate monthly returns
    data['Monthly Return'] = data['Adj Close'].pct_change(periods=22)  # Calculate monthly returns (approx. 22 trading days)

    # Drop NA values for monthly data
    monthly_data = data[22:]

    # Calculate mean and standard deviation of monthly returns
    mean_monthly_return = monthly_data['Monthly Return'].mean()
    std_monthly_dev = monthly_data['Monthly Return'].std()

    # Calculate the probability that monthly return is greater than 0%
    prob_monthly_greater_than_0 = 1 - norm.cdf(0, mean_monthly_return, std_monthly_dev)

    # Append the monthly results
    monthly_results.append({
        'Stock': stock,
        'Mean Monthly Return': mean_monthly_return,
        'Standard Deviation (Monthly)': std_monthly_dev,
        'Probability (Monthly Return > 0%)': prob_monthly_greater_than_0
    })

# Create DataFrames from the results
daily_results_df = pd.DataFrame(daily_results)
weekly_results_df = pd.DataFrame(weekly_results)
monthly_results_df = pd.DataFrame(monthly_results)

# Sort by probability of return > 0% for daily returns
daily_results_df = daily_results_df.sort_values(by='Probability (Daily Return > 0%)', ascending=False)

# Sort by probability of return > 0% for weekly returns
weekly_results_df = weekly_results_df.sort_values(by='Probability (Weekly Return > 0%)', ascending=False)

# Sort by probability of return > 0% for monthly returns
monthly_results_df = monthly_results_df.sort_values(by='Probability (Monthly Return > 0%)', ascending=False)

# Print title before displaying the results
print("Top 10 Stocks with Probability of Daily Return Greater Than 0%:\n")
# Format output for daily results
d1 = daily_results_df[['Stock', 'Mean Daily Return', 'Standard Deviation (Daily)', 'Probability (Daily Return > 0%)']].head(10).to_string(index=False, float_format='%.6f')
d1_df = pd.DataFrame(d1) 
print(daily_results_df[['Stock', 'Mean Daily Return', 'Standard Deviation (Daily)', 'Probability (Daily Return > 0%)']].head(10).to_string(index=False, float_format='%.6f'))
st.table(d1_df)

print("\nTop 10 Stocks with Probability of Weekly Return Greater Than 0%:\n")
# Format output for weekly results
d2 = weekly_results_df[['Stock', 'Mean Weekly Return', 'Standard Deviation (Weekly)', 'Probability (Weekly Return > 0%)']].head(10).to_string(index=False, float_format='%.6f')
print(weekly_results_df[['Stock', 'Mean Weekly Return', 'Standard Deviation (Weekly)', 'Probability (Weekly Return > 0%)']].head(10).to_string(index=False, float_format='%.6f'))
d2_df = pd.DataFrame(d2) 
st.table(d2_df)

print("\nTop 10 Stocks with Probability of Monthly Return Greater Than 0%:\n")
# Format output for monthly results
d3= monthly_results_df[['Stock', 'Mean Monthly Return', 'Standard Deviation (Monthly)', 'Probability (Monthly Return > 0%)']].head(10).to_string(index=False, float_format='%.6f')
print(monthly_results_df[['Stock', 'Mean Monthly Return', 'Standard Deviation (Monthly)', 'Probability (Monthly Return > 0%)']].head(10).to_string(index=False, float_format='%.6f'))
d3_df = pd.DataFrame(d3) 
st.table(d3_df)



