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

# Get today's date
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Title of the Streamlit app
st.title(" DESSI - Stock Price Simulation")

# Stock selection (user can choose the stock)
stock_symbol = st.selectbox(
    "Select a stock symbol",
    ('RACE.MI','GOOGL', 'AAPL', 'MSFT', 'META', 'NVDA', 'SPY', 'TSLA', 'AMZN','XLC')  # You can add more symbols if needed
)

# Fetch historical data for the selected stock
googl_hist = yf.download(stock_symbol, start='2020-01-01', end=yesterday)

# Calculate daily returns
googl_hist['Return'] = googl_hist['Close'].pct_change().dropna()

# Estimate drift (annualized) and volatility (annualized)
returns = googl_hist['Return'].dropna()
mu = returns.mean() * 252  # Annualize the mean
sigma = returns.std() * (252 ** 0.5)  # Annualize the standard deviation

#################################################################################################################################################

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

# Create a second plot for the distribution of daily returns
plt.figure(figsize=(10, 5))
plt.hist(returns, bins=30, color='orange', alpha=0.7, edgecolor='black')
plt.title(f"Distribution of Daily Returns for {stock_symbol} - update {yesterday}")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.grid()
st.pyplot(plt)  # Display the second plot in Streamlit


#############################################################################################################################

def fetch_tesla_stock_data():
    """
    Fetch Tesla's historical stock data from Yahoo Finance.

    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices indexed by date.
    """
    # Fetch data for Tesla (TSLA) from Yahoo Finance
    ticker = stock_symbol
    start_date = "2020-01-01"
    end_date = yesterday
    tesla = yf.download(ticker, start=start_date, end=end_date)

    # Return a DataFrame with the adjusted close prices
    tesla_data = tesla[['Adj Close']].rename(columns={"Adj Close": "adjClose"})
    tesla_data.index.name = "date"
    return tesla_data

# Fetch Tesla stock data
tesla_data = fetch_tesla_stock_data()

# Display the first few rows of data
print(tesla_data.head(10))


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
# Define the window size and prediction time
window_size = 20
prediction_steps = 10
# Function to create sequences
def create_sequences(data, window_size, prediction_steps):
    # Indented block within the function
    X = []
    y = []
    for i in range(window_size, len(data) - prediction_steps):
        X.append(data[i-window_size:i, 0]) # input sequence
        y.append(data[i+prediction_steps-1, 0]) # target value (price at the next timestep)
    return np.array(X), np.array(y)
# Fetch Tesla stock data
data = tesla_data[['adjClose']].values
# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
# Create sequences for the model
X, y = create_sequences(scaled_data, window_size, prediction_steps)
# Reshape input data to be in the shape [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Assuming tesla_data contains your original DataFrame with a 'date' index
test_dates = tesla_data.index[len(tesla_data) - len(y_test):]


################################################################################ LSTM Model

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Attention, Add, LayerNormalization, Layer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Define a custom attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[2], input_shape[2]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        q = tf.matmul(inputs, self.W)
        a = tf.matmul(q, inputs, transpose_b=True)
        attention_weights = tf.nn.softmax(a, axis=-1)
        return tf.matmul(attention_weights, inputs)

# LSTM model with attention and early stopping
def build_lstm_model_with_attention(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Attention layer
    model.add(AttentionLayer())
    model.add(LayerNormalization())

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for prediction

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Build the LSTM model with attention
model = build_lstm_model_with_attention(X_train.shape[1:])

# Implement EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with EarlyStopping and 50 epochs
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Inverse scale the actual stock prices
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_scaled, predicted_stock_price)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Attention, Add, LayerNormalization, Layer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Define a custom attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[2], input_shape[2]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        q = tf.matmul(inputs, self.W)
        a = tf.matmul(q, inputs, transpose_b=True)
        attention_weights = tf.nn.softmax(a, axis=-1)
        return tf.matmul(attention_weights, inputs)

# LSTM model with attention and early stopping
def build_lstm_model_with_attention(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Attention layer
    model.add(AttentionLayer())
    model.add(LayerNormalization())

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for prediction

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Build the LSTM model with attention
model = build_lstm_model_with_attention(X_train.shape[1:])

# Implement EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with EarlyStopping and 50 epochs
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Inverse scale the actual stock prices
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test_scaled, predicted_stock_price)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

#future predictions

prediction_steps = 30
def create_sequences(data, window_size, prediction_steps):
    X = []
    y = []
    for i in range(window_size, len(data) - prediction_steps):
        X.append(data[i - window_size:i, 0])  # Input sequence
        y.append(data[i + prediction_steps - 1, 0])  # Target value (price 30 days ahead)
    return np.array(X), np.array(y)

last_window_data = scaled_data[-window_size:]
last_window_data = last_window_data.reshape(1, window_size, 1)

future_predictions = []
for i in range(30):  # Predict for 30 days
    prediction = model.predict(last_window_data)
    future_predictions.append(prediction[0, 0])
    last_window_data = np.append(last_window_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

import datetime
today = datetime.date.today()
future_dates = [today + datetime.timedelta(days=i) for i in range(1, 31)]  # 30 days

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_scaled, label="Actual Stock Price", color='blue')
plt.plot(future_dates, future_predictions, label="Predicted Stock Price (LSTM) - Next 30 Days", color='green', linestyle='--') # Future 30-day predictions
plt.plot(test_dates, predicted_stock_price, label="Predicted Stock Price (LSTM)", color='red')
plt.title('Stock Price Prediction with LSTM (Including 30-Day Prediction)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Scaled Stock Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
st.pyplot(plt)  # Display the  plot in Streamlit



############################# RNN


from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout

# Define the RNN model
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for prediction

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build the RNN model
rnn_model = build_rnn_model(X_train.shape[1:])

# Train the model
rnn_history = rnn_model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
predicted_stock_price_rnn = rnn_model.predict(X_test)
predicted_stock_price_rnn = scaler.inverse_transform(predicted_stock_price_rnn)

# Inverse scale the actual stock prices
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAPE for RNN
mape_rnn = mean_absolute_percentage_error(y_test_scaled, predicted_stock_price_rnn)
print(f"Mean Absolute Percentage Error (MAPE) for RNN: {mape_rnn:.2f}%")
# Plot the results for RNN model
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_scaled, label="Actual Tesla Stock Price", color='blue')
plt.plot(test_dates, predicted_stock_price_rnn, label="Predicted Tesla Stock Price", color='red')
plt.title('Tesla Stock Price Prediction with RNN', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel(' Scaled Stock Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
st.pyplot(plt) 


last_window_data = scaled_data[-window_size:]  # Get the last window_size values
last_window_data = last_window_data.reshape(1, window_size, 1)  # Reshape

future_predictions = []
for i in range(30):  
    prediction = rnn_model.predict(last_window_data) # Make prediction
    future_predictions.append(prediction[0, 0]) # Store prediction
    
    # Update the input sequence for the next prediction
    last_window_data = np.append(last_window_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
import datetime

today = datetime.date.today()
future_dates = [today + datetime.timedelta(days=i) for i in range(1, 31)]  # 30 days

plt.plot(tesla_data.index, tesla_data['adjClose'], label="Actual Tesla Stock Price")
plt.plot(future_dates, future_predictions, label="Predicted Tesla Stock Price (RNN)") 
plt.title('Tesla Stock Price Prediction with RNN (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt) 



