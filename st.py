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


###################################################################################################################################### Fred Normer Approach
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft, ifft
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fetching Reliance stock data from yfinance
ticker = stock_symbol
data = yf.download(ticker, start="2020-01-01", end=yesterday, interval="1d")
close_prices = data['Close'].values


# Function to apply Fourier Transform and return stable frequencies
def frednormer_transform(data, threshold=0.05):  # Lower threshold to preserve more details
    # Apply Fourier Transform
    freq_data = fft(data)

    # Compute amplitude
    amplitude = np.abs(freq_data)

    # Stability measure: Coefficient of Variation
    stable_freqs = np.where(amplitude > threshold * np.max(amplitude), freq_data, freq_data)  # Retain more frequencies

    # Return filtered data in time domain using Inverse FFT
    filtered_data = np.real(ifft(stable_freqs))

    return filtered_data

# Applying FredNormer on closing prices
filtered_prices = frednormer_transform(close_prices)

# Plotting the transformed data

plt.plot(data.index, close_prices)
plt.plot(data.index, filtered_prices)
plt.title(f'FredNormer Transformed Prices {ticker} - updated {today}')
plt.legend(['Original Prices', 'Filtered Prices'])
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

# Prepare data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(filtered_prices.reshape(-1, 1)).flatten()

# Create sequences for LSTM input
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 7  # 60 days sequence
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):  # Adjust hidden size and layers
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last LSTM cell
        return out

# Initialize model, loss function, and optimizer
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 150  # Increase epochs for better learning
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            test_loss += loss.item()

    test_losses.append(test_loss / len(test_loader))

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")


# Testing loop and metric evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch).squeeze()
        predictions.append(output.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Inverse scale the predictions and actuals
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

# Calculate MAE and MAPE
mae = mean_absolute_error(actuals, predictions)
mape = mean_absolute_percentage_error(actuals, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

dates = [yesterday - datetime.timedelta(days=i) for i in range(len(predictions))]

# Get the last sequence of data
last_sequence = scaled_data[-SEQ_LENGTH:]

# Make predictions for the next 30 days
future_predictions = []
for _ in range(30):
    # Reshape the input sequence for the model
    input_sequence = last_sequence.reshape(1, SEQ_LENGTH)  
    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).to(device)

    # Make the prediction
    prediction = model(input_sequence).cpu().detach().numpy()[0, 0]  

    # Append the prediction to the list of future predictions
    future_predictions.append(prediction)  

    # Update the input sequence for the next prediction
    last_sequence = np.append(last_sequence[1:], prediction)  

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Get future dates for plotting
future_dates = [today + datetime.timedelta(days=i) for i in range(1, 31)]




# Get yesterday's closing value
yesterday_closing_value = data['Close'].iloc[-1]
# Extract the closing price for 'RACE.MI'
closing_price = yesterday_closing_value[ticker]

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.axhline(y=data['Close'].iloc[-1].item(), color='red', linestyle='--', label=f" {yesterday}'s Price {closing_price:.2f}")
plt.plot(data.index, close_prices, label='Actual Prices')
plt.plot(future_dates, future_predictions, label='Predicted Prices (Next 30 Days)')
plt.title('FredNormer LSTM Stock Price Prediction - Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(True)
plt.legend()
st.pyplot(plt) 

st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape}")


##################################################################################
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



# Plot the results for RNN model
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_scaled, label="Actual Stock Price", color='blue')
plt.plot(test_dates, predicted_stock_price_rnn, label="Predicted Stock Price", color='red')
plt.plot(future_dates, future_predictions, label="Predicted Stock Price (RNN)", color='green', linestyle='--') 
plt.title(f'Stock Price Prediction with RNN for {stock_symbol}', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel(' Scaled Stock Price (USD)', fontsize=14)
plt.legend()
plt.grid(True)
st.pyplot(plt) 
