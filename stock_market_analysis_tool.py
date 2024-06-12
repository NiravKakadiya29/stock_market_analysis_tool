import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkcalendar import DateEntry
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fetch historical data for a given stock
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Prepare the data for linear regression model
def prepare_data_linear(stock_data):
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_numeric(stock_data['Date'])
    X = stock_data[['Date']]
    y = stock_data['Close']
    return X, y

# Prepare the data for LSTM model
def prepare_data_lstm(stock_data, time_step=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Train the linear regression model and make predictions
def train_and_predict_linear(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Linear Regression Mean Squared Error: {mse}")
    return model, predictions, X_test, y_test

# Train the LSTM model and make predictions
def train_and_predict_lstm(X, y, scaler, epochs=100, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, predictions)
    print(f"LSTM Mean Squared Error: {mse}")
    return model, predictions, X_test, y_test

# Visualize the results
def visualize_results(y_test, predictions, ticker, model_type):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='black', label='Actual Prices')
    plt.plot(predictions, color='blue', linewidth=2, label='Predicted Prices')
    plt.title(f'{model_type} Stock Price Prediction for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# GUI to select stock and dates
class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Analysis Tool")
        self.create_widgets()

    def create_widgets(self):
        self.ticker_label = tk.Label(self.root, text="Stock Ticker Symbol:")
        self.ticker_label.grid(row=0, column=0, padx=10, pady=5)

        self.ticker_entry = ttk.Combobox(self.root, values=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
        self.ticker_entry.grid(row=0, column=1, padx=10, pady=5)

        self.start_date_label = tk.Label(self.root, text="Start Date:")
        self.start_date_label.grid(row=1, column=0, padx=10, pady=5)

        self.start_date_entry = DateEntry(self.root, date_pattern='yyyy-mm-dd')
        self.start_date_entry.grid(row=1, column=1, padx=10, pady=5)

        self.end_date_label = tk.Label(self.root, text="End Date:")
        self.end_date_label.grid(row=2, column=0, padx=10, pady=5)

        self.end_date_entry = DateEntry(self.root, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=2, column=1, padx=10, pady=5)

        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze_stock)
        self.analyze_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def analyze_stock(self):
        ticker = self.ticker_entry.get()
        start_date = self.start_date_entry.get_date().strftime('%Y-%m-%d')
        end_date = self.end_date_entry.get_date().strftime('%Y-%m-%d')

        if not ticker:
            messagebox.showerror("Input Error", "Please select a stock ticker.")
            return

        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        if stock_data.empty:
            messagebox.showerror("Data Error", "No data found for the given stock ticker and date range.")
            return

        # Linear Regression
        X_linear, y_linear = prepare_data_linear(stock_data)
        _, predictions_linear, _, y_test_linear = train_and_predict_linear(X_linear, y_linear)
        visualize_results(y_test_linear, predictions_linear, ticker, 'Linear Regression')
        
        # LSTM
        time_step = 60
        X_lstm, y_lstm, scaler = prepare_data_lstm(stock_data, time_step)
        _, predictions_lstm, _, y_test_lstm = train_and_predict_lstm(X_lstm, y_lstm, scaler)
        visualize_results(y_test_lstm, predictions_lstm, ticker, 'LSTM')

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()
