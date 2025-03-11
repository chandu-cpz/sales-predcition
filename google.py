import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from io import StringIO # for demonstration with string data, replace with file read in real use
import warnings
warnings.filterwarnings('ignore') 
# --- Data Loading and Preprocessing ---
def preprocess_data(data):
    """
    Preprocesses the sales data to daily aggregated quantities per StockCode.

    Args:
        data: Raw sales data (string or pandas DataFrame).

    Returns:
        pandas DataFrame: Daily sales data.
    """
    if isinstance(data, str): # if data is string, read from string (for demonstration)
        df = pd.read_csv(StringIO(data)) # in real use, read from file: pd.read_csv('your_file.csv')
    elif isinstance(data, pd.DataFrame): # if data is already a DataFrame
        df = data.copy()
    else:
        raise ValueError("Input data must be a string or pandas DataFrame")

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    daily_sales = df.groupby(['StockCode', 'Date'])['Quantity'].sum().reset_index()
    return daily_sales

# --- ARIMA Model ---
def train_arima_model(timeseries_data):
    """
    Trains an ARIMA model on the given time series data.

    Args:
        timeseries_data (pandas Series): Time series data for a single StockCode (indexed by date).

    Returns:
        statsmodels.tsa.arima.model.ARIMAResults: Trained ARIMA model.
    """
    try: # Use try-except to handle potential ARIMA fitting issues
        model = ARIMA(timeseries_data, order=(5,1,0)) # You might need to tune the order (p, d, q)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"ARIMA model fitting failed: {e}")
        return None


def predict_arima(model_fit, steps=1):
    """
    Makes ARIMA predictions.

    Args:
        model_fit (statsmodels.tsa.arima.model.ARIMAResults): Trained ARIMA model.
        steps (int): Number of steps to forecast into the future.

    Returns:
        numpy.ndarray: ARIMA predictions.
    """
    if model_fit:
        predictions = model_fit.forecast(steps=steps)
        return predictions
    else:
        return None


# --- XGBoost Model ---
def create_lagged_features(timeseries_data, lags=3):
    """
    Creates lagged features for time series data.

    Args:
        timeseries_data (pandas Series): Time series data.
        lags (int): Number of lag periods to create.

    Returns:
        pandas DataFrame: DataFrame with lagged features.
    """
    df = pd.DataFrame(timeseries_data)
    df['Quantity'] = df['Quantity'].astype(float) # Ensure float type for shift operation
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Quantity'].shift(i) # Assuming 'Quantity' is the column name
    df.dropna(inplace=True) # Drop rows with NaN due to shifting
    return df

def train_xgboost_model(feature_data, target_data):
    """
    Trains an XGBoost model on the given feature and target data.

    Args:
        feature_data (pandas DataFrame): Features for training.
        target_data (pandas Series): Target variable.

    Returns:
        xgboost.sklearn.XGBRegressor: Trained XGBoost model.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror',  # Regression objective
                             n_estimators=100, # You can tune hyperparameters
                             seed=42)
    model.fit(feature_data, target_data)
    return model

def predict_xgboost(model, last_lags):
    """
    Makes XGBoost predictions.

    Args:
        model (xgboost.sklearn.XGBRegressor): Trained XGBoost model.
        last_lags (numpy.ndarray): Array of last lagged values for prediction.

    Returns:
        numpy.ndarray: XGBoost predictions.
    """
    return model.predict(last_lags.reshape(1, -1)) # Reshape for single prediction


# --- Main Execution ---
if __name__ == "__main__":
    data = pd.read_csv("online_retail_II.csv")

    daily_sales_data = preprocess_data(data)


    stock_codes = daily_sales_data['StockCode'].unique() # Get unique stock codes
    print(f"Unique Stock Codes: {stock_codes}")

    results = {} # Store results for comparison

    for stock_code in stock_codes:
        print(f"\n--- Processing StockCode: {stock_code} ---")
        stock_data = daily_sales_data[daily_sales_data['StockCode'] == stock_code].copy()
        if stock_data.empty:
            print(f"No data found for StockCode: {stock_code}")
            continue # Skip to the next stock code

        stock_data['Date'] = pd.to_datetime(stock_data['Date']) # Ensure Date is datetime for indexing
        stock_data.set_index('Date', inplace=True)
        stock_timeseries = stock_data['Quantity'] # Extract time series

        if len(stock_timeseries) < 2: # Need at least 2 data points to train models
            print(f"Insufficient data points for StockCode: {stock_code} for model training.")
            continue


        # --- ARIMA Model ---
        print("Training ARIMA Model...")
        arima_model = train_arima_model(stock_timeseries)
        if arima_model:
            arima_prediction = predict_arima(arima_model, steps=1) # Predict next day
            print(f"ARIMA Prediction (Next Day): {arima_prediction}")
            results.setdefault(stock_code, {})['arima_prediction'] = arima_prediction.iloc[0] # Store prediction


        # --- XGBoost Model ---
        print("Training XGBoost Model...")
        lagged_data = create_lagged_features(stock_timeseries, lags=3) # Create lagged features (using 3 days lag)

        print(f"Shape of lagged_data for StockCode {stock_code}: {lagged_data.shape}") # Debugging print
        print(f"First few rows of lagged_data:\n{lagged_data.head()}") # Debugging print


        if lagged_data.shape[0] < 5: # Check if enough data after lagging (adjust 5 as needed)
            print(f"Insufficient data for XGBoost training for StockCode: {stock_code} after creating lagged features.")
            continue # Skip XGBoost for this stock code

        X = lagged_data[[col for col in lagged_data.columns if 'lag' in col]] # Features are lag columns
        y = lagged_data['Quantity'] # Target is 'Quantity'

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Time series split, no shuffle


        if len(X_train) > 0: # Ensure there's training data
            xgboost_model = train_xgboost_model(X_train, y_train)
            last_lags = X.iloc[-1].values # Use last available lags for prediction
            xgboost_prediction = predict_xgboost(xgboost_model, last_lags)
            print(f"XGBoost Prediction (Next Day): {xgboost_prediction}")
            results.setdefault(stock_code, {})['xgboost_prediction'] = xgboost_prediction[0] # Store prediction
        else:
            print(f"Insufficient training data for XGBoost for StockCode: {stock_code} even after splitting.")


    print("\n--- Prediction Results ---")
    for code, preds in results.items():
        print(f"StockCode: {code}")
        if 'arima_prediction' in preds:
            print(f"  ARIMA Predicted Stock Required: {preds['arima_prediction']:.2f}") # Format to 2 decimal places
        if 'xgboost_prediction' in preds:
            print(f"  XGBoost Predicted Stock Required: {preds['xgboost_prediction']:.2f}")
