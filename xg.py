import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 
# --- Data Loading and Preprocessing ---
def preprocess_data(data):
    """
    Preprocesses the sales data to daily aggregated quantities per StockCode.
    Now includes People_Count and Time_Spent features.

    Args:
        data: Raw sales data (string or pandas DataFrame).

    Returns:
        pandas DataFrame: Daily sales data with additional features.
    """
    if isinstance(data, str):
        df = pd.read_csv(StringIO(data))
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input data must be a string or pandas DataFrame")

    # Parse datetime with error handling
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['InvoiceDate'])
    
    # Convert Quantity to numeric, replacing invalid values with NaN
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    # Remove rows with invalid quantities
    df = df.dropna(subset=['Quantity'])
    
    df['Date'] = df['InvoiceDate'].dt.date
    
    # Ensure People_Count and Time_Spent are numeric
    df['People_Count'] = pd.to_numeric(df['People_Count'], errors='coerce')
    df['Time_Spent'] = pd.to_numeric(df['Time_Spent'], errors='coerce')
    
    # Aggregate daily data including new features
    daily_sales = df.groupby(['StockCode', 'Date']).agg({
        'Quantity': 'sum',
        'People_Count': 'mean',  # Taking average people count per day
        'Time_Spent': 'mean'     # Taking average time spent per day
    }).reset_index()
    
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
    Creates lagged features for time series data including People_Count and Time_Spent.

    Args:
        timeseries_data (pandas DataFrame): Time series data with multiple features.
        lags (int): Number of lag periods to create.

    Returns:
        pandas DataFrame: DataFrame with lagged features.
    """
    df = timeseries_data.copy()
    
    # Create lags for each feature
    features = ['Quantity', 'People_Count', 'Time_Spent']
    for feature in features:
        df[feature] = df[feature].astype(float)
        for i in range(1, lags + 1):
            df[f'{feature}_lag_{i}'] = df[feature].shift(i)
    
    df.dropna(inplace=True)
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

def analyze_people_impact(stock_data, stock_code):
    """
    Analyzes and visualizes how People_Count affects sales predictions
    """
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot of People_Count vs Quantity
    plt.subplot(1, 2, 1)
    plt.scatter(stock_data['People_Count'], stock_data['Quantity'], alpha=0.5)
    plt.xlabel('People Count')
    plt.ylabel('Quantity Sold')
    plt.title(f'People Count vs Sales (StockCode: {stock_code})')
    
    # Calculate correlation
    correlation = stock_data['People_Count'].corr(stock_data['Quantity'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
             transform=plt.gca().transAxes, fontsize=10)
    
    # Show sales prediction for different people counts
    lagged_data = create_lagged_features(stock_data, lags=3)
    feature_cols = [col for col in lagged_data.columns if 'lag' in col]
    X = lagged_data[feature_cols]
    y = lagged_data['Quantity']
    
    if len(X) > 5:  # Ensure enough data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = train_xgboost_model(X_train, y_train)
        
        # Create predictions for different people counts
        last_row = X.iloc[-1].copy()
        people_counts = np.linspace(stock_data['People_Count'].min(), 
                                  stock_data['People_Count'].max(), 10)
        predictions = []
        
        for count in people_counts:
            # Update people count in the features
            test_features = last_row.copy()
            people_cols = [col for col in feature_cols if 'People_Count' in col]
            for col in people_cols:
                test_features[col] = count
            pred = predict_xgboost(model, test_features.values)
            predictions.append(pred[0])
        
        # Plot predictions vs people count
        plt.subplot(1, 2, 2)
        plt.plot(people_counts, predictions, 'r-', label='Predicted Sales')
        plt.xlabel('People Count')
        plt.ylabel('Predicted Quantity')
        plt.title('Sales Prediction vs People Count')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'people_impact_{stock_code}.png')
    plt.close()
    
    return correlation

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = pd.read_csv("retail_data_full_with_people_time.csv")
    daily_sales_data = preprocess_data(data)

    stock_codes = daily_sales_data['StockCode'].unique()[:5]  # Taking first 5 stock codes for demonstration
    print(f"\nAnalyzing first 5 stock codes: {stock_codes}")

    results = {}
    people_correlations = {}

    for stock_code in stock_codes:
        print(f"\n=== Processing StockCode: {stock_code} ===")
        stock_data = daily_sales_data[daily_sales_data['StockCode'] == stock_code].copy()
        if len(stock_data) < 10:
            print(f"Insufficient data for StockCode: {stock_code}")
            continue

        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        
        # Analyze people impact
        print(f"Analyzing people impact for StockCode {stock_code}...")
        correlation = analyze_people_impact(stock_data, stock_code)
        people_correlations[stock_code] = correlation
        print(f"Correlation between People Count and Sales: {correlation:.2f}")
        
        # XGBoost predictions with different people counts
        print("\nGenerating predictions for different people counts...")
        lagged_data = create_lagged_features(stock_data, lags=3)
        if len(lagged_data) > 5:
            feature_cols = [col for col in lagged_data.columns if 'lag' in col]
            X = lagged_data[feature_cols]
            y = lagged_data['Quantity']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            xgboost_model = train_xgboost_model(X_train, y_train)
            
            # Show predictions for low, medium, and high people counts
            current_people = stock_data['People_Count'].iloc[-1]
            low_people = stock_data['People_Count'].quantile(0.25)
            high_people = stock_data['People_Count'].quantile(0.75)
            
            last_features = X.iloc[-1].copy()
            scenarios = {
                'Low People Count': low_people,
                'Current People Count': current_people,
                'High People Count': high_people
            }
            
            print("\nSales Predictions for Different Scenarios:")
            for scenario, people_count in scenarios.items():
                test_features = last_features.copy()
                people_cols = [col for col in feature_cols if 'People_Count' in col]
                for col in people_cols:
                    test_features[col] = people_count
                pred = predict_xgboost(xgboost_model, test_features.values)
                print(f"{scenario} ({people_count:.0f} people): {pred[0]:.0f} units")
        
        print(f"\nVisualization saved as 'people_impact_{stock_code}.png'")
        print("-" * 50)
