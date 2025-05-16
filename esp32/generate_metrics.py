import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split # Might be needed for variations
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import os
import sys
import warnings
import traceback
from tqdm import tqdm # For progress bars during backtesting

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
# Paths relative to this script's location (esp32/)
SALES_DATA_FILE = '../retail_data_full_with_people_time.csv'
RFID_MAPPING_FILE = 'rfid_item_mapping.csv'
METRICS_DIR = 'metrics' # Output directory for plots

# --- Target Products for Limited Analysis ---
# Limit analysis to these StockCodes found in rfid_item_mapping.csv to reduce memory usage
TARGET_STOCK_CODES = ['85123A', '84077', '85099B', '21212']

# Constants potentially needed from store_monitor.py for backtesting consistency
PREDICTION_LAGS = 3
MIN_HISTORICAL_DAYS_FOR_TRAINING = 15 # Min days needed to train a model

# --- Create Output Directory ---
try:
    os.makedirs(METRICS_DIR, exist_ok=True)
    print(f"Created/Ensured metrics directory exists: '{METRICS_DIR}'")
except OSError as e:
    print(f"Error creating directory {METRICS_DIR}: {e}")
    # Decide if this is fatal - maybe exit? For now, continue and plots might fail.

# --- Data Loading Functions ---

def load_raw_sales_data(filename=SALES_DATA_FILE):
    """Loads the raw sales data."""
    print(f"Loading raw sales data from {filename}...")
    try:
        if not os.path.exists(filename):
             print(f"Error: Sales data file not found at {os.path.abspath(filename)}")
             return None
        df = pd.read_csv(filename, encoding='ISO-8859-1')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        # Basic type conversions and cleaning
        df['StockCode'] = df['StockCode'].astype(str).str.strip()
        df['Description'] = df['Description'].astype(str).str.strip()
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce') # Keep NaNs for raw analysis
        df['People_Count'] = pd.to_numeric(df['People_Count'], errors='coerce')
        df['Time_Spent'] = pd.to_numeric(df['Time_Spent'], errors='coerce')
        df.dropna(subset=['InvoiceDate'], inplace=True) # Drop rows where date couldn't be parsed
        print(f"Raw sales data loaded: {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading raw sales data: {e}")
        traceback.print_exc()
        return None

def preprocess_and_aggregate_sales(df_raw):
    """Preprocesses raw data and aggregates it daily, similar to store_monitor."""
    print("Preprocessing and aggregating sales data daily...")
    if df_raw is None:
        return None
    df = df_raw.copy()
    # Handle missing numeric values before aggregation (fill with 0 for sum/mean)
    df['Quantity'] = df['Quantity'].fillna(0)
    df['People_Count'] = df['People_Count'].fillna(0)
    df['Time_Spent'] = df['Time_Spent'].fillna(0)

    df['Date'] = df['InvoiceDate'].dt.date
    daily_sales = df.groupby(['StockCode', 'Description', 'Date']).agg(
        Quantity=('Quantity', 'sum'),
        Avg_People_Count=('People_Count', 'mean'),
        Avg_Time_Spent=('Time_Spent', 'mean')
    ).reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values(by=['StockCode', 'Date'])

    # Ensure daily frequency by reindexing and filling missing days with 0
    daily_sales = daily_sales.set_index('Date').groupby('StockCode').resample('D').agg(
         {'Quantity': 'sum', 'Avg_People_Count': 'mean', 'Avg_Time_Spent': 'mean'}
    ).fillna(0).reset_index()

    # Re-fetch description after resampling
    desc_map = df_raw[['StockCode', 'Description']].drop_duplicates(subset=['StockCode']).set_index('StockCode')
    daily_sales['Description'] = daily_sales['StockCode'].map(desc_map['Description']).fillna('Unknown')

    print(f"Daily aggregated sales data created: {len(daily_sales)} records.")
    return daily_sales


def load_rfid_mapping(filename=RFID_MAPPING_FILE):
    """Loads the RFID to StockCode mapping."""
    print(f"Loading RFID mapping from {filename}...")
    try:
        if os.path.exists(filename):
            rfid_map_df = pd.read_csv(filename)
            rfid_map_df['RFID_UID'] = rfid_map_df['RFID_UID'].astype(str)
            rfid_map_df['StockCode'] = rfid_map_df['StockCode'].astype(str)
            rfid_map_df['Description'] = rfid_map_df['Description'].astype(str)
            print(f"RFID mapping loaded: {len(rfid_map_df)} entries.")
            return rfid_map_df
        else:
            print(f"Warning: RFID mapping file '{filename}' not found. Returning empty map.")
            return pd.DataFrame(columns=['RFID_UID', 'StockCode', 'Description'])
    except Exception as e:
        print(f"Error loading RFID mapping: {e}")
        return pd.DataFrame(columns=['RFID_UID', 'StockCode', 'Description'])

# --- Metric Calculation Functions ---

def calculate_dataset_metrics(df_raw, df_aggregated):
    """Calculates and prints dataset characteristic metrics."""
    print("\n--- Dataset Characteristics ---")
    if df_raw is None or df_aggregated is None:
        print("Sales data not loaded. Cannot calculate metrics.")
        return

    # Raw Data Metrics
    print("\n[Raw Sales Data]")
    num_records = len(df_raw)
    time_span_start = df_raw['InvoiceDate'].min().strftime('%Y-%m-%d')
    time_span_end = df_raw['InvoiceDate'].max().strftime('%Y-%m-%d')
    num_unique_products = df_raw['StockCode'].nunique()
    print(f"Dataset Size (Records): {num_records}")
    print(f"Time Span Covered: {time_span_start} to {time_span_end}")
    print(f"Unique Products (StockCodes): {num_unique_products}")

    print("\nFeature Count & Types:")
    print(df_raw.dtypes)

    print("\nCompleteness (% Missing Values):")
    missing_perc = (df_raw.isnull().sum() / num_records) * 100
    print(missing_perc[missing_perc > 0]) # Only show columns with missing values
    print("Handling Method: Numeric coerced/filled with 0 before aggregation, Date errors dropped.")

    print("\nData Density:")
    avg_transactions_per_product = df_raw.groupby('StockCode').size().mean()
    print(f"Average Transactions per Product: {avg_transactions_per_product:.2f}")

    # Check for Category column (adjust column name if different)
    category_col = None
    potential_cat_cols = ['Category', 'ProductType', 'Department'] # Add other likely names
    for col in potential_cat_cols:
        if col in df_raw.columns:
            category_col = col
            break

    if category_col:
        print(f"\nDistribution Across Categories ('{category_col}'):")
        print(df_raw[category_col].value_counts())
    else:
        print("\nCategory Information: No obvious category column found.")

    # Aggregated Data Metrics
    print("\n[Daily Aggregated Data]")
    print("Descriptive Statistics (Aggregated Daily):")
    print(df_aggregated[['Quantity', 'Avg_People_Count', 'Avg_Time_Spent']].describe())

    # Stockout Analysis (Definition: Aggregated Daily Quantity <= 0)
    print("\nStockout Analysis (Aggregated Daily Quantity <= 0):")
    stockout_days = df_aggregated[df_aggregated['Quantity'] <= 0]
    total_possible_days = len(df_aggregated)
    overall_stockout_perc = (len(stockout_days) / total_possible_days) * 100 if total_possible_days > 0 else 0
    print(f"Overall Percentage of Stockout Days: {overall_stockout_perc:.2f}%")

    # Per-product stockout (for top 10 by total records)
    product_day_counts = df_aggregated['StockCode'].value_counts()
    top_10_products = product_day_counts.head(10).index.tolist()
    stockout_by_product = stockout_days[stockout_days['StockCode'].isin(top_10_products)]['StockCode'].value_counts()
    product_stockout_perc = (stockout_by_product / product_day_counts[top_10_products]) * 100
    print("\nStockout Percentage for Top 10 Products (by # of days present):")
    print(product_stockout_perc.fillna(0))


def calculate_rfid_metrics(rfid_map_df, sales_df_raw):
    """Calculates and prints RFID mapping metrics."""
    print("\n--- RFID Mapping Analysis ---")
    if rfid_map_df is None or sales_df_raw is None:
        print("RFID map or sales data not loaded.")
        return {}

    total_mapped_tags = len(rfid_map_df)
    unique_mapped_products = rfid_map_df['StockCode'].nunique()
    print(f"Total Mapped RFID Tags: {total_mapped_tags}")
    print(f"Unique Products Mapped: {unique_mapped_products}")

    all_products_in_sales = set(sales_df_raw['StockCode'].unique())
    mapped_products_set = set(rfid_map_df['StockCode'].unique())
    unmapped_products = all_products_in_sales - mapped_products_set
    print(f"Products in Sales Data but NOT Mapped: {len(unmapped_products)}")
    # print(f"Unmapped StockCodes: {list(unmapped_products)}") # Optional: print list

    return {"total_mapped": total_mapped_tags, "unique_mapped": unique_mapped_products, "unmapped_count": len(unmapped_products)}


# --- Visualization Functions ---

def plot_histograms(df_raw, columns, titles, filename_prefix):
    """Generates and saves histograms for specified columns."""
    print(f"Generating histograms: {', '.join(columns)}...")
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1: axes = [axes] # Make it iterable if only one plot

    for i, col in enumerate(columns):
        if col in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[col]):
            sns.histplot(df_raw[col].dropna(), kde=False, ax=axes[i])
            axes[i].set_title(titles[i])
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
        else:
            print(f"Warning: Column '{col}' not found or not numeric. Skipping histogram.")
            axes[i].set_title(f"{titles[i]}\n(Column not found/numeric)")

    plt.tight_layout()
    filepath = os.path.join(METRICS_DIR, f"{filename_prefix}_histograms.png")
    plt.savefig(filepath)
    print(f"Saved histograms to {filepath}")
    plt.close(fig)

def plot_aggregated_timeseries(df_aggregated, filename="aggregated_daily_quantity.png"):
    """Plots the total aggregated daily quantity over time."""
    print("Generating aggregated daily quantity time series plot...")
    if df_aggregated is None or df_aggregated.empty:
        print("Aggregated data not available. Skipping time series plot.")
        return

    daily_total = df_aggregated.groupby('Date')['Quantity'].sum()
    fig, ax = plt.subplots(figsize=(15, 6))
    daily_total.plot(ax=ax)
    ax.set_title("Total Daily Quantity Sold Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Quantity")
    ax.grid(True)

    filepath = os.path.join(METRICS_DIR, filename)
    plt.savefig(filepath)
    print(f"Saved time series plot to {filepath}")
    plt.close(fig)

def plot_category_distribution(df_raw, category_col, filename="category_distribution.png"):
    """Plots the distribution of records across categories."""
    print(f"Generating category distribution plot for '{category_col}'...")
    if df_raw is None or category_col not in df_raw.columns:
        print(f"Raw data or category column '{category_col}' not available. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(y=df_raw[category_col], order=df_raw[category_col].value_counts().index, ax=ax)
    ax.set_title(f"Distribution of Records by Category ('{category_col}')")
    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Category")
    plt.tight_layout()

    filepath = os.path.join(METRICS_DIR, filename)
    plt.savefig(filepath)
    print(f"Saved category distribution plot to {filepath}")
    plt.close(fig)

def plot_stockout_distribution(df_aggregated, top_n=20, filename="stockout_distribution.png"):
     """Plots the stockout percentage for the top N products."""
     print("Generating stockout distribution plot...")
     if df_aggregated is None or df_aggregated.empty:
         print("Aggregated data not available. Skipping stockout plot.")
         return

     stockout_days = df_aggregated[df_aggregated['Quantity'] <= 0]
     product_day_counts = df_aggregated['StockCode'].value_counts()
     top_products = product_day_counts.head(top_n).index.tolist()

     stockout_by_product = stockout_days[stockout_days['StockCode'].isin(top_products)]['StockCode'].value_counts()
     product_stockout_perc = (stockout_by_product / product_day_counts[top_products]) * 100
     product_stockout_perc = product_stockout_perc.fillna(0).sort_values(ascending=False)

     fig, ax = plt.subplots(figsize=(12, 8))
     sns.barplot(x=product_stockout_perc.values, y=product_stockout_perc.index, ax=ax)
     ax.set_title(f"Stockout Percentage (Daily Qty <= 0) for Top {top_n} Products")
     ax.set_xlabel("Stockout Percentage (%)")
     ax.set_ylabel("StockCode")
     plt.tight_layout()

     filepath = os.path.join(METRICS_DIR, filename)
     plt.savefig(filepath)
     print(f"Saved stockout distribution plot to {filepath}")
     plt.close(fig)


def plot_rfid_mapping_summary(rfid_metrics, filename="rfid_mapping_summary.png"):
    """Plots a bar chart summarizing RFID mapping."""
    print("Generating RFID mapping summary plot...")
    if not rfid_metrics:
        print("RFID metrics not available. Skipping plot.")
        return

    labels = ['Mapped Products', 'Unmapped Products']
    counts = [rfid_metrics.get('unique_mapped', 0), rfid_metrics.get('unmapped_count', 0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=counts, ax=ax)
    ax.set_title("RFID Mapping Summary (Products in Sales Data)")
    ax.set_ylabel("Number of Unique Products")
    for i, count in enumerate(counts):
        ax.text(i, count + 0.1, str(count), ha='center')

    filepath = os.path.join(METRICS_DIR, filename)
    plt.savefig(filepath)
    print(f"Saved RFID mapping summary plot to {filepath}")
    plt.close(fig)


# --- XGBoost Backtesting Functions ---

# Reuse/Adapt from store_monitor.py
def create_lagged_features(timeseries_data, lags=PREDICTION_LAGS):
    """Creates lagged features for time series data including People_Count and Time_Spent."""
    df = timeseries_data.copy()
    features_to_lag = ['Quantity', 'Avg_People_Count', 'Avg_Time_Spent']
    missing_cols = [col for col in features_to_lag if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for lagging: {missing_cols}. Skipping them.")
        features_to_lag = [col for col in features_to_lag if col in df.columns]
        if not features_to_lag:
             print("Error: No valid columns found for lagging.")
             return None, None

    df_lags = df[features_to_lag].copy()
    for feature in features_to_lag:
        df_lags[feature] = df_lags[feature].astype(float)
        for i in range(1, lags + 1):
            df_lags[f'{feature}_lag_{i}'] = df_lags[feature].shift(i)
    df_lags.dropna(inplace=True)
    if df_lags.empty: return None, None
    target = df_lags['Quantity']
    features = df_lags.drop(columns=features_to_lag)
    return features, target

def train_xgboost_model(feature_data, target_data):
    """Trains an XGBoost model."""
    if feature_data is None or target_data is None or feature_data.empty or target_data.empty:
        print("Warning: Cannot train XGBoost model with empty data.")
        return None
    try:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                                 max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, n_jobs=-1)
        model.fit(feature_data, target_data)
        return model
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        return None

def perform_xgboost_backtest(df_aggregated, product_stockcode, test_days=30):
    """Performs a rolling forecast backtest for a given product."""
    print(f"\n--- XGBoost Backtest for Product: {product_stockcode} ---")
    product_data = df_aggregated[df_aggregated['StockCode'] == product_stockcode].copy()
    product_data = product_data.set_index('Date').sort_index()

    if len(product_data) < MIN_HISTORICAL_DAYS_FOR_TRAINING + test_days + PREDICTION_LAGS:
        print(f"Insufficient data for {product_stockcode} to perform backtest with {test_days} test days.")
        return None, None

    predictions = []
    actuals = []
    feature_importances = []

    print(f"Backtesting over {test_days} days...")
    for i in tqdm(range(test_days)):
        split_point = len(product_data) - test_days + i
        train_data = product_data.iloc[:split_point]
        test_actual = product_data.iloc[split_point] # The day we want to predict

        if len(train_data) < MIN_HISTORICAL_DAYS_FOR_TRAINING:
            # print(f"Skipping day {i+1}: Not enough training data ({len(train_data)} days)")
            continue # Skip if not enough history even at this point

        # Create features for training
        X_train_features, y_train_target = create_lagged_features(train_data, lags=PREDICTION_LAGS)
        if X_train_features is None or X_train_features.empty:
            # print(f"Skipping day {i+1}: Could not create training features.")
            continue

        # Train model
        model = train_xgboost_model(X_train_features, y_train_target)
        if model is None:
            # print(f"Skipping day {i+1}: Model training failed.")
            continue
        feature_importances.append(model.feature_importances_)

        # Prepare features for prediction (using data up to the day *before* test_actual)
        pred_feature_data = train_data.tail(PREDICTION_LAGS)
        if len(pred_feature_data) < PREDICTION_LAGS:
            # print(f"Skipping day {i+1}: Not enough recent data for prediction features.")
            continue

        prediction_features = {}
        feature_names_ordered = []
        for lag in range(1, PREDICTION_LAGS + 1):
            hist_index = PREDICTION_LAGS - lag
            for base_feature in ['Quantity', 'Avg_People_Count', 'Avg_Time_Spent']:
                 lag_col_name = f'{base_feature}_lag_{lag}'
                 prediction_features[lag_col_name] = pred_feature_data[base_feature].iloc[hist_index]
                 feature_names_ordered.append(lag_col_name)

        # Create DataFrame in the correct order
        expected_feature_order = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_names_ordered
        try:
            feature_vector_df = pd.DataFrame([prediction_features])[expected_feature_order]
        except KeyError as e:
             print(f"\nError: Feature mismatch on day {i+1}: {e}")
             continue
        except Exception as e:
             print(f"\nError creating feature vector on day {i+1}: {e}")
             continue


        # Make prediction
        try:
            prediction = model.predict(feature_vector_df)
            predicted_quantity = max(0, round(prediction[0]))
            predictions.append(predicted_quantity)
            actuals.append(test_actual['Quantity'])
        except Exception as e:
            print(f"\nError during prediction on day {i+1}: {e}")
            # Append NaN or handle differently? For now, skip this day's metrics
            continue

    if not actuals or not predictions:
        print("No successful predictions made during backtest.")
        return None, None

    # Calculate Metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    print(f"\nBacktest Results ({product_stockcode}):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Average Feature Importances
    avg_feature_importance = np.mean(feature_importances, axis=0)
    feature_names = expected_feature_order # Use the order from the last successful model
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': avg_feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nAverage Feature Importances:")
    print(importance_df)

    # Store results for plotting
    results = {
        'actuals': actuals,
        'predictions': predictions,
        'dates': product_data.index[-len(actuals):], # Dates corresponding to actuals
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'feature_importance': importance_df
    }
    return results, product_stockcode


def plot_backtest_results(results, stockcode, filename_prefix):
    """Plots the results of the XGBoost backtest."""
    if not results: return

    print(f"Generating backtest plots for {stockcode}...")
    actuals = results['actuals']
    predictions = results['predictions']
    dates = results['dates']
    importance_df = results['feature_importance']

    # 1. Predicted vs Actual Scatter Plot
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.scatter(actuals, predictions, alpha=0.6)
    ax1.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], '--', color='red', label='Ideal')
    ax1.set_title(f'Predicted vs Actual Quantity ({stockcode})')
    ax1.set_xlabel('Actual Quantity')
    ax1.set_ylabel('Predicted Quantity')
    ax1.legend()
    ax1.grid(True)
    filepath1 = os.path.join(METRICS_DIR, f"{filename_prefix}_{stockcode}_pred_vs_actual.png")
    plt.savefig(filepath1)
    print(f"Saved plot: {filepath1}")
    plt.close(fig1)

    # 2. Time Series Plot
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(dates, actuals, label='Actual Quantity', marker='.')
    ax2.plot(dates, predictions, label='Predicted Quantity', marker='x', linestyle='--')
    ax2.set_title(f'Actual vs Predicted Quantity Over Time ({stockcode})')
    ax2.set_xlabel('Date') # Corrected ax to ax2
    ax2.set_ylabel('Quantity')
    ax2.legend()
    ax2.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    filepath2 = os.path.join(METRICS_DIR, f"{filename_prefix}_{stockcode}_timeseries.png")
    plt.savefig(filepath2)
    print(f"Saved plot: {filepath2}")
    plt.close(fig2)

    # 3. Feature Importance Bar Chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax3)
    ax3.set_title(f'Average Feature Importance ({stockcode})')
    plt.tight_layout()
    filepath3 = os.path.join(METRICS_DIR, f"{filename_prefix}_{stockcode}_feature_importance.png")
    plt.savefig(filepath3)
    print(f"Saved plot: {filepath3}")
    plt.close(fig3)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Metrics Generation Script...")
    print(f"Limiting analysis to products: {TARGET_STOCK_CODES}")

    # Load Data
    sales_raw_df_full = load_raw_sales_data()

    # Filter raw data BEFORE aggregation and most analysis
    if sales_raw_df_full is not None:
        print(f"Filtering raw sales data for target products ({len(TARGET_STOCK_CODES)})...")
        sales_raw_df = sales_raw_df_full[sales_raw_df_full['StockCode'].isin(TARGET_STOCK_CODES)].copy()
        print(f"Filtered raw data size: {len(sales_raw_df)} records.")
        if sales_raw_df.empty:
            print("Warning: No data found for the target StockCodes in the raw sales file.")
            sales_raw_df = None # Ensure downstream checks handle this
    else:
        sales_raw_df = None

    # Aggregate only the filtered data
    sales_aggregated_df = preprocess_and_aggregate_sales(sales_raw_df) # Pass filtered data

    # Load RFID mapping (still load full map for context)
    rfid_df = load_rfid_mapping()

    # Calculate and Print Metrics (will use filtered data where applicable)
    calculate_dataset_metrics(sales_raw_df, sales_aggregated_df)
    rfid_metrics_dict = calculate_rfid_metrics(rfid_df, sales_raw_df)

    # Generate Visualizations
    if sales_raw_df is not None:
        plot_histograms(sales_raw_df,
                        columns=['Quantity', 'People_Count', 'Time_Spent'],
                        titles=['Raw Quantity Distribution', 'Raw People Count Distribution', 'Raw Time Spent Distribution'],
                        filename_prefix="raw_data")
        # Check for category column before plotting
        category_col = None
        potential_cat_cols = ['Category', 'ProductType', 'Department']
        for col in potential_cat_cols:
            if col in sales_raw_df.columns:
                category_col = col
                break
        if category_col:
            plot_category_distribution(sales_raw_df, category_col)

    if sales_aggregated_df is not None:
        plot_aggregated_timeseries(sales_aggregated_df)
        plot_stockout_distribution(sales_aggregated_df)

    if rfid_metrics_dict:
        plot_rfid_mapping_summary(rfid_metrics_dict)

    # Perform XGBoost Backtest (only on the target products)
    if sales_aggregated_df is not None:
        print("\n--- XGBoost Backtesting ---")
        print(f"Performing backtest only for target products: {TARGET_STOCK_CODES}")
        for product in TARGET_STOCK_CODES:
            # Check if product exists in the aggregated (filtered) data
            if product in sales_aggregated_df['StockCode'].unique():
                backtest_results, tested_stockcode = perform_xgboost_backtest(sales_aggregated_df, product)
                if backtest_results:
                    plot_backtest_results(backtest_results, tested_stockcode, filename_prefix="xgboost_backtest")
            else:
                print(f"Skipping backtest for {product}: No data found after filtering/aggregation.")

    print("\nMetrics Generation Script Finished.")
    print(f"Plots saved in '{METRICS_DIR}' directory.")
