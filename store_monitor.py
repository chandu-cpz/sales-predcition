import threading
import time
import serial
import pandas as pd
import numpy as np
import queue
import os
import xgboost as xgb # Import XGBoost directly
from sklearn.model_selection import train_test_split # Needed if splitting within training
from datetime import datetime, timedelta
import traceback
import warnings

warnings.filterwarnings('ignore') # Suppress warnings during ARIMA/XGBoost operations

# Import the refactored YoloTracker class
try:
    from yolo_tracker import YoloTracker
except ImportError:
    print("Error: Could not import YoloTracker from yolo_tracker.py.")
    print("Ensure yolo_tracker.py is in the same directory.")
    exit(1)

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'  # *** CHANGE THIS to your ESP32's serial port ***
BAUD_RATE = 115200
RFID_MAPPING_FILE = 'esp32/rfid_item_mapping.csv'
SALES_DATA_FILE = 'retail_data_full_with_people_time.csv'

# YOLO Tracker Config
YOLO_MODEL_PATH = 'best.pt' # Or 'yolov8n.pt' etc.
VIDEO_SOURCE = '0' # Webcam index or video file path
SHOW_YOLO_VIDEO = False # Set to True to see the YOLO tracking window
AVG_DWELL_PERIOD_MINUTES = 5 # Period for calculating average dwell time

# Prediction Config
PREDICTION_LAGS = 3 # Number of lagged features for the XGBoost model
MIN_HISTORICAL_DAYS_FOR_TRAINING = 15 # Min days needed to train a model (lags + buffer)

# --- Global Variables & Shared Resources ---
rfid_map_df = pd.DataFrame(columns=['RFID_UID', 'StockCode', 'Description'])
sales_df = pd.DataFrame()
# Dictionary to cache trained models in memory {stock_code: (model, train_time)}
# Retrain if model is older than, say, 1 hour
trained_models_cache = {}
MODEL_CACHE_EXPIRY_SECONDS = 3600 # 1 hour

map_lock = threading.Lock() # To protect access to rfid_map_df
dwell_time_queue = queue.Queue(maxsize=1) # Queue for avg dwell time from YOLO tracker

# --- Helper Functions (Copied/Adapted from time_series.ipynb & previous version) ---

def load_sales_data(filename=SALES_DATA_FILE):
    """Loads and preprocesses the historical sales data."""
    global sales_df
    try:
        print(f"Loading sales data from {filename}...")
        df = pd.read_csv(filename, encoding='ISO-8859-1') # Common encoding for retail data
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        required_cols = ['StockCode', 'Description', 'Quantity', 'InvoiceDate', 'People_Count', 'Time_Spent']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Sales data CSV must contain columns: {required_cols}")

        df['StockCode'] = df['StockCode'].astype(str).str.strip()
        df['Description'] = df['Description'].astype(str).str.strip()
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        # Use fillna(0) or a more sophisticated imputation if needed
        df['People_Count'] = pd.to_numeric(df['People_Count'], errors='coerce').fillna(0)
        df['Time_Spent'] = pd.to_numeric(df['Time_Spent'], errors='coerce').fillna(0)

        df['Date'] = df['InvoiceDate'].dt.date
        daily_sales = df.groupby(['StockCode', 'Description', 'Date']).agg(
            Quantity=('Quantity', 'sum'),
            Avg_People_Count=('People_Count', 'mean'),
            Avg_Time_Spent=('Time_Spent', 'mean')
        ).reset_index()
        daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
        daily_sales = daily_sales.sort_values(by=['StockCode', 'Date'])
        # Ensure daily frequency by reindexing (important for time series lags)
        daily_sales = daily_sales.set_index('Date').groupby('StockCode').resample('D').asfreq().fillna(0).reset_index()
        # Need to re-fetch description after resampling
        desc_map = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')
        daily_sales['Description'] = daily_sales['StockCode'].map(desc_map['Description'])
        daily_sales = daily_sales.fillna({'Description': 'Unknown'}) # Handle cases where description might be lost


        sales_df = daily_sales
        print(f"Sales data loaded and preprocessed: {len(sales_df)} daily records.")
        return True
    except FileNotFoundError:
        print(f"Error: Sales data file not found at {filename}")
        return False
    except Exception as e:
        print(f"Error loading sales data: {e}")
        traceback.print_exc()
        return False

def load_rfid_mapping(filename=RFID_MAPPING_FILE):
    """Loads the RFID to StockCode mapping."""
    global rfid_map_df
    try:
        with map_lock:
            if os.path.exists(filename):
                rfid_map_df = pd.read_csv(filename)
                rfid_map_df['RFID_UID'] = rfid_map_df['RFID_UID'].astype(str)
                rfid_map_df['StockCode'] = rfid_map_df['StockCode'].astype(str)
                rfid_map_df['Description'] = rfid_map_df['Description'].astype(str)
                print(f"RFID mapping loaded: {len(rfid_map_df)} entries.")
            else:
                print(f"RFID mapping file not found at {filename}. Starting with empty map.")
                rfid_map_df = pd.DataFrame(columns=['RFID_UID', 'StockCode', 'Description'])
        return True
    except Exception as e:
        print(f"Error loading RFID mapping: {e}")
        return False

def save_rfid_mapping(filename=RFID_MAPPING_FILE):
    """Saves the current RFID mapping to CSV."""
    try:
        with map_lock:
            # Sort by RFID_UID for consistency
            rfid_map_df_sorted = rfid_map_df.sort_values(by='RFID_UID')
            rfid_map_df_sorted.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error saving RFID mapping: {e}")

def find_best_unmapped_item():
    """Finds the best-selling item (total quantity) not yet in the RFID map."""
    global rfid_map_df, sales_df
    if sales_df.empty:
        print("Warning: Sales data not loaded. Cannot find best item.")
        return None, None

    with map_lock:
        mapped_codes = set(rfid_map_df['StockCode'].unique())

    total_sales = sales_df.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
    total_sales = total_sales.sort_values('Quantity', ascending=False)

    for _, row in total_sales.iterrows():
        stock_code = str(row['StockCode'])
        if stock_code not in mapped_codes:
            print(f"Found best unmapped item: {stock_code} - {row['Description']}")
            return stock_code, row['Description']

    print("Warning: No unmapped items found in sales data.")
    return None, None

# --- XGBoost Training and Prediction Logic (Adapted from time_series.ipynb) ---

def create_lagged_features(timeseries_data, lags=PREDICTION_LAGS):
    """Creates lagged features for time series data including People_Count and Time_Spent."""
    df = timeseries_data.copy()
    # Ensure we are working with the relevant columns if Description is present
    features_to_lag = ['Quantity', 'Avg_People_Count', 'Avg_Time_Spent']
    df_lags = df[features_to_lag].copy()

    for feature in features_to_lag:
        df_lags[feature] = df_lags[feature].astype(float)
        for i in range(1, lags + 1):
            df_lags[f'{feature}_lag_{i}'] = df_lags[feature].shift(i)

    df_lags.dropna(inplace=True)
    # Return only the target ('Quantity') and the generated lag features
    target = df_lags['Quantity']
    features = df_lags.drop(columns=features_to_lag) # Drop original non-lagged cols
    return features, target

def train_xgboost_model(feature_data, target_data):
    """Trains an XGBoost model on the given feature and target data."""
    if feature_data.empty or target_data.empty:
        print("Warning: Cannot train XGBoost model with empty data.")
        return None
    try:
        # Basic XGBoost Regressor - parameters can be tuned further
        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 n_estimators=100, # Number of boosting rounds
                                 learning_rate=0.1, # Step size shrinkage
                                 max_depth=5,       # Maximum depth of a tree
                                 subsample=0.8,     # Subsample ratio of the training instance
                                 colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree
                                 random_state=42,   # Seed for reproducibility
                                 n_jobs=-1)         # Use all available CPU cores

        model.fit(feature_data, target_data)
        return model
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        traceback.print_exc()
        return None

def predict_stock_need(stock_code, current_avg_dwell):
    """
    Trains model (if needed/cache expired) and predicts stock need for the next day
    using historical data and current average dwell time.
    """
    global sales_df, trained_models_cache
    if sales_df.empty:
        print("Warning: Sales data not loaded. Cannot predict.")
        return None

    # --- Get or Train Model ---
    model = None
    current_time = time.time()
    retrain_needed = True

    if stock_code in trained_models_cache:
        cached_model, train_time = trained_models_cache[stock_code]
        if (current_time - train_time) < MODEL_CACHE_EXPIRY_SECONDS:
            model = cached_model
            retrain_needed = False
            # print(f"Using cached model for {stock_code}") # Optional debug

    if retrain_needed:
        print(f"Training model for StockCode: {stock_code}...")
        product_history = sales_df[sales_df['StockCode'] == stock_code].copy()

        if len(product_history) < MIN_HISTORICAL_DAYS_FOR_TRAINING:
            print(f"Insufficient history for {stock_code} ({len(product_history)} days). Need at least {MIN_HISTORICAL_DAYS_FOR_TRAINING}.")
            return None # Cannot train

        # Create features and target for training
        X_train_features, y_train_target = create_lagged_features(product_history, lags=PREDICTION_LAGS)

        if X_train_features is None or X_train_features.empty:
             print(f"Could not create training features for {stock_code}.")
             return None

        model = train_xgboost_model(X_train_features, y_train_target)
        if model:
            trained_models_cache[stock_code] = (model, current_time) # Update cache
            print(f"Model trained and cached for {stock_code}.")
        else:
            print(f"Failed to train model for {stock_code}.")
            # Remove potentially outdated model from cache if training failed
            if stock_code in trained_models_cache:
                 del trained_models_cache[stock_code]
            return None # Stop if training failed

    # --- Prepare Features for Prediction ---
    # We need the latest PREDICTION_LAGS days of data to create the *next* feature set
    product_history_pred = sales_df[sales_df['StockCode'] == stock_code].tail(PREDICTION_LAGS).copy()
    if len(product_history_pred) < PREDICTION_LAGS:
         print(f"Warning: Not enough recent data ({len(product_history_pred)} days) to form prediction features for {stock_code}.")
         return None

    # Create the single row feature vector for the next prediction point
    prediction_features = {}
    feature_names_ordered = [] # Keep track of order for DataFrame creation
    for i in range(1, PREDICTION_LAGS + 1):
        # Lags are relative to the *prediction point*, so lag 1 is the last actual data point
        hist_index = PREDICTION_LAGS - i
        qty_lag_col = f'Quantity_lag_{i}'
        ppl_lag_col = f'Avg_People_Count_lag_{i}'
        time_lag_col = f'Avg_Time_Spent_lag_{i}'

        prediction_features[qty_lag_col] = product_history_pred['Quantity'].iloc[hist_index]
        prediction_features[ppl_lag_col] = product_history_pred['Avg_People_Count'].iloc[hist_index]
        prediction_features[time_lag_col] = product_history_pred['Avg_Time_Spent'].iloc[hist_index]
        feature_names_ordered.extend([qty_lag_col, ppl_lag_col, time_lag_col])


    # *** Replace the most recent 'People Count' lag with the LIVE dwell time ***
    # This assumes the model learned that recent activity (represented by people count/dwell)
    # is predictive. We use the live dwell time here.
    live_feature_col = f'Avg_People_Count_lag_1' # Or Avg_Time_Spent_lag_1 based on model training
    if live_feature_col in prediction_features:
         print(f"Injecting live dwell time ({current_avg_dwell:.2f}s) into feature '{live_feature_col}'")
         prediction_features[live_feature_col] = current_avg_dwell
    else:
         print(f"Warning: Feature '{live_feature_col}' not found in prediction features. Cannot inject live dwell time.")


    # Create DataFrame in the correct order expected by the model
    # If model has feature_names_in_, use that order, otherwise use generated order
    expected_feature_order = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_names_ordered
    try:
        feature_vector_df = pd.DataFrame([prediction_features])[expected_feature_order]
    except KeyError as e:
         print(f"Error: Mismatch between generated features and expected model features: {e}")
         print(f"Generated: {list(prediction_features.keys())}")
         print(f"Expected: {expected_feature_order}")
         return None


    # --- Make Prediction ---
    try:
        prediction = model.predict(feature_vector_df)
        predicted_quantity = max(0, round(prediction[0])) # Ensure non-negative integer prediction
        print(f"Prediction for {stock_code}: Need {predicted_quantity} units tomorrow.")
        return predicted_quantity
    except Exception as e:
        print(f"Error during prediction for {stock_code}: {e}")
        traceback.print_exc()
        return None


# --- Thread Tasks ---

def serial_listener_task():
    """Listens to serial port for RFID tags and handles mapping/prediction."""
    global rfid_map_df
    print("Serial listener thread started.")
    serial_conn = None
    while True:
        try:
            if serial_conn is None or not serial_conn.is_open:
                print(f"Attempting to connect to serial port {SERIAL_PORT}...")
                serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
                print("Serial port connected.")
                time.sleep(2) # Allow ESP32 to initialize after connection

            if serial_conn.in_waiting > 0:
                line = serial_conn.readline().decode('utf-8', errors='ignore').strip()
                print(f"Serial RX: {line}") # Debug output

                if line.startswith("RFID_UID:"):
                    rfid_uid = line.split(":", 1)[1].strip()
                    if not rfid_uid:
                        print("Warning: Received empty RFID UID.")
                        continue

                    print(f"Processing RFID UID: {rfid_uid}")
                    stock_code = None
                    description = None

                    # Check if UID is already mapped
                    with map_lock:
                        mapping_entry = rfid_map_df[rfid_map_df['RFID_UID'] == rfid_uid]

                    if not mapping_entry.empty:
                        stock_code = mapping_entry['StockCode'].iloc[0]
                        description = mapping_entry['Description'].iloc[0]
                        print(f"RFID {rfid_uid} mapped to: {stock_code} - {description}")
                    else:
                        print(f"New RFID UID detected: {rfid_uid}. Finding best item to map...")
                        new_stock_code, new_description = find_best_unmapped_item()
                        if new_stock_code and new_description:
                            with map_lock:
                                # Add new mapping using pd.concat
                                new_entry = pd.DataFrame([{'RFID_UID': rfid_uid, 'StockCode': new_stock_code, 'Description': new_description}])
                                rfid_map_df = pd.concat([rfid_map_df, new_entry], ignore_index=True)
                            save_rfid_mapping() # Save immediately after adding
                            stock_code = new_stock_code
                            description = new_description
                            print(f"Mapped new RFID {rfid_uid} to: {stock_code} - {description}")
                        else:
                            print(f"Could not find an unmapped item for RFID {rfid_uid}.")
                            continue # Skip prediction if no item could be mapped

                    # --- Trigger Prediction ---
                    if stock_code:
                        try:
                            # Get the latest average dwell time (non-blocking)
                            current_avg_dwell = dwell_time_queue.get_nowait()
                            print(f"Using latest average dwell time: {current_avg_dwell:.2f}s for prediction.")
                        except queue.Empty:
                            print("Warning: Dwell time queue is empty. Using 0 for prediction.")
                            current_avg_dwell = 0.0

                        # Call prediction function
                        predicted_need = predict_stock_need(stock_code, current_avg_dwell)
                        if predicted_need is not None:
                            # Placeholder for action: Log, display, send alert, etc.
                            print(f"ACTION >> StockCode: {stock_code}, Description: {description}, Predicted Need (Next Day): {predicted_need}")
                        else:
                            print(f"Prediction failed for {stock_code}.")

        except serial.SerialException as se:
            print(f"Serial Error: {se}. Retrying connection...")
            if serial_conn and serial_conn.is_open:
                serial_conn.close()
            serial_conn = None
            time.sleep(5) # Wait before retrying connection
        except KeyboardInterrupt:
            print("Serial listener stopping...")
            break
        except Exception as e:
            print(f"Error in serial listener: {e}")
            traceback.print_exc()
            time.sleep(2) # Prevent rapid error loops

    if serial_conn and serial_conn.is_open:
        serial_conn.close()
    print("Serial listener thread finished.")


def yolo_tracker_task():
    """Runs the YOLO tracker in a separate thread."""
    print("YOLO tracker thread starting.")
    try:
        tracker = YoloTracker(
            model_path=YOLO_MODEL_PATH,
            video_source=VIDEO_SOURCE,
            show_video=SHOW_YOLO_VIDEO,
            dwell_queue=dwell_time_queue, # Pass the shared queue
            avg_dwell_period_minutes=AVG_DWELL_PERIOD_MINUTES
        )
        tracker.run() # This will block until tracking stops or errors out
    except Exception as e:
        print(f"Error initializing or running YOLO tracker: {e}")
        traceback.print_exc()
    print("YOLO tracker thread finished.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Store Monitor Application...")

    # Load initial data
    if not load_sales_data():
        print("Exiting due to sales data loading error.")
        exit(1)
    if not load_rfid_mapping():
        print("Continuing with empty RFID map due to loading error.")
        # Decide if this is critical - exit(1) might be appropriate

    # --- Start Threads ---
    print("Creating threads...")
    serial_thread = threading.Thread(target=serial_listener_task, daemon=True)
    yolo_thread = threading.Thread(target=yolo_tracker_task, daemon=True)

    print("Starting YOLO tracker thread...")
    yolo_thread.start()
    # Give YOLO time to initialize model before starting serial listener
    print("Waiting for YOLO model to load (approx 10-20s)...")
    time.sleep(15)

    print("Starting serial listener thread...")
    serial_thread.start()

    # Keep main thread alive while daemon threads run
    try:
        while True:
            # Check if threads are alive (optional)
            if not yolo_thread.is_alive():
                print("Error: YOLO tracker thread has stopped unexpectedly.")
                # Optionally try to restart it or handle the error
                break
            if not serial_thread.is_alive():
                 print("Error: Serial listener thread has stopped unexpectedly.")
                 # Optionally try to restart it or handle the error
                 break
            time.sleep(5) # Check every 5 seconds
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")

    # Wait briefly for threads to potentially finish cleanup (though daemons might exit abruptly)
    # yolo_thread.join(timeout=2)
    # serial_thread.join(timeout=2)

    print("Store Monitor Application finished.")