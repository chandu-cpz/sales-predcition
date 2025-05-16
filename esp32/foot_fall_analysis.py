import streamlit as st
import pandas as pd
import time
import os
import sys
from datetime import datetime
import threading
import queue
import serial
import traceback
import importlib
import colorama # Needed by store_monitor functions potentially
from colorama import Fore, Style

# --- Add parent directory to sys.path ---
SCRIPT_DIR_FOOTFALL = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR_FOOTFALL)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if SCRIPT_DIR_FOOTFALL not in sys.path:
     sys.path.insert(0, SCRIPT_DIR_FOOTFALL)

# --- Initialize colorama ---
colorama.init(autoreset=True)

# --- Import Functions & Classes ---
# Import cautiously, avoiding running __main__ blocks if possible
try:
    import store_monitor
    from store_monitor import (
        load_sales_data, load_rfid_mapping, predict_stock_need,
        save_rfid_mapping, find_best_unmapped_item, map_lock,
        # Access global vars via the module name after import
        # rfid_map_df, sales_df, trained_models_cache # Avoid direct import of mutable globals
        SERIAL_PORT, BAUD_RATE, # Get constants
        MIN_HISTORICAL_DAYS_FOR_TRAINING, PREDICTION_LAGS, MODEL_CACHE_EXPIRY_SECONDS # Needed by predict_stock_need
    )
    from suggestions import get_average_sales, generate_suggestion
    from yolo_tracker import YoloTracker # Import the class

    # Configure Google AI
    import google.generativeai as genai
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAlV6jPPwaj669vdEigWnfyLBrel4xKWns")
    GOOGLE_MODEL_NAME = "gemini-1.5-flash-latest"
    if not GOOGLE_API_KEY:
         st.error("GOOGLE_API_KEY not set.")
         st.stop()
    try:
         genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
         st.error(f"Error configuring Google Generative AI: {e}")
         st.stop()

except ImportError as e:
    st.error(f"Failed to import necessary modules: {e}")
    st.stop()
except Exception as e:
     st.error(f"An unexpected error occurred during imports: {e}")
     st.stop()

# --- Constants & Config (from store_monitor where needed) ---
YOLO_MODEL_PATH = '../best.pt' # Relative to esp32/
VIDEO_SOURCE = '0'
SHOW_YOLO_VIDEO = False # Cannot show video window easily in Streamlit background thread
AVG_DWELL_PERIOD_MINUTES = 5
YOLO_OUTPUT_CSV = 'yolov8_dwell_times_st.csv' # Separate output for streamlit run

# --- Queues for Thread Communication ---
rfid_queue = queue.Queue()
dwell_queue = queue.Queue(maxsize=1) # Same as in store_monitor

# --- Streamlit App State ---
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'rfid_map_df' not in st.session_state:
        st.session_state.rfid_map_df = pd.DataFrame(columns=['RFID_UID', 'StockCode', 'Description'])
    if 'sales_df' not in st.session_state:
        st.session_state.sales_df = pd.DataFrame()
    if 'trained_models_cache' not in st.session_state:
        st.session_state.trained_models_cache = {}
    if 'serial_thread_running' not in st.session_state:
        st.session_state.serial_thread_running = False
    if 'yolo_thread_running' not in st.session_state:
        st.session_state.yolo_thread_running = False
    if 'last_rfid' not in st.session_state:
        st.session_state.last_rfid = "N/A"
    if 'last_stock_code' not in st.session_state:
        st.session_state.last_stock_code = "N/A"
    if 'last_description' not in st.session_state:
        st.session_state.last_description = "N/A"
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = "N/A"
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = "N/A"
    if 'last_suggestion' not in st.session_state:
        st.session_state.last_suggestion = "Waiting for scan..."
    if 'last_dwell_time' not in st.session_state:
        st.session_state.last_dwell_time = 0.0
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Initializing..."
    if 'error_message' not in st.session_state:
        st.session_state.error_message = ""

init_session_state()

# --- Background Task Functions ---

def serial_listener_thread_func(rfid_q):
    """Listens to serial port and puts RFID UIDs into the queue."""
    serial_conn = None
    st.session_state.status_message = "Serial Listener: Starting..."
    print("Serial listener thread started.") # Log to console
    while True:
        try:
            if serial_conn is None or not serial_conn.is_open:
                st.session_state.status_message = f"Serial Listener: Connecting to {SERIAL_PORT}..."
                print(f"Attempting to connect to serial port {SERIAL_PORT}...")
                serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
                st.session_state.status_message = "Serial Listener: Connected."
                print("Serial port connected.")
                time.sleep(2)

            if serial_conn.in_waiting > 0:
                line = serial_conn.readline().decode('utf-8', errors='ignore').strip()
                print(f"Serial RX: {line}") # Debug output

                if line.startswith("RFID_UID:"):
                    rfid_uid = line.split(":", 1)[1].strip()
                    if rfid_uid:
                        print(f"Putting RFID {rfid_uid} into queue.")
                        rfid_q.put(rfid_uid) # Put the UID into the queue for main thread
                        st.session_state.status_message = f"Serial Listener: Detected {rfid_uid}"
                    else:
                        print("Warning: Received empty RFID UID.")
                        st.session_state.status_message = "Serial Listener: Received empty UID."

        except serial.SerialException as se:
            st.session_state.error_message = f"Serial Error: {se}. Retrying..."
            print(f"Serial Error: {se}. Retrying connection...")
            if serial_conn and serial_conn.is_open:
                serial_conn.close()
            serial_conn = None
            time.sleep(5)
        except Exception as e:
            st.session_state.error_message = f"Error in serial listener: {e}"
            print(f"Error in serial listener: {e}")
            traceback.print_exc()
            time.sleep(2)
        # Small sleep to prevent high CPU usage if not waiting on serial
        time.sleep(0.1)


def yolo_tracker_thread_func(dwell_q):
    """Runs the YOLO tracker and puts average dwell time into the queue."""
    st.session_state.status_message = "YOLO Tracker: Starting..."
    print("YOLO tracker thread starting.")
    try:
        # Ensure model path is correct relative to this script's location (esp32/)
        tracker = YoloTracker(
            model_path=YOLO_MODEL_PATH,
            video_source=VIDEO_SOURCE,
            show_video=SHOW_YOLO_VIDEO, # Should be False
            dwell_queue=dwell_q, # Use the shared queue
            avg_dwell_period_minutes=AVG_DWELL_PERIOD_MINUTES,
            output_csv=YOLO_OUTPUT_CSV
        )
        st.session_state.status_message = "YOLO Tracker: Running..."
        tracker.run() # This blocks until stopped or error
    except Exception as e:
        st.session_state.error_message = f"Error running YOLO tracker: {e}"
        print(f"Error initializing or running YOLO tracker: {e}")
        traceback.print_exc()
    finally:
        st.session_state.status_message = "YOLO Tracker: Stopped."
        print("YOLO tracker thread finished.")


# --- Data Loading ---
def initialize_data_st():
    """Loads data using imported functions and stores in session state."""
    if not st.session_state.data_loaded:
        st.session_state.status_message = "Loading initial data..."
        st.info("Loading Sales Data...")
        # Need to manage the global state used by store_monitor functions
        if store_monitor.load_sales_data():
             st.session_state.sales_df = store_monitor.sales_df.copy()
             st.success("Sales data loaded.")
        else:
             st.error("Failed to load sales data.")
             st.stop()

        st.info("Loading RFID Mapping...")
        if store_monitor.load_rfid_mapping():
             st.session_state.rfid_map_df = store_monitor.rfid_map_df.copy()
             st.success("RFID mapping loaded.")
        else:
             st.warning("Failed to load RFID mapping.")
             st.session_state.rfid_map_df = store_monitor.rfid_map_df.copy() # Store empty df

        st.session_state.data_loaded = True
        st.session_state.status_message = "Data loaded. Ready."
        st.rerun()

# --- Main App Logic ---
st.set_page_config(page_title="Live Store Monitor", layout="wide")
st.title("🛒 Live Store Monitor & Foot Fall Analysis")

# Initialize data if needed
initialize_data_st()

# Start background threads if not already running
if not st.session_state.serial_thread_running:
    st.info("Starting Serial Listener Thread...")
    serial_thread = threading.Thread(target=serial_listener_thread_func, args=(rfid_queue,), daemon=True)
    serial_thread.start()
    st.session_state.serial_thread_running = True
    time.sleep(1) # Give thread time to start
    st.rerun()

if not st.session_state.yolo_thread_running:
    st.info("Starting YOLO Tracker Thread...")
    yolo_thread = threading.Thread(target=yolo_tracker_thread_func, args=(dwell_queue,), daemon=True)
    yolo_thread.start()
    st.session_state.yolo_thread_running = True
    time.sleep(5) # Give YOLO more time to initialize
    st.rerun()


# --- Display Area ---
st.sidebar.header("Status")
st.sidebar.info(st.session_state.status_message)
if st.session_state.error_message:
    st.sidebar.error(st.session_state.error_message)

st.sidebar.header("RFID Mapping")
st.sidebar.dataframe(st.session_state.rfid_map_df, height=200)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Live Metrics")
    st.metric(label="Average Customer Dwell Time", value=f"{st.session_state.last_dwell_time:.1f} s")

with col2:
    st.subheader("🏷️ Last Scanned Item")
    st.text(f"RFID UID: {st.session_state.last_rfid}")
    st.text(f"Stock Code: {st.session_state.last_stock_code}")
    st.text(f"Description: {st.session_state.last_description}")
    st.text(f"Scan Time: {st.session_state.last_scan_time}")

st.divider()
st.subheader("🔮 Prediction & Suggestion")
col3, col4 = st.columns([1, 2])
with col3:
     st.metric(label="Predicted Need (Tomorrow)", value=st.session_state.last_prediction)
with col4:
    st.markdown("**AI Sales Suggestion:**")
    st.info(st.session_state.last_suggestion)

# --- Process Queues and Update State ---
new_rfid_received = False
try:
    # Check RFID queue
    rfid_uid = rfid_queue.get_nowait()
    st.session_state.last_rfid = rfid_uid
    st.session_state.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.status_message = f"Processing RFID: {rfid_uid}"
    st.session_state.last_prediction = "Processing..." # Indicate processing
    st.session_state.last_suggestion = "Processing..."
    new_rfid_received = True
    print(f"Streamlit: Processing RFID {rfid_uid} from queue.")

except queue.Empty:
    pass # No new RFID tag

try:
    # Check Dwell time queue
    dwell_time = dwell_queue.get_nowait()
    st.session_state.last_dwell_time = dwell_time
    print(f"Streamlit: Updated dwell time to {dwell_time:.2f}s")
except queue.Empty:
    pass # No new dwell time

# --- Trigger Prediction/Suggestion if new RFID ---
if new_rfid_received:
    stock_code = None
    description = None
    # Check mapping using session state df
    with map_lock:
        mapping_entry = st.session_state.rfid_map_df[st.session_state.rfid_map_df['RFID_UID'] == st.session_state.last_rfid]

    if not mapping_entry.empty:
        stock_code = mapping_entry['StockCode'].iloc[0]
        description = mapping_entry['Description'].iloc[0]
        st.session_state.last_stock_code = stock_code
        st.session_state.last_description = description
        st.session_state.status_message = f"RFID {st.session_state.last_rfid} mapped to {stock_code}"
        print(f"RFID {st.session_state.last_rfid} mapped to {stock_code}")

        # --- Prediction ---
        print(f"Predicting for {stock_code} with dwell {st.session_state.last_dwell_time:.2f}s")
        # Ensure predict_stock_need uses session state data/cache
        store_monitor.sales_df = st.session_state.sales_df
        store_monitor.rfid_map_df = st.session_state.rfid_map_df
        store_monitor.trained_models_cache = st.session_state.trained_models_cache
        predicted_need = predict_stock_need(stock_code, st.session_state.last_dwell_time)
        st.session_state.trained_models_cache = store_monitor.trained_models_cache # Update cache back

        if predicted_need is not None:
            st.session_state.last_prediction = predicted_need
            print(f"Prediction: {predicted_need}")

            # --- Suggestion ---
            print("Generating suggestion...")
            avg_sales = get_average_sales(stock_code)
            if avg_sales is not None:
                st.session_state.last_suggestion = generate_suggestion(stock_code, predicted_need, avg_sales)
                print("Suggestion generated.")
            else:
                st.session_state.last_suggestion = "Could not calculate average sales."
                print("Could not get average sales.")
            st.session_state.status_message = f"Processed: {stock_code}"

        else:
            st.session_state.last_prediction = "Error"
            st.session_state.last_suggestion = "Prediction failed."
            st.session_state.status_message = f"Prediction failed for {stock_code}"
            print("Prediction failed.")

    else:
        # Handle unmapped RFID - potentially call find_best_unmapped_item and update map
        st.session_state.last_stock_code = "Not Mapped"
        st.session_state.last_description = "Not Mapped"
        st.session_state.last_prediction = "N/A"
        st.session_state.last_suggestion = "RFID not mapped. Scan item to map."
        st.session_state.status_message = f"RFID {st.session_state.last_rfid} not mapped."
        print(f"RFID {st.session_state.last_rfid} not mapped.")
        # Add logic here if you want auto-mapping on the fly

    st.rerun() # Rerun to show updated prediction/suggestion

# --- Auto-refresh ---
# Periodically rerun to check queues even if no interaction happened
# Adjust sleep time as needed
time.sleep(2) # Check queues every 2 seconds
st.rerun()