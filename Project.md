# Project: RFID & Dwell Time Based Stock Prediction System

## 1. Overview

This project aims to create a system that predicts the near-term stock requirement for retail items by correlating RFID tag scans with real-time customer dwell time information gathered via webcam analysis. When an RFID tag (associated with a specific product) is scanned, the system leverages historical sales data, recent customer dwell time near product areas (approximated by overall store dwell time), and machine learning (XGBoost) to predict the quantity of that item likely needed the next day.

The core idea is that combining the explicit interaction signal (RFID scan) with an implicit interest signal (customer dwell time) can lead to more accurate short-term demand forecasting compared to using historical sales alone.

## 2. System Architecture

The system comprises hardware components (ESP32, RFID reader, webcam) and a central Python application (`store_monitor.py`) running on a host computer. The application orchestrates data flow using concurrent threads:

*   **Hardware Interface:** An ESP32 microcontroller reads RFID tag UIDs using an RC522 module and transmits them over a serial (USB) connection to the host computer.
*   **Webcam Analysis:** A separate thread runs a YOLOv8 object detection model to track people in the webcam feed, calculate individual dwell times, and compute a rolling average dwell time for all currently tracked individuals over the last 5 minutes. This average dwell time is shared with the main application via a thread-safe queue.
*   **Serial Listener & Prediction:** Another thread listens to the serial port for incoming RFID UIDs.
    *   **Mapping:** If a UID is new, it's mapped to the highest-selling (by total historical quantity) currently unmapped item from the sales data. This mapping is stored persistently in `esp32/rfid_item_mapping.csv`.
    *   **Prediction:** If a UID is known (mapped to a `StockCode`), the system retrieves the latest average dwell time, prepares features (including historical lags and the live dwell time), trains an XGBoost model on-the-fly (using cached models for efficiency), and predicts the required quantity for the associated `StockCode` for the next day.

```mermaid
graph TD
    subgraph Hardware
        Webcam[Webcam] -->|Video Stream| HostPC[Host PC]
        RFID_Reader[RC522 RFID Reader] -->|SPI| ESP32[ESP32 Dev Board]
        ESP32 -->|Serial TX (UID)| Serial_Port[/dev/ttyUSB0] -->|USB| HostPC
    end

    subgraph HostPC [Host PC - Python Application]
        style HostPC fill:#lightblue,stroke:#333,stroke-width:2px
        MonitorScript[store_monitor.py]

        subgraph MonitorScript Threads
            YOLOThread[YOLO Tracker Thread (yolo_tracker.py)]
            SerialThread[Serial Listener Thread]
        end

        MonitorScript -->|Manages| YOLOThread
        MonitorScript -->|Manages| SerialThread

        YOLOThread -->|Reads| Webcam
        YOLOThread -->|Calculates & Puts| DwellQueue[(Shared Avg Dwell Queue)]

        SerialThread -->|Reads| Serial_Port
        SerialThread -->|Reads/Writes| MappingCSV[esp32/rfid_item_mapping.csv]
        SerialThread -->|Reads| SalesCSV[../retail_data_full_with_people_time.csv]
        SerialThread -->|Gets| DwellQueue
        SerialThread -- Known Tag -->|Triggers| XGBoostLogic[XGBoost Training & Prediction]
        SerialThread -- New Tag -->|Updates| MappingCSV

        XGBoostLogic -->|Reads| SalesCSV
        XGBoostLogic -->|Outputs| PredictionResult[Prediction Output (Console)]

    end

    subgraph Data & Models
        MappingCSV
        SalesCSV
        YOLOModel[YOLOv8 Model (e.g., best.pt)]
        XGBoostCache[(In-Memory Model Cache)]
    end

    YOLOThread -- Uses --> YOLOModel
    XGBoostLogic -- Uses/Updates --> XGBoostCache

```

## 3. Hardware Setup

### 3.1. Components

*   **ESP32 DevKit V1 (or similar):** Microcontroller responsible for reading the RFID module.
*   **RC522 RFID Reader Module:** Reads 13.56 MHz RFID tags (e.g., MiFare). Communicates via SPI.
*   **USB Webcam:** Provides the video feed for person detection and dwell time analysis. Connected to the host PC.
*   **Host PC:** Runs the main Python application (`store_monitor.py`). Requires Python environment and necessary libraries.
*   **RFID Tags:** Associated with products.

### 3.2. Connections

*### 3.2. Connections (Specific to Provided Board Image)

*   **ESP32 <-> RC522 (SPI Interface):**
    *   RC522 `SDA` (SS/CS) <-> ESP32 **`D5`** (Left side, 8th pin from top)
    *   RC522 `SCK` <-> ESP32 **`D18`** (Left side, 9th pin from top)
    *   RC522 `MOSI` <-> ESP32 **`D23`** (Left side, 15th pin from top)
    *   RC522 `MISO` <-> ESP32 **`D19`** (Left side, 10th pin from top)
    *   RC522 `RST` <-> ESP32 **`D4`** (Left side, 5th pin from top)
    *   RC522 `GND` <-> ESP32 **`GND`** (Left side, 2nd pin OR Right side, 2nd pin)
    *   RC522 `3.3V` <-> ESP32 **`3V3`** (Left side, 1st pin from top)
    *   RC522 `IRQ` -> *Not Connected*
*   **ESP32 <-> Host PC (Serial Interface):**
    *   Connect the ESP32 to the Host PC via its USB port. This provides power and establishes a serial communication channel (e.g., `/dev/ttyUSB0` on Linux, `COMx` on Windows).
*   **ESP32 <-> Host PC (Serial Interface):**
    *   Connect the ESP32 to the Host PC via its USB port. This provides power and establishes a serial communication channel (e.g., `/dev/ttyUSB0` on Linux, `COMx` on Windows).

## 4. Software Components (Located in `esp32/` directory)

### 4.1. `rfid_reader.ino` (Arduino Sketch for ESP32)

*   **Purpose:** Initializes the RC522 module via SPI and continuously scans for RFID tags.
*   **Functionality:**
    *   Includes `SPI.h` and `MFRC522.h` libraries.
    *   Defines SPI pins (`SS_PIN`, `RST_PIN`).
    *   Initializes Serial communication at 115200 baud.
    *   Initializes the SPI bus and the MFRC522 module.
    *   In the main loop (`loop()`):
        *   Checks for new RFID cards (`PICC_IsNewCardPresent()`).
        *   Selects a card and reads its serial number (`PICC_ReadCardSerial()`).
        *   Formats the Unique Identifier (UID) as a space-separated hexadecimal string (e.g., " 04 AB CD EF").
        *   Prints the UID to the Serial port, prefixed with "RFID_UID:" (e.g., `RFID_UID: 04 AB CD EF`).
        *   Halts the card (`PICC_HaltA()`) to allow reading other cards.
        *   Includes a short delay (`delay(1000)`) to prevent rapid re-reading of the same card.

### 4.2. `yolo_tracker.py` (Python Module)

*   **Purpose:** Encapsulates the YOLOv8-based person tracking and dwell time calculation logic. Designed to be run in a separate thread.
*   **Functionality:**
    *   Uses the `ultralytics` library for YOLOv8 object detection and tracking (`model.track()`).
    *   Configurable model path (`best.pt`, `yolov8n.pt`, etc.), video source (webcam index or file path), confidence threshold, and tracker type (`bytetrack.yaml`).
    *   Tracks objects belonging to the 'person' class (COCO index 0).
    *   Maintains a history (`track_history`) of when each person (track ID) was first and last seen.
    *   Maintains a dictionary (`active_tracks`) of tracks seen within the `avg_dwell_period`.
    *   Calculates the *current* dwell time for each tracked person (time since first seen).
    *   **Average Dwell Time Calculation:** Implements `get_average_dwell_time()` which calculates the mean dwell time (duration from first seen to last seen) of all tracks whose `last_seen_time` falls within the specified `avg_dwell_period` (e.g., last 5 minutes). Old tracks are pruned from `active_tracks`.
    *   **Data Sharing:** Accepts a `queue.Queue` object during initialization (`dwell_queue`). Periodically (e.g., every second), it calculates the average dwell time and `put`s the latest value into this queue (clearing any previous value first).
    *   Optionally displays the video feed with bounding boxes, track IDs, and current dwell times (`show_video=True`).
    *   Saves final aggregated dwell times per track ID to a CSV file upon completion/termination.
    *   Handles model loading relative path adjustments assuming it runs from within `esp32/`.

### 4.3. `store_monitor.py` (Main Python Application)

*   **Purpose:** Orchestrates the entire system, managing threads, data flow, mapping, and prediction.
*   **Functionality:**
    *   **Initialization:**
        *   Loads historical sales data using `preprocess_sales_data()` from `../retail_data_full_with_people_time.csv`.
        *   Loads the existing RFID-StockCode mapping from `rfid_item_mapping.csv`.
        *   Initializes the shared `dwell_time_queue`.
        *   Initializes an empty dictionary `trained_models_cache` to store XGBoost models in memory.
    *   **Threading:**
        *   Starts `yolo_tracker_task` in a daemon thread, passing the `dwell_time_queue`. This thread runs the `YoloTracker` class.
        *   Starts `serial_listener_task` in a daemon thread.
    *   **`serial_listener_task`:**
        *   Continuously attempts to connect to the specified `SERIAL_PORT`.
        *   Reads lines from the serial port.
        *   If a line starts with `RFID_UID:`, extracts the UID.
        *   Looks up the UID in the `rfid_map_df`.
        *   **Mapping Logic:** If the UID is new, calls `find_best_unmapped_item()` to get the `StockCode` and `Description` of the highest total quantity selling item not yet mapped. Adds the new mapping to `rfid_map_df` (using `pd.concat`) and saves the updated CSV using `save_rfid_mapping()`.
        *   **Prediction Trigger:** If the UID is mapped to a `StockCode`, it retrieves the latest average dwell time from `dwell_time_queue` (using `get_nowait()`, defaulting to 0 if empty). It then calls `predict_stock_need(stock_code, current_avg_dwell)`.
        *   Prints the prediction result or mapping actions to the console.
        *   Includes error handling for serial communication issues.
    *   **`predict_stock_need(stock_code, current_avg_dwell)`:**
        *   **Model Caching:** Checks if a valid (non-expired) model for the `stock_code` exists in `trained_models_cache`.
        *   **On-the-Fly Training:** If no valid cached model exists, it retrieves the product's historical data, creates lagged features using `create_lagged_features()`, trains a new XGBoost model using `train_xgboost_model()`, and stores the trained model and timestamp in the cache. Requires `MIN_HISTORICAL_DAYS_FOR_TRAINING`.
        *   **Feature Preparation:** Creates the feature vector for the *next* prediction step using the most recent `PREDICTION_LAGS` days of historical data.
        *   **Dwell Time Injection:** Replaces the value of the most recent people count lag feature (e.g., `Avg_People_Count_lag_1`) in the prepared feature vector with the `current_avg_dwell` value obtained from the queue.
        *   **Prediction:** Uses the trained/cached XGBoost model's `.predict()` method on the prepared feature vector to get the predicted quantity for the next day.
        *   Returns the rounded, non-negative predicted quantity.
    *   **Helper Functions:** Includes `load_sales_data`, `load_rfid_mapping`, `save_rfid_mapping`, `find_best_unmapped_item`, `create_lagged_features`, `train_xgboost_model`.
    *   **Main Loop:** Keeps the main thread alive while the daemon threads run, handling `KeyboardInterrupt` for graceful shutdown.

### 4.4. `rfid_item_mapping.csv` (Data File)

*   **Purpose:** Persistently stores the mapping between scanned RFID tag UIDs and corresponding product `StockCode` and `Description`.
*   **Format:** CSV file with columns: `RFID_UID`, `StockCode`, `Description`.

## 5. Data Sources

*   **`../retail_data_full_with_people_time.csv`:** The primary source of historical data. Located in the parent directory relative to the `esp32/` scripts. Expected columns:
    *   `InvoiceDate`: Timestamp of the transaction.
    *   `StockCode`: Unique identifier for the product.
    *   `Description`: Text description of the product.
    *   `Quantity`: Number of units sold (can be negative for returns).
    *   `People_Count`: Average number of people detected during the transaction period (presumably added externally).
    *   `Time_Spent`: Average time spent by people during the transaction period (presumably added externally).
    *   Other columns (like `CustomerID`, `Price`, `Country`) might exist but are not directly used in the core prediction logic shown here.

## 6. Prediction Methodology

*   **Model:** XGBoost (eXtreme Gradient Boosting) Regressor. Chosen for its performance on structured data and ability to handle complex relationships.
*   **Features:**
    *   Lagged historical values (default `lags=3`) of:
        *   `Quantity` sold daily.
        *   `Avg_People_Count` daily.
        *   `Avg_Time_Spent` daily.
    *   **Live Dwell Time:** The most recent average dwell time (over the last 5 minutes, obtained from `yolo_tracker.py`) is injected into the feature set, typically replacing the `Avg_People_Count_lag_1` feature before making a prediction. This assumes the model learns a relationship between recent customer presence/interest and next-day sales.
*   **Training:** Performed on-the-fly for a specific `StockCode` when a prediction is requested and no valid cached model exists. The model is trained on all available historical data for that product up to the point of prediction, using the lagged features.
*   **Caching:** Trained models are stored in memory (`trained_models_cache`) with a timestamp to avoid retraining for every prediction. Models are retrained if the cached version is older than `MODEL_CACHE_EXPIRY_SECONDS` (e.g., 1 hour).
*   **Target:** Predicts the `Quantity` needed for the next day.

## 7. Assumptions & Prerequisites

*   **Hardware:** ESP32, RC522, and Webcam are correctly connected and functional.
*   **ESP32 Flashed:** The ESP32 is flashed with the `esp32/rfid_reader.ino` sketch or equivalent firmware sending UIDs in the format "RFID_UID: [UID]".
*   **Serial Port:** The correct serial port for the ESP32 connection is identified and configured in `store_monitor.py` (`SERIAL_PORT`). Permissions might be needed on Linux (`sudo usermod -a -G dialout $USER`).
*   **Python Environment:** A Python environment (e.g., `.venv` activated via `source .venv/bin/activate`) is set up in the parent directory (`sales-predcition/`).
*   **Libraries:** Required Python libraries are installed in the environment: `pyserial`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `ultralytics`, `opencv-python`.
*   **Data Files:**
    *   `../retail_data_full_with_people_time.csv` exists and contains the necessary columns.
    *   `esp32/rfid_item_mapping.csv` exists (can be initially empty with just the header).
*   **YOLO Model:** The specified YOLO model file (e.g., `../best.pt`) exists relative to the `esp32/` directory, or is a standard model name (`yolov8n.pt`) that `ultralytics` can download.
*   **Webcam Access:** The system has permission to access the webcam specified by `VIDEO_SOURCE`.

## 8. Execution Flow

1.  Activate the Python environment (e.g., `source .venv/bin/activate`).
2.  Navigate to the `esp32/` directory in the terminal (`cd esp32`).
3.  Run the main application: `python store_monitor.py`.
4.  The script loads initial data (sales history, RFID map).
5.  The YOLO tracker thread starts, initializes the model, and begins processing the webcam feed, periodically updating the shared dwell time queue.
6.  The serial listener thread starts and attempts to connect to the ESP32.
7.  The ESP32 reads RFID tags and sends UIDs over serial.
8.  The serial listener receives a UID:
    *   If new: Finds the best unmapped item, updates `rfid_item_mapping.csv`, and prints the mapping.
    *   If known: Retrieves the latest average dwell time, triggers `predict_stock_need`.
9.  `predict_stock_need` checks the model cache, trains a model if necessary, prepares features (injecting live dwell time), predicts the next day's quantity, and returns the result.
10. The prediction result is printed to the console.
11. The process continues until manually stopped (Ctrl+C).