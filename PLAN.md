# Plan: Integrated RFID, Dwell Time, and Stock Prediction System

**Overall Goal:** Build a Python application (`store_monitor.py`) that runs two main processes concurrently:
1.  **Webcam Monitoring:** Uses the YOLO script to track people via webcam, calculate dwell times, and maintain a running average dwell time over the last 5 minutes.
2.  **RFID Monitoring:** Listens to a serial port for RFID tag IDs sent by the ESP32, maps new tags to best-selling items, and triggers stock predictions for known tags using the dwell time data and an XGBoost model.

**Detailed Plan:**

1.  **Refactor YOLO Script (`main.py` -> `yolo_tracker.py`):** Make it importable, add average dwell time calculation (last 5 mins), and implement data sharing (e.g., queue).
2.  **Create Orchestrator Script (`store_monitor.py`):**
    *   Use `threading` for concurrency.
    *   Implement `serial_listener_task` (reads serial, handles mapping, triggers prediction).
    *   Implement `yolo_tracker_task` (runs refactored tracker, updates shared dwell time).
    *   Implement `predict_stock_need` (uses XGBoost with historical data + current avg dwell time).
    *   Handle mapping logic (load/save `esp32/rfid_item_mapping.csv`, find best unmapped item based on total historical quantity).
    *   Load pre-trained XGBoost model.
    *   Manage threads and graceful shutdown.
3.  **Create Initial `esp32/rfid_item_mapping.csv`:** Header: `RFID_UID,StockCode,Description`.
4.  **Update `time_series.ipynb`:** Reflect final code, explain model saving/loading.

**System Architecture Diagram:**

```mermaid
graph TD
    subgraph Hardware
        Webcam[Webcam]
        RFID_Reader[RC522 RFID Reader] -->|SPI| ESP32[ESP32 Dev Board]
        ESP32 -->|Serial TX| Serial_Port[/dev/ttyUSB0]
    end

    subgraph Python App (store_monitor.py)
        style Python App fill:#lightblue,stroke:#333,stroke-width:2px
        Main[Main Thread] -->|Starts| Thread_YOLO[YOLO Tracker Thread]
        Main -->|Starts| Thread_Serial[Serial Listener Thread]

        Thread_YOLO -->|Reads Video| Webcam
        Thread_YOLO -->|Updates| SharedDwellTime[(Shared Avg Dwell Time)]
        Thread_YOLO -- Uses --> YoloTrackerCode[yolo_tracker.py Logic]

        Thread_Serial -->|Reads Serial| Serial_Port
        Thread_Serial -->|Reads/Writes| MappingCSV[esp32/rfid_item_mapping.csv]
        Thread_Serial -->|Reads| SalesCSV[retail_data_full_with_people_time.csv]
        Thread_Serial -- Known Tag -->|Triggers| PredictionFunc[Prediction Function (XGBoost)]
        Thread_Serial -- New Tag -->|Updates| MappingCSV

        PredictionFunc -->|Reads| SharedDwellTime
        PredictionFunc -->|Reads| SalesCSV
        PredictionFunc -- Uses --> XGBoostModel[Pre-trained XGBoost Model]
        PredictionFunc -->|Outputs| ConsoleOutput[Prediction Result (Console)]

    end

    subgraph Files & Data
        MappingCSV
        SalesCSV
        YoloTrackerCode
        XGBoostModel
    end

    subgraph Notebook
        TimeSeriesNB[time_series.ipynb] -- Reflects --> PredictionFunc
        TimeSeriesNB -- Explains Training --> XGBoostModel
    end