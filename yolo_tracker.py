import cv2
import time
import os
from collections import defaultdict, deque
import argparse
import numpy as np
from ultralytics import YOLO
import csv
import traceback
import queue
import threading
from datetime import datetime, timedelta

class YoloTracker:
    """
    Tracks persons using YOLOv8, calculates dwell times, and provides
    the average dwell time over a specified recent period via a shared queue.
    """
    TARGET_CLASS_INDEX = 0 # COCO index for 'person'

    def __init__(self, model_path='best.pt', video_source='0', conf_threshold=0.4,
                 tracker_config='bytetrack.yaml', show_video=False,
                 output_csv='yolov8_dwell_times.csv', dwell_queue=None,
                 avg_dwell_period_minutes=5):
        """
        Initializes the tracker.

        Args:
            model_path (str): Path to the YOLOv8 model file.
            video_source (str): Path to video file or camera index.
            conf_threshold (float): Object detection confidence threshold.
            tracker_config (str): Tracker configuration file.
            show_video (bool): Whether to display the video output.
            output_csv (str): Path to save final dwell time results CSV.
            dwell_queue (queue.Queue): Thread-safe queue to put average dwell time.
            avg_dwell_period_minutes (int): The period (in minutes) over which to calculate avg dwell time.
        """
        self.model_path = model_path
        self.video_source_str = video_source
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.show_video = show_video
        self.output_csv = output_csv
        self.dwell_queue = dwell_queue
        self.avg_dwell_period = timedelta(minutes=avg_dwell_period_minutes)

        self.video_source = self._get_video_source()
        self.fps = self._get_fps()
        self.model = self._load_model()

        # Stores track_id -> {'first_seen_time': datetime, 'last_seen_time': datetime, 'total_frames': int}
        self.track_history = defaultdict(lambda: {'first_seen_time': None, 'last_seen_time': None, 'total_frames': 0})
        # Stores track_id -> last_seen_time for tracks currently active or seen recently
        self.active_tracks = {}
        self.frame_no = 0
        self.total_unique_ids = set()
        self.track_colors = {}
        self.last_avg_dwell_update_time = time.monotonic()

        # Lock for thread safety when accessing shared track data if needed (though queue is primary)
        self.lock = threading.Lock()


    def _get_video_source(self):
        """Determines if the video source is a webcam or file."""
        if self.video_source_str.isdigit():
            try:
                return int(self.video_source_str)
            except ValueError:
                print(f"Error: Invalid camera index '{self.video_source_str}'. Treating as file path.")
                return self.video_source_str
        return self.video_source_str

    def _get_fps(self):
        """Estimates FPS from video file or assumes default for webcam."""
        fps = 30.0 # Default FPS
        if isinstance(self.video_source, str): # It's a file path
            try:
                cap = cv2.VideoCapture(self.video_source)
                if cap.isOpened():
                    _fps = cap.get(cv2.CAP_PROP_FPS)
                    if _fps and _fps > 0:
                        fps = _fps
                        print(f"Video FPS: {fps:.2f}")
                    else:
                        print(f"Warning: Could not determine video FPS from file '{self.video_source}'. Assuming {fps} FPS.")
                else:
                     print(f"Warning: Could not open video file '{self.video_source}' to get FPS. Assuming {fps} FPS.")
                cap.release()
            except Exception as e:
                print(f"Warning: Error getting FPS for '{self.video_source}': {e}. Assuming {fps} FPS.")
        else: # It's a webcam index
            print(f"Using webcam. Assuming {fps} FPS for dwell time calculation.")
        return fps

    def _load_model(self):
        """Loads the YOLOv8 model."""
        print(f"Loading YOLOv8 model: {self.model_path}")
        try:
            model = YOLO(self.model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading YOLO model '{self.model_path}': {e}")
            print("Ensure the model file exists or internet connectivity is available for download.")
            raise # Re-raise the exception to stop execution if model fails

    def _get_color(self, track_id):
        """Generates a consistent color for a track ID."""
        if track_id not in self.track_colors:
            np.random.seed(track_id)
            self.track_colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
            np.random.seed(None)
        return self.track_colors[track_id]

    def get_average_dwell_time(self):
        """
        Calculates the average dwell time of tracks active within the
        defined recent period.
        """
        with self.lock: # Ensure thread safety if accessing shared data directly
            now = datetime.now()
            recent_tracks = []
            total_dwell_seconds = 0
            count = 0

            # Clean up old tracks from active_tracks first
            cutoff_time = now - self.avg_dwell_period - timedelta(seconds=10) # Add buffer
            current_active_ids = list(self.active_tracks.keys())
            for track_id in current_active_ids:
                 if self.active_tracks[track_id] < cutoff_time:
                     del self.active_tracks[track_id]


            # Calculate dwell for tracks seen recently
            for track_id, last_seen in self.active_tracks.items():
                if last_seen >= (now - self.avg_dwell_period):
                    if track_id in self.track_history and self.track_history[track_id]['first_seen_time']:
                        dwell_duration = last_seen - self.track_history[track_id]['first_seen_time']
                        total_dwell_seconds += dwell_duration.total_seconds()
                        count += 1

            if count == 0:
                return 0.0 # No active tracks recently
            else:
                return total_dwell_seconds / count

    def run(self):
        """Starts the tracking process."""
        print("Starting video processing with YOLOv8 tracking...")
        print(f"Tracking target class index: {self.TARGET_CLASS_INDEX} (Person)")
        print(f"Using tracker configuration: {self.tracker_config}")
        if self.show_video:
            print("Displaying output window. Press 'q' in the window to stop.")

        loop_start_time = time.time()

        try:
            results_generator = self.model.track(
                source=self.video_source,
                show=False, # Manual display handled below
                conf=self.conf_threshold,
                classes=[self.TARGET_CLASS_INDEX],
                persist=True,
                tracker=self.tracker_config,
                stream=True
            )

            for results in results_generator:
                current_time = datetime.now()
                frame_to_show = results.orig_img.copy() if self.show_video else None
                active_in_frame = set()

                if results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results.boxes.id.cpu().numpy().astype(int)

                    with self.lock: # Lock when modifying shared track data
                        for box, track_id in zip(boxes, track_ids):
                            self.total_unique_ids.add(track_id)
                            active_in_frame.add(track_id)
                            self.active_tracks[track_id] = current_time # Update last seen time

                            # Update history
                            if self.track_history[track_id]['first_seen_time'] is None:
                                self.track_history[track_id]['first_seen_time'] = current_time
                            self.track_history[track_id]['last_seen_time'] = current_time
                            self.track_history[track_id]['total_frames'] += 1

                            # --- Drawing (only if show_video is True) ---
                            if self.show_video:
                                x1, y1, x2, y2 = box
                                color = self._get_color(track_id)
                                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color, 2)
                                id_text = f"ID: {track_id}"
                                cv2.putText(frame_to_show, id_text, (x1, y1 - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                # Calculate current dwell time for display
                                current_dwell_sec = (current_time - self.track_history[track_id]['first_seen_time']).total_seconds()
                                dwell_text = f"Dwell: {current_dwell_sec:.1f}s"
                                cv2.putText(frame_to_show, dwell_text, (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow dwell text

                self.frame_no += 1

                # --- Update Average Dwell Time Queue ---
                # Update queue periodically (e.g., every second)
                current_mono_time = time.monotonic()
                if self.dwell_queue and (current_mono_time - self.last_avg_dwell_update_time >= 1.0):
                    avg_dwell = self.get_average_dwell_time()
                    try:
                        # Clear the queue before putting the new value if it's not empty
                        while not self.dwell_queue.empty():
                            self.dwell_queue.get_nowait()
                        self.dwell_queue.put(avg_dwell) # Put the latest average
                        # print(f"Debug: Put avg dwell {avg_dwell:.2f}s into queue.") # Optional debug
                    except queue.Full:
                        print("Warning: Dwell queue is full. Skipping update.")
                    except Exception as q_err:
                         print(f"Error updating dwell queue: {q_err}")
                    self.last_avg_dwell_update_time = current_mono_time


                # --- Display ---
                if self.show_video:
                    info_text = f"Frame: {self.frame_no}"
                    cv2.putText(frame_to_show, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("YOLOv8 Person Tracking", frame_to_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing stopped by user ('q' pressed).")
                        break

                # Print progress
                if self.frame_no % 100 == 0:
                    print(f"Processed frame {self.frame_no}...")

        except KeyboardInterrupt:
            print("\nProcessing stopped by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")
            traceback.print_exc()
        finally:
            loop_end_time = time.time()
            if self.show_video:
                cv2.destroyAllWindows()

            print("\nTracking finished or stopped.")
            total_time = loop_end_time - loop_start_time
            avg_fps_proc = self.frame_no / total_time if total_time > 0 else 0
            print(f"Total frames processed: {self.frame_no}")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Average processing FPS: {avg_fps_proc:.2f}")

            self._save_final_dwell_times()

    def _save_final_dwell_times(self):
        """Calculates final dwell times for all tracked IDs and saves to CSV."""
        print(f"\nTotal unique persons tracked: {len(self.total_unique_ids)}")
        final_results = []
        print(f"Calculating final dwell times...")

        with self.lock: # Access history safely
            for track_id in sorted(list(self.total_unique_ids)):
                data = self.track_history[track_id]
                if data['first_seen_time'] and data['last_seen_time']:
                    start_time = data['first_seen_time']
                    end_time = data['last_seen_time']
                    dwell_time_sec = (end_time - start_time).total_seconds()

                    # Ensure minimum dwell time if seen only briefly
                    if dwell_time_sec <= 0:
                         dwell_time_sec = 1.0 / self.fps if self.fps > 0 else 0.1

                    final_results.append({
                        'id': track_id,
                        'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'dwell_time_sec': round(dwell_time_sec, 2),
                        'total_frames': data['total_frames']
                    })

        print("\nFinal Dwell Time Results (ID, Start Time, End Time, Dwell Duration, Frames Seen):")
        if final_results:
            for result in final_results:
                 print(f"  ID: {result['id']:<4} Start: {result['start_time']} End: {result['end_time']} Dwell: {result['dwell_time_sec']:<7.2f}s ({result['total_frames']} frames)")
        else:
            print("  No persons were tracked.")

        if self.output_csv:
            try:
                print(f"\nSaving results to {self.output_csv}...")
                output_dir = os.path.dirname(self.output_csv)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created output directory: {output_dir}")

                with open(self.output_csv, 'w', newline='') as csvfile:
                    fieldnames = ['id', 'start_time', 'end_time', 'dwell_time_sec', 'total_frames']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(final_results)
                print("Results saved successfully.")
            except Exception as e:
                print(f"Error saving CSV file '{self.output_csv}': {e}")

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Person Tracking and Dwell Time using YOLOv8.')
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index.')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model file.') # Use standard yolov8n for testing
    parser.add_argument('--output_csv', type=str, default='yolov8_dwell_times_test.csv', help='Path to save dwell time results CSV.')
    parser.add_argument('--conf', type=float, default=0.4, help='Object detection confidence threshold.')
    parser.add_argument('--show', action='store_true', help='Display video output.')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker configuration file.')
    parser.add_argument('--dwell_period', type=int, default=5, help='Period in minutes for avg dwell time calculation.')

    args = parser.parse_args()

    # Create a dummy queue for testing
    test_queue = queue.Queue(maxsize=1)

    tracker = YoloTracker(
        model_path=args.model,
        video_source=args.video,
        conf_threshold=args.conf,
        tracker_config=args.tracker,
        show_video=args.show,
        output_csv=args.output_csv,
        dwell_queue=test_queue,
        avg_dwell_period_minutes=args.dwell_period
    )

    # Example of how to run in a thread and get data from queue (optional test)
    tracker_thread = threading.Thread(target=tracker.run, daemon=True)
    tracker_thread.start()

    # Main thread can periodically check the queue
    last_print_time = time.time()
    while tracker_thread.is_alive():
        try:
            # Check queue non-blockingly or with timeout
            avg_dwell = test_queue.get(timeout=0.1)
            if time.time() - last_print_time > 5: # Print every 5 seconds
                 print(f"[Main Thread Test] Latest Avg Dwell Time ({args.dwell_period} min): {avg_dwell:.2f} seconds")
                 last_print_time = time.time()
        except queue.Empty:
            pass # No new data yet
        except KeyboardInterrupt:
             print("[Main Thread Test] Stopping...")
             break
        time.sleep(0.05) # Small sleep to prevent busy-waiting

    print("[Main Thread Test] Tracker finished.")