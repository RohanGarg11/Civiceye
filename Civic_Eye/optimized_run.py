import cv2
import threading
import time
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- SHARED MEMORY PIPELINE ---
shared_data = {
    "latest_frame": None,        
    "new_garbage_found": False,  # Flag to trigger the Handoff
    "garbage_bbox": None,        # The raw coordinates from SAHI
    "sniper_working": False      
}

def sniper_thread(weights_path):
    """BACKGROUND THREAD: The SAHI Heavy Lifter"""
    print("[SNIPER] Booting SAHI Micro-Engine...")
    
    micro_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weights_path,
        confidence_threshold=0.25, # Slightly higher confidence to avoid tracking shadows
        device="cuda:0"  
    )
    
    print("[SNIPER] Online. Hunting for targets.")
    
    while True:
        # Wait until the main thread gives us a frame, or wait if we are busy
        if shared_data["latest_frame"] is None or shared_data["sniper_working"]:
            time.sleep(0.05)
            continue
            
        shared_data["sniper_working"] = True
        frame_to_scan = shared_data["latest_frame"].copy()
        
        try:
            micro_results = get_sliced_prediction(
                frame_to_scan, micro_model,
                slice_height=256, slice_width=256,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type="NMS", postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1, verbose=0 
            )
            
            # Find the most confident piece of garbage
            best_garbage = None
            highest_conf = 0
            
            for pred in micro_results.object_prediction_list:
                if pred.category.id == 1 and pred.score.value > highest_conf: 
                    best_garbage = map(int, pred.bbox.to_xyxy())
                    highest_conf = pred.score.value
            
            # If we found garbage, trigger the handoff flag
            if best_garbage:
                x1, y1, x2, y2 = best_garbage
                # OpenCV trackers need (x, y, width, height), not (x1,y1,x2,y2)
                shared_data["garbage_bbox"] = (x1, y1, x2 - x1, y2 - y1)
                shared_data["new_garbage_found"] = True
            
        except Exception as e:
            pass
        finally:
            shared_data["sniper_working"] = False

def run_optimized_pipeline():
    print("--- INITIATING OPTIMIZED OPTICAL EDGE PIPELINE ---")
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt"
    
    # 1. Start Sniper
    threading.Thread(target=sniper_thread, args=(weights_path,), daemon=True).start()
    
    # 2. Load Macro Brain (No tracking, just fast prediction)
    macro_model = YOLO(weights_path)
    
    # 3. Initialize OpenCV Optical Tracker (KCF is very fast for edge devices)
    optical_tracker = None
    is_tracking_garbage = False

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Feed the sniper
        if not shared_data["sniper_working"]:
            shared_data["latest_frame"] = frame.copy()

        # --- PHASE A: Macro Pass (Dustbins - Stateless & Fast) ---
        macro_results = macro_model.predict(source=frame, conf=0.40, classes=[0], stream=True, half=False, verbose=False)
        
        for r in macro_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                cv2.putText(frame, "Dustbin", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- PHASE B: The Optical Handoff (Garbage) ---
        # If the Sniper just found new garbage, initialize the optical tracker on those pixels
        if shared_data["new_garbage_found"]:
            try:
                optical_tracker = cv2.TrackerKCF_create() # Fast, lightweight pixel tracker
            except AttributeError:
                optical_tracker = cv2.TrackerKCF_create() # Fallback for older OpenCV versions
            optical_tracker.init(frame, shared_data["garbage_bbox"])
            is_tracking_garbage = True
            tracker_lifespan = 60
            shared_data["new_garbage_found"] = False # Reset flag

        # If we have an active tracker, update its position based on pixel movement
        if is_tracking_garbage and optical_tracker is not None:
            success_track, bbox = optical_tracker.update(frame)
            if success_track and tracker_lifespan > 0:
                # Tracking succeeded, draw the box as it falls
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, "Garbage (Tracked)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                tracker_lifespan -= 1 # Drain the lifespan
            else:
                # Tracking lost (e.g., garbage left the screen or hit the ground)
                is_tracking_garbage = False
                optical_tracker = None

        # --- UI Overlay ---
        status_text = "Sniper: SCANNING..." if shared_data["sniper_working"] else "Sniper: READY"
        status_color = (0, 165, 255) if shared_data["sniper_working"] else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("CivicEye Optimized Setup", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_optimized_pipeline()