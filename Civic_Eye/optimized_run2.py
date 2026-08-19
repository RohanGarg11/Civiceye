import cv2
import threading
import time
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- SHARED MEMORY PIPELINE ---
shared_data = {
    "latest_frame": None,        
    "garbage_boxes": [],         # Stores [x1, y1, x2, y2, conf, time_to_live]
    "sniper_working": False      
}

def sniper_thread(weights_path):
    """BACKGROUND THREAD: The SAHI Heavy Lifter"""
    print("[SNIPER] Booting SAHI Micro-Engine...")
    
    micro_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weights_path,
        confidence_threshold=0.20, # Balanced confidence
        device="cuda:0"  
    )
    
    print("[SNIPER] Online. Hunting for targets.")
    
    while True:
        if shared_data["latest_frame"] is None or shared_data["sniper_working"]:
            time.sleep(0.05)
            continue
            
        shared_data["sniper_working"] = True
        
        # --- CRITICAL FIX: COLOR SPACE CONVERSION ---
        # OpenCV uses BGR. YOLO/SAHI were trained on RGB. We MUST convert it.
        bgr_frame = shared_data["latest_frame"].copy()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Feed the color-corrected RGB frame to SAHI
            micro_results = get_sliced_prediction(
                rgb_frame, micro_model,
                slice_height=256, slice_width=256,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type="NMS", postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1, verbose=0 
            )
            
            # Store new boxes with a "Time-To-Live" (TTL) of 15 frames
            new_garbage = []
            for pred in micro_results.object_prediction_list:
                if pred.category.id == 1: # Garbage
                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    conf = pred.score.value
                    # Append coordinates, confidence, and 15 frames of life
                    new_garbage.append([x1, y1, x2, y2, conf, 15]) 
                    
            shared_data["garbage_boxes"] = new_garbage
            
        except Exception as e:
            print(f"[SNIPER ERROR]: {e}")
        finally:
            shared_data["sniper_working"] = False

def run_stabilized_pipeline():
    print("--- INITIATING STABILIZED HYBRID EDGE PIPELINE ---")
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt"
    
    # 1. Start Sniper
    threading.Thread(target=sniper_thread, args=(weights_path,), daemon=True).start()
    
    # 2. Load Macro Brain (Dustbins)
    macro_model = YOLO(weights_path)

    cap = cv2.VideoCapture(0) # Or change to your iVCam index (e.g., 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Feed the sniper the BGR frame (it will convert it to RGB internally)
        if not shared_data["sniper_working"]:
            shared_data["latest_frame"] = frame.copy()

        # --- PHASE A: Macro Pass (Dustbins) ---
        # YOLO auto-converts BGR to RGB, so passing raw 'frame' here is safe
        macro_results = macro_model.predict(source=frame, conf=0.40, classes=[0], stream=True, half=False, verbose=False)
        
        for r in macro_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                cv2.putText(frame, "Dustbin", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- PHASE B: The Persistence Buffer (Garbage) ---
        # Draw the SAHI boxes, and slowly count down their TTL so they don't vanish instantly
        alive_garbage = []
        for box_data in shared_data["garbage_boxes"]:
            x1, y1, x2, y2, conf, time_to_live = box_data
            
            if time_to_live > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"Garbage {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                box_data[5] -= 1 # Drain 1 frame of life
                alive_garbage.append(box_data)
                
        shared_data["garbage_boxes"] = alive_garbage

        # --- UI Overlay ---
        status_text = "Sniper: SCANNING..." if shared_data["sniper_working"] else "Sniper: READY"
        status_color = (0, 165, 255) if shared_data["sniper_working"] else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("CivicEye Stabilized", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stabilized_pipeline()