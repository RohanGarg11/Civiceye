import cv2
import threading
import time
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- SHARED MEMORY PIPELINE ---
shared_data = {
    "latest_frame": None,        
    "garbage_boxes": [],         # Stores target coordinates from SAHI
    "sniper_working": False,
    "new_data_ready": False # Flag to indicate new data from SAHI
}

def sniper_thread(weights_path):
    """BACKGROUND THREAD: The SAHI Heavy Lifter"""
    print("[SNIPER] Booting SAHI Micro-Engine...")
    
    micro_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=weights_path,
        confidence_threshold=0.15, 
        device="cuda:0"  
    )
    
    print("[SNIPER] Online. Hunting for targets.")
    
    while True:
        if shared_data["latest_frame"] is None or shared_data["sniper_working"]:
            time.sleep(0.05)
            continue
            
        shared_data["sniper_working"] = True
        
        # 1. COLOR SPACE CORRECTION (Fixes the blind AI)
        bgr_frame = shared_data["latest_frame"].copy()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        try:
            # 2. MICRO-SLICES (Fixes the pixel crushing)
            micro_results = get_sliced_prediction(
                rgb_frame, micro_model,
                slice_height=256, slice_width=256,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type="NMS", postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1, verbose=0 
            )
            
            new_garbage = []
            for pred in micro_results.object_prediction_list:
                if pred.category.id == 1: # Garbage
                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    conf = pred.score.value
                    new_garbage.append({"target": [x1, y1, x2, y2], "conf": conf, "ttl": 20})
                    
            shared_data["garbage_boxes"] = new_garbage
            shared_data["new_data_ready"] = True # Signal that new data is ready for the main thread
            
        except Exception as e:
            shared_data["garbage_boxes"] = [] # Clear boxes on error to prevent stale data
            shared_data["new_data_ready"] = True # Still signal to clear any existing boxes
        finally:
            shared_data["sniper_working"] = False

def run_tracked_pipeline():
    print("--- INITIATING BYTETRACK HYBRID PIPELINE ---")
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt"
    
    threading.Thread(target=sniper_thread, args=(weights_path,), daemon=True).start()
    macro_model = YOLO(weights_path)

    cap = cv2.VideoCapture(0) # Change to 1 if using iVCam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Active boxes currently being drawn on screen (used for smooth gliding)
    display_garbage = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        if not shared_data["sniper_working"]:
            shared_data["latest_frame"] = frame.copy()

        # --- PHASE A: Macro Pass (ByteTrack for Dustbins) ---
        # .track() activates ByteTrack to cure amnesia on large objects
        macro_results = macro_model.track(
            source=frame, 
            conf=0.30, 
            imgsz = 1280,
            classes=[0], 
            persist=True, 
            tracker="bytetrack.yaml", 
            stream=True, 
            half=False, 
            verbose=False
        )
        
        for r in macro_results:
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                    cv2.putText(frame, f"Dustbin #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- PHASE B: The Micro Pass (Smooth Glide Buffer) ---
        
        # Only process targets and reset lifespans IF the data is fresh from SAHI
        if shared_data["new_data_ready"]:
            sahi_targets = shared_data["garbage_boxes"]
            shared_data["new_data_ready"] = False # Consume the flag so we don't read it again
            
            alive_display = []
            for target in sahi_targets:
                tx1, ty1, tx2, ty2 = target["target"]
                conf = target["conf"]
                
                matched = False
                for d_box in display_garbage:
                    dx1, dy1, dx2, dy2 = d_box["current"]
                    # If centers are close, it's the same moving object
                    if abs(tx1 - dx1) < 100 and abs(ty1 - dy1) < 100:
                        d_box["destination"] = [tx1, ty1, tx2, ty2]
                        d_box["conf"] = conf
                        d_box["ttl"] = 20  # ONLY reset TTL on fresh data
                        matched = True
                        alive_display.append(d_box)
                        break
                        
                if not matched:
                    # Brand new garbage spotted
                    alive_display.append({
                        "current": [tx1, ty1, tx2, ty2], 
                        "destination": [tx1, ty1, tx2, ty2],
                        "conf": conf, 
                        "ttl": 20
                    })
            
            # Drain life from boxes SAHI didn't see in this fresh pass
            for d_box in display_garbage:
                if d_box not in alive_display:
                    d_box["ttl"] -= 1
                    if d_box["ttl"] > 0:
                        alive_display.append(d_box)
            
            display_garbage = alive_display

        # If no fresh data, just glide the boxes and DRAIN their life constantly
        else:
            alive_display = []
            for d_box in display_garbage:
                cx1, cy1, cx2, cy2 = d_box["current"]
                dx1, dy1, dx2, dy2 = d_box["destination"]
                
                # Glide 30% closer to destination
                d_box["current"] = [
                    cx1 + (dx1 - cx1) * 0.3,
                    cy1 + (dy1 - cy1) * 0.3,
                    cx2 + (dx2 - cx2) * 0.3,
                    cy2 + (dy2 - cy2) * 0.3
                ]
                
                d_box["ttl"] -= 1 # Constantly age the box
                if d_box["ttl"] > 0:
                    alive_display.append(d_box)
                    
            display_garbage = alive_display

        # Draw the garbage boxes
        for box in display_garbage:
            x1, y1, x2, y2 = map(int, box["current"])
            conf = box["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Garbage {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # UI
        status_text = "Sniper: SCANNING..." if shared_data["sniper_working"] else "Sniper: READY"
        status_color = (0, 165, 255) if shared_data["sniper_working"] else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("CivicEye Tracked Hybrid", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracked_pipeline()