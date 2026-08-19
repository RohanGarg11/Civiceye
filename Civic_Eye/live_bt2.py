import cv2
import threading
import time
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================================
# --- SHARED MEMORY PIPELINE ---
# ==========================================================
shared_data = {
    "latest_frame": None,        
    "garbage_boxes": [],         
    "sniper_working": False,
    "new_data_ready": False      # The Freshness Handshake
}

def sniper_thread():
    """BACKGROUND THREAD: The Hyper-Optimized SAHI Micro-Engine"""
    print("[SNIPER] Booting Optimized SAHI Engine...")
    
    # Utilizing the optimized ONNX engine for the RTX 4050
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.onnx" 
    
    micro_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', # The version mismatch fix
        model_path=weights_path,
        confidence_threshold=0.45, # Strict gate to kill hallucinations
        device="cuda:0"  
    )
    
    print("[SNIPER] Online. Hunting for targets.")
    
    while True:
        if shared_data["latest_frame"] is None or shared_data["sniper_working"]:
            time.sleep(0.01) 
            continue
            
        shared_data["sniper_working"] = True
        
        # Color Space Fix (BGR to RGB)
        bgr_frame = shared_data["latest_frame"].copy()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # The Region of Interest (RoI) Crop (Saves 30% Compute Power)
        ceiling_cutoff = 240
        roi_frame = rgb_frame[ceiling_cutoff:720, 0:1280]
        
        try:
            # Lean Slicing Math
            micro_results = get_sliced_prediction(
                roi_frame, micro_model,
                slice_height=320, slice_width=320,  
                overlap_height_ratio=0.1, overlap_width_ratio=0.1, 
                postprocess_type="NMS", postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1, verbose=0 
            )
            
            new_garbage = []
            for pred in micro_results.object_prediction_list:
                if pred.category.id == 1: 
                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    
                    # Shift coordinates back down due to the ceiling crop
                    global_y1 = y1 + ceiling_cutoff
                    global_y2 = y2 + ceiling_cutoff
                    
                    conf = pred.score.value
                    new_garbage.append({"target": [x1, global_y1, x2, global_y2], "conf": conf})
                    
            shared_data["garbage_boxes"] = new_garbage
            shared_data["new_data_ready"] = True 
            
        except Exception as e:
            # Failsafe to prevent UI freezing on errors
            shared_data["garbage_boxes"] = []
            shared_data["new_data_ready"] = True 
        finally:
            shared_data["sniper_working"] = False


def run_tracked_pipeline():
    print("--- INITIATING CIVIC-EYE ENTERPRISE EDGE PIPELINE ---")
    
    # 1. Boot the Sniper
    threading.Thread(target=sniper_thread, daemon=True).start()
    
    # 2. Boot the Watcher (Standard YOLO runs flawlessly with ByteTrack on .pt)
    macro_weights = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt" 
    macro_model = YOLO(macro_weights)

    # 3. Secure the Camera
    cap = cv2.VideoCapture(0) # Change to your iVCam index if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    display_garbage = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Feed the Sniper
        if not shared_data["sniper_working"]:
            shared_data["latest_frame"] = frame.copy()

        # ==========================================================
        # --- PHASE A: Macro Pass (High-Res ByteTrack for Dustbins) ---
        # ==========================================================
        # imgsz=1280 forces full resolution to catch distant dustbins
        macro_results = macro_model.track(
            source=frame, conf=0.25, imgsz=1280, classes=[0], 
            persist=True, tracker="bytetrack.yaml", 
            stream=True, half=False, verbose=False
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

        # ==========================================================
        # --- PHASE B: The Micro Pass (Smooth Glide Buffer) ---
        # ==========================================================
        if shared_data["new_data_ready"]:
            sahi_targets = shared_data["garbage_boxes"]
            shared_data["new_data_ready"] = False 
            
            alive_display = []
            for target in sahi_targets:
                tx1, ty1, tx2, ty2 = target["target"]
                conf = target["conf"]
                
                matched = False
                for d_box in display_garbage:
                    dx1, dy1, dx2, dy2 = d_box["current"]
                    if abs(tx1 - dx1) < 100 and abs(ty1 - dy1) < 100:
                        d_box["destination"] = [tx1, ty1, tx2, ty2]
                        d_box["conf"] = conf
                        d_box["ttl"] = 20  
                        matched = True
                        alive_display.append(d_box)
                        break
                        
                if not matched:
                    alive_display.append({
                        "current": [tx1, ty1, tx2, ty2], 
                        "destination": [tx1, ty1, tx2, ty2],
                        "conf": conf, 
                        "ttl": 20
                    })
            
            for d_box in display_garbage:
                if d_box not in alive_display:
                    d_box["ttl"] -= 1
                    if d_box["ttl"] > 0:
                        alive_display.append(d_box)
            
            display_garbage = alive_display

        else:
            # Glide the boxes while SAHI computes the next frame
            alive_display = []
            for d_box in display_garbage:
                cx1, cy1, cx2, cy2 = d_box["current"]
                dx1, dy1, dx2, dy2 = d_box["destination"]
                
                d_box["current"] = [
                    cx1 + (dx1 - cx1) * 0.3,
                    cy1 + (dy1 - cy1) * 0.3,
                    cx2 + (dx2 - cx2) * 0.3,
                    cy2 + (dy2 - cy2) * 0.3
                ]
                
                d_box["ttl"] -= 1 
                if d_box["ttl"] > 0:
                    alive_display.append(d_box)
                    
            display_garbage = alive_display

        # Draw the garbage boxes
        for box in display_garbage:
            x1, y1, x2, y2 = map(int, box["current"])
            conf = box["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Garbage {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ==========================================================
        # --- UI OVERLAY & DISPLAY ---
        # ==========================================================
        status_text = "Sniper: SCANNING..." if shared_data["sniper_working"] else "Sniper: READY"
        status_color = (0, 165, 255) if shared_data["sniper_working"] else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("CivicEye Enterprise Edge", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracked_pipeline()