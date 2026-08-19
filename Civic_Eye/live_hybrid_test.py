import cv2
import threading
import time
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- SHARED MEMORY PIPELINE ---
shared_data = {
    "latest_frame": None,        
    "garbage_boxes": [],         
    "sniper_working": False      
}

def sniper_thread(weights_path):
    """BACKGROUND THREAD: The SAHI Heavy Lifter (Garbage Only)"""
    print("[SNIPER] Booting SAHI Micro-Engine...")
    
    micro_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weights_path,
        confidence_threshold=0.25,
        device="cuda:0"  
    )
    
    print("[SNIPER] Online and waiting for targets.")
    
    while True:
        if shared_data["latest_frame"] is None or shared_data["sniper_working"]:
            time.sleep(0.05)
            continue
            
        shared_data["sniper_working"] = True
        frame_to_scan = shared_data["latest_frame"].copy()
        
        try:
            # 9-slice SAHI scan
            micro_results = get_sliced_prediction(
                frame_to_scan,
                micro_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1,
                verbose=0 
            )
            
            # Extract only Class 1 (Garbage)
            new_garbage_boxes = []
            for pred in micro_results.object_prediction_list:
                if pred.category.id == 1: 
                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    conf = pred.score.value
                    new_garbage_boxes.append((x1, y1, x2, y2, conf))
            
            shared_data["garbage_boxes"] = new_garbage_boxes
            
        except Exception as e:
            print(f"[SNIPER ERROR] {e}")
            
        finally:
            shared_data["sniper_working"] = False

def run_async_pipeline():
    print("--- INITIATING STABILIZED ASYNC HYBRID PIPELINE ---")
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt"
    
    # 1. Start Background Sniper
    sniper = threading.Thread(target=sniper_thread, args=(weights_path,), daemon=True)
    sniper.start()
    
    # 2. Load Macro Brain 
    print("[WATCHER] Booting YOLO Macro-Engine...")
    try:
        macro_model = YOLO(weights_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot find best.pt!\n{e}")
        return

    # 3. Tap Local Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- LIVE SECURED. Press 'q' to terminate. ---")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if not shared_data["sniper_working"]:
            shared_data["latest_frame"] = frame.copy()

        # --- PHASE A: Macro Pass (Dustbins Only) ---
        # CRITICAL FIX: half=False to stop the RTX 4050 FP16 glitch
        macro_results = macro_model.predict(source=frame, conf=0.40, classes=[0], stream=True, half=False, verbose=False)
        
        for r in macro_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # RED
                cv2.putText(frame, f"Dustbin {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- PHASE B: Micro Pass (Garbage Overlay) ---
        for (x1, y1, x2, y2, conf) in shared_data["garbage_boxes"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2) # CYAN
            cv2.putText(frame, f"Garbage {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- UI: Sniper Status Indicator ---
        status_text = "Sniper: SCANNING..." if shared_data["sniper_working"] else "Sniper: READY"
        status_color = (0, 165, 255) if shared_data["sniper_working"] else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("CivicEye Live Async Deployment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- PIPELINE TERMINATED ---")

if __name__ == "__main__":
    run_async_pipeline()