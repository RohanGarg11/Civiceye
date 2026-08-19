"""
=======================================================================
  CivicEye Unified Pipeline — v1.0
  
  A single, production-ready live detection + tracking system that:
    1. Tracks BOTH dustbins AND garbage via YOLO ByteTrack
    2. Boosts small garbage detection via background SAHI sniper
    3. Deduplicates overlapping YOLO + SAHI detections (IoU)
    4. Raises alarms for:
       - Garbage detected with NO dustbin in the scene
       - Garbage detected NEAR a dustbin but not placed inside

    ONNX runtime bug solution misscalculation and drawing of random boxes on the screen. The fix was to ensure that the input frame is converted from BGR to RGB before being fed into the SAHI model, as the model was trained on RGB images. This conversion is critical for accurate predictions and prevents the erratic behavior caused by color misinterpretation. 
    fix -  hard mathematical gate in sniper thread to delete boxes in particular origin coordinates 
=======================================================================
"""

import cv2
import threading
import time
import math
import numpy as np
import winsound
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================================
# --- CONFIGURATION ---
# ==========================================================
CONFIG = {
    # Model
    "weights_path": "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt",
    
    # Camera
    "camera_index": 0,          # Change to 1 for iVCam / external camera
    "frame_width": 1280,
    "frame_height": 720,
    
    # YOLO Tracking (Main Thread)
    "yolo_imgsz": 1280,         # Full resolution for catching distant objects
    "yolo_conf": 0.20,          # Low threshold — we post-filter per class below
    "dustbin_min_conf": 0.15,   # Minimum confidence to accept a dustbin detection
    "garbage_min_conf": 0.45,   # Lower threshold for garbage (harder to detect)
    
    # SAHI Sniper (Background Thread)
    "sahi_conf": 0.20,          # Sensitive — catches faint garbage SAHI is meant for
    "sahi_slice_size": 256,     # Small slices to magnify tiny garbage
    "sahi_overlap": 0.2,        # 20% overlap between slices
    
    # Deduplication
    "iou_dedup_threshold": 0.25, # If SAHI box overlaps YOLO box by >25%, skip it
    
    # Alarm System
    "proximity_radius": 250,    # Pixels — garbage within this distance of a dustbin
    "alarm_cooldown_sec": 3.0,  # Minimum seconds between alarm beeps
    "alarm_freq_hz": 1800,      # Beep frequency
    "alarm_duration_ms": 400,   # Beep duration
    
    # SAHI Detection Persistence (TTL buffer for smooth display)
    "sahi_ttl_frames": 25,      # How many frames a SAHI-only detection persists
    "sahi_match_radius": 80,    # Pixel distance to consider same object across frames
}

# Class IDs from the training data
CLASS_DUSTBIN = 0
CLASS_GARBAGE = 1

# ==========================================================
# --- SHARED DATA (Thread-Safe Pipeline) ---
# ==========================================================
shared_data = {
    "latest_frame": None,
    "sahi_garbage": [],          # List of {"bbox": [x1,y1,x2,y2], "conf": float}
    "sniper_working": False,
    "new_data_ready": False,
}

# ==========================================================
# --- UTILITY FUNCTIONS ---
# ==========================================================

def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    if inter_area == 0:
        return 0.0
    
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    return inter_area / (area_a + area_b - inter_area)


def centroid(box):
    """Compute center point of a [x1, y1, x2, y2] box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def distance(pt_a, pt_b):
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2)


def deduplicate_sahi(sahi_boxes, yolo_boxes, iou_threshold):
    """
    Filter out SAHI detections that overlap with existing YOLO tracked boxes.
    Returns only the novel SAHI detections.
    """
    novel = []
    for s_box in sahi_boxes:
        is_duplicate = False
        for y_box in yolo_boxes:
            if compute_iou(s_box["bbox"], y_box) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            novel.append(s_box)
    return novel


# ==========================================================
# --- SAHI SNIPER THREAD ---
# ==========================================================

def sniper_thread():
    """Background thread: SAHI sliced inference for catching small garbage."""
    print("[SNIPER] Booting SAHI Micro-Engine...")
    
    micro_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',       # FIXED: was 'yolov8' (deprecated)
        model_path=CONFIG["weights_path"],
        confidence_threshold=CONFIG["sahi_conf"],
        device="cuda:0"
    )
    
    print("[SNIPER] Online. Scanning for micro-targets.")
    
    while True:
        # Wait for a frame and for the previous scan to finish
        if shared_data["latest_frame"] is None or shared_data["sniper_working"]:
            time.sleep(0.01)
            continue
        
        shared_data["sniper_working"] = True
        
        # CRITICAL FIX: Convert BGR (OpenCV) to RGB (what the model was trained on)
        bgr_frame = shared_data["latest_frame"].copy()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        try:
            micro_results = get_sliced_prediction(
                rgb_frame,
                micro_model,
                slice_height=CONFIG["sahi_slice_size"],
                slice_width=CONFIG["sahi_slice_size"],
                overlap_height_ratio=CONFIG["sahi_overlap"],
                overlap_width_ratio=CONFIG["sahi_overlap"],
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.1,
                verbose=0
            )
            
            new_garbage = []
            for pred in micro_results.object_prediction_list:
                if pred.category.id == CLASS_GARBAGE:
                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())

                    #the ghost box filter
                    #if onnx darws a ghost box at top origin delete it 
                    if x1 <= 5 and y1 <= 5:
                        continue 

                    conf = pred.score.value
                    new_garbage.append({"bbox": [x1, y1, x2, y2], "conf": conf})
            
            shared_data["sahi_garbage"] = new_garbage
            shared_data["new_data_ready"] = True
            
        except Exception as e:
            # On error, clear stale data
            shared_data["sahi_garbage"] = []
            shared_data["new_data_ready"] = True
        finally:
            shared_data["sniper_working"] = False


# ==========================================================
# --- ALARM THREAD ---
# ==========================================================

def alarm_beep():
    """Play alarm sound in a separate thread to avoid blocking the main loop."""
    try:
        winsound.Beep(CONFIG["alarm_freq_hz"], CONFIG["alarm_duration_ms"])
    except Exception:
        pass  # Silently ignore if sound fails


# ==========================================================
# --- MAIN PIPELINE ---
# ==========================================================

def run_civiceye():
    print("=" * 60)
    print("  CivicEye Unified Pipeline — Starting Up")
    print("=" * 60)
    
    # ---- 1. Boot the SAHI Sniper Thread ----
    threading.Thread(target=sniper_thread, daemon=True).start()
    
    # ---- 2. Load the Main YOLO Tracker ----
    print("[TRACKER] Loading YOLO model with ByteTrack...")
    model = YOLO(CONFIG["weights_path"])
    
    # ---- 3. Open Camera ----
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check camera_index in CONFIG.")
        return
    
    print(f"[CAMERA] Live feed secured at {CONFIG['frame_width']}x{CONFIG['frame_height']}.")
    print("Press 'q' to quit.\n")
    
    # ---- State Variables ----
    # SAHI persistence buffer (smooth display of micro-detections between scans)
    sahi_display_buffer = []
    
    # Alarm control
    last_alarm_time = 0.0
    alarm_active = False
    alarm_reason = ""
    
    # FPS tracker
    prev_frame_time = time.time()
    fps = 0.0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Feed interrupted.")
            break
        
        # --- FPS Calculation ---
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_frame_time, 0.001)
        prev_frame_time = current_time
        
        # --- Feed frame to Sniper (if idle) ---
        if not shared_data["sniper_working"]:
            shared_data["latest_frame"] = frame.copy()
        
        # ===========================================================
        # PHASE A: YOLO ByteTrack — Track ALL Classes (Dustbin + Garbage)
        # ===========================================================
        tracked_dustbins = []   # List of [x1, y1, x2, y2]
        tracked_garbage = []    # List of [x1, y1, x2, y2]
        dustbin_count = 0
        garbage_count = 0
        
        track_results = model.track(
            source=frame,
            conf=CONFIG["yolo_conf"],
            imgsz=CONFIG["yolo_imgsz"],
            classes=[CLASS_DUSTBIN, CLASS_GARBAGE],  # BOTH classes
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            half=False,
            verbose=False
        )
        
        for r in track_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            # Track IDs may not always be assigned (first frame, etc.)
            has_ids = r.boxes.id is not None
            track_ids = r.boxes.id.int().cpu().numpy() if has_ids else [None] * len(boxes)
            
            for box, cls_id, conf, track_id in zip(boxes, classes, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                if cls_id == CLASS_DUSTBIN and conf >= CONFIG["dustbin_min_conf"]:
                    # --- DUSTBIN (Red) ---
                    tracked_dustbins.append([x1, y1, x2, y2])
                    dustbin_count += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"Dustbin #{track_id}" if track_id is not None else "Dustbin"
                    label += f" {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                elif cls_id == CLASS_GARBAGE and conf >= CONFIG["garbage_min_conf"]:
                    # --- GARBAGE (Cyan/Yellow) ---
                    tracked_garbage.append([x1, y1, x2, y2])
                    garbage_count += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f"Garbage #{track_id}" if track_id is not None else "Garbage"
                    label += f" {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ===========================================================
        # PHASE B: SAHI Micro-Detection Merge + Deduplication
        # ===========================================================
        
        # Process fresh SAHI data when available
        if shared_data["new_data_ready"]:
            raw_sahi = shared_data["sahi_garbage"]
            shared_data["new_data_ready"] = False
            
            # Deduplicate: remove SAHI boxes that overlap with YOLO tracked boxes
            novel_sahi = deduplicate_sahi(
                raw_sahi, tracked_garbage, CONFIG["iou_dedup_threshold"]
            )
            
            # Build the new display buffer with matching + new detections
            new_buffer = []
            for detection in novel_sahi:
                det_center = centroid(detection["bbox"])
                
                # Check if this matches an existing buffered box (same object moving)
                matched = False
                for existing in sahi_display_buffer:
                    if distance(centroid(existing["bbox"]), det_center) < CONFIG["sahi_match_radius"]:
                        existing["bbox"] = detection["bbox"]
                        existing["conf"] = detection["conf"]
                        existing["ttl"] = CONFIG["sahi_ttl_frames"]
                        new_buffer.append(existing)
                        matched = True
                        break
                
                if not matched:
                    new_buffer.append({
                        "bbox": detection["bbox"],
                        "conf": detection["conf"],
                        "ttl": CONFIG["sahi_ttl_frames"]
                    })
            
            # Drain life from boxes SAHI didn't re-detect in this pass
            for existing in sahi_display_buffer:
                if existing not in new_buffer:
                    existing["ttl"] -= 1
                    if existing["ttl"] > 0:
                        new_buffer.append(existing)
            
            sahi_display_buffer = new_buffer
        
        else:
            # No fresh SAHI data — just drain TTL on existing buffer
            alive = []
            for existing in sahi_display_buffer:
                existing["ttl"] -= 1
                if existing["ttl"] > 0:
                    alive.append(existing)
            sahi_display_buffer = alive
        
        # Re-deduplicate buffer against current YOLO boxes (YOLO may have caught up)
        final_sahi = deduplicate_sahi(
            sahi_display_buffer, tracked_garbage, CONFIG["iou_dedup_threshold"]
        )
        
        # Draw SAHI-only detections
        sahi_garbage_count = 0
        for det in final_sahi:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["conf"]
            
            # Add to the garbage list for alarm calculations
            tracked_garbage.append([x1, y1, x2, y2])
            garbage_count += 1
            sahi_garbage_count += 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Garbage(S) {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        
        # ===========================================================
        # PHASE C: ALARM SYSTEM
        # ===========================================================
        alarm_active = False
        alarm_reason = ""
        
        if garbage_count > 0:
            if dustbin_count == 0:
                # --- ALARM TYPE 1: Garbage detected with NO dustbin in scene ---
                alarm_active = True
                alarm_reason = "GARBAGE DETECTED — NO DUSTBIN IN SIGHT!"
            
            else:
                # --- ALARM TYPE 2: Garbage near a dustbin but not inside it ---
                for g_box in tracked_garbage:
                    g_x1, g_y1, g_x2, g_y2 = g_box
                    g_center = centroid(g_box)
                    near_any_dustbin = False
                    
                    for d_box in tracked_dustbins:
                        d_x1, d_y1, d_x2, d_y2 = d_box
                        d_center = centroid(d_box)
                        dist = distance(g_center, d_center)
                        
                        # Check if garbage is NEAR the dustbin 
                        if dist < CONFIG["proximity_radius"]:
                            near_any_dustbin = True
                            
                            d_height = d_y2 - d_y1

                            #is garbage outside dustbins width

                            is_outside_width = g_x2 <  d_x1 or g_x1 > d_x2

                            # spatial physics check - garbage lower than the dustbins base
                            # floor zone if bottom of garbae is anywhere near in the bottom 30% it is on the floor 
                            floor_zone_y = d_y2 - (d_height * 0.3) 
                            is_on_floor = g_y2 > floor_zone_y           
                            
                            if is_on_floor and is_outside_width:
                                alarm_active = True
                                alarm_reason = "LITTERING NEAR DUSTBIN DETECTED!"
                                
                                # Draw proximity line
                                cv2.line(frame, 
                                         (int(g_center[0]), int(g_center[1])),
                                         (int(d_center[0]), int(d_center[1])),
                                         (0, 0, 255), 2)
                            
                            # Check if garbage overlaps with dustbin (being placed inside)
                            # overlap = compute_iou(g_box, d_box)
                            # if overlap < 0.3:
                            #     # Garbage is near but NOT inside the dustbin
                            #     alarm_active = True
                            #     alarm_reason = "LITTERING NEAR DUSTBIN DETECTED!"
                                
                            #     # Draw proximity line
                            #     cv2.line(frame, 
                            #              (int(g_center[0]), int(g_center[1])),
                            #              (int(d_center[0]), int(d_center[1])),
                            #              (0, 0, 255), 2)
                    
                    if not near_any_dustbin:
                        # Garbage is far from all dustbins — still littering
                        alarm_active = True
                        alarm_reason = "GARBAGE DUMPED AWAY FROM DUSTBIN!"
        
        # --- Trigger Alarm Sound (with cooldown) ---
        if alarm_active and (current_time - last_alarm_time) > CONFIG["alarm_cooldown_sec"]:
            last_alarm_time = current_time
            # Fire beep in a daemon thread so it doesn't freeze the video
            threading.Thread(target=alarm_beep, daemon=True).start()
        
        # --- Visual Alarm: Red border flash ---
        if alarm_active:
            # Draw thick red border around the entire frame
            h, w = frame.shape[:2]
            border = 8
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
            cv2.rectangle(frame, (border, border), (w - 1 - border, h - 1 - border), (0, 0, 255), border)
            
            # Alarm text banner
            banner_y = h - 50
            cv2.rectangle(frame, (0, banner_y - 40), (w, banner_y + 10), (0, 0, 180), -1)
            cv2.putText(frame, f"!! {alarm_reason} !!", 
                        (20, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # ===========================================================
        # PHASE D: UI OVERLAY
        # ===========================================================
        
        # --- Top-left: FPS and Sniper Status ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        sniper_text = "SAHI: SCANNING..." if shared_data["sniper_working"] else "SAHI: READY"
        sniper_color = (0, 165, 255) if shared_data["sniper_working"] else (0, 255, 0)
        cv2.putText(frame, sniper_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sniper_color, 2)
        
        # --- Top-right: Detection Counts ---
        count_x = CONFIG["frame_width"] - 300
        cv2.putText(frame, f"Dustbins: {dustbin_count}", (count_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Garbage: {garbage_count} (SAHI: {sahi_garbage_count})", (count_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # --- Alarm status indicator ---
        if alarm_active:
            cv2.circle(frame, (count_x - 30, 30), 12, (0, 0, 255), -1)  # Red dot
            cv2.putText(frame, "ALARM", (count_x - 30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (count_x - 30, 30), 12, (0, 200, 0), -1)  # Green dot
        
        # ===========================================================
        # DISPLAY
        # ===========================================================
        cv2.imshow("CivicEye Unified Pipeline", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Graceful shutdown
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- CivicEye Pipeline Terminated ---")


# ==========================================================
# --- ENTRY POINT ---
# ==========================================================
if __name__ == "__main__":
    run_civiceye()
