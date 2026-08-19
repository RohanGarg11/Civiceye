import cv2
from ultralytics import YOLO

def test_raw_yolo_vision():
    print("--- INITIATING PURE YOLO VISION TEST ---")
    
    # Load the RAW model (No SAHI wrapper)
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt" 
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        height, width, _ = frame.shape
        
        # --- THE MANUAL MAGNIFYING GLASS ---
        # We physically crop a 400x400 square from the direct center of the webcam
        center_x, center_y = width // 2, height // 2
        crop_size = 200 # 200 pixels in every direction from center
        
        y1 = max(0, center_y - crop_size)
        y2 = min(height, center_y + crop_size)
        x1 = max(0, center_x - crop_size)
        x2 = min(width, center_x + crop_size)
        
        # This is our zoomed-in "Slice"
        zoomed_crop = frame[y1:y2, x1:x2]
        
        # Run pure YOLO on the zoomed crop
        results = model.predict(source=zoomed_crop, conf=0.10, verbose=False)
        
        # Draw results on the crop
        for r in results:
            for box in r.boxes:
                # Class 1 is Garbage
                if int(box.cls[0]) == 1: 
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(zoomed_crop, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                    cv2.putText(zoomed_crop, f"Garbage {conf:.2f}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show the zoomed-in feed
        cv2.imshow("RAW YOLO VISION (Center Crop)", zoomed_crop)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_raw_yolo_vision()