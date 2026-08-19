import cv2
from ultralytics import YOLO

def run_live_edge_inference():
    print("--- INITIATING ZERO-LATENCY EDGE FEED ---")
    
    # 1. Load your downloaded production brain
    # (Ensure best.pt is in the same folder as this script on your Windows machine)
    try:
        model = YOLO("runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load weights. Is best.pt in this folder?\n{e}")
        return

    # 2. Tap into the local Windows Webcam (Port 0)
    cap = cv2.VideoCapture(0)
    
    # Force the webcam hardware to push a stable 720p feed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not connect to the webcam.")
        return

    print("--- LIVE FEED SECURED. Press 'q' to terminate. ---")

    # 3. The Real-Time Event Loop
    while True:
        success, frame = cap.read()
        if not success:
            print("Feed interrupted.")
            break

        # 4. The Zero-Latency Prediction Parameters
        # stream=True: Prevents RAM overflow during continuous video
        # half=True: Uses FP16 precision, doubling inference speed on modern GPUs
        # imgsz=640: Drops resolution from 1024 to 640 to guarantee 30+ FPS
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            stream=True,      
            half=True,        
            verbose=False     # Silences the terminal so it doesn't spam text every frame
        )

        # 5. Render the bounding boxes
        for r in results:
            annotated_frame = r.plot()

        # 6. Push the frame to your Windows screen
        cv2.imshow("CivicEye Live Deployment Test", annotated_frame)

        # 7. Kill switch
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Graceful shutdown
    cap.release()
    cv2.destroyAllWindows()
    print("--- FEED TERMINATED ---")

if __name__ == "__main__":
    run_live_edge_inference()