from ultralytics import YOLO
import os

def run_live_fire_test():
    print("--- INITIATING LIVE FIRE VISION TEST ---")
    
    # 1. Load the specific "best" brain you just finished training
    weights_path = "runs/detect/CivicEye_Vision_Audit/yolo11_quality_run/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Cannot find weights at {weights_path}")
        return

    model = YOLO(weights_path)
    
    # 2. Point it at your unseen test dataset
    # (Assuming your dataset is here, update if necessary)
    test_images_folder = "/home/faculty1/civiceye/rapidtest/test/images"
    
    # 3. Execute the Inference Protocol
    results = model.predict(
        source=test_images_folder,
        conf=0.30,             # Only show boxes if it is 30%+ confident
        imgsz=1024,            # Match the training resolution
        save=True,             # Visually draw the boxes and save the image
        project='CivicEye_Testing',
        name='unseen_data_test'
    )
    
    print("--- LIVE FIRE TEST COMPLETE ---")
    print("Open the CivicEye_Testing/unseen_data_test folder to view the AI's vision.")

if __name__ == "__main__":
    run_live_fire_test()