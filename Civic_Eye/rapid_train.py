from ultralytics import YOLO
import multiprocessing

def validate_dataset_quality():
    print("--- INITIATING VISION DATASET QUALITY TEST (YOLO11) ---")
    
    # Upgraded to YOLO11 Medium. 
    # It will automatically download the new yolo11m.pt weights on the first run.
    model = YOLO('yolo11m.pt') 
    
    # CRITICAL: Path to your data.yaml on the Linux server
    dataset_yaml = "/home/faculty1/civiceye/rapidtest/data.yaml"
    
    # The Validation Training Protocol
    results = model.train(
        data=dataset_yaml,
        epochs=50,             # 50 epochs to expose bad data labels
        patience=15,            
        imgsz=1024,            # High-res to test distance detection
        batch=8,               # Hard-capped to 8 to protect your 16GB VRAM
        device=0,               
        workers=4,              
        
        # --- Auditing Tools ---
        plots=True,            # Auto-generates Confusion Matrices
        save_conf=True,        # Saves the confidence scores
        
        project='CivicEye_Vision_Audit',
        name='yolo11_quality_run'
    )
    
    print("--- DATASET VALIDATION COMPLETE ---")
    print("Open the CivicEye_Vision_Audit/yolo11_quality_run folder to view your accuracy graphs.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    validate_dataset_quality()