from ultralytics import YOLO
import multiprocessing

def train_production_model():
    print("--- INITIATING PRODUCTION VISION TRAINING (YOLO11) ---")
    
    # Starting fresh with the base YOLO11 Medium weights
    model = YOLO('yolo11m.pt') 
    
    # CRITICAL: Verify this path matches your transferred dataset
    dataset_yaml = "/home/faculty1/civiceye/Phase1/data.yaml"
    
    # The Production Training Protocol
    results = model.train(
        data=dataset_yaml,
        epochs=150,            
        patience=25,           
        imgsz=1024,                    
        batch=8,               
        device=0,               
        workers=4,              
        
        # --- Auditing Tools ---
        plots=True,            
        save_conf=True,        
        
        project='CivicEye_Production',
        name='final_weights_v1'
    )
    
    print("--- PRODUCTION TRAINING COMPLETE ---")
    print("Your hardened weights are saved in CivicEye_Production/final_weights_v1/weights/best.pt")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_production_model()