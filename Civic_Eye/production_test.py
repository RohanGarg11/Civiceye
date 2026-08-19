from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

def run_production_validation():
    print("--- INITIATING PRODUCTION SAHI VALIDATION ---")
    
    # 1. Load your newly forged 150-Epoch Production Weights
    weights_path = "runs/detect/CivicEye_Production/final_weights_v1/weights/best.pt"
    
    # CRITICAL: Replace this with the absolute path to your NEW test images
    test_images_folder = "Phase1/test/images"
    output_directory = "CivicEye_Testing/production_vision_test"
    
    os.makedirs(output_directory, exist_ok=True)
    
    # 2. Ignite the RTX 2000 with the tuned thresholds
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',            # Keeps SAHI using the Ultralytics engine
        model_path=weights_path,
        confidence_threshold=0.25,      # The sweet spot to catch transparent bags
        device="cuda:0"             
    )
    
    image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} unseen images. Commencing Sliced Inference...")
    
    # 3. The Slicing Protocol
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(test_images_folder, image_name)
        
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",          
            postprocess_match_metric="IOU",  
            postprocess_match_threshold=0.1  # Aggressively delete ghost boxes
        )
        
        clean_name = os.path.splitext(image_name)[0]
        result.export_visuals(export_dir=output_directory, file_name=clean_name)
        
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"Analyzed {i + 1}/{len(image_files)} images...")
            
    print("--- VALIDATION COMPLETE ---")
    print(f"Open {output_directory} to audit the final vision.")

if __name__ == "__main__":
    run_production_validation()