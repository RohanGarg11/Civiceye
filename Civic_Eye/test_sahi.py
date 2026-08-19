from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

def run_sahi_magnifying_glass():
    print("--- INITIATING SAHI SLICED INFERENCE (CORE API) ---")
    
    weights_path = "runs/detect/CivicEye_Vision_Audit/yolo11_quality_run/weights/best.pt"
    test_images_folder = "/home/faculty1/civiceye/rapidtest/test/images"
    output_directory = "CivicEye_Testing/sahi_vision_test"
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # 1. Load the model explicitly (Bypassing the wrapper bug)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weights_path,
        confidence_threshold=0.25,  # The Timidity Dial
        device="cuda:0"             # Ignite the RTX 2000
    )
    
    # 2. Map the dataset
    image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images. Igniting Tensor Cores...")
    
    # 3. Feed the images through the Slicer manually
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(test_images_folder, image_name)
        
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=416,
            slice_width=416,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",          # Tell SAHI to use Non-Maximum Suppression
            postprocess_match_metric="IOU",  # Use Intersection-Over-Union math
            postprocess_match_threshold=0.1  # If boxes overlap more than 20%, delete the weaker one
        )
        
        # Save the stitched image with bounding boxes
        # (Stripping the extension so SAHI doesn't save it as "image.jpg.png")
        clean_name = os.path.splitext(image_name)[0]
        result.export_visuals(export_dir=output_directory, file_name=clean_name)
        
        # Real-time terminal tracking
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"Processed {i + 1}/{len(image_files)} images...")
            
    print("--- SAHI TEST COMPLETE ---")
    print(f"Open {output_directory} to view the enhanced vision.")

if __name__ == "__main__":
    run_sahi_magnifying_glass()