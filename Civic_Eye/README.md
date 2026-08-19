# CivicEye: Autonomous Edge Surveillance System

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO26-yellow)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Edge%2FWeb-ff69b4)

**CivicEye** is an enterprise-grade, autonomous computer vision pipeline designed to detect public offenses (such as littering) in real-time. Built for constrained edge environments, it features a specialized Python-based vision engine that natively triggers event logging and analytics into a full-stack dashboard.

---

## System Architecture

CivicEye bridges the gap between raw edge inference and actionable system-of-record logging through a **Hybrid AI Architecture**:

1. **The Vision Engine (Edge):** Processes 1080p CCTV feeds locally on Intel i7 CPUs. Instead of relying on cloud inference, the system uses compiled `.pt` weights trained on Kaggle Cloud GPUs, ensuring low latency and high privacy.
2. **The Intelligence Layer (Agentic Triggering):** Upon detecting a high-confidence offense, the vision engine acts as an autonomous agent, formatting the detection data (timestamps, bounding boxes, event types) and pushing it to the backend.
3. **The System of Record (MERN Dashboard):** A centralized command center that ingests the real-time API payloads, allowing administrators to review flagged events, monitor camera health, and analyze trends.

---

##  Core Technical Features :

###  Slicing Aided Hyper Inference (SAHI)
Detecting small objects (like discarded wrappers or bottles) in high-resolution 1080p feeds traditionally leads to pixel-blur degradation and feature loss during standard model downscaling. CivicEye implements **SAHI** to dynamically slice the inference grid, maintaining bounding box integrity and drastically improving small-object recall without requiring massive compute overhead.

###  Advanced Model Training & Dataset Curation
The core detection model utilizes **YOLO26**, optimized for rapid edge inference. To combat model hallucinations in "forced-choice" background environments, the dataset was curated using **Negative Example Injection**. By purposefully introducing 15% null images (backgrounds with no targets) into the training pipeline, the model's false-positive rate is severely reduced.

---

##  Performance & Results

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/35dce19f-b825-4e30-b820-2dc9a9bd60e7" /> \
<img width="1920" height="1108" alt="val_batch2_labels" src="https://github.com/user-attachments/assets/9041ee89-92b0-4512-99d9-9da8c4e626bc" /> \
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/026dcef1-0021-4f84-a429-dab122350de1" /> 

### Inference Benchmarks
* **Hardware:** Local Intel i7 CPU
* **Resolution:** 1080p (Sliced via SAHI)
* **Average Inference Time:** 100 ms / frame
* **mAP@50:** <img width="489" height="612" alt="results" src="https://github.com/user-attachments/assets/0158115e-6e8a-459f-ba54-c81a717bf6d3" />

### Precision vs. Recall
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/c2d7a76f-81d7-4080-8bc8-c43a5cabdd1e" />



### Real-World Detection Samples
<img width="1920" height="1108" alt="val_batch1_pred" src="https://github.com/user-attachments/assets/70291beb-8ea3-421b-b24a-9c18d3336340" /> \
<img width="1920" height="1108" alt="val_batch1_labels" src="https://github.com/user-attachments/assets/9364888f-5510-4a3a-a03a-cb46b71d518e" /> \
<img width="1920" height="876" alt="val_batch0_pred" src="https://github.com/user-attachments/assets/cd9b5a43-d64e-4000-aefa-3d91167ee2ad" /> \
<img width="1920" height="876" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/e763ab4e-4ed3-4442-88c6-f258d54be2fd" /> \
<img width="1920" height="1920" alt="train_batch1" src="https://github.com/user-attachments/assets/4378fcae-64cf-40eb-ac17-b78ac0a28293" /> \
<img width="1920" height="1108" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/8d0bb99d-c24d-4220-a11f-f6696a6be54d" /> \

---


<div align="center">
  <video src="https://github.com/user-attachments/assets/907823ce-e936-4982-99a0-82c8f6430371" width="800" muted playsinline></video>
</div>




---

##  Tech Stack

**Edge AI Pipeline:**
* Python
* YOLO26
* SAHI
* OpenCV / NumPy / Pandas

**Full-Stack Dashboard (being worked on) :**
* MongoDB
* Express.js
* React.js
* Node.js
* REST APIs

---

##  Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Meet2909/CivicEye.git](https://github.com/Meet2909/CivicEye.git)
cd CivicEye
```

### 2. Setup the Vision Environment
```bash
python -m venv edge_env
source edge_env/bin/activate  # On Windows use: edge_env\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Edge Inference Agent
```bash
# Execute the optimized run script on a target video or camera feed
python optimized_run.py --source data/test_video.mp4 --weights runs/train/weights/best.p
```

# Future Roadmap
As CivicEye evolves from a detection pipeline into a fully autonomous system, the following features are planned for upcoming iterations:

● Agentic Action Protocols: Integrating KOGO-style autonomous workflows where the system doesn't just log offenses, but drafts and sends automated alert reports to relevant maintenance or security teams via Slack/Email APIs.

● Multi-Class Offense Expansion: Extending the dataset to recognize complex behavioral offenses like vandalism, unauthorized access, or loitering using spatio-temporal tracking.

● Hardware Quantization: Exporting the YOLO26 weights to TensorRT (.engine) or ONNX formats to achieve ultra-low latency on edge TPUs (like NVIDIA Jetson or Google Coral).

● Multi-Camera Tracking (ReID): Implementing DeepSORT or ByteTrack to maintain consistent object IDs across overlapping CCTV feeds.

Developed by Meet Anand visit -  [LinkedIn](https://www.linkedin.com/in/meet-anand-693354320/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BMd7HHEoZSjy8dhSclpJIMQ%3D%3D)
