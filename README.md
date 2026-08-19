# CivicEye: Autonomous Edge Data & Stream Processing Pipeline

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO26-yellow)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Edge%2FWeb-ff69b4)

**CivicEye** is an end-to-end computer vision data engineering and stream processing pipeline designed to ingest, process, and structure high-throughput edge CCTV telemetry in real time. Built for distributed edge environments, it orchestrates continuous video stream ingestion, metadata extraction, dynamic image slicing, and downstream analytical data publishing into a centralized operational store.

---

## System Architecture & Data Flow

CivicEye bridges edge stream ingestion and operational data warehouses through a robust **Hybrid Data Architecture**:

1. **Edge Stream Ingestion & Processing:** Ingests uncompressed 1080p RTSP/CCTV video feeds locally on Intel i7 nodes. Applies localized inference pipelines using compiled `.pt` weights trained via cloud GPU pipelines, ensuring low-latency feature extraction and data filtering directly at the edge source.
2. **Metadata Transformation & Event Dispatch:** The transformation layer cleans raw frame metadata, formats structured JSON event payloads (timestamps, bounding box coordinates, class labels, confidence scores), and dispatches them via REST/streaming APIs to upstream ingestion brokers.
3. **Analytical Store & Dashboard (MERN Layer):** Centralized document store and indexing layer that ingests real-time event payloads, manages temporal indexing, schema validation, and powers analytical dashboards for historical aggregation and monitoring.

---

## Core Data Engineering Features

### High-Resolution Stream Tiling & Data Slicing (SAHI)
Processing raw 1080p video feeds at scale often creates computational bottlenecks and loss of granular spatial data during downscaling. CivicEye implements a **SAHI-based dynamic frame-slicing pipeline**, partitioning high-resolution streams into micro-batches to preserve spatial feature resolution without exceeding edge memory limits.

### Dataset Engineering & Distribution Balancing
To mitigate class imbalance and false-positive skew in downstream analytical models, the data pipeline implements automated **Negative Example Injection**. By programmatically injecting 15% null background frames into the training/validation data split, the ingestion pipeline guarantees robust distribution calibration and eliminates domain-drift bias.

---

## Performance & Data Evaluation Metrics

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/35dce19f-b825-4e30-b820-2dc9a9bd60e7" /> \
<img width="1920" height="1108" alt="val_batch2_labels" src="https://github.com/user-attachments/assets/9041ee89-92b0-4512-99d9-9da8c4e626bc" /> \
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/026dcef1-0021-4f84-a429-dab122350de1" /> 

### Stream Ingestion & Processing Benchmarks
* **Processing Hardware:** Local Intel i7 CPU (Edge Compute Node)
* **Ingestion Resolution:** 1080p (Sliced via SAHI Pipeline)
* **Pipeline Latency:** 100 ms / frame (End-to-End Extraction)
* **Model Quality Metric (mAP@50):** <img width="489" height="612" alt="results" src="https://github.com/user-attachments/assets/0158115e-6e8a-459f-ba54-c81a717bf6d3" />

### Precision vs. Recall Distribution
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/c2d7a76f-81d7-4080-8bc8-c43a5cabdd1e" />

### Data Validation & Pipeline Batches
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

## Tech Stack

**Data & Stream Ingestion Pipeline:**
* Python
* YOLO26
* SAHI (Stream Slicing Engine)
* OpenCV / NumPy / Pandas (Array & Vector Transformation)

**Storage & Analytics Layer (In Progress):**
* MongoDB (NoSQL Document Store & Time-Series Collections)
* Express.js / Node.js (Data Ingestion API Layer)
* React.js (Telemetry & Operational Analytics Dashboard)
* REST / Webhook Protocols

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Meet2909/CivicEye.git](https://github.com/Meet2909/CivicEye.git)
cd CivicEye
