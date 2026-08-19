# Unified CivicEye Detection + Tracking Pipeline

## Problem Analysis

After analyzing all existing scripts, I've identified **three root causes** for the issues:

### Root Cause 1: Garbage Not Detected in SAHI Scripts

The macro YOLO pass in `optimized_run.py`, `live_bt.py`, and `live_bt2.py` filters to **`classes=[0]` (dustbin only)**. Garbage detection is delegated entirely to a background SAHI thread. However, several issues cripple the SAHI garbage path:

| File | SAHI Issue |
|---|---|
| `optimized_run.py` | Uses `model_type='yolov8'` (deprecated). Feeds raw BGR frame to SAHI (color mismatch). |
| `live_bt2.py` | Uses `.onnx` weights + crops top 240px off. High `conf=0.45` kills weak garbage detections. |
| `hybrid_live_test.py`, `live_hybrid_test.py` | Same `model_type='yolov8'` bug. No BGR→RGB conversion. |
| `optimized_run2.py` | Same `model_type='yolov8'` bug. |

Meanwhile in `model.py`, garbage detection works well because:
- It uses **raw YOLO `.predict()`** directly (no SAHI)
- It uses a **low confidence threshold (`conf=0.10`)**
- It crops a center region giving the model a "zoomed-in" view of small garbage

### Root Cause 2: SAHI Cannot Track Objects

SAHI is a **sliced inference** tool — it runs one-shot detections on image tiles. It has **no temporal state**, so it cannot assign persistent IDs across frames (i.e., tracking). Your existing workarounds (TTL buffers, glide interpolation, KCF handoff) are clever but fragile.

### Root Cause 3: Architecture Split

Dustbins use YOLO `.track()` with ByteTrack (works great), but garbage is stuck in the SAHI background thread without tracking. This creates an asymmetric pipeline where one class is tracked and the other isn't.

---

## Proposed Solution: Unified Dual-Pass YOLO Pipeline

> [!IMPORTANT]
> The key insight: **YOLO's `.track()` with ByteTrack already works flawlessly for dustbins.** The fix is to use `.track()` for **both classes simultaneously** in the main thread, plus add a secondary SAHI "sniper" scan specifically to catch **small/distant garbage** that standard YOLO misses.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN THREAD (30 FPS)                  │
│                                                         │
│  Webcam Frame                                           │
│      │                                                  │
│      ├──► YOLO .track() [ALL CLASSES: 0+1]              │
│      │    ByteTrack assigns persistent IDs               │
│      │    ├── Dustbin detections (RED boxes + ID)        │
│      │    └── Garbage detections (CYAN boxes + ID)      │
│      │                                                  │
│      └──► Feed frame to Sniper (if idle)                │
│                                                         │
│  Merge: YOLO tracked boxes + SAHI micro-detections      │
│  Deduplicate via IoU matching                           │
│  Draw + Display + Alarm Logic                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              SNIPER THREAD (Background, ~2-5 FPS)       │
│                                                         │
│  BGR→RGB conversion                                    │
│  SAHI sliced prediction (256×256 tiles, overlap 0.2)    │
│  Extract garbage detections (Class 1)                   │
│  Hand coordinates back to main thread                   │
│  ► Catches SMALL garbage that full-frame YOLO misses    │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **YOLO `.track()` handles BOTH classes** — removes the artificial split. ByteTrack gives persistent IDs to dustbins AND garbage simultaneously.
2. **SAHI is demoted to "small object booster"** — it only supplements garbage detections when YOLO misses tiny items. Its results are merged with a deduplication step (IoU-based).
3. **No more KCF/optical tracker hacks** — ByteTrack is a proper multi-object tracker built for this.
4. **Alarm system** — when garbage is detected near a dustbin (within a configurable radius) but not being placed in it, an alert is raised.

---

## Proposed Changes

### [NEW] [civiceye_unified.py](file:///c:/Users/Meet/Desktop/AI%20PROJECT/Phase%201%20Full/civiceye_unified.py)

A single, clean file containing:

#### 1. Configuration Block
- Weights path, camera index, resolution, confidence thresholds
- Alarm parameters (proximity radius, alarm cooldown)

#### 2. SAHI Sniper Thread (background)
- `model_type='ultralytics'` (fixes deprecated `yolov8` issue)
- BGR→RGB conversion before feeding to SAHI
- Slice size 256×256, overlap 0.2
- Low confidence threshold (`conf=0.15`) to catch faint garbage
- Thread-safe data handoff with freshness flag

#### 3. IoU Deduplication Function
- When SAHI finds garbage, compare against YOLO's existing tracked boxes
- If IoU > 0.3 with an existing box, skip (already tracked)
- If novel, draw as "Garbage (SAHI)" with a TTL buffer for persistence

#### 4. Main Pipeline Loop
- **Single YOLO `.track()` call** for classes `[0, 1]` (dustbin + garbage)
- ByteTrack with `persist=True` for continuous ID assignment
- Confidence: `0.20` for garbage, `0.30` for dustbin (class-aware post-filter)
- Merge SAHI detections via dedup
- Alarm logic: compute distance between garbage centroids and nearest dustbin centroid

#### 5. Alarm System
- Visual alarm: frame border flashes red + "⚠ LITTERING DETECTED" text
- Audio alarm: `winsound.Beep()` on Windows (non-blocking via a flag to avoid spam)
- Cooldown timer to prevent continuous alarm spam

#### 6. UI Overlay
- FPS counter
- Sniper status indicator
- Detection counts (dustbins / garbage)
- Alarm status indicator
- Tracking ID labels on all objects

---

## Open Questions

> [!IMPORTANT]  
> **Camera Index**: Your files use both `cv2.VideoCapture(0)` and `cv2.VideoCapture(1)`. Which camera index should I use? I'll default to `0` with a config variable at the top so you can easily change it.

> [!NOTE]
> **Alarm Sound**: I'll use `winsound.Beep()` for Windows audio alerts. If you want a custom alarm sound file (.wav) instead, let me know.

> [!NOTE]
> **Proximity Threshold**: For the "garbage near dustbin" alarm, I'll use 200 pixels as the default. Is there a specific real-world distance you want to calibrate to?

---

## Verification Plan

### Manual Verification
1. Run `python civiceye_unified.py` with webcam
2. Verify **dustbins** are detected with red boxes and persistent track IDs
3. Verify **garbage** is detected with cyan boxes and persistent track IDs
4. Place garbage near a dustbin to trigger the proximity alarm
5. Verify SAHI sniper status indicator toggles between SCANNING/READY
6. Verify no duplicate boxes (YOLO + SAHI overlap) on the same garbage item
7. Check FPS stays above 15 on your hardware
