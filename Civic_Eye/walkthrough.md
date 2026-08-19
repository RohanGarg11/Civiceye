# CivicEye v2.0 — Bug Fixes & Alarm Redesign

## Changes Made to [civiceye_unified.py](file:///c:/Users/Meet/Desktop/AI%20PROJECT/Phase%201%20Full/civiceye_unified.py)

---

### 🔧 Fix 1: Ghost SAHI Boxes (Top-Left Blue Boxes)

The phantom boxes in the top-left corner were caused by SAHI creating tiny artifact detections at slice boundaries that persisted for too long (25 frames TTL).

**What changed:**
- Added **minimum area filter** (`sahi_min_area: 900` = 30×30 px) — boxes smaller than this are rejected
- Added **edge margin ghost detector** (`sahi_edge_margin: 15`) — boxes fully within 15px of any frame edge are rejected
- Added **SAHI confidence floor** (`sahi_min_conf_display: 0.25`) — low-conf SAHI ghosts are filtered
- **Clamped SAHI box coordinates** to frame boundaries (no more negative/out-of-bounds coords)
- **Reduced TTL** from 25 → 10 frames (ghosts die 2.5× faster)
- **Raised SAHI base confidence** from 0.15 → 0.20 to reduce phantom generations
- **Increased slice size** from 256 → 320 (fewer slice boundaries = fewer artifacts)

New utility functions: `box_area()`, `is_ghost_box()`

---

### 🔧 Fix 2: Alarm Logic Redesigned (Proximity → Containment)

**Old (v1.0):** Alarm checked if garbage was *near* a dustbin via pixel distance. Problem: garbage dropped right outside a dustbin but within the proximity radius wouldn't trigger an alarm.

**New (v2.0):** Alarm checks if the garbage bounding box is **geometrically contained inside** the dustbin bounding box using a new `compute_containment()` function.

- `compute_containment(garbage_box, dustbin_box)` → returns fraction (0.0–1.0) of garbage area inside dustbin
- If containment ≥ 50% OR IoU ≥ 30% → garbage is "inside" → **no alarm** (green line drawn)
- Otherwise → garbage is NOT inside → **alarm fires** (red line drawn to nearest dustbin)
- Old proximity-based code is **fully preserved as comments** (lines 467–505)

---

### 🔧 Fix 3: Improved Far-Away / Multi-Object Detection

Lowered confidence thresholds to catch more distant and multiple objects:

| Parameter | Old (v1.0) | New (v2.0) |
|-----------|-----------|-----------|
| `yolo_conf` | 0.20 | **0.15** |
| `dustbin_min_conf` | 0.30 | **0.25** |
| `garbage_min_conf` | 0.15 | **0.12** |

---

### ⚠️ Garbage-in-Hand Detection Limitation

> [!IMPORTANT]
> Detecting garbage held in someone's hand is primarily a **model training limitation**, not a code issue. The model was trained on stationary garbage on the ground. To properly fix this, you'd need to:
> 1. Collect training images of people holding garbage bags
> 2. Annotate them as "garbage" class
> 3. Retrain the model
>
> The lowered `garbage_min_conf` (0.12) may help catch *some* handheld cases if the model has partial confidence, but the real fix requires retraining.

---

### All Old Code Preserved

Per request, all replaced logic is **commented out, not deleted**. The old proximity alarm logic is at lines 467–505 with clear `OLD` / `END OLD` markers.
