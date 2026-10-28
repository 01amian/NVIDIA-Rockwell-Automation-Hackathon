# Rockwell Automation + NVIDIA Innovation Lab · Fall 2026
## Final Vial Inspection — Automated Industrial Vision Pipeline

**Team Repository:** Sana and Saja Elsharef's Team  
**Challenge Track:** Dataset 1: Final Vial Inspection  
**Partners:** Rockwell Automation, NVIDIA, MSOE AI Club, UWM Connected Systems Institute (CSI)

---

### 🚀 Quick Start & How to Run (Summary)

Anyone cloning this repository can run all workflows immediately with the commands below.

#### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-sana-and-saja-elsharef-s-tea-bc5aad3e39.git
cd msoe-ai-club-rockwell-automation-nvidia-hackath-sana-and-saja-elsharef-s-tea-bc5aad3e39

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

#### 2. Run Inspection on REAL Factory Images (Production Runs)
To inspect any real 1920×1080 camera capture from the factory floor:

```bash
# Run on the included real factory test scene:
python3 inspect_vials.py --image test_scenes/my_custom_scene.jpg --output-dir results/

# Or run on any image from the official CSI dataset:
python3 inspect_vials.py --image /path/to/csi-vial-fill-dataset/cap_seal/images/tray_000001.jpg --output-dir results/
```

**What this outputs:**
- `results/inspection_summary.json`: Complete structured audit containing per-slot classifications, confidence scores, and factory tray rules (`complete_order`, `missing_vial`, `bad_misclassified_as_good`).
- `results/annotated_*.jpg`: High-resolution visual overlay with color-coded inspection rings (Green = Pass, Red = Defect / Damaged / Missing Cap, Blue = Empty Slot).

---

#### 3. Run the Massive 100-Scene Automated Benchmark (Backtest Audit)
To evaluate the AI model across 100 distinct fixtures (2,400 individual slot inspections) spanning all 5 defect classes:

```bash
# Run 100-scene benchmark remotely against the cloud GPU:
python3 benchmark_100_scenes.py --remote --count 100

# Or run locally against a directory of rendered scenes:
python3 benchmark_100_scenes.py --data-dir path/to/synthetic_100 --count 100
```

**Benchmark Results:**
- **Defects Correctly Caught:** 220 / 221 (**99.55%**)
- **🚨 Escaped Defects (Missed):** **0 (0.00%)**
- **Overall System Accuracy:** **99.88%**
- **Recall by Class:**
  - `damaged_vial`: **100.00%** (57/57)
  - `no_vial`: **100.00%** (58/58)
  - `missing_cap`: **100.00%** (54/54)
  - `crooked_cap`: **98.08%** (51/52)
  - `pass`: **99.91%** (2,177/2,179)

---

#### 4. Run Custom Digital Twin Defect Simulations (Interactive NVIDIA Isaac Sim)
Inject any custom defect combination into the 3D ray-traced digital twin on GPU:

```bash
python3 run_custom_inspection.py --name scenario_demo --defects G05:crooked_cap G12:damaged_vial G20:no_vial G24:missing_cap
```

**What this does automatically:**
1. Dispatches USD geometry modifications to NVIDIA Isaac Sim running headless on the Brev GPU.
2. Ray-traces the 3D scene with RTX lighting, glass refraction, and authentic camera distortion ($k_1 = -0.1527$).
3. Downloads the rendered image and ground truth JSON to `output_scenes/scenario_demo/`.
4. Runs automated inspection and prints a live Discrepancy & Match Audit report.

---

#### 5. Run Unit & Integration Tests
```bash
pytest tests/ -v
```

---

#### 6. Evaluate Predictions Against Ground Truth (Scoring Metric)
```bash
python3 evaluate_run.py --ground-truth path/to/ground_truth.json --predictions results/inspection_summary.json
```

---

### 1. Problem Overview
On high-mix, low-volume (HMLV) pharmaceutical filling lines, vials are picked and placed into two fixtures at the final packaging station:
- **Good Tray (`G01`–`G30`)**: Holds acceptable, non-defective product.
- **Scrap Tray (`S01`–`S30`)**: Holds rejected or defect-suspected vials.

An overhead 1920×1080 camera inspects both fixtures simultaneously. In industrial automation, **a missed defect (false negative) is catastrophic**, while a false alarm is a manageable operational cost.

Our pipeline automatically monitors all 60 fixture apertures to:
1. **Detect 5 target conditions per slot**:
   - `pass`: Visible cap and vial body acceptable.
   - `missing_cap`: Vial present, but cap is missing / open mouth.
   - `crooked_cap`: Cap is tilted or unseated relative to the vial mouth.
   - `damaged_vial`: Vial body crushed, broken, or deformed.
   - `no_vial`: Empty fixture aperture.
2. **Apply Factory Decision Semantics (`rules.py`)**:
   - **Good Tray**: Evaluates `complete_order`, `missing_vial` (internal skipped placements), and `bad_misclassified_as_good` (defects entering good stream).
   - **Scrap Tray**: Evaluates `missing_vial`, `full`, and `good_misclassified_as_bad` (misrouted good product).
3. **Produce Visual & Structured Inspection Reports**:
   - Generates compliant `inspection_summary.json` and annotated visual inspection overlays with color-coded bounding rings.

---

### 2. Architecture

```
                 1920x1080 Fixture Scene
                            │
                            ▼
               ┌─────────────────────────┐
               │    TrayGeometry Engine  │  (Calibration & Alignment)
               └────────────┬────────────┘
                            │ 60 Crops (G01-G30, S01-S30)
                            ▼
               ┌─────────────────────────┐
               │  Defect Vision Engine   │  (Occupancy, Cap Tilt, Damage)
               └────────────┬────────────┘
                            │ Slot Predictions + Confidences
                            ▼
               ┌─────────────────────────┐
               │  Factory Rules Engine   │  (Good & Scrap Tray Decisions)
               └────────────┬────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   Structured JSON Report        Annotated Visual Overlay
```

---

### 3. Repository Structure
```
├── README.md                  # Project documentation & execution guide
├── calibration.json           # 1920x1080 camera calibration & grid offsets
├── inspect_vials.py           # CLI entry point for full inspection of REAL factory images
├── benchmark_100_scenes.py    # Large-scale 100-scene automated backtest & audit suite
├── run_custom_inspection.py   # Interactive Isaac Sim 3D ray-traced defect generator & inspector
├── generate_damaged_vials.py  # 3D USD crushed/deformed vial generator for Isaac Sim
├── evaluate_run.py            # CLI benchmark scoring predictions against ground truth
├── requirements.txt           # Python dependencies
├── models/
│   └── vial_mobilenet.pth     # Trained PyTorch MobileNetV3 deep learning weights
├── src/
│   ├── classifier.py          # Deep learning classifier with confidence thresholds
│   ├── evaluate.py            # Hackathon evaluation metrics (penalty scores)
│   ├── geometry.py            # Coordinate transformations & crop extraction
│   ├── pipeline.py            # End-to-end automated inspection pipeline
│   ├── rules.py               # Factory decision & tray-level semantics
│   └── train_torch.py         # PyTorch MobileNetV3 deep learning training script
└── tests/
    └── test_inspection.py     # Unit & integration tests
```
