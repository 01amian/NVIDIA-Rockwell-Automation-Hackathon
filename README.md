# Rockwell Automation + NVIDIA Innovation Lab · Fall 2026
## Final Vial Inspection — Automated Industrial Vision Pipeline

**Team Repository:** Sana and Saja Elsharef's Team  
**Challenge Track:** Dataset 1: Final Vial Inspection  
**Partners:** Rockwell Automation, NVIDIA, MSOE AI Club, UWM Connected Systems Institute (CSI)

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

### 3. Installation & Quick Start

#### Requirements
- Python 3.10+ (macOS, Linux, or Windows)
- Dependencies: `numpy`, `Pillow`, `opencv-python-headless`, `torch`, `torchvision`, `scikit-learn`, `pytest`

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Run Automated Inspection
Inspect any fixture image with one command:
```bash
python inspect_vials.py --image /path/to/tray_000001.jpg --output-dir results/
```

This generates:
- `results/inspection_summary.json`: Detailed machine-readable classifications and findings.
- `results/annotated_tray_000001.jpg`: Visual image with color-coded rings for quick operator verification.

#### Run Test Suite
```bash
pytest tests/ -v
```

---

### 4. Repository Structure
```
├── README.md                  # Project documentation & team guide
├── calibration.json           # 1920x1080 camera calibration & grid offsets
├── inspect_vials.py           # CLI entry point for full inspection of any scene
├── generate_test_scene.py     # CLI to inject synthetic defects into fixture scenes
├── evaluate_run.py            # CLI benchmark scoring predictions against ground truth
├── test_vial_system.py        # All-in-one test cycle (Generate -> Inspect -> Benchmark -> Browser Review)
├── requirements.txt           # Python dependencies
├── src/
│   ├── classifier.py          # Defect & occupancy computer vision classifier
│   ├── evaluate.py            # Hackathon evaluation metrics (penalty scores)
│   ├── geometry.py            # Coordinate transformations & crop extraction
│   ├── pipeline.py            # End-to-end automated inspection pipeline
│   ├── rules.py               # Factory decision & tray-level semantics
│   ├── synthetic_generator.py # Photorealistic defect injector for local testing
│   └── train_torch.py         # PyTorch MobileNetV3 deep learning training script
└── tests/
    └── test_inspection.py     # Unit & integration tests
```

---

### 5. Training with the Digital Twin (Brev / Isaac Sim)
Per the competition guidelines, models should be trained on synthetic data rendered from the **Isaac Sim Digital Twin** (`csi_digital_twin_v1`):
1. **Generate synthetic renders**: Render tray scenes or slot crops with varying lighting, vial tilt, and defects using Isaac Sim.
2. **Train Deep Classifier**:
   ```bash
   python -m src.train_torch --data-dir path/to/synthetic_crops/ --epochs 20 --output models/vial_model.pth
   ```
   The training script automatically applies class weighting to heavily penalize defect escapes (false negatives).
