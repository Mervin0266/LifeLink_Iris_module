# Raspberry Pi Visible-Light Iris Recognition Pipeline

A complete, production-quality, modular **Raspberry Pi-based visible-spectrum iris recognition system** built from scratch using `LightIrisNet` (MobileNetV3 iris segmentation) and `VIS-IrisFormer` (Vision Transformer matcher).

---

## 🌟 Architecture Overview

```text
Raspberry Pi Camera / Video Input Stream
                    ↓
       Frame Acquisition Loop
                    ↓
    Stage A: Fast Quality Screening (Sharpness, Exposure)
                    ↓
       Candidate Frame Selection
                    ↓
        Eye Region Crop (Fixed / Extensible ROI)
                    ↓
   Stage B: LightIrisNet Segmentation (Iris + Pupil Masks)
                    ↓
      Segmentation Quality Validation
                    ↓
  Daugman Rubber-Sheet Polar Normalization (64 × 512)
                    ↓
   VIS-IrisFormer Feature Extractor (Patch Embeddings)
                    ↓
     Iris Template Storage & 1:1 Cosine Matcher
```

---

## 🚀 Key Features

* **Hardware & Development Backends**: Supports native Raspberry Pi `Picamera2` / `libcamera` hardware as well as OpenCV video files, static images, and webcams for PC development.
* **Two-Stage Frame Quality Selector**: Evaluates Laplacian variance sharpness, brightness histogram, specular glare, and segmentation mask sanity before selecting the single best frame.
* **Deep Segmentation Engine**: Integrates MobileNetV3-based `LightIrisNet` with pupil refinement, ellipse refitting, and containment gating.
* **Daugman Polar Normalization**: Unwraps non-concentric annular iris regions into fixed $64 \times 512$ grayscale representations.
* **Transformer Feature Matching**: Uses `VIS-IrisFormer` Vision Transformer with 2D Relative Positional Encoding (`RoPE2D`) to extract 128 patch-wise embeddings and compute patch cosine similarity.
* **Privacy-Aware Enrolment & Storage**: Enrolls templates locally under `templates/<subject_id>/<eye>_template.npy` without saving full-face images or retaining raw camera frames by default.
* **Offline Local Operation**: Operates 100% locally on Raspberry Pi / PC without requiring external cloud/BIQT servers.

---

## 📂 Project Structure

```text
.
├── config/
│   ├── config.yaml                   # Subsystem parameters & paths
│   └── matching_threshold.json        # Calibrated score threshold
├── docs/
│   ├── repository_analysis.md         # Forensic analysis of source repos
│   ├── architecture.md                # System architecture & flowchart
│   ├── installation.md                # Setup & dependency instructions
│   ├── model_integration.md           # Model definitions & tensor contracts
│   ├── performance.md                 # Latency & performance benchmarks
│   └── troubleshooting.md            # Common issues & diagnostic steps
├── src/
│   ├── camera/                        # Picamera2 & OpenCV camera backends
│   ├── models/                        # LightIrisNet & VIS-IrisFormer wrappers
│   ├── quality/                       # Quality evaluator & multi-frame selector
│   ├── vision/                        # Eye ROI, validator, & rubber-sheet normalizer
│   ├── recognition/                   # Iris template & cosine matcher
│   ├── storage/                       # Privacy-aware template store
│   └── pipeline/                      # High-level pipeline orchestrator
├── tools/
│   └── calibrate_threshold.py         # ROC/EER threshold calibration utility
├── tests/                             # Automated test suite
├── models/
│   └── vis_irisformer/sota.pth        # VIS-IrisFormer checkpoint
├── LightIrisNet/
│   └── model/best_mobilenetv3.pt      # LightIrisNet checkpoint
├── sample_data/                       # Test image fixtures
├── debug/                             # Saved visualization outputs
├── main.py                            # CLI entrypoint
└── requirements.txt                   # Dependency definitions
```

---

## ⚡ Quickstart Commands

### 1. Launch Desktop GUI Dashboard (Recommended)
```bash
python gui_app.py
```

### 2. Test Camera Initialization
```bash
python main.py camera-test --backend opencv_video --source 0
```

### 3. Process Single Image File
```bash
python main.py process-image --input sample_data/eye_001.jpg --save-debug
```

### 4. Enroll Subject Template
```bash
python main.py enroll --subject-id person_001 --eye right --source sample_data/eye_001.jpg
```

### 5. Verify Probe Image Against Enrolled Template
```bash
python main.py verify --subject-id person_001 --eye right --source sample_data/eye_001.jpg
```

### 5. Run Automated Unit Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 6. Calibrate Score Thresholds
```bash
python tools/calibrate_threshold.py
```

---

## 🧪 Verification Matrix

| Component | Implemented | Software-Tested | Hardware-Tested | Status |
|-----------|-------------|-----------------|-----------------|--------|
| OpenCV Camera Backend | Yes | Yes | N/A | VERIFIED |
| Frame Selector | Yes | Yes | N/A | VERIFIED |
| Eye ROI Crop | Yes | Yes | N/A | VERIFIED |
| LightIrisNet Segmenter | Yes | Yes | N/A | VERIFIED |
| Daugman Iris Normalizer | Yes | Yes | N/A | VERIFIED |
| VIS-IrisFormer Matcher | Yes | Yes | N/A | VERIFIED |
| Template Storage | Yes | Yes | N/A | VERIFIED |
| 1:1 Verification Matcher | Yes | Yes | N/A | VERIFIED |
| Picamera2 Raspberry Pi Camera | Yes | Structure Valid | No (Host PC) | CODE COMPLETE — NOT HARDWARE TESTED |

---

## 📄 License & Citation

Licensed under MIT. Built upon published research repositories:
* `LightIrisNet`: https://github.com/naveengv7/LightIrisNet
* `Vis-IrisFormer`: https://github.com/naveengv7/Vis-IrisFormer
* `VISIrisHub`: https://github.com/naveengv7/VISIrisHub
# LifeLink_Iris_module
