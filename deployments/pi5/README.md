# Raspberry Pi 5 — Main Decision & Processing Server

This package contains all backend scripts, model checkpoints, and recognition modules for operating the **Raspberry Pi 5** server. (No GUI required).

## Directory Structure
```text
pi5/
├── server_pi5.py         # Main server daemon (FastAPI / Native HTTP on port 8000)
├── config/
│   └── config.yaml       # System pipeline configuration
├── models/
│   ├── best_mobilenetv3.pt  # LightIrisNet segmentation model weights (17.2 MB)
│   └── sota.pth             # VIS-IrisFormer recognition weights (453.4 MB)
├── LightIrisNet/         # LightIrisNet model architecture
├── Vis-IrisFormer/       # VIS-IrisFormer Transformer model architecture
├── src/                  # Core pipeline processing modules
└── data/templates/       # Enrolled iris biometric template store
```

## How to Run

1. **Install Python dependencies**:
   ```bash
   pip install torch torchvision numpy opencv-python fastapi uvicorn pyyaml
   ```

2. **Start Server Daemon**:
   ```bash
   python server_pi5.py --port 8000
   ```

The server listens on port 8000 for incoming camera payloads from the Raspberry Pi Zero.
