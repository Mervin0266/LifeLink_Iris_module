# Raspberry Pi Zero — Autonomous Edge Camera Unit

This package contains the lightweight camera client and Stage 1 ISO quality evaluation code for the **Raspberry Pi Zero**.

## Directory Structure
```text
pizero/
├── client_pizero.py         # Autonomous edge camera client & transmission daemon
├── step1_find_best_frame.py  # Stage 1 fast ISO sharpness & glare frame selector
├── lifelink-client.service  # Systemd auto-boot unit file
└── src/                     # Lightweight edge camera & quality modules
```

## How to Setup & Run

1. **Install Python dependencies** (OpenCV & NumPy only - no heavy PyTorch required):
   ```bash
   pip install opencv-python numpy
   ```

2. **Run Client Manually**:
   ```bash
   python client_pizero.py --source 0
   ```

3. **Install Auto-Boot Systemd Service**:
   ```bash
   sudo cp lifelink-client.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable lifelink-client.service
   sudo systemctl start lifelink-client.service
   ```

Upon powering on, the Pi Zero will automatically:
1. Discover the Pi 5 server at `http://main-pi.local:8000`.
2. Start camera acquisition and select the highest quality eye frame.
3. Immediately stop camera hardware.
4. Calculate SHA-256 hash and transmit payload to Pi 5.
5. Execute clean operating system shutdown (`sudo shutdown -h now`) upon receiving `SUCCESS`.
