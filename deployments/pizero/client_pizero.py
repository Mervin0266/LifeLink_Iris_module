#!/usr/bin/env python3
"""
Raspberry Pi Zero - Autonomous Camera Client Application for LifeLink.

Functions:
- Discovers Raspberry Pi 5 server via mDNS (http://main-pi.local:8000) or fallback IP (http://192.168.50.1:8000).
- Waits until Pi 5 server is available at GET /health.
- Initializes camera and continuously scans eye frames.
- Runs Stage 1 quality evaluation (sharpness, exposure, position, visibility).
- Selects the best quality frame and IMMEDIATELY STOPS camera acquisition.
- Calculates SHA-256 hash and sends POST /capture payload to Pi 5.
- Handles application-level responses:
    • SUCCESS -> Shuts down Pi Zero cleanly.
    • RESCAN  -> Restarts camera & captures a new frame.
    • RETRY   -> Re-transmits the existing frame.
"""

import os
import sys
import time
import json
import base64
import hashlib
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step1_find_best_frame import find_best_frame, FrameCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PiZeroClient")

# Configuration Defaults
DEFAULT_SERVER_HOSTS = ["main-pi.local", "192.168.50.1", "localhost", "127.0.0.1"]
DEFAULT_PORT = 8000
DEVICE_ID = "PIZERO-CAM-01"


class PiZeroCameraClient:
    """Autonomous Camera Unit Manager for Raspberry Pi Zero."""

    def __init__(
        self,
        server_hosts: Optional[list] = None,
        port: int = DEFAULT_PORT,
        device_id: str = DEVICE_ID,
        video_source: str = "0",
        simulate_shutdown: bool = False,
    ):
        self.server_hosts = server_hosts or DEFAULT_SERVER_HOSTS
        self.port = port
        self.device_id = device_id
        self.video_source = video_source
        self.simulate_shutdown = simulate_shutdown
        self.active_server_url: Optional[str] = None
        self.capture_counter = 1

    def discover_server(self, timeout_sec: float = 2.0) -> str:
        """Discovers Raspberry Pi 5 server via mDNS or fallback IP."""
        logger.info("Discovering Raspberry Pi 5 server via mDNS / local IP...")
        while True:
            for host in self.server_hosts:
                url = f"http://{host}:{self.port}"
                try:
                    req = urllib.request.Request(f"{url}/health", method="GET")
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            logger.info(f"✅ Found Pi 5 Server at '{url}' (System: {data.get('system')})")
                            self.active_server_url = url
                            return url
                except Exception:
                    pass

            logger.info("Pi 5 server unavailable. Retrying mDNS discovery in 3 seconds...")
            time.sleep(3.0)

    def capture_best_frame(self) -> Tuple[FrameCandidate, Dict[str, Any], bytes]:
        """Captures eye frames and selects best frame using Stage 1 algorithm."""
        logger.info("🎥 Initializing camera & starting frame acquisition...")
        
        # Execute Step 1 best frame finder
        cand, report = find_best_frame(
            source=self.video_source,
            max_frames=30,
            early_stop=True,
            use_segmentation=True,
        )

        logger.info(f"🛑 Best frame selected (Frame #{cand.frame_index}, Score: {cand.quality_metrics.quality_score:.1f}). Camera acquisition STOPPED.")

        # Encode eye crop image as JPEG
        eye_bgr = cv2.cvtColor(cand.eye_crop, cv2.COLOR_RGB2BGR)
        ret, jpg_bytes = cv2.imencode(".jpg", eye_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ret:
            raise RuntimeError("Failed to encode eye crop image as JPEG.")

        return cand, report, jpg_bytes.tobytes()

    def send_capture_payload(
        self, capture_id: str, report: Dict[str, Any], image_bytes: bytes
    ) -> Dict[str, Any]:
        """Calculates SHA-256 hash and posts capture payload to Pi 5 server."""
        if not self.active_server_url:
            self.discover_server()

        # Compute SHA-256 Checksum
        img_hash = hashlib.sha256(image_bytes).hexdigest()
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "capture_id": capture_id,
            "device_id": self.device_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "eye_side": "right",
            "image_hash": img_hash,
            "image_data": img_b64,
            "quality_metrics": report.get("quality_metrics", {}),
        }

        endpoint_url = f"{self.active_server_url}/capture"
        logger.info(f"Sending capture payload '{capture_id}' to Pi 5 at '{endpoint_url}' (Size: {len(image_bytes)} bytes, SHA256: {img_hash[:12]}...)...")

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint_url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10.0) as resp:
            resp_body = resp.read().decode("utf-8")
            response_json = json.loads(resp_body)

        return response_json

    def shutdown_system(self):
        """Executes safe system shutdown upon receiving SUCCESS from Pi 5."""
        logger.info("🎉 SUCCESS response received from Pi 5!")
        logger.info("Executing safe operating system shutdown...")

        if self.simulate_shutdown:
            logger.info("[Simulated Shutdown] Exiting client application cleanly.")
            sys.exit(0)

        try:
            os.system("sudo shutdown -h now")
        except Exception:
            try:
                os.system("sudo poweroff")
            except Exception as e:
                logger.error(f"Failed to trigger system shutdown: {e}")
                sys.exit(0)

    def run_autonomous_loop(self):
        """Executes the complete autonomous loop: SCAN -> SELECT -> STOP -> SEND -> WAIT -> RESPONSE."""
        logger.info("==================================================")
        logger.info("  Raspberry Pi Zero Autonomous Camera Unit Start  ")
        logger.info("==================================================")

        # 1. Discover Pi 5 server on network
        self.discover_server()

        # Main Scan & Transmit Loop
        while True:
            capture_id = f"CAPTURE-{self.capture_counter:04d}"
            logger.info(f"\n--- Starting Scan Sequence [{capture_id}] ---")

            try:
                # 2. Capture Best Frame
                cand, report, img_bytes = self.capture_best_frame()

                # Loop for retry/transmission handling
                while True:
                    try:
                        # 3. Send image payload to Pi 5
                        response = self.send_capture_payload(capture_id, report, img_bytes)
                        status = response.get("status", "RETRY").upper()
                        msg = response.get("message", "")

                        logger.info(f"Received Response from Pi 5: status='{status}', message='{msg}'")

                        if status == "SUCCESS":
                            self.shutdown_system()
                            return

                        elif status == "RESCAN":
                            logger.warning(f"⚠️ RESCAN requested by Pi 5 ('{msg}'). Capturing new best frame...")
                            self.capture_counter += 1
                            break  # Break inner retry loop to capture fresh frame

                        elif status == "RETRY":
                            logger.warning(f"🔄 RETRY requested by Pi 5 ('{msg}'). Retrying transmission in 2 seconds...")
                            time.sleep(2.0)
                            continue

                    except Exception as err:
                        logger.error(f"Network error during transmission: {err}. Retrying transmission in 3 seconds...")
                        time.sleep(3.0)
                        continue

            except Exception as e:
                logger.error(f"Error during scan sequence: {e}. Retrying scan in 3 seconds...")
                time.sleep(3.0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Raspberry Pi Zero Autonomous Camera Client for LifeLink.")
    parser.add_argument("--source", type=str, default="0", help="Video source path or webcam index ('0').")
    parser.add_argument("--host", type=str, default=None, help="Specific Pi 5 server host IP or mDNS name.")
    parser.add_argument("--port", type=int, default=8000, help="Pi 5 server port (default: 8000).")
    parser.add_argument("--simulate-shutdown", action="store_true", help="Simulate OS shutdown on SUCCESS.")
    args = parser.parse_args()

    hosts = [args.host] if args.host else DEFAULT_SERVER_HOSTS

    client = PiZeroCameraClient(
        server_hosts=hosts,
        port=args.port,
        video_source=args.source,
        simulate_shutdown=args.simulate_shutdown,
    )
    client.run_autonomous_loop()


if __name__ == "__main__":
    main()
