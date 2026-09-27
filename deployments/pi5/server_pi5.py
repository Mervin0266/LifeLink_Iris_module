#!/usr/bin/env python3
"""
Raspberry Pi 5 - Main Processing & Decision Server for LifeLink Iris Recognition.

Functions:
- Runs HTTP server on port 8000 (supports FastAPI or built-in HTTP server fallback).
- Listens for device discovery at GET /health.
- Ingests captured eye frames at POST /capture.
- Verifies SHA-256 image integrity.
- Executes Stage 2 Deep Learning processing (LightIrisNet segmentation + Daugman polar normalization + VIS-IrisFormer extraction + 1:N Identification).
- Returns application-level response: SUCCESS, RESCAN, or RETRY.
"""

import os
import sys
import time
import json
import base64
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.iris_pipeline import IrisRecognitionPipeline
from src.storage.template_store import TemplateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Pi5Server")

# Directory to persist received captures
CAPTURES_DIR = Path("incoming_captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Global Pipeline Instance
pipeline: Optional[IrisRecognitionPipeline] = None


def get_pipeline() -> IrisRecognitionPipeline:
    global pipeline
    if pipeline is None:
        logger.info("Initializing Pi 5 IrisRecognitionPipeline models (LightIrisNet & VIS-IrisFormer)...")
        pipeline = IrisRecognitionPipeline()
    return pipeline


def process_capture_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Processes incoming capture payload from Raspberry Pi Zero.

    Args:
        payload: Dictionary containing capture_id, image_data, image_hash, etc.

    Returns:
        Response dictionary with status: SUCCESS, RESCAN, or RETRY.
    """
    capture_id = payload.get("capture_id", f"CAPTURE-{int(time.time())}")
    device_id = payload.get("device_id", "UNKNOWN_DEVICE")
    received_hash = payload.get("image_hash", "")
    image_b64 = payload.get("image_data", "")
    eye_side = payload.get("eye_side", "right")

    logger.info(f"Processing '{capture_id}' from device '{device_id}' (Eye: {eye_side})...")

    if not image_b64:
        logger.warning(f"Empty image_data in payload for {capture_id}")
        return {
            "capture_id": capture_id,
            "status": "RETRY",
            "message": "Missing image_data payload field.",
        }

    # 1. Decode Base64 Image Data
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        logger.error(f"Base64 decode failed for {capture_id}: {e}")
        return {
            "capture_id": capture_id,
            "status": "RETRY",
            "message": f"Invalid Base64 payload: {e}",
        }

    # 2. Verify SHA-256 Image Integrity
    computed_hash = hashlib.sha256(image_bytes).hexdigest()
    if received_hash and computed_hash.lower() != received_hash.lower():
        logger.error(f"SHA-256 hash mismatch for {capture_id}! Computed: {computed_hash}, Received: {received_hash}")
        return {
            "capture_id": capture_id,
            "status": "RETRY",
            "message": "SHA-256 image checksum mismatch (data corruption in transit).",
        }

    # 3. Decode OpenCV Image Array & Save to Disk
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        logger.error(f"cv2.imdecode failed for {capture_id}")
        return {
            "capture_id": capture_id,
            "status": "RETRY",
            "message": "Failed to decode JPEG image bytes.",
        }

    save_path = CAPTURES_DIR / f"{capture_id}.jpg"
    cv2.imwrite(str(save_path), frame_bgr)
    logger.info(f"Saved capture image to '{save_path}'")

    # 4. Perform Stage 2 Deep Learning AI/ML Processing
    try:
        pipe = get_pipeline()
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process through LightIrisNet + Daugman + VIS-IrisFormer
        template, diag = pipe.process_eye_image(
            img_rgb, subject_id=capture_id, eye_side=eye_side
        )

        seg_res = diag.get("segmentation_result", {})
        iris_mask = seg_res.get("iris_mask")
        iris_pixels = int((iris_mask > 0).sum()) if iris_mask is not None else 0

        # If full camera frame was passed (large image) and initial segmentation had low pixels, apply EyeROICropper
        if iris_pixels < 50 and (img_rgb.shape[0] > 400 and img_rgb.shape[1] > 400):
            logger.info("Received full frame image. Applying EyeROICropper...")
            eye_crop, _ = pipe.cropper.crop(img_rgb)
            template, diag = pipe.process_eye_image(
                eye_crop, subject_id=capture_id, eye_side=eye_side
            )
            seg_res = diag.get("segmentation_result", {})
            iris_mask = seg_res.get("iris_mask")
            iris_pixels = int((iris_mask > 0).sum()) if iris_mask is not None else 0

        # Query enrolled templates database for 1:N identification
        gallery = pipe.store.list_templates()
        if not gallery:
            logger.info(f"No enrolled gallery templates. Auto-enrolling capture as '{capture_id}'.")
            pipe.store.save_template(template)
            return {
                "capture_id": capture_id,
                "status": "SUCCESS",
                "message": f"First subject auto-enrolled successfully as {capture_id}.",
                "identified_subject": capture_id,
                "score": "100.0%",
            }

        matches = pipe.matcher.identify(template, gallery, threshold=0.80)
        top_match = matches[0] if matches else None

        if top_match and top_match.match:
            match_pct = f"{top_match.similarity * 100.0:.2f}%"
            logger.info(f"🎉 SUCCESS! Identified '{top_match.subject_id}' ({top_match.eye_side}) with score {match_pct}")
            return {
                "capture_id": capture_id,
                "status": "SUCCESS",
                "message": f"Iris recognized successfully as '{top_match.subject_id}'",
                "identified_subject": top_match.subject_id,
                "eye_side": top_match.eye_side,
                "score": match_pct,
            }
        else:
            best_id = top_match.subject_id if top_match else "None"
            best_score = f"{(top_match.similarity * 100.0):.1f}%" if top_match else "0%"
            logger.warning(f"Non-match for {capture_id} (Best candidate '{best_id}' score {best_score} < 84.0%). Requesting RESCAN.")
            return {
                "capture_id": capture_id,
                "status": "RESCAN",
                "message": f"Unrecognized iris candidate (Best score: {best_score}). Please rescan.",
                "best_candidate": best_id,
                "best_score": best_score,
            }

    except Exception as e:
        logger.error(f"Error during AI/ML processing of {capture_id}: {e}", exc_info=True)
        return {
            "capture_id": capture_id,
            "status": "RETRY",
            "message": f"Server processing error: {e}",
        }


# -----------------------------------------------------------------------------
# HTTP SERVER IMPLEMENTATION (FastAPI or Built-in HTTP Server Fallback)
# -----------------------------------------------------------------------------
def run_fastapi_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="LifeLink Pi 5 Server", version="2.0.0")

    @app.on_event("startup")
    def startup_event():
        get_pipeline()

    @app.get("/health")
    def health_check():
        return {"status": "ok", "system": "Raspberry Pi 5 Main Server", "timestamp": time.time()}

    @app.post("/capture")
    async def capture_endpoint(request: Request):
        try:
            body = await request.json()
            response_data = process_capture_payload(body)
            return JSONResponse(content=response_data)
        except Exception as e:
            logger.error(f"Malformed HTTP request: {e}")
            return JSONResponse(
                content={"status": "RETRY", "message": f"Malformed JSON request: {e}"},
                status_code=400,
            )

    logger.info(f"Starting FastAPI server on http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port)


def run_builtin_server(host: str = "0.0.0.0", port: int = 8000):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Pi5RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))

        def do_GET(self):
            if self.path == "/health":
                res = json.dumps({"status": "ok", "system": "Raspberry Pi 5 Main Server", "timestamp": time.time()}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res)))
                self.end_headers()
                self.wfile.write(res)
            else:
                self.send_error(404, "Not Found")

        def do_POST(self):
            if self.path == "/capture":
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length)
                try:
                    payload = json.loads(body_bytes.decode("utf-8"))
                    res_dict = process_capture_payload(payload)
                except Exception as e:
                    res_dict = {"status": "RETRY", "message": f"Malformed request: {e}"}

                res_bytes = json.dumps(res_dict).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)
            else:
                self.send_error(404, "Not Found")

    get_pipeline()
    logger.info(f"Starting built-in HTTP server on http://{host}:{port}...")
    httpd = HTTPServer((host, port), Pi5RequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Main Processing Server for LifeLink.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding IP address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    try:
        import fastapi
        import uvicorn
        run_fastapi_server(host=args.host, port=args.port)
    except ImportError:
        logger.info("FastAPI/Uvicorn not installed. Falling back to native Python HTTPServer...")
        run_builtin_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
