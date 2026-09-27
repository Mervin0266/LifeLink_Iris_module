#!/usr/bin/env python3
"""
Step 1: Best Frame Finder Script for Visible-Light Iris Recognition Pipeline.

This script executes Stage 1 of the pipeline:
1. Ingests a video file, image sequence, or live camera stream.
2. Crops the Eye Region of Interest (ROI) via EyeROICropper.
3. Stage A Quality Filtering: Evaluates ISO/IEC 29794-6 metrics (sharpness, exposure, glare, shadows).
4. Stage B Deep Validation: Runs LightIrisNet segmentation and geometric containment validation.
5. Selects, saves, and reports the single best quality frame candidate.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_manager import RaspberryPiCamera
from src.vision.eye_roi import EyeROICropper
from src.quality.iris_quality import IrisQualityEvaluator
from src.quality.frame_selector import FrameSelector, FrameCandidate
from src.models.lightirisnet_wrapper import LightIrisNetSegmenter
from src.vision.segmentation_validator import SegmentationValidator


def find_best_frame(
    source: str,
    max_frames: int = 30,
    early_stop: bool = True,
    use_segmentation: bool = True,
    segmenter_ckpt: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[FrameCandidate, Dict[str, Any]]:
    """Scans an input source and extracts the highest-quality eye frame candidate.

    Args:
        source: File path to video/image or camera index ('0').
        max_frames: Maximum number of frames to inspect.
        early_stop: If True, stops as soon as a frame >= 80.0 quality score is found.
        use_segmentation: If True, uses LightIrisNet for Stage B validation.
        segmenter_ckpt: Optional path to LightIrisNet weights checkpoint.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        best_candidate: Selected FrameCandidate object.
        report: Summary metadata dictionary.
    """
    t_start = time.time()

    # 1. Initialize Components
    cropper = EyeROICropper()
    evaluator = IrisQualityEvaluator()
    validator = SegmentationValidator()
    
    segmenter = None
    if use_segmentation:
        print("[Step 1] Loading LightIrisNet segmentation model...")
        segmenter = LightIrisNetSegmenter(checkpoint_path=segmenter_ckpt, device=device)

    selector = FrameSelector(
        cropper=cropper,
        quality_evaluator=evaluator,
        segmenter=segmenter,
        validator=validator,
    )

    # 2. Check if source is a single static image file
    ext = Path(source).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        print(f"[Step 1] Processing static image input: '{source}'")
        img_bgr = cv2.imread(source)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image file: '{source}'")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        cand = selector.process_frame(img_rgb, frame_index=0)
        if segmenter is not None:
            seg_res = segmenter.segment(cand.eye_crop)
            val_res = validator.validate(
                seg_res["iris_mask"], seg_res["pupil_mask"], seg_res["diagnostics"]
            )
            cand.segmentation_result = seg_res
            cand.validation = val_res

        total_time = (time.time() - t_start) * 1000.0
        report = _build_report(cand, total_time, total_frames=1, source=source)
        return cand, report

    # 3. Stream frames from Video / Camera Source
    print(f"[Step 1] Ingesting video stream from: '{source}' (Scanning up to {max_frames} frames)...")
    camera = RaspberryPiCamera(backend="opencv_video", source=source)
    camera.start()

    frames: List[np.ndarray] = []
    best_early_cand: Optional[FrameCandidate] = None

    try:
        frame_cnt = 0
        while frame_cnt < max_frames:
            ret, frame_rgb = camera.read()
            if not ret or frame_rgb is None:
                break
            
            frames.append(frame_rgb)
            frame_cnt += 1

            # Early stopping check if high quality frame obtained
            if early_stop:
                cand = selector.process_frame(frame_rgb, frame_index=frame_cnt - 1)
                if cand.quality_metrics.valid and cand.quality_metrics.quality_score >= 80.0:
                    if segmenter is not None:
                        seg_res = segmenter.segment(cand.eye_crop)
                        val_res = validator.validate(
                            seg_res["iris_mask"], seg_res["pupil_mask"], seg_res["diagnostics"]
                        )
                        if val_res.valid:
                            cand.segmentation_result = seg_res
                            cand.validation = val_res
                            best_early_cand = cand
                            print(f"[Step 1] Early stop triggered at frame #{frame_cnt}! Quality Score: {cand.quality_metrics.quality_score:.1f}")
                            break
    finally:
        camera.stop()

    if not frames and best_early_cand is None:
        raise RuntimeError(f"No frames could be read from video source: '{source}'")

    if best_early_cand is not None:
        best_cand = best_early_cand
    else:
        print(f"[Step 1] Running 2-stage multi-frame quality selection across {len(frames)} frames...")
        best_cand = selector.select_best_frame(frames, segmenter_override=segmenter)

    total_time = (time.time() - t_start) * 1000.0
    report = _build_report(best_cand, total_time, total_frames=len(frames), source=source)
    return best_cand, report


def _build_report(cand: FrameCandidate, total_ms: float, total_frames: int, source: str) -> Dict[str, Any]:
    """Formats candidate results into a structured dictionary."""
    valid_seg = False
    iris_ratio = 0.0
    pupil_ratio = 0.0

    if cand.validation:
        valid_seg = cand.validation.valid
        iris_ratio = cand.validation.iris_area_ratio
        pupil_ratio = cand.validation.pupil_inside_ratio

    return {
        "source": str(source),
        "total_frames_scanned": total_frames,
        "selected_frame_index": cand.frame_index,
        "processing_time_ms": round(total_ms, 2),
        "quality_metrics": {
            "sharpness": round(cand.quality_metrics.sharpness, 2),
            "brightness_mean": round(cand.quality_metrics.brightness_mean, 2),
            "contrast": round(cand.quality_metrics.contrast, 2),
            "highlight_ratio": round(cand.quality_metrics.highlight_ratio, 4),
            "shadow_ratio": round(cand.quality_metrics.shadow_ratio, 4),
            "stage_a_valid": cand.quality_metrics.valid,
            "stage_a_reason": cand.quality_metrics.reason,
            "quality_score": round(cand.quality_metrics.quality_score, 2),
        },
        "segmentation_validation": {
            "stage_b_valid": valid_seg,
            "iris_area_ratio": round(iris_ratio, 4),
            "pupil_inside_ratio": round(pupil_ratio, 4),
        },
        "roi_bbox": list(cand.roi_bbox),
        "final_selection_score": round(cand.final_score, 2),
    }


def save_output_artifacts(
    candidate: FrameCandidate,
    report: Dict[str, Any],
    output_dir: str = "output_best_frame",
):
    """Saves the best frame image, cropped eye ROI, overlay, and metadata report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save Full Frame
    frame_bgr = cv2.cvtColor(candidate.frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path / "best_full_frame.jpg"), frame_bgr)

    # 2. Save Cropped Eye ROI
    eye_bgr = cv2.cvtColor(candidate.eye_crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path / "best_eye_crop.jpg"), eye_bgr)

    # 3. Draw Segmentation Overlay if available
    overlay_bgr = frame_bgr.copy()
    x, y, w, h = candidate.roi_bbox
    cv2.rectangle(overlay_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if candidate.segmentation_result:
        seg = candidate.segmentation_result
        if seg.get("iris_ellipse") is not None:
            (cx_i, cy_i), (ma_i, mi_i), angle_i = seg["iris_ellipse"]
            cv2.ellipse(
                overlay_bgr,
                ((int(cx_i + x), int(cy_i + y)), (int(ma_i), int(mi_i)), angle_i),
                (255, 255, 0),
                2,
            )
        if seg.get("pupil_ellipse") is not None:
            (cx_p, cy_p), (ma_p, mi_p), angle_p = seg["pupil_ellipse"]
            cv2.ellipse(
                overlay_bgr,
                ((int(cx_p + x), int(cy_p + y)), (int(ma_p), int(mi_p)), angle_p),
                (255, 0, 255),
                2,
            )

    cv2.imwrite(str(out_path / "best_frame_overlay.jpg"), overlay_bgr)

    # 4. Save JSON metadata report
    with open(out_path / "best_frame_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Step 1 Artifacts Saved to '{out_path}']:")
    print(f"  • Full Frame:  {out_path / 'best_full_frame.jpg'}")
    print(f"  • Eye ROI:     {out_path / 'best_eye_crop.jpg'}")
    print(f"  • Overlay:     {out_path / 'best_frame_overlay.jpg'}")
    print(f"  • Metadata:    {out_path / 'best_frame_report.json'}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Best Frame Selection Script for Visible-Light Iris Recognition."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sample_videos/eye_sample_01.mp4",
        help="Path to input video/image or webcam index ('0').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_best_frame",
        help="Directory to save extracted best frame artifacts.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum frames to scan from input stream.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping and scan all max-frames.",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable Stage B LightIrisNet segmentation validation.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="LightIrisNet/model/best_mobilenetv3.pt",
        help="Path to LightIrisNet PyTorch checkpoint (.pt file). Default: LightIrisNet/model/best_mobilenetv3.pt",
    )

    args = parser.parse_args()

    # Discover sample fallback if default file missing
    if not os.path.exists(args.source) and not args.source.isdigit():
        for fallback in ["sample_data/eye_001.jpg", "sample_videos/sample_eye.mp4"]:
            if os.path.exists(fallback):
                args.source = fallback
                break

    try:
        best_cand, report = find_best_frame(
            source=args.source,
            max_frames=args.max_frames,
            early_stop=not args.no_early_stop,
            use_segmentation=not args.no_segmentation,
            segmenter_ckpt=args.ckpt,
        )

        save_output_artifacts(best_cand, report, output_dir=args.output_dir)

        print("=== STEP 1: BEST FRAME FINDER SUMMARY ===")
        print(f"Source:              {report['source']}")
        print(f"Frames Scanned:      {report['total_frames_scanned']}")
        print(f"Selected Frame #:    {report['selected_frame_index']}")
        print(f"Sharpness Score:     {report['quality_metrics']['sharpness']}")
        print(f"Quality Score:       {report['quality_metrics']['quality_score']} / 100.0")
        print(f"Stage A Valid:       {report['quality_metrics']['stage_a_valid']}")
        print(f"Stage B Valid:       {report['segmentation_validation']['stage_b_valid']}")
        print(f"Total Time:          {report['processing_time_ms']} ms")
        print("==========================================")

    except Exception as e:
        print(f"[Error in Step 1 Script]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
