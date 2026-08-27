# -*- coding: utf-8 -*-
"""
SAM3-based facade opening detector -- a drop-in replacement for
Mask_RCNN-2.1/facade_batch.py's 'batch' command, producing the exact same
output contract so the rest of the pipeline (initpara_batch3_stage.py ->
inverse_facade_stage.getparaset()) needs no changes:

  {save}/{imagename}.pkl        -- joblib.dump((rois, class_ids))
                                    rois[i]      = [y1, x1, y2, x2]  (pixel coords,
                                                    Mask R-CNN's (row, col) convention --
                                                    SAM3 natively returns (x1, y1, x2, y2),
                                                    so this script converts it)
                                    class_ids[i] = 1 window / 2 balcony / 3 door
  {save}/{imagename}_splash.png -- initpara_batch3_stage.py only reads this file's
                                    (H, W) shape via cv2.imread, not its pixel content,
                                    so the source image is reused as-is.

Usage (mirrors facade_batch.py's batch command):
    python facade_batch_sam3.py batch --image <rectified_images_dir> --save <ces2_dir>

Requires the 'sam3' conda environment (Python 3.12+, PyTorch 2.7+/cu128) and an
HF_TOKEN with approved access to the gated facebook/sam3 checkpoint.
"""

import argparse
import os

import cv2
import joblib
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# SAM3 is used for "window" ONLY (class_id 1, matching
# Mask_RCNN-2.1/facade_batch.py's user_settings['classes'] indexing). SAM3
# dramatically improved window detection on large/complex facades (one wall
# went from 8 windows/1 column with Mask R-CNN to 680 windows/258 columns),
# but its generic "door" prompt confidently (score > 0.85, even with a
# sharpened "building entrance door" prompt) mismatches repeating ground-floor
# windows/storefront glass as doors -- up to 28 "doors" on a single wall.
# Doors/balconies were never the problem that motivated adopting SAM3, so
# those two classes are left to the original, purpose-trained Mask R-CNN
# model instead (see merge_detections.py, which combines both models' output
# into one final .pkl per image).
CLASSES = [
    (1, "window", 0.5),
]


def detect_facade(processor, image_path, savepath):
    imagename = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGB")

    all_rois = []
    all_class_ids = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for class_id, prompt, min_score in CLASSES:
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            boxes = output["boxes"]
            scores = output["scores"]
            if boxes is None or len(boxes) == 0:
                continue
            boxes = boxes.detach().float().cpu().numpy()
            scores = scores.detach().float().cpu().numpy()
            for (x1, y1, x2, y2), score in zip(boxes, scores):
                if score < min_score:
                    continue
                all_rois.append([y1, x1, y2, x2])
                all_class_ids.append(class_id)

    if all_rois:
        rois = np.array(all_rois, dtype=np.float32)
        class_ids = np.array(all_class_ids, dtype=np.int32)
    else:
        rois = np.zeros((0, 4), dtype=np.float32)
        class_ids = np.zeros((0,), dtype=np.int32)

    joblib.dump((rois, class_ids), os.path.join(savepath, imagename + ".pkl"))

    img_bgr = cv2.imread(image_path)
    cv2.imwrite(os.path.join(savepath, imagename + "_splash.png"), img_bgr)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3-based facade opening detector (drop-in replacement for facade_batch.py's 'batch' command)."
    )
    parser.add_argument("command", help="'batch' (only supported mode)")
    parser.add_argument("--image", required=True, help="Directory of rectified facade images")
    parser.add_argument("--save", required=True, help="Directory to save .pkl + _splash.png outputs")
    args = parser.parse_args()

    if args.command != "batch":
        raise SystemExit(f"Unsupported command '{args.command}' -- only 'batch' is implemented")

    os.makedirs(args.save, exist_ok=True)

    print("Building SAM3 image model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    files = sorted(os.listdir(args.image))
    ct = 0
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        print(ct, "filename", file)
        image_path = os.path.join(args.image, file)
        detect_facade(processor, image_path, args.save)
        ct += 1

    print(f"Done. Processed {ct} images -> {args.save}")


if __name__ == "__main__":
    main()
