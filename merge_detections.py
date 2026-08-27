# -*- coding: utf-8 -*-
"""
Merge two Step-2 detector outputs into one final per-image .pkl, combining
each model's strongest class:

  - SAM3      (sam3_dir)  -> keep class_id 1 (window) only
  - Mask R-CNN (mrcnn_dir) -> keep class_id 2 (balcony) and 3 (door) only

Rationale: SAM3 dramatically improved window detection on large/complex
facades, but its generic "door" prompt confidently over-detects (mismatching
repeating ground-floor windows/storefront glass as doors). Doors/balconies
were never the problem that motivated adopting SAM3, so the original,
purpose-trained Mask R-CNN model's detections are kept for those two classes.

Usage:
    python merge_detections.py --sam3 <ces2_sam3_dir> --mrcnn <ces2_mrcnn_dir> --out <ces2_dir>

Writes the merged (rois, class_ids) tuple to {out}/{imagename}.pkl for every
image present in sam3_dir, using its own _splash.png (already correct, just
the source image reused for its dimensions). Images present in mrcnn_dir but
not sam3_dir are skipped -- SAM3 is expected to have run over the same image
set.
"""

import argparse
import os

import joblib
import numpy as np


def merge_one(sam3_pkl_path, mrcnn_pkl_path):
    sam3_rois, sam3_class_ids = joblib.load(sam3_pkl_path)
    window_mask = sam3_class_ids == 1
    rois = [sam3_rois[window_mask]] if window_mask.any() else []
    class_ids = [sam3_class_ids[window_mask]] if window_mask.any() else []

    if os.path.isfile(mrcnn_pkl_path):
        mrcnn_rois, mrcnn_class_ids = joblib.load(mrcnn_pkl_path)
        keep_mask = (mrcnn_class_ids == 2) | (mrcnn_class_ids == 3)
        if keep_mask.any():
            rois.append(mrcnn_rois[keep_mask])
            class_ids.append(mrcnn_class_ids[keep_mask])

    if rois:
        return np.concatenate(rois, axis=0), np.concatenate(class_ids, axis=0)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3", required=True, help="Directory of SAM3 .pkl/_splash.png outputs")
    parser.add_argument("--mrcnn", required=True, help="Directory of Mask R-CNN .pkl outputs")
    parser.add_argument("--out", required=True, help="Directory to write merged .pkl (usually same as --sam3)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pkl_files = sorted(f for f in os.listdir(args.sam3) if f.endswith(".pkl"))
    ct = 0
    for f in pkl_files:
        imagename = f[:-4]
        sam3_pkl_path = os.path.join(args.sam3, f)
        mrcnn_pkl_path = os.path.join(args.mrcnn, f)
        rois, class_ids = merge_one(sam3_pkl_path, mrcnn_pkl_path)
        joblib.dump((rois, class_ids), os.path.join(args.out, f))
        ct += 1

    print(f"Merged {ct} images -> {args.out}")


if __name__ == "__main__":
    main()
