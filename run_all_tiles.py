"""
run_all_tiles.py  —  Run the full LoD3 pipeline for every tile in Data/.

Tiles that already have a finished GML in lod3/ are skipped automatically.
Tiles are processed one at a time (the intermediate folders appearence_rect2/
and ces2/ are reused and overwritten each run; only the final lod3/*.gml
files accumulate).

Usage:
    python run_all_tiles.py                   # process all remaining tiles
    python run_all_tiles.py --tile 6528       # process a single specific tile
    python run_all_tiles.py --start 6529      # resume from a given tile name
"""

import os
import subprocess
import argparse

# ---------------------------------------------------------------------------
# Paths – edit these if the conda environments were installed elsewhere
# ---------------------------------------------------------------------------
WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR    = WORKSPACE  # repo layout is flat -- no LoD3Framework--main/ wrapper
MRCNN_DIR   = os.path.join(MAIN_DIR,  "Mask_RCNN-2.1")
DATA_DIR    = os.path.join(WORKSPACE, "Data")
LOD3_DIR    = os.path.join(WORKSPACE, "lod3")
RECT_DIR    = os.path.join(WORKSPACE, "appearence_rect2")
CES2_DIR    = os.path.join(WORKSPACE, "ces2")
CES2_MRCNN_DIR = os.path.join(WORKSPACE, "ces2_mrcnn")  # temp dir, sam3 hybrid mode only

LOD3_PYTHON  = r"C:\Users\dmz-user\.conda\envs\lod3\python.exe"
MRCNN_PYTHON = r"C:\Users\dmz-user\.conda\envs\mrcnn\python.exe"
SAM3_PYTHON  = r"C:\Users\dmz-user\.conda\envs\sam3\python.exe"
WEIGHTS      = os.path.join(MAIN_DIR, "logs", "mask_rcnn_facade_0299.h5")

# Which Step 2 opening-detector to use: "mrcnn" (TF/Mask R-CNN, trained on facade
# data) or "sam3" (hybrid: Meta SAM3 for window detection -- see facade_batch_sam3.py
# and merge_detections.py -- since it dramatically outperforms Mask R-CNN on large/
# complex facades, PLUS Mask R-CNN for balcony/door, since SAM3's generic "door"
# prompt badly over-detects and doors/balconies were never the problem SAM3 was
# adopted to fix). "sam3" mode runs BOTH models and merges their output, so it
# needs both the mrcnn and sam3 conda envs, and HF_TOKEN with approved access to
# facebook/sam3.
DETECTOR = "sam3"

# When DETECTOR == "sam3", also run Mask R-CNN for balcony/door and merge it in.
# Currently this changes nothing in the output: the pipeline discards balcony and
# door detections entirely (inverse_facade_stage.getparaset() hardcodes
# eachfloor_balc/eachfloor_door to {}, and creat1.multiSurfaceWithEmbrasure()
# skips balconies when writing), and the doors that do appear are SYNTHESISED by
# amendwindow22_stage_PATCH.adddoor() from window geometry rather than detected.
# So it costs ~4-5 min/tile for output that is thrown away. Set True only if
# door/balcony parametrisation is re-enabled in getparaset() -- and re-measure
# which detector is actually better for those classes first, since SAM3's door
# quality was never fairly tested.
SAM3_WITH_MRCNN_OPENINGS = False

# PATH prefixes so Windows can resolve conda env DLLs without full activation
LOD3_ENV_BIN = (
    r"C:\Users\dmz-user\.conda\envs\lod3;"
    r"C:\Users\dmz-user\.conda\envs\lod3\Library\mingw-w64\bin;"
    r"C:\Users\dmz-user\.conda\envs\lod3\Library\usr\bin;"
    r"C:\Users\dmz-user\.conda\envs\lod3\Library\bin;"
    r"C:\Users\dmz-user\.conda\envs\lod3\Scripts;"
)
MRCNN_ENV_BIN = (
    r"C:\Users\dmz-user\.conda\envs\mrcnn;"
    r"C:\Users\dmz-user\.conda\envs\mrcnn\Library\mingw-w64\bin;"
    r"C:\Users\dmz-user\.conda\envs\mrcnn\Library\usr\bin;"
    r"C:\Users\dmz-user\.conda\envs\mrcnn\Library\bin;"
    r"C:\Users\dmz-user\.conda\envs\mrcnn\Scripts;"
)
SAM3_ENV_BIN = (
    r"C:\Users\dmz-user\.conda\envs\sam3;"
    r"C:\Users\dmz-user\.conda\envs\sam3\Library\mingw-w64\bin;"
    r"C:\Users\dmz-user\.conda\envs\sam3\Library\usr\bin;"
    r"C:\Users\dmz-user\.conda\envs\sam3\Library\bin;"
    r"C:\Users\dmz-user\.conda\envs\sam3\Scripts;"
)

# Tiles to always exclude from a run (e.g. known-bad data)
SKIP_TILES = set()
# ---------------------------------------------------------------------------


def get_tiles():
    return sorted(d.name for d in os.scandir(DATA_DIR) if d.is_dir())


def is_done(tile):
    return os.path.isfile(os.path.join(LOD3_DIR, f"LOD3.5_{tile}.gml"))


def clear_dir(path):
    """Remove all files in a directory without deleting the directory itself."""
    if os.path.isdir(path):
        for f in os.scandir(path):
            os.remove(f.path)


def run(cmd, cwd, env=None, label=""):
    print(f"  >> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"  [ERROR] {label} exited with code {result.returncode}")
        return False
    return True


def process_tile(tile):
    print(f"\n{'='*60}")
    print(f"  TILE: {tile}")
    print(f"{'='*60}")

    lod3_env = os.environ.copy()
    lod3_env["PATH"] = LOD3_ENV_BIN + lod3_env.get("PATH", "")

    # ---- Step 1: extract & rectify facade images ----
    print(f"\n[Step 1] Extracting facade images for tile {tile} ...")
    clear_dir(RECT_DIR)
    ok = run(
        [LOD3_PYTHON, "Main_auto.py", "--tile", tile, "--step", "1"],
        cwd=MAIN_DIR,
        env=lod3_env,
        label="Step 1",
    )
    if not ok:
        return False

    # ---- Step 2: opening detection (Mask R-CNN or SAM3) ----
    clear_dir(CES2_DIR)
    os.makedirs(CES2_DIR, exist_ok=True)

    if DETECTOR == "sam3":
        print(f"\n[Step 2a] SAM3 window detection for tile {tile} ...")
        sam3_env = os.environ.copy()
        sam3_env["PATH"] = SAM3_ENV_BIN + sam3_env.get("PATH", "")
        ok = run(
            [SAM3_PYTHON, "facade_batch_sam3.py", "batch",
             "--image", RECT_DIR,
             "--save",  CES2_DIR + "/"],
            cwd=MAIN_DIR,
            env=sam3_env,
            label="Step 2a (SAM3)",
        )
        if not ok:
            return False

        if SAM3_WITH_MRCNN_OPENINGS:
            print(f"\n[Step 2b] Mask R-CNN balcony/door detection for tile {tile} ...")
            clear_dir(CES2_MRCNN_DIR)
            os.makedirs(CES2_MRCNN_DIR, exist_ok=True)
            mrcnn_env = os.environ.copy()
            mrcnn_env["PATH"] = MRCNN_ENV_BIN + mrcnn_env.get("PATH", "")
            ok = run(
                [MRCNN_PYTHON, "facade_batch.py", "batch",
                 "--weights", WEIGHTS,
                 "--image",   RECT_DIR,
                 "--save",    CES2_MRCNN_DIR + "/"],
                cwd=MRCNN_DIR,
                env=mrcnn_env,
                label="Step 2b (Mask R-CNN)",
            )
            if not ok:
                return False

            print(f"\n[Step 2c] Merging SAM3 windows + Mask R-CNN balcony/door for tile {tile} ...")
            ok = run(
                [LOD3_PYTHON, "merge_detections.py",
                 "--sam3",  CES2_DIR,
                 "--mrcnn", CES2_MRCNN_DIR,
                 "--out",   CES2_DIR],
                cwd=MAIN_DIR,
                env=lod3_env,
                label="Step 2c (merge)",
            )
    else:
        print(f"\n[Step 2] Mask R-CNN segmentation for tile {tile} ...")
        mrcnn_env = os.environ.copy()
        mrcnn_env["PATH"] = MRCNN_ENV_BIN + mrcnn_env.get("PATH", "")
        ok = run(
            [MRCNN_PYTHON, "facade_batch.py", "batch",
             "--weights", WEIGHTS,
             "--image",   RECT_DIR,
             "--save",    CES2_DIR + "/"],
            cwd=MRCNN_DIR,
            env=mrcnn_env,
            label="Step 2 (Mask R-CNN)",
        )
    if not ok:
        return False

    # ---- Steps 3-5: parametric model → optimise → write LoD3 GML ----
    print(f"\n[Steps 3-5] Parametric model + GML writing for tile {tile} ...")
    os.makedirs(LOD3_DIR, exist_ok=True)
    ok = run(
        [LOD3_PYTHON, "Main_auto.py", "--tile", tile, "--step", "2"],
        cwd=MAIN_DIR,
        env=lod3_env,
        label="Steps 3-5",
    )
    if not ok:
        return False

    out = os.path.join(LOD3_DIR, f"LOD3.5_{tile}.gml")
    size_mb = os.path.getsize(out) / 1e6 if os.path.isfile(out) else 0
    print(f"\n[DONE] tile {tile}  ->  lod3/LOD3.5_{tile}.gml  ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run LoD3 pipeline for all tiles.")
    parser.add_argument("--tile",  help="Process a single tile only")
    parser.add_argument("--start", help="Skip tiles before this name (resume mode)")
    args = parser.parse_args()

    os.makedirs(RECT_DIR, exist_ok=True)
    os.makedirs(CES2_DIR, exist_ok=True)
    os.makedirs(LOD3_DIR, exist_ok=True)

    tiles = get_tiles()
    if args.tile:
        tiles = [args.tile]
    elif args.start:
        tiles = [t for t in tiles if t >= args.start]

    skipped = [t for t in tiles if t in SKIP_TILES]
    tiles   = [t for t in tiles if t not in SKIP_TILES]
    if skipped:
        print(f"Skipping tiles: {skipped}")

    total  = len(tiles)
    done   = sum(is_done(t) for t in tiles)
    todo   = [t for t in tiles if not is_done(t)]
    print(f"Tiles found: {total}  |  Already done: {done}  |  Remaining: {len(todo)}")

    failed = []
    for i, tile in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}]", end="")
        if not process_tile(tile):
            failed.append(tile)
            print(f"  Skipping tile {tile} after error — continuing with next tile.")

    print(f"\n{'='*60}")
    print(f"Finished.  Processed: {len(todo)-len(failed)}  |  Failed: {len(failed)}")
    if failed:
        print("Failed tiles:", failed)


if __name__ == "__main__":
    main()
