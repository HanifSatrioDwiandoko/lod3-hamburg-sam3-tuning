# LoD3 Hamburg — SAM3 detector + facade-placement fixes

Work on the LoD3 Hamburg CityGML facade-reconstruction pipeline (Wang et al. 2024,
ISPRS J P&RS). Two separate things live here:

1. **A SAM3-based window detector** replacing Mask R-CNN for Step 2.
2. **Six geometry/logic fixes** in the framework's own code that were causing
   openings to be placed in the wrong place ("overshooting" past wall ends and
   roof edges) or to be deleted outright.

The geometry fixes are **independent of which detector you use** — they matter
just as much with the original Mask R-CNN.

Input tile data and generated outputs are not included; only code.

---

## 1. Why SAM3

The original Mask R-CNN model was fine-tuned on a limited local facade-photo set
and generalised poorly to some large or complex Hamburg facades. Worst measured
case: an 85 m wall produced **8 windows, all at a single (x,y) position** — one
degenerate vertical column.

SAM3 (Meta, Nov 2025) does text-prompted concept segmentation — prompt `"window"`
and it segments every window instance. On that same wall it produced **~640
windows across ~240 distinct horizontal positions**. It is also faster here
(~3 min vs ~8 min per tile), because PyTorch 2.7+/cu128 has native Blackwell
support while the TensorFlow stack must JIT-compile CUDA kernels.

### Doors, balconies, and why Mask R-CNN is switched off

`run_all_tiles.py` briefly ran **both** models, taking windows from SAM3 and
balconies/doors from Mask R-CNN via `merge_detections.py`. That is now disabled
(`SAM3_WITH_MRCNN_OPENINGS = False`), because Mask R-CNN was **measured to
contribute nothing to the output**.

The split was originally decided on evidence that turned out to be **wrong**. A
tile showed 224 doors (up to 28 on one wall), attributed to SAM3's `"door"`
prompt over-detecting. It was later traced to
`amendwindow22_stage_PATCH.adddoor()`, which **synthesises** doors from
window/stairwell-window geometry, entirely independently of any detector. Raw
detections were only ~20 doors tile-wide from either model.

The pipeline discards door and balcony detections completely:

```python
# inverse_facade_stage.getparaset() -- the real computation sits commented out
eachfloor_balc={}
eachfloor_door={}
```
```python
# creat1.multiSurfaceWithEmbrasure() -- balconies are never written
if label != 'balcony':
```

**Experimentally confirmed** (only valid once the RNG was seeded, see §2): with
identical SAM3 windows and a fixed seed, stripping 33 balcony + 19 door
detections and regenerating produced a **byte-identical set of openings** —
13,223 windows and 118 doors either way, zero openings unique to either version.

So only class 1 (window) affects the output. Mask R-CNN cost ~4-5 min/tile for
output that is provably discarded. It is retained as a fallback (`DETECTOR =
"mrcnn"`) and as the published baseline, but is not run. Note SAM3's own door
quality was never fairly measured — that would need testing before enabling
door/balcony parametrisation with either model.

## 2. Reproducibility — the pipeline was not deterministic

The parametric facade fitting in `amendwindow22_stage_PATCH` samples window
width, height, spacing and floor gap with `np.random.uniform`/`randint`
(`get_windows_edge()` and the layout initialisation, ~lines 117-130 and
431-441). This was **unseeded**, so results were not reproducible:

| | unseeded | seeded |
|---|---|---|
| Windows, two identical runs | 13,151 vs 13,052 | **13,223 vs 13,223** |
| Window geometries differing | ~9,100 (**69 %**) | **0** |

`Main_auto.py` now sets `RANDOM_SEED = 0` at module load, seeding `np.random`
and `random` for both `--step 1` and `--step 2`. Change it to explore a
different layout sampling deliberately.

This matters beyond tidiness: **any A/B comparison run before seeding was
measuring RNG noise**, since run-to-run variance dwarfed most real effects.
Comparisons made against fixed intermediate files (the overshoot bound and the
roofline profile below) are unaffected, as they never compared two runs.

## 3. Environment

Needs its own conda env (Python 3.12+, PyTorch 2.7+, CUDA 12.8):

```powershell
conda create -n sam3 python=3.12 -y
conda install -n sam3 pip -y
<env>\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
<env>\python.exe -m pip install opencv-python pillow numpy huggingface_hub matplotlib scikit-image joblib
cd LoD3Framework--main/sam3-main
<env>\python.exe -m pip install -e ".[train,notebooks,dev]" psutil einops triton-windows pycocotools
```

`triton-windows`, `einops`, `pycocotools` and `psutil` need explicit installs —
the package imports them at runtime but does not declare them all.

The `sam3.pt` checkpoint is **gated**: request access at
[facebook/sam3](https://huggingface.co/facebook/sam3), then set a token —
`build_sam3_image_model()` downloads and caches it automatically.

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_xxxx", "User")
```

Inference must run inside `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
or it raises a dtype mismatch — the upstream README's basic example omits this.

## 4. Contents

| Path | Role |
|---|---|
| `run_all_tiles.py` | Batch driver. `DETECTOR = "mrcnn" \| "sam3"` switch near the top |
| `LoD3Framework--main/facade_batch_sam3.py` | SAM3 Step-2 detector. **All SAM3 tuning is the `CLASSES` list**: `(class_id, prompt, min_score)` |
| `LoD3Framework--main/merge_detections.py` | Merges SAM3 windows + Mask R-CNN balcony/door |
| `LoD3Framework--main/coor_opening4.py` | Opening placement — fixes 1, 2, 3, 4 below |
| `LoD3Framework--main/rect_im.py` | Facade rectification — fix 5 |
| `LoD3Framework--main/Main_auto.py`, `readgml1.py` | Wall→image matching — fix 6 |
| `LoD3Framework--main/inverse_facade_stage.py`, `amendwindow22_stage_PATCH.py` | Parametric model — hang guards + door caps |
| `LoD3Framework--main/creat1.py` | GML writer — missing-`gml:id` guards |
| `LoD3Framework--main/sam3-main/` | Vendored SAM3 (Meta, MIT). Upstream demo `assets/` and `examples/` removed (61 MB, unused) |

`facade_batch_sam3.py` reproduces Mask R-CNN's output contract exactly, so nothing
downstream changes: `{save}/{name}.pkl` = `joblib.dump((rois, class_ids))` with
`rois[i] = [y1, x1, y2, x2]` (SAM3 natively returns `x1,y1,x2,y2` — converted) and
`class_ids` 1=window/2=balcony/3=door, plus `{name}_splash.png`, which downstream
reads only for its pixel dimensions.

## 5. The geometry fixes

Measured on tile 6632. Root cause of most of it: **corner selection used
`s = sqrt(x²+y²) + z`**. In a projected CRS (EPSG:25832) `sqrt(x²+y²)` is ~5.96e6 m
and varies by only a metre or two along one wall, while `z` varies by up to 30 m —
so `s` was dominated by height, and `argmin`/`argmax` returned the **lowest and
highest vertex** rather than the wall's two ends.

1. **Wall axis / origin** (`coor_opening4.wall_axis`) — replaces that metric with
   the true XY diameter, ordered low-radius → high-radius to match how
   `rect_im.order_points()` builds the rectified image (image-left = low radius).
   *Before: origin sat up to 15.5 m inside the wall on 27 walls, pushing whole
   window grids up to 13.6 m off the far end. After: 0 walls.*

2. **Rotation direction** (`get_rotation`/`rotator`) — the original
   `atan(dy/dx)` + sign-of-`tan` reconstruction cannot distinguish a wall running
   `(dx,dy)` from `(-dx,-dy)`, flipping placement 180°. Now a unit vector.
   (Subsumed by fix 1, which computes the direction correctly by construction.)

3. **Embrasure depth** (`rotator`) — `sqrt(x² + depth²)` folded the 0.5 m depth
   into the *along-wall* direction, skewing openings sideways instead of
   recessing them. Openings are now flush, so the hole matches the opening.

4. **Roofline clipping** (`clip_to_roofline` / `wall_height_profile`) — drops
   openings above the wall's true roofline, for walls where a lower and a taller
   wing share one `WallSurface`. **The profile must be built by rasterising the
   ring's edges, not by sampling its vertices**: many walls carry extra vertices
   along their *base* at positions where the top edge has none, so a
   vertex-sampled profile reports near-zero height there and deletes every
   opening in that stretch. *Vertex version wrongly deleted 973 valid windows
   (1033 clipped vs 60 genuinely above the roof); edge version deletes 0.*

5. **Rectification scale** (`rect_im.four_point_transform`) — `scale` was derived
   from an `int()`-truncated pixel height, inflating the wall's apparent metric
   width by up to `1/maxHeight`. On short, wide walls the rectified image is only
   a few pixels tall, so widths were overstated by up to 14 % (45.48 m reported
   for a 39.80 m wall). Now uses the untruncated ratio.
   *Before: 99 walls with `W` too large by >0.5 m. After: 0.*

6. **Degenerate ring filter** (`Main_auto` + `readgml1.polygon_area_3d` /
   `wall_span_length`) — requires area > 10 m², height > 1.5 m and corner span >
   1.5 m before a ring is treated as a photographable facade. Excludes thin
   parapet/coping strips and malformed rings. Note it also drops ~164 legitimate
   but small facades (~3.6 × 3.2 m) that fall just under the area threshold — a
   tunable if blank facades are observed.

Plus, from earlier work: empty-array and iteration guards in `adj_floor()`, an
O(n²) guard in `check_and_remove_overlapping()` (both prevent hangs on degenerate
facades), missing-`gml:id` guards in `creat1.py`, and door-count caps
(`MAX_STAIRWELL_COLUMNS`, `MAX_DOORS_PER_FACADE`) that took synthesised doors from
224 to ~117 tile-wide.

### Verified state (tile 6632)

| Metric | Before | After |
|---|---|---|
| Openings past the wall end (>0.5 m) | 23 walls, worst 13.56 m | **0, worst −0.00 m** |
| `para['W']` vs true wall width | median 1.011 | **median 1.0000, 99.2 % within 1 %** |
| Windows wrongly deleted by roofline clip | 973 | **0** |
| Synthesised doors, tile-wide | 224 (28 on one wall) | **~117 (13 max)** |
| Known-bad 85 m wall | 8 windows / 1 position | **~640 / ~240 positions** |

## 6. Running

```powershell
# set DETECTOR in run_all_tiles.py, then:
python run_all_tiles.py --tile 6632
```

## 7. Status / open items

- Validated on **tile 6632 only**. The other ~89 tiles have not been re-run.
- Mask R-CNN is switched off (see §1); re-enable via `SAM3_WITH_MRCNN_OPENINGS`.
- The 10 m² area threshold (fix 6) may be worth lowering.
- No automated tests exist; all validation has been manual measurement against
  the source geometry.
