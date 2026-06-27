"""
crop_nuclei_sam.py — LiveFISH nucleus auto-cropping via µSAM (vit_b_lm, APG mode)

Usage:
    conda run -n base python "code (being modified)/crop_nuclei_sam.py" \
        "data for analysis/FOV (.nd2 files)" \
        --nucleus-channel 0 --margin 30 \
        --min-area 1000 --max-area 200000 \
        --segmentation-mode apg --model-type vit_b_lm

    # Single file:
    conda run -n base python "code (being modified)/crop_nuclei_sam.py" \
        "data for analysis/FOV (.nd2 files)/....nd2" ...
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # matplotlib will not plot to the screen, only to the file
import matplotlib.pyplot as plt
import numpy as np
from nd2 import ND2File
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter # pre and post processing of masks
from skimage.feature import peak_local_max # peak detection for watershed split
from skimage.measure import regionprops # region properties for mask filtering
from skimage.segmentation import watershed # watershed segmentation for mask (adjacent nuclei) splitting

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_BORDER_MARGIN_PX = 5 # minimum distance from the border of the image to the nucleus center
MIN_SOLIDITY     = 0.70 # minimum solidity of the nucleus to not be passed to watershed split (by raising this, we allow more fragmented nuclei to be passed to watershed split)
SPLIT_SIGMA      = 5 # sigma for gaussian filtering of the distance transform for peak detection
SPLIT_MIN_DIST   = 20 # minimum distance between peaks of nuclei to be considered as separate nuclei (and therefore passed to watershed split)
MERGE_PROXIMITY  = 2 # maximum distance between nuclei to be considered as part of the same nucleus (and therefore passed to merging)
MERGE_MIN_SOLID  = 0.60 # minimum solidity of the nucleus to not be passed to merging (by raising this, we allow more fragmented nuclei to be passed to merging)
MIN_CIRC         = 0.3 # minimum circularity 4*pi*area/perimeter^2 of the nucleus to not be passed to merging (by raising this, we allow more elongated nuclei to be passed to merging)
IOU_THRESH       = 0.3 # maximum allowed overlap between nuclei to be considered as part of the same nucleus and discarded since it's redundant
CONTAIN_THRESH   = 0.5 # minimum containment of the nucleus to not be passed to merging (raising this allows more elongated nuclei to be passed to merging)


# ── helpers ────────────────────────────────────────────────────────────────────
"""
load_fov() loads the .nd2 file and returns a numpy array of the FOV (T, C, Y, X) uint16
input: path to the .nd2 file
output: numpy array of the FOV (T, C, Y, X) uint16

1. open the .nd2 file
2. get the array from the .nd2 file
3. get the axes from the .nd2 file
4. max-project the z-axis if it exists
5. transpose the array to the target order
6. return the array
"""
def load_fov(path: Path) -> np.ndarray:
    """Load nd2 → (T, Z, C, Y, X) uint16, preserving all axes.
    The nd2 library squeezes out size-1 axes — we reinsert them so the
    array always has the same number of dims regardless of acquisition mode."""
    with ND2File(path) as f:
        arr = f.asarray()
        keys = list(f.sizes.keys())

    axes = [k.upper() for k in keys]
    print(f"  nd2 axes string: {''.join(axes)}, raw shape: {arr.shape}")

    # Reinsert any size-1 axes the nd2 library squeezed out
    for ax in ('T', 'Z', 'C'):
        if ax not in axes:
            arr = np.expand_dims(arr, axis=0)
            axes.insert(0, ax)
            print(f"  Reinserted size-1 '{ax}' axis (squeezed out by nd2 library)")

    target = ['T', 'Z', 'C', 'Y', 'X']
    order = [axes.index(a) for a in target]
    arr = np.transpose(arr, order).astype(np.uint16)
    print(f"  Final shape TZCYX: {arr.shape}")
    return arr

"""
norm_u8() normalizes the image to the uint8 range (0-255) for the µSAM model
input: numpy array of the image
output: numpy array of the image in the uint8 range (0-255)

1. get the percentile values of the image
2. clip the image to the percentile values
3. normalize the image to the uint8 range (0-255)
4. return the image
"""
def norm_u8(img: np.ndarray, lo_pct=1, hi_pct=99) -> np.ndarray:
    """Percentile-stretch float/uint → uint8."""
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    if hi == lo:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


"""
circularity() calculates the circularity of the nucleus
input: area and perimeter of the nucleus
output: circularity of the nucleus

1. if the perimeter is 0, return 0.0
2. calculate the circularity as 4*pi*area/perimeter^2
3. return the circularity
"""
def circularity(area, perimeter):
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)



"""
try_split_mask() attempts to split a low-solidity mask into two pieces using watershed segmentation
input: mask of the nucleus, minimum area of the nucleus
output: list of bool arrays (1 or 2 pieces) or None on failure

1. calculate the distance transform of the mask
2. smooth the distance transform with a gaussian filter
3. find the peaks in the distance transform
4. create markers for the watershed segmentation
5. perform the watershed segmentation
6. get the properties of the nucleus
7. if the area is too small, we discard the nucleus
8. if the circularity is low, (too elongated) we discard the nucleus
9. return the pieces
"""
def try_split_mask(mask: np.ndarray, min_area: int):
    dist = distance_transform_edt(mask) # this modifies the mask by creating a distance transform of the mask
    dist_smooth = gaussian_filter(dist, sigma=SPLIT_SIGMA)
    coords = peak_local_max(dist_smooth, min_distance=SPLIT_MIN_DIST,
                            num_peaks=2, labels=mask)
    if len(coords) < 2:
        return None
    markers = np.zeros_like(mask, dtype=int)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labels_ws = watershed(-dist_smooth, markers, mask=mask)
    pieces = []
    for lbl in [1, 2]:
        piece = labels_ws == lbl # get the mask for the nucleus
        props = regionprops(piece.astype(np.uint8)) # get the properties of the nucleus
        if not props: # if no properties, the nucleus isn't valid
            continue
        p = props[0]
        if p.area < min_area: # if the area is too small, we discard the nucleus
            continue
        if circularity(p.area, p.perimeter) < MIN_CIRC: # if the circularity is low, (too elongated) we discard the nucleus
            continue 
        pieces.append(piece) # add the piece to the list
    if len(pieces) == 2: # if there are 2 pieces, we return the pieces
        return pieces
    return None # if there are not 2 pieces, we return None (keeping the original nucleus, probably malformed)

"""
filter_masks() filters the masks based on the area, circularity, and border
input: list of masks, image shape, minimum area, maximum area, border margin
output: list of masks

1. get the height and width of the image
2. create a list to keep the masks
3. create a dictionary to keep the statistics
4. for each mask:
5. get the properties of the mask
6. if the area is too small, we discard the mask
7. if the area is too large, we discard the mask
8. if the centroid is too close to the border, we discard the mask
9. if the solidity is too low, we attempt to split the mask
10. if the split is successful, we keep the pieces and add the statistics
11. if the split is unsuccessful, we keep the original mask and add the statistics
12. return the kept masks and the statistics
"""
def filter_masks(masks, img_shape, min_area, max_area, border_margin):
    H, W = img_shape 
    kept = [] # list to keep the masks
    stats = {"too_small": 0, "too_large": 0, "border": 0, # dictionary to keep the statistics
             "split_ok": 0, "split_fail_kept": 0, "split_fail_dropped": 0, # you can see what global variables might need to be changed based on the respective statistics
             "kept": 0}  
    for mask in masks:
        props = regionprops(mask.astype(np.uint8)) # get the properties of the mask
        if not props:
            continue # if no properties, the mask isn't valid
        p = props[0] # get the first property
        area = p.area # get the area of the mask
        cy, cx = p.centroid # get the centroid of the mask

        # check if the area is too small, too large, or if centroid is too close to the border
        if area < min_area:
            stats["too_small"] += 1
            continue
        if area > max_area * 2: # if the area is too large, we discard the mask (2x allows leeway for merged pairs)
            stats["too_large"] += 1
            continue
        if cy < border_margin or cy > H - border_margin:
            stats["border"] += 1
            continue
        if cx < border_margin or cx > W - border_margin:
            stats["border"] += 1
            continue

        if p.solidity < MIN_SOLIDITY: # if the solidity is too low, we attempt to split the mask
            pieces = try_split_mask(mask, min_area)
            if pieces: # if the split is successful, we keep the pieces and add the statistics
                kept.extend(pieces)
                stats["split_ok"] += len(pieces)
                continue
            else: # if the split is unsuccessful, we keep the original mask and add the statistics
                if area <= max_area:
                    kept.append(mask)
                    stats["split_fail_kept"] += 1
                else:
                    stats["split_fail_dropped"] += 1
                continue

        kept.append(mask)
        stats["kept"] += 1

    return kept, stats

"""
deduplicate_masks() removes masks that substantially overlap with an already-accepted larger mask.

input: list of masks
output: list of surviving masks

1. Sort masks largest-first so that when two masks overlap, the larger one is
   accepted and the smaller one is rejected (not the reverse).
2. For each candidate (largest to smallest), compute IoU and containment against
   every already-accepted mask. Discard if either exceeds its threshold.
3. Return the surviving masks.
"""

def deduplicate_masks(masks):
    """Remove masks that substantially overlap with larger masks."""
    if not masks:
        return masks
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]
    accepted = []
    for i in order:
        m = masks[i]
        discard = False
        for a in accepted:
            inter = (m & a).sum()
            union = (m | a).sum()
            iou = inter / union if union else 0
            containment = inter / m.sum() if m.sum() else 0  # fraction of m's pixels covered by already-accepted a
            if iou > IOU_THRESH or containment > CONTAIN_THRESH:
                discard = True  # m is a duplicate or subset of a — keep a, drop m
                break
        if not discard:
            accepted.append(m)
    return accepted


"""
merge_adjacent_masks() fuses pairs of µSAM fragment masks that are close
enough to be adjacent and whose union passes nucleus validity checks.

µSAM occasionally splits a single nucleus into two separate masks. This
function identifies and fuses such pairs.

1. Build a square dilation kernel of radius MERGE_PROXIMITY to use as a
   proximity detector.
2. Repeat until a full pass produces no new merges (handles chains: if
   A+B merge, then B+C can merge in the next pass):
   a. For each unmerged mask i, dilate it and check whether any other
      unmerged mask j overlaps the dilation (i.e. is within MERGE_PROXIMITY
      px of i).
   b. If adjacent, form the union and validate: area ≤ max_area AND
      solidity ≥ MERGE_MIN_SOLID. Low solidity means the union is still
      two-blob shaped — not a real fused nucleus — so reject it.
   c. Accept the first valid partner found (greedy). Mark both used; add
      union to result.
   d. Unmerged masks carry through to result unchanged.
3. Return the updated mask list and total merge count.
"""
def merge_adjacent_masks(masks, max_area):
    """Merge pairs of µSAM fragment masks that are adjacent and together form a valid nucleus."""
    if len(masks) < 2:
        return masks, 0

    # square kernel: dilating by struct expands a mask by MERGE_PROXIMITY px in all 8 directions
    struct = np.ones((MERGE_PROXIMITY * 2 + 1, MERGE_PROXIMITY * 2 + 1), dtype=bool)
    total_merges = 0

    changed = True
    while changed:  # repeat until a full pass produces no merges; needed so chains of ≥3 fragments can fully fuse
        changed = False
        used = [False] * len(masks)
        result = []
        for i in range(len(masks)):
            if used[i]:
                continue
            dil_i = binary_dilation(masks[i], structure=struct)
            merged_with = None
            for j in range(i + 1, len(masks)):
                if used[j]:
                    continue
                if not (dil_i & masks[j]).any():
                    continue  # j has no pixel within MERGE_PROXIMITY px of i — not adjacent
                union = masks[i] | masks[j]
                props = regionprops(union.astype(np.uint8))
                if not props:
                    continue
                p = props[0]
                if p.area > max_area:
                    continue
                if p.solidity < MERGE_MIN_SOLID:
                    continue  # union is still two-blob shaped; rejecting avoids fusing unrelated nearby nuclei
                merged_with = j
                result.append(union)
                used[i] = True
                used[j] = True
                total_merges += 1
                changed = True
                break  # greedy: take first valid partner; remaining masks re-evaluated next pass
            if merged_with is None and not used[i]:  # not used[i] guards against i already consumed as j in an earlier merge this pass
                result.append(masks[i])
                used[i] = True
        masks = result

    return masks, total_merges


"""
crop_with_suppression() extracts a padded bounding-box crop for one nucleus
and zeros every pixel that belongs to any other nucleus across all T and C.

1. Find the bounding box of the target mask and expand by margin px, clamping
   to image edges.
2. Build a full-FOV suppression mask: union of all other accepted masks.
3. Copy the FOV subarray (copy is required — the slice is a view into fov;
   zeroing in-place would corrupt fov for later crops that share pixels).
4. Apply the suppression mask to zero all other nuclei across every timepoint
   and channel simultaneously via boolean indexing.
5. Return the (T, C, cropH, cropW) crop and the bbox tuple (r0, r1, c0, c1).
"""
def crop_with_suppression(fov: np.ndarray, mask: np.ndarray,
                          all_masks: list, margin: int):
    """Return (T, Z, C, cropH, cropW) uint16 with all other nuclei zeroed."""
    T, Z, C, H, W = fov.shape
    rows, cols = np.where(mask)
    r0 = max(0, rows.min() - margin)
    r1 = min(H, rows.max() + margin + 1)
    c0 = max(0, cols.min() - margin)
    c1 = min(W, cols.max() + margin + 1)

    suppress = np.zeros((H, W), dtype=bool)
    for other in all_masks:
        if other is mask:
            continue
        suppress |= other

    crop = fov[:, :, :, r0:r1, c0:c1].copy()
    crop[:, :, :, suppress[r0:r1, c0:c1]] = 0
    return crop, (r0, r1, c0, c1)


# ── visualisation ──────────────────────────────────────────────────────────────

def _draw_masks_numbered(ax, nuc_u8, masks, title):
    ax.imshow(nuc_u8, cmap='gray', alpha=0.5)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(masks), 1))
    for idx, m in enumerate(masks):
        color = cmap(idx % cmap.N)[:3]  # % cmap.N wraps if more masks than colors; [:3] drops alpha
        rgba = np.zeros((*m.shape, 4))
        rgba[m] = [*color, 0.5]
        ax.imshow(rgba)
        props = regionprops(m.astype(np.uint8))
        if props:
            cy, cx = props[0].centroid
            ax.text(cx, cy, str(idx + 1), color='white',
                    fontsize=7, ha='center', va='center', fontweight='bold')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def save_seg_overview(nuc_u8, segmentation, filtered_masks, merged_masks,
                      deduped_masks, out_path: Path):
    """4-panel figure: raw image | µSAM raw | after filter+merge | final deduped."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(nuc_u8, cmap='gray')
    axes[0].set_title("Time-averaged nucleus channel (uint8)", fontsize=10)
    axes[0].axis('off')

    overlay_raw = np.zeros((*nuc_u8.shape, 3), dtype=float)
    n_inst = int(segmentation.max())
    cmap_inst = matplotlib.colormaps.get_cmap('tab20').resampled(max(n_inst, 1))
    for i in range(1, n_inst + 1):
        color = cmap_inst(i % cmap_inst.N)[:3]
        for ch in range(3):
            overlay_raw[:, :, ch] += (segmentation == i) * color[ch]
    overlay_raw = np.clip(overlay_raw, 0, 1)  # adjacent instances can additively exceed 1.0
    axes[1].imshow(nuc_u8, cmap='gray', alpha=0.5)
    axes[1].imshow(overlay_raw, alpha=0.5)
    axes[1].set_title(f"µSAM raw: {n_inst} instances", fontsize=10)
    axes[1].axis('off')

    _draw_masks_numbered(axes[2], nuc_u8, merged_masks,
                         f"After filter+merge: {len(merged_masks)}")
    _draw_masks_numbered(axes[3], nuc_u8, deduped_masks,
                         f"Final (deduped): {len(deduped_masks)}")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved segmentation overview → {out_path.name}")


def save_crop_grid(crops_info, nuc_u8, out_path: Path):
    """Grid of nucleus-channel thumbnails (max-T) for all accepted crops."""
    n = len(crops_info)
    if n == 0:
        return
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols  # ceiling division

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.5))
    axes = np.array(axes).reshape(-1)  # flatten 2D axes grid → 1D for uniform indexing

    for idx, (crop, bbox, mask) in enumerate(crops_info):
        nuc_crop = crop[:, :, 0].max(axis=1).max(axis=0).astype(float) 
        ax = axes[idx]
        ax.imshow(nuc_crop, cmap='gray',
                  vmin=np.percentile(nuc_crop, 1),
                  vmax=np.percentile(nuc_crop, 99))
        ax.set_title(f"#{idx+1}\n{crop.shape[2]}×{crop.shape[3]}px", fontsize=7)
        ax.axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"All {n} nucleus crops — nucleus channel (max-T projection)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved crop grid → {out_path.name}")


def save_suppression_demo(crops_info, fov, deduped_masks, out_path: Path,
                          max_show=6):
    """Side-by-side before/after suppression for up to max_show nuclei."""
    n = min(len(crops_info), max_show)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]  # plt.subplots(1,2) returns 1D; add dim for consistent [row,col] indexing

    for row, (crop, bbox, mask) in enumerate(crops_info[:n]):
        r0, r1, c0, c1 = bbox
        # max-project Z (axis 1) then T (axis 0)
        raw_region  = fov[:, :, 0, r0:r1, c0:c1].max(axis=1).max(axis=0).astype(float)
        supp_region = crop[:, :, 0].max(axis=1).max(axis=0).astype(float)

        vmin = np.percentile(raw_region[raw_region > 0], 1) if raw_region.max() > 0 else 0 # exclude zeroed suppression pixels from contrast floor 
        vmax = np.percentile(raw_region, 99)

        axes[row, 0].imshow(raw_region,  cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"#{row+1} Before suppression", fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(supp_region, cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"#{row+1} After suppression", fontsize=8)
        axes[row, 1].axis('off')

    plt.suptitle("Neighbour suppression demo (nucleus channel, max-T)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved suppression demo → {out_path.name}")


def save_all_channels_demo(crops_info, out_path: Path, max_show=4):
    """Show all 4 channels for the first max_show crops."""
    n = min(len(crops_info), max_show)
    if n == 0:
        return
    n_chan = crops_info[0][0].shape[2]
    chan_labels = ['C0 Nucleus (DAPI)', 'C1 640nm', 'C2 488nm', 'C3 561nm']

    fig, axes = plt.subplots(n, n_chan, figsize=(n_chan * 2.8, n * 2.8))
    if n == 1:
        axes = np.atleast_2d(np.array(axes)).reshape(-1, n_chan)  # plt.subplots(1,n_chan) returns 1D; add dim for consistent [row,col] indexing

    for row, (crop, bbox, mask) in enumerate(crops_info[:n]):
        for ch in range(n_chan):
            img = crop[:, :, ch].max(axis=1).max(axis=0).astype(float)
            axes[row, ch].imshow(img, cmap='gray',
                                 vmin=np.percentile(img, 1),
                                 vmax=np.percentile(img, 99))
            if row == 0:
                axes[row, ch].set_title(
                    chan_labels[ch] if ch < len(chan_labels) else f"C{ch}",
                    fontsize=8)
            axes[row, ch].set_ylabel(f"#{row+1}", fontsize=8)
            axes[row, ch].axis('off')

    plt.suptitle("All channels — first crops (max-T projection)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved all-channels demo → {out_path.name}")


# ── main processing ────────────────────────────────────────────────────────────

"""
process_file() runs the full nucleus-cropping pipeline on one .nd2 FOV file.

1. Load and Z-max-project the .nd2 → (T, C, H, W) uint16.
2. Time-average the nucleus channel → (H, W) float, normalize to uint8.
   Time-averaging accumulates stationary locus signal while washing out
   diffuse unbound probes — gives µSAM a clean nucleus silhouette.
3. Run µSAM automatic instance segmentation → (H, W) int label map.
4. Convert labels → bool masks; label 0 is background, so start at 1.
5. filter_masks: drop debris/border nuclei; watershed-split low-solidity
   (merged-pair) masks.
6. merge_adjacent_masks: fuse µSAM fragments back into whole nuclei.
7. deduplicate_masks: remove masks that substantially overlap a larger one.
8. Crop each nucleus with neighbour suppression; emit visualizations.
9. Write <stem>_crops.json encoding bboxes and suppression pixel coords
   for save_crops.py to consume without re-running µSAM.
"""
def process_file(nd2_path: Path, predictor, segmenter, args):
    print(f"\n── {nd2_path.name} ──")

    fov = load_fov(nd2_path)          # (T, Z, C, Y, X)
    T, Z, C, H, W = fov.shape
    print(f"  Loaded: T={T} Z={Z} C={C} Y={H} X={W}")

    ch_means = [fov[:, :, c].mean() for c in range(C)]
    print(f"  Channel means: {[f'{m:.1f}' for m in ch_means]}")

    # Max-project Z only for segmentation input — µSAM needs 2D
    # The full Z info is preserved in fov for cropping and TIFF export
    nuc_avg = fov[:, :, args.nucleus_channel].max(axis=1).mean(axis=0).astype(float)
    nuc_u8  = norm_u8(nuc_avg)

    from micro_sam.automatic_segmentation import automatic_instance_segmentation  # deferred import — µSAM has heavy import-time side effects; keep it after model load
    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=nuc_u8,
        ndim=2,
    )

    n_inst = int(segmentation.max())
    print(f"  µSAM found {n_inst} instances")

    raw_masks = [segmentation == i for i in range(1, n_inst + 1)]  # label 0 is background — skip it

    filtered_masks, stats = filter_masks(raw_masks, (H, W), args.min_area, args.max_area, args.border_margin)
    print(f"  {len(filtered_masks)} masks after filter "
          f"(small={stats['too_small']}, large={stats['too_large']}, "
          f"border={stats['border']}, split={stats['split_ok']}, "
          f"split_kept={stats['split_fail_kept']}, "
          f"split_dropped={stats['split_fail_dropped']})")

    merged_masks, n_merges = merge_adjacent_masks(filtered_masks, args.max_area)
    if n_merges:
        print(f"  {n_merges} fragment pair(s) merged → {len(merged_masks)} masks")

    deduped_masks = deduplicate_masks(merged_masks)
    print(f"  {len(deduped_masks)} masks after deduplication")

    stem    = nd2_path.stem
    out_dir = nd2_path.parent / stem
    out_dir.mkdir(exist_ok=True)
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    save_seg_overview(nuc_u8, segmentation, filtered_masks, merged_masks,
                      deduped_masks, viz_dir / "seg_overview.png")

    crops_info = []
    for mask in deduped_masks:
        crop, bbox = crop_with_suppression(fov, mask, deduped_masks, args.margin)
        crops_info.append((crop, bbox, mask))

    save_crop_grid(crops_info, nuc_u8, viz_dir / "crop_grid.png")
    save_suppression_demo(crops_info, fov, deduped_masks, viz_dir / "suppression_demo.png")
    save_all_channels_demo(crops_info, viz_dir / "all_channels_demo.png")

    # Save bboxes + suppression masks to JSON for save_crops.py to consume
    crops_json = []
    for idx, (crop, bbox, mask) in enumerate(crops_info, start=1):
        r0, r1, c0, c1 = bbox
        suppress_full = np.zeros((H, W), dtype=bool)
        for other_mask in deduped_masks:
            if not np.array_equal(other_mask, mask):
                suppress_full |= other_mask
        suppress_crop = suppress_full[r0:r1, c0:c1]
        rows, cols = np.where(suppress_crop)  # JSON can't serialize numpy arrays; coordinate lists are compact and load-friendly
        crops_json.append({
            'idx':              idx,
            'bbox':             [int(r0), int(r1), int(c0), int(c1)],
            'suppression_rows': rows.tolist(),
            'suppression_cols': cols.tolist(),
        })

    json_path = out_dir / f"{stem}_crops.json"
    with open(json_path, 'w') as f:
        json.dump({'nd2_path': str(nd2_path.resolve()), 'stem': stem,
                   'crops': crops_json}, f)

    print(f"  ✓ {len(crops_info)} crops → {out_dir}/")
    print(f"  ✓ Visualizations → {viz_dir}/")
    return len(crops_info)


"""
main() parses CLI args, loads µSAM once, then batch-processes all .nd2 files.
The predictor and segmenter are shared across files — model load (~30s) happens once.
"""
def main():
    parser = argparse.ArgumentParser(
        description="Crop nuclei from LiveFISH .nd2 FOV files using µSAM")
    parser.add_argument("input", help=".nd2 file or directory of .nd2 files")
    parser.add_argument("--nucleus-channel",   type=int, default=0)
    parser.add_argument("--margin",            type=int, default=30)
    parser.add_argument("--min-area",          type=int, default=1000)
    parser.add_argument("--max-area",          type=int, default=200000)
    parser.add_argument("--border-margin",     type=int, default=DEFAULT_BORDER_MARGIN_PX, help="Min distance (px) from image border to nucleus centroid")
    parser.add_argument("--segmentation-mode", default="apg", choices=["apg", "amg"])
    parser.add_argument("--model-type",        default="vit_b_lm")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        nd2_files = sorted(input_path.rglob("*.nd2"))
        if not nd2_files:
            print(f"No .nd2 files found in {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        nd2_files = [input_path]

    print(f"Files to process: {len(nd2_files)}")
    for f in nd2_files:
        print(f"  {f.name}")

    print(f"\nLoading µSAM model: {args.model_type}, "
          f"mode={args.segmentation_mode}, device=mps")
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter  # deferred — same reason as in process_file
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        device='mps',  # hardcoded for Apple Silicon; change to 'cuda' on Linux GPU machines
        segmentation_mode=args.segmentation_mode,
    )
    print("  µSAM loaded.")

    total_crops = 0
    failed = []
    for nd2_path in nd2_files:
        try:
            n = process_file(nd2_path, predictor, segmenter, args)
            total_crops += n
        except Exception as e:
            import traceback
            print(f"  ERROR on {nd2_path.name}: {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append(nd2_path.name)

    print(f"\n{'='*60}")
    print(f"Done. {len(nd2_files) - len(failed)}/{len(nd2_files)} files OK, "
          f"{total_crops} total crops.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
