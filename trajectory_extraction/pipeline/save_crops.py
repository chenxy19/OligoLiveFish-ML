"""
save_crops.py — read JSON files produced by crop_nuclei_sam.py and save TIFFs.

Usage:
    conda run -n base python "code (being modified)/save_crops.py" \
        "data for analysis/FOV (.nd2 files)"

Finds all *_crops.json files under the given directory tree, then for each crop:
  1. Loads the original .nd2 → (T, C, Y, X) uint16 via Z max-projection
  2. Reads per-channel LUT colors from nd2 metadata
  3. Slices the bbox, zeros suppression pixels
  4. Saves as an ImageJ composite hyperstack TIFF with per-channel LUTs
     so Fiji opens with the same colors as the original nd2
"""

import sys
import json
import argparse
import numpy as np
import tifffile
from pathlib import Path
from nd2 import ND2File


def per_channel_ranges(crop: np.ndarray, lo_pct=0.1, hi_pct=99.9):
    """Flat (min0, max0, min1, max1, ...) tuple for ImageJ 'Ranges' metadata.
    crop is (T, Z, C, H, W) — iterate over C axis (axis 2)."""
    T, Z, C, H, W = crop.shape
    flat = []
    for c in range(C):
        ch = crop[:, :, c]
        # use nonzero pixels so suppressed regions don't drag the min to 0
        nz = ch[ch > 0]
        if nz.size == 0:
            flat.extend([0.0, 1.0])
        else:
            flat.extend([float(np.percentile(nz, lo_pct)),
                         float(np.percentile(nz, hi_pct))])
    return tuple(flat)
    

def load_fov_with_metadata(nd2_path: Path):
    """Load nd2 → (T, Z, C, Y, X) uint16 plus physical metadata for TIFF export.
    The nd2 library squeezes out size-1 axes — we reinsert them so the array
    always has 5 dims regardless of acquisition mode."""
    with ND2File(nd2_path) as f:
        arr = f.asarray()
        keys = list(f.sizes.keys())
        channels = f.metadata.channels
 
        # Physical pixel size in microns. nd2 returns (x, y, z).
        try:
            vx = f.voxel_size()
            px_um_x, px_um_y, px_um_z = float(vx.x), float(vx.y), float(vx.z)
        except Exception:
            px_um_x = px_um_y = px_um_z = 1.0
 
        # Frame interval in seconds, if a T loop exists.
        finterval_s = None
        try:
            for loop in f.experiment:
                if getattr(loop, 'type', '').lower().startswith('time'):
                    period_ms = getattr(loop.parameters, 'periodMs', None) \
                                or getattr(loop.parameters, 'period', None)
                    if period_ms:
                        finterval_s = float(period_ms) / 1000.0
                        break
        except Exception:
            pass
 
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
    fov = np.transpose(arr, order).astype(np.uint16)
    T, Z, C, H, W = fov.shape
    print(f"  Final shape TZCYX: {fov.shape}")
 
    # Channel names
    chan_names = []
    for i in range(C):
        if i < len(channels):
            name = channels[i].channel.name or f"C{i}"
        else:
            name = f"C{i}"
        chan_names.append(str(name))
 
    # Per-channel LUTs
    ramp = np.arange(256, dtype=np.float32) / 255.0
    luts = []
    for i in range(C):
        if i < len(channels):
            col = channels[i].channel.color
            r, g, b = int(col.r), int(col.g), int(col.b)
        else:
            r, g, b = 255, 255, 255
        lut = np.zeros((3, 256), dtype=np.uint8)
        lut[0] = (ramp * r).astype(np.uint8)
        lut[1] = (ramp * g).astype(np.uint8)
        lut[2] = (ramp * b).astype(np.uint8)
        luts.append(lut)
 
    return fov, {
        'px_um_x':     px_um_x,
        'px_um_y':     px_um_y,
        'px_um_z':     px_um_z,
        'finterval_s': finterval_s,
        'chan_names':  chan_names,
        'luts':        luts,
    }


def process_json(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
 
    nd2_path = Path(data['nd2_path'])
    stem = data['stem']
    crops = data['crops']
 
    if not nd2_path.exists():
        print(f"  ERROR: nd2 not found: {nd2_path}", file=sys.stderr)
        return 0
 
    print(f"\n── {nd2_path.name} ──")
    fov, meta = load_fov_with_metadata(nd2_path)
    T, Z, C, H, W = fov.shape
    print(f"  Loaded: T={T} Z={Z} C={C} Y={H} X={W}")
    print(f"  Pixel size: {meta['px_um_x']:.4f} × {meta['px_um_y']:.4f} µm/px"
          f"  Z-step: {meta['px_um_z']:.4f} µm")
    print(f"  Frame interval: {meta['finterval_s']} s")
    print(f"  Channels: {meta['chan_names']}")
 
    # ImageJ 'resolution' is pixels-per-unit. 1/(µm per px) gives px per µm.
    xres = 1.0 / meta['px_um_x'] if meta['px_um_x'] else 1.0
    yres = 1.0 / meta['px_um_y'] if meta['px_um_y'] else 1.0
 
    out_dir = json_path.parent
    saved = 0
 
    for crop_info in crops:
        idx = crop_info['idx']
        r0, r1, c0, c1 = crop_info['bbox']
        sup_rows = np.array(crop_info['suppression_rows'], dtype=np.int32)
        sup_cols = np.array(crop_info['suppression_cols'], dtype=np.int32)
 
        # Slice spatial dims (Y, X) — preserve T and Z fully
        crop = fov[:, :, :, r0:r1, c0:c1].copy()
        if sup_rows.size > 0:
            # suppression mask is 2D (Y, X) — broadcast over T, Z, C
            crop[:, :, :, sup_rows, sup_cols] = 0
 
        # ImageJ Labels: one entry per T*Z*C plane, in T-major → Z-major → C-major order
        labels = [
            meta['chan_names'][c]
            for t in range(T)
            for z in range(Z)
            for c in range(C)
        ]
        ranges = per_channel_ranges(crop)
 
        ij_metadata = {
            'axes':   'TZCYX',
            'mode':   'color',
            'unit':   'um',
            'LUTs':   meta['luts'],
            'Labels': labels,
            'Ranges': ranges,
            'spacing': meta['px_um_z'],  # Z-step in µm for Fiji's z-calibration
        }
        if meta['finterval_s'] is not None:
            ij_metadata['finterval'] = meta['finterval_s']
            ij_metadata['fps'] = 1.0 / meta['finterval_s']
 
        tif_path = out_dir / f"{stem}_{idx}.tif"
        tifffile.imwrite(
            tif_path,
            crop,
            imagej=True,
            resolution=(xres, yres),
            metadata=ij_metadata,
        )
        saved += 1
 
    print(f"  ✓ {saved} TIFFs → {out_dir}/")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Save TIFFs from crop JSON files produced by crop_nuclei_sam.py")
    parser.add_argument("input", help="directory containing *_crops.json files (searched recursively)")
    args = parser.parse_args()
 
    root = Path(args.input)
    json_files = sorted(root.rglob("*_crops.json"))
    if not json_files:
        print(f"No *_crops.json files found under {root}", file=sys.stderr)
        sys.exit(1)
 
    print(f"Found {len(json_files)} JSON file(s):")
    for j in json_files:
        print(f"  {j}")
 
    total = 0
    for j in json_files:
        total += process_json(j)
 
    print(f"\nDone. {total} total TIFFs saved.")
 
 
if __name__ == "__main__":
    main()

