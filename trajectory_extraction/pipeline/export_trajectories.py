#!/usr/bin/env python3
"""
Export SPT trajectories from a .mat file to individual CSVs.
Usage: python export_trajectories.py <path_to_mat_file>
Output: matlab_trajectory/ folder next to the .mat file, one CSV per trajectory.
"""

import sys
import os
import re
import numpy as np
import scipy.io as sio


def get_channel_prefix(filename):
    fname = filename.lower()
    if 'green'  in fname: return 'G'
    if 'red'    in fname: return 'R'
    if 'purple' in fname: return 'P'
    if 'blue'   in fname: return 'B'
    return 'X'


def main():
    if len(sys.argv) != 2:
        print("Usage: python export_trajectories.py <path_to_mat_file>")
        sys.exit(1)

    mat_path = sys.argv[1]
    if not os.path.isfile(mat_path):
        print(f"Error: file not found: {mat_path}")
        sys.exit(1)

    mat = sio.loadmat(mat_path)
    traj    = mat['traj']
    sptpara = mat['sptpara'][0, 0]
    pixl_nm = float(sptpara['pixl'].flat[0]) * 1000

    prefix = get_channel_prefix(os.path.basename(mat_path))
    outdir = os.path.join(os.path.dirname(mat_path), 'matlab_trajectory')
    os.makedirs(outdir, exist_ok=True)

    n_traj = traj.shape[1]
    for i in range(n_traj):
        pos = traj[0, i]['pos']
        if pos.shape == (1, 1):
            pos = pos[0, 0]

        frames = pos[:, 2].astype(int)
        x_nm   = pos[:, 0] * pixl_nm
        y_nm   = pos[:, 1] * pixl_nm

        outfile = os.path.join(outdir, f"{prefix}_loci{i+1}_trajectory.csv")
        with open(outfile, 'w') as f:
            f.write("frame,x_nm,y_nm\n")
            for frame, x, y in zip(frames, x_nm, y_nm):
                f.write(f"{frame},{x:.2f},{y:.2f}\n")

    print(f"Done. {n_traj} trajectory files written to:\n  {outdir}")


if __name__ == "__main__":
    main()
