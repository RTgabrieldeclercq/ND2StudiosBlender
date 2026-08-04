"""dvc — shared catalog helpers."""

from __future__ import annotations

import numpy as np

from typing import Dict, Optional

def _dvc_rows(res, vox, *, m: int, t: int, c: int, z_plane: Optional[int]) -> Dict[str, np.ndarray]:
    """Flatten one :class:`DVCResult` (a coarse subset grid) into a column-dict of Point
    rows. ``vox`` = the ``voxel_size_um`` passed to the kernel (component order matches
    the displacement/grid axes, slowest-first) so displacement voxels → µm is a
    component-wise multiply. 2D stamps ``z = z_plane`` (plane index); 3D reads z from the
    grid. Strain ``[i,j] = ∂u_i/∂x_j`` becomes ``strain_<axis_i><axis_j>`` columns."""
    d = int(res.dim)
    grid = np.asarray(res.grid_coords, dtype=float).reshape(-1, d)          # voxel coords
    disp = np.asarray(res.displacement_field, dtype=float).reshape(-1, d)   # voxels
    disp_um = disp * np.asarray(vox, dtype=float)                            # → µm
    n = grid.shape[0]
    anames = ("z", "y", "x") if d == 3 else ("y", "x")
    cols: Dict[str, np.ndarray] = {}
    if d == 3:
        z, y, x = grid[:, 0], grid[:, 1], grid[:, 2]
        cols["disp_z"], cols["disp_y"], cols["disp_x"] = (disp_um[:, 0], disp_um[:, 1],
                                                          disp_um[:, 2])
    else:
        y, x = grid[:, 0], grid[:, 1]
        z = np.full(n, float(z_plane if z_plane is not None else 0))
        cols["disp_y"], cols["disp_x"] = disp_um[:, 0], disp_um[:, 1]
    cols["disp_mag_um"] = np.sqrt((disp_um ** 2).sum(axis=1))
    if res.strain_field is not None:
        s = np.asarray(res.strain_field, dtype=float).reshape(-1, d, d)
        for i in range(d):
            for j in range(d):
                cols[f"strain_{anames[i]}{anames[j]}"] = s[:, i, j]
    if res.qfactor is not None:
        cols["qfactor"] = np.asarray(res.qfactor, dtype=float).reshape(-1)
    row = {
        "m": np.full(n, m, dtype=np.int64), "t": np.full(n, t, dtype=np.int64),
        "c": np.full(n, c, dtype=np.int64),
        "z": z.astype(float), "y": y.astype(float), "x": x.astype(float),
    }
    row.update(cols)
    return row
