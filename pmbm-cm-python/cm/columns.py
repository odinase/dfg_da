"""Track-file column layout (``inCol``) for the cluster-management PMBM filter.

Port of the ``inCol`` struct built in ``script_pmbm91.m:516-527``. A track is a
single column of the ``trackFile`` matrix; ``inCol`` maps named fields to the
rows of that column.

Internally we use 0-based row indices (for indexing numpy ``trackFile`` arrays);
:meth:`InCol.to_matlab_struct` returns the original 1-based field values so the
dumped ``.mat`` matches the MATLAB ``inCol`` exactly.
"""

from dataclasses import dataclass

import numpy as np

DIM_TAR = 4   # single-target state dimension [px, py, vx, vy]
DIM_Z = 2     # measurement dimension [px, py]
MAX_LAG = 8   # measurement-history depth (script_pmbm91.m:568)


@dataclass(frozen=True)
class InCol:
    """0-based row indices into a trackFile column (dimTar = 4)."""

    dim_tar: int = DIM_TAR

    @property
    def n_cov(self):
        d = self.dim_tar
        return d + d * (d - 1) // 2  # 10 for d=4

    # 0-based python slices / indices
    @property
    def tarX(self):
        return slice(0, self.dim_tar)               # 0:4

    @property
    def tarP(self):
        return slice(self.dim_tar, self.dim_tar + self.n_cov)  # 4:14

    @property
    def meaLast(self):
        return self.dim_tar + self.n_cov            # 14

    @property
    def cost(self):
        return self.meaLast + 1                     # 15

    @property
    def contrib(self):
        return self.cost + 1                        # 16

    @property
    def exi(self):
        return self.contrib + 1                     # 17

    @property
    def visi(self):
        return self.exi + 1                         # 18

    @property
    def label(self):
        return self.visi + 1                        # 19

    @property
    def last(self):
        """Number of rows in a trackFile column (= MATLAB inCol.last = 21)."""
        return self.label + 2                       # 21

    def to_matlab_struct(self):
        """Return a dict matching the MATLAB ``inCol`` struct (1-based)."""
        d = self.dim_tar
        n_cov = self.n_cov
        return {
            "tarX": np.arange(1, d + 1, dtype=float).reshape(1, -1),
            "tarP": np.arange(d + 1, d + n_cov + 1, dtype=float).reshape(1, -1),
            "meaLast": float(d + n_cov + 1),
            "cost": float(d + n_cov + 2),
            "contrib": float(d + n_cov + 3),
            "exi": float(d + n_cov + 4),
            "visi": float(d + n_cov + 5),
            "label": float(d + n_cov + 6),
            "last": float(d + n_cov + 7),
        }
