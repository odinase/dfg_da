"""Decode PMBM cluster membership from a ``priorLikelihood*.mat`` file.

Clusters are stored as a two-level **CSR / "jagged" array** nesting

    cluster  ->  hypotheses  ->  tracks

encoded in four flat arrays (a ``values`` array + a per-group ``card``inality
array at each level):

    clusters / clustersCard : cluster     -> hypothesis ids (1-based)
    hypos    / hyposCard     : hypothesis -> track numbers  (1-based)

There are no explicit boundaries; group ``i`` owns ``values[off[i]:off[i+1]]``
with ``off = prefix_sum(card)``. This module implements a *generic* jagged
decoder (:class:`Jagged` + :func:`resolve`) and composes two layers to recover,
for one scan:

    - ``cluster_tracks``   : cluster index (0-based) -> sorted track numbers
    - ``track_to_cluster`` : track number -> cluster index (0-based)
    - ``cluster_to_supercluster`` : cluster -> supercluster (from ``assocLocal``)

Run as a script for a demonstration on a single ``.mat`` file:

    python cluster_membership.py [path/to/priorLikelihood5.mat]

The pure functions are import-friendly so tests can assert against them.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio


# --------------------------------------------------------------------------- #
# Generic CSR / jagged-array primitive
# --------------------------------------------------------------------------- #
class Jagged:
    """A list-of-lists stored as a flat ``values`` array + prefix ``offsets``.

    Group ``i`` owns ``values[offsets[i]:offsets[i + 1]]``.
    """

    def __init__(self, offsets: np.ndarray, values: np.ndarray):
        self.offsets = offsets
        self.values = values

    @classmethod
    def from_card(cls, values, card) -> "Jagged":
        """Build from a flat ``values`` array and per-group cardinalities."""
        values = np.asarray(values).ravel()
        card = np.asarray(card).ravel().astype(int)
        offsets = np.zeros(card.size + 1, dtype=int)
        np.cumsum(card, out=offsets[1:])
        if offsets[-1] != values.size:
            raise ValueError(
                f"malformed jagged array: sum(card)={offsets[-1]} != "
                f"len(values)={values.size}"
            )
        return cls(offsets, values)

    def __len__(self) -> int:
        return self.offsets.size - 1

    def row(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i] : self.offsets[i + 1]]


def resolve(top: Jagged, leaf: Jagged) -> List[List[int]]:
    """Compose two layers: each ``top`` group holds **1-based** row ids into
    ``leaf``; return the deduped, sorted leaf items for each top group."""
    out: List[List[int]] = []
    for g in range(len(top)):
        items = set()
        for idx in top.row(g):
            items.update(leaf.row(int(idx) - 1).tolist())  # 1-based -> 0-based
        out.append(sorted(items))
    return out


# --------------------------------------------------------------------------- #
# Cluster-specific composition
# --------------------------------------------------------------------------- #
def cluster_tracks(clusters, clusters_card, hypos, hypos_card) -> List[List[int]]:
    """Cluster index (0-based) -> sorted track numbers (1-based).

    This is the two-level jagged decode; equivalent to
    ``cm.clouds.cluster2tracks`` for every cluster.
    """
    hypos_j = Jagged.from_card(hypos, hypos_card)
    clusters_j = Jagged.from_card(clusters, clusters_card)
    return resolve(clusters_j, hypos_j)


def track_to_cluster(cluster_tracks_list: List[List[int]]) -> Dict[int, int]:
    """Invert: track number -> cluster index (0-based)."""
    return {t: c for c, ts in enumerate(cluster_tracks_list) for t in ts}


def cluster_to_supercluster(assoc_local: np.ndarray) -> Dict[int, int]:
    """Cluster index (0-based) -> supercluster (master) id, from ``assocLocal``.

    ``assocLocal[0, c]`` is the 1-based master-cluster id (clusters sharing a
    master form one supercluster); ``assocLocal[1, c]`` flags the masters.
    """
    return {c: int(master) for c, master in enumerate(assoc_local[0])}


# --------------------------------------------------------------------------- #
# .mat loading + structural verification
# --------------------------------------------------------------------------- #
def load_cloud(matfile: str) -> dict:
    """Load the four cloud arrays (+ helpers) from a ``priorLikelihood*.mat``.

    ``squeeze_me`` collapses length-1 arrays to scalars, so every field is
    forced back to at least 1-D (and ``assocLocal`` to its ``(2, nC)`` shape).
    """
    d = sio.loadmat(matfile, squeeze_me=True)
    ints = lambda k: np.atleast_1d(d[k]).astype(int)
    assoc = None
    if "assocLocal" in d:
        assoc = np.asarray(d["assocLocal"])
        if assoc.ndim == 1:  # single-cluster scan squeezed (2,1) -> (2,)
            assoc = assoc.reshape(2, -1)
        assoc = assoc.astype(int)
    return {
        "clusters": ints("clusters"),
        "clustersCard": ints("clustersCard"),
        "hypos": ints("hypos"),
        "hyposCard": ints("hyposCard"),
        "assocLocal": assoc,
        "n_tracks": int(d["trackFile"].shape[1]) if "trackFile" in d else None,
        "k": int(d["k"]) if "k" in d else None,
    }


def verify_structure(cloud: dict) -> None:
    """Assert the CSR contract + that clusters form a track partition.

    This is the structural contract any consumer (Rust included) can rely on:
    both layers well-formed, hypothesis ids in range, and the decoded clusters
    are disjoint and cover ``1..n_tracks``.
    """
    cl, cc = cloud["clusters"], cloud["clustersCard"]
    hy, hc = cloud["hypos"], cloud["hyposCard"]
    assert cc.sum() == cl.size == hc.size, "cluster layer malformed"
    assert hc.sum() == hy.size, "hypothesis layer malformed"
    if cl.size:
        assert cl.min() >= 1 and cl.max() <= hc.size, "hypothesis id out of range"
    ct = cluster_tracks(cl, cc, hy, hc)
    flat = [t for ts in ct for t in ts]
    assert len(flat) == len(set(flat)), "clusters overlap (not disjoint)"
    if cloud["n_tracks"] is not None:
        assert set(flat) == set(range(1, cloud["n_tracks"] + 1)), "tracks not fully covered"


# --------------------------------------------------------------------------- #
# Demonstration entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Decode cluster membership from a priorLikelihood*.mat file."
    )
    p.add_argument(
        "matfile",
        nargs="?",
        default="data/pmbm_output_files/priorLikelihood5.mat",
        help="path to a priorLikelihood*.mat (default: scan 5)",
    )
    p.add_argument("--no-verify", action="store_true", help="skip structural checks")
    args = p.parse_args(argv)

    cloud = load_cloud(args.matfile)
    if not args.no_verify:
        verify_structure(cloud)

    ct = cluster_tracks(
        cloud["clusters"], cloud["clustersCard"], cloud["hypos"], cloud["hyposCard"]
    )
    t2c = track_to_cluster(ct)
    c2s = cluster_to_supercluster(cloud["assocLocal"]) if cloud["assocLocal"] is not None else None

    name = Path(args.matfile).name
    extra = f"  (trackFile: {cloud['n_tracks']} tracks)" if cloud["n_tracks"] else ""
    print(f"{name}: k={cloud['k']}  {len(ct)} clusters  {len(t2c)} tracks{extra}")

    print("\ncluster -> tracks:")
    for c, ts in enumerate(ct):
        sup = f"  [supercluster {c2s[c]}]" if c2s is not None else ""
        print(f"  cluster {c + 1:2d} ({int(cloud['clustersCard'][c]):2d} hypos): {ts}{sup}")

    first = sorted(t2c)[:15]
    print("\ntrack -> cluster (first 15, 1-based cluster ids):",
          {t: t2c[t] + 1 for t in first})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
