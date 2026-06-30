"""Write the per-step ``priorLikelihood{k}.mat`` dump.

The variable set and names match ``script_pmbm91.m:1237`` exactly. Values are
the closest reproduction the Python CM port can produce (equivalent, not
bit-exact). Index-valued variables (``hypos``, ``clusters``,
``trackNumberLookup``, ``indicesOfNewbornTracks``, ``meaLast``) are written
1-based to match MATLAB semantics.
"""

import os

import numpy as np
import scipy.io as sio

# The exact ordered variable list saved at script_pmbm91.m:1237.
DUMP_VARS = [
    "hypos", "hyposCard", "clusters", "clustersCard", "probLogHypos",
    "assocLocal", "gainMatPostC", "indicesOfNewbornTracks", "nHypoMax",
    "nHypoTotalMax", "trackNumberLookup", "k", "trackFile", "inCol", "hTrue",
    "trackFileShadow", "measurements", "predX", "predZ", "predP", "predS",
    "pD", "meaHistCol", "meaHistColNew",
]


def save_priorlikelihood(dump, k, outdir):
    """Write ``{outdir}/priorLikelihood{k}.mat`` from a pipeline dump dict."""
    missing = [v for v in DUMP_VARS if v not in dump]
    if missing:
        raise KeyError(f"dump is missing variables: {missing}")
    matdict = {v: dump[v] for v in DUMP_VARS}
    path = os.path.join(outdir, f"priorLikelihood{k}.mat")
    sio.savemat(path, matdict, do_compression=True)
    return path
