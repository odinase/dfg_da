# `cluster_partition` — efficient cluster marginalization with exact inclusion-exclusion

A self-contained, pure-Python implementation of the cluster-partitioning /
cluster-conditioning marginalization of Chapter 7 (Sections 7.1–7.5) of

> O. A. Severinsen, *Efficient cluster marginalization for multi-cluster,
> multi-hypothesis data association*, master's thesis, NTNU.

It is structurally compatible with the reference code base
[`odinase/dfg_da`](https://github.com/odinase/dfg_da) (the `cluster_bayes_tree`
and `cluster_conditioning_lbp` modules and their `solver(R, prior_hypotheses) ->
(marginals, theta_posterior, likelihood)` contract), but has **no dependency on
the compiled `py_dfg_da` pybind module** — the prior-hypothesis containers are
re-implemented in `prior_hypotheses.py`.

The headline addition over the thesis is the **closed-form inclusion-exclusion
correction of Section 7.5.1**, which the thesis derives the need for but leaves
as future work. With an exact inner solver it turns the thesis's *approximate*
overlapping-event-space estimate into an **exact** marginalization that matches
the disjoint exact path and the brute-force reference to machine precision.

---

## 1. The problem and the overparameterization trick

At one filter step the multi-target posterior factors over **prior clusters**
that are independent *a priori* (thesis Eq. (5.21)). Clusters become coupled
only through **linking measurements**: a measurement gated by tracks belonging
to more than one cluster. A linking measurement can be claimed by at most one
track across the whole scene, and that single global "at most one" constraint is
the only thing that prevents the posterior from factoring.

The factor graph of Chapter 5 is deliberately **over-parameterized**: each track
carries its own association variable `a^t` and each measurement its own variable
`b^j`, tied together by per-edge consistency factors `γ^{jt}`. Because the
coupling is localized in the measurement variables, we can *split* it. For every
linking measurement `l` we introduce a **delegating variable** `d_l` (Section
7.1) that decides which cluster is allowed to "own" the measurement. Conditioned
on `d = (d_l)_l`, the clusters are independent again, so (Eq. (7.10)–(7.12), with
a uniform prior on `d`):

```
Pr{a^t | Z} ∝ Σ_d Pr{a^t | Z, d} · p(Z | d),     p(Z | d) = Π_c p(Z^c | d).
```

Each `p(Z^c | d)` is a single-cluster, multi-hypothesis sub-problem solved
independently and cached by which linking measurements the cluster owns
(dynamic programming, Section 7.3: `2^{|L_c|}` distinct sub-problems per cluster).

## 2. Two event spaces for the delegating variable

* **Disjoint** (Eq. (7.5)): `d_l ∈ {{0}} ∪ {T^c : c gates l}`. Delegating to
  cluster `c` *forces* the measurement to be detected by a track in `c` (false
  alarm forbidden there). This is exact, but enforcing a detection drives the
  effective SNR of that measurement to infinity, which destabilizes loopy belief
  propagation and voids its convergence guarantees (Sections 7.4–7.5).

* **Overlapping** (Eq. (7.29)): `d_l ∈ {{0}} ∪ {{0} ∪ T^c : c gates l}`.
  Delegating to `c` lets the measurement be *either* a detection in `c` *or* a
  misdetection — no infinite SNR, LBP stays well behaved. The price is that the
  misdetection outcome `{0}` now lies inside **every** event, so naively summing
  over `d` over-counts it.

The thesis ships only the **first stage** of the overlapping sum (no correction
for the over-counted `{0}`) and leaves the full correction as future work.

## 3. The inclusion-exclusion closed form (the key derivation)

Because every track belongs to exactly one cluster, the track sets `T^c` are
disjoint and *any* intersection of two or more overlapping events `{0} ∪ T^c`
collapses to exactly `{0}`. The conditioned mass `P(·)` (a sum over the global
assignments allowed by the event) is **additive over disjoint allowed-sets**, so
the inclusion-exclusion principle applies *exactly* and telescopes. For one
linking measurement gated by `K_l = |C_l|` clusters:

```
P(⋃_c ({0} ∪ T^c)) = Σ_c P({0} ∪ T^c) − (K_l − 1) · P({0})
                    = Σ_c P({0} ∪ T^c) + (1 − K_l) · P({0}).
```

The middle line is the inclusion-exclusion sum: each pairwise (and higher)
intersection is `{0}`, there are `C(K,2) − C(K,3) + … = K − 1` net duplicate
copies of `P({0})` to remove. Over the enumeration tuple the code base already
uses — `i_l ∈ {none} ∪ C_l` written as `[-1, *C_l]` — this is exactly a
**signed weight** on each first-stage term:

```
ω_l(c)    = +1            (c ∈ C_l, i.e. "measurement delegated to cluster c")
ω_l(none) = 1 − K_l       (the {0}-everywhere outcome, value -1 in the code).
```

With several linking measurements the conditioned mass is **multilinear** in the
per-measurement choices, so the weights multiply and the exact union is

```
Z_exact  =  Σ_d ( Π_l ω_l(i_l) ) · Q(d),
```

where `Q(d)` is precisely the thesis first-stage conditioned term. Setting all
`ω ≡ 1` recovers the published first-stage approximation; using the signed
weights is the exact correction. This is implemented in
`ConditionalSuperclusterMarginals._ie_weights` and applied in
`compute_marginals_likelihood` (mode `"overlap_ie"`).

### Worked check (thesis Fig. 7.7)

Two single-track clusters share one measurement, `R = [[0, 1.3], [0, 0.7]]`,
`K = 2`. Then `P({0}) = e^0 · e^0 = 1`, and

```
first stage = P({0}) + P({0}∪T^0) + P({0}∪T^1) = 8.683049
exact (IE)  =          P({0}∪T^0) + P({0}∪T^1) − P({0}) = 6.683049,
```

so the first stage over-counts by `K · P({0}) = 2`, and the IE form recovers the
brute-force reference `Z = 6.683049` exactly. (The general over-count of the
first stage relative to exact is `Σ_l K_l · P(·)`-type terms; for a single
linking measurement it is `K · P({0})`.)

## 4. How the Section 7.5.1 approximation affects results

The thesis analyses the first-stage approximation through the **Bonferroni
inequalities**: truncating an inclusion-exclusion expansion after an *odd*
number of stages **over-estimates** the union, after an *even* number
**under-estimates** it. The first-stage-only estimate is a single (odd) stage,
so it **over-estimates the unnormalized mass / Bethe normalization constant**.
Every case in the verification table below shows `Z_firststage > Z_exact`,
confirming the bound. Concretely the over-estimate is the discarded
`(K_l − 1)·P({0})` mass per linking measurement.

For the **normalization constant** this matters directly: if you use the
constant to gauge how much posterior mass your enumerated hypotheses capture
(Section 6.2.1), an over-estimate is the unsafe direction. The IE correction
removes the bias entirely (exact inner solver) or de-biases it (LBP inner
solver).

For the **association marginals** the effect is subtler, exactly as the thesis
notes. Marginals are re-normalized per track, so a uniform over-count of `{0}`
partly cancels; what survives is a systematic tilt of probability mass toward
the misdetection / non-existence outcomes (the `{0}` event), because that is the
outcome being over-counted. The comparison table quantifies this: with the exact
inner solver the first-stage marginals differ from exact by up to ~0.2 in
absolute probability on these small cases, while `overlap_ie` reproduces them to
machine precision.

A caveat worth stating plainly: when the **inner** solver is itself approximate
(LBP+Bethe), the IE weights are applied to approximate `Q(d)` terms. The
correction then reliably fixes the *normalization constant* (its design goal),
but because the signed weights can amplify per-track marginal error, the
already-normalized marginals are not guaranteed to be uniformly closer than the
first-stage ones on every track — fixing the constant is the robust guarantee,
not fixing every marginal. On cases where each cluster's own graph is a tree
(e.g. the thesis 5×7 `upstream` case) LBP is exact per cluster and `overlap_ie`
recovers the exact constant outright.

## 5. Package layout

```
cluster_partition/
├── prior_hypotheses.py   Hypothesis / Hypotheses / HypothesesList containers
│                         + make_hypotheses(); upstream call contract.
├── cluster_links.py      ClusterLinks: discover merging clusters & linking
│                         measurements from gating; LinkingMappings; the
│                         delegating-variable event space (Eq. (7.5)).
├── solvers.py            Single-cluster multi-hypothesis solvers, all sharing
│                         solver(R, ph, enforce_meas=(), reindex=True) ->
│                         (marginals, theta_post, likelihood):
│                           • ExactEnumerationSolver  (ground truth)
│                           • EHM2Solver              (pyehm baseline)
│                           • LBPBetheSolver          (Williams-LBP + Bethe
│                                                      pseudodual, Corollary 2)
├── partitioning.py       ConditionedCluster (DP cache by owned-measurement
│                         mask), ConditionalSuperclusterMarginals (the d-sum
│                         with the three modes + IE weights),
│                         MulticlusterPartitionedMarginals (top-level driver).
├── reference.py          multicluster_exact_reference: brute-force full-graph
│                         ground truth.
├── demo_comparison.py    modes × solvers × cases results table.
└── tests/
    └── test_partitioning.py   33 tests (see §7).
```

### Modes

| mode                 | event space            | exactness                                   | upstream analogue                       |
|----------------------|------------------------|---------------------------------------------|-----------------------------------------|
| `disjoint_exact`     | Eq. (7.5), enforce det | exact                                        | `MulticlusterEfficientMarginals`        |
| `overlap_firststage` | Eq. (7.29), 1 stage    | biased (Bonferroni over-estimate)            | `MulticlusterEfficientMarginalsLBP`     |
| `overlap_ie`         | Eq. (7.29) + IE        | exact (exact solver) / de-biased (LBP)       | the thesis "future work"                |

## 6. Usage

```python
import numpy as np
from cluster_partition import (
    make_hypotheses, MulticlusterPartitionedMarginals,
    ExactEnumerationSolver, EHM2Solver, LBPBetheSolver,
    multicluster_exact_reference,
)

# (n, m+1) log-reward matrix: column 0 is misdetection, columns 1..m are
# per-measurement detection log-rewards (-inf where a track does not gate a
# measurement).  Tracks are 1-based globally; row t-1 is track t.
R = np.array([
    [3.0, -np.inf, -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
    [3.2, -np.inf, -np.inf, -0.56, -np.inf, -np.inf, -np.inf],
    [-3.0,    1.2, -np.inf, -np.inf, -0.46, -np.inf, -np.inf],
    [-np.inf, 3.0, -np.inf, -np.inf, -np.inf, -0.62, -np.inf],
    [-np.inf, -0.4, -np.inf, -np.inf, -np.inf, -np.inf, -0.55],
])

# One Hypotheses object per prior cluster; each prior hypothesis is
# (existing track ids, linear prior probability).
prior = [
    make_hypotheses([([1, 2], 0.5), ([1, 3], 0.5)]),
    make_hypotheses([([4],    0.5), ([5],    0.5)]),
]

m = MulticlusterPartitionedMarginals(R, prior, ExactEnumerationSolver(),
                                     mode="overlap_ie")
marginals, theta_posteriors, Z = m.compute_marginals_likelihood()

# marginals: (n, m+2) array, columns [misdetect, meas_1..m, nonexistence],
#            row-normalized per track.
# theta_posteriors: {cluster -> posterior over that cluster's prior hypotheses}.
# Z: the (exact, here) joint normalization constant.
```

Swap `ExactEnumerationSolver()` for `EHM2Solver()` to use the `pyehm` EHM2
baseline, or `LBPBetheSolver()` for the loopy-BP + Bethe-pseudodual estimate the
thesis advocates for large clusters.

## 7. Verification

`python -m pytest cluster_partition/tests/ -v` runs 33 tests:

1. **`ClusterLinks` discovery** — merging clusters and linking measurements are
   recovered from gating on the upstream 5×7 case, a three-cluster (`K=3`) case,
   a two-linking-measurement case, and a no-merge case.
2. **Delegating-variable event space** — disjoint space is `{{0}} ∪ {T^c}`;
   enumeration columns are `[-1, *clusters]`.
3. **IE weights** — reproduce `ω(c)=+1`, `ω(none)=1−K` (incl. `1−3=−2`), and the
   Fig. 7.7 identity `Z_exact = P({0}∪T^0)+P({0}∪T^1)−P({0})`.
4. **Exactness** — `disjoint_exact == overlap_ie == reference` for **both**
   marginals and likelihood on all four cases, with the exact inner solver.
5. **Bonferroni** — `overlap_firststage` likelihood is `> exact` on every case.
6. **EHM2 comparison** — `EHM2Solver` single-cluster marginals equal
   `ExactEnumerationSolver`; the full pipeline with EHM2 inner solver stays exact.
7. **LBP + Bethe** — the Bethe constant is exact on a tree, a sound underestimate
   on a loopy cluster, and the IE correction de-biases the first-stage constant.
8. **DP cache** — `ConditionedCluster` never stores more than `2^{|owned|}`
   conditioned sub-problems.

`python -m cluster_partition.demo_comparison` prints the modes × solvers × cases
table. Summary of the constant `Z` versus the exact reference:

| case            | exact `Z`   | `overlap_firststage` `Z` | `overlap_ie` `Z` (exact solver) |
|-----------------|-------------|--------------------------|---------------------------------|
| fig77           | 6.683049    | 8.683049  (+2.000)       | 6.683049  (exact)               |
| upstream (5×7)  | 2928.193287 | 3224.784566 (+296.59)    | 2928.193287 (exact)             |
| three_clusters  | 33.766259   | 51.332568 (+17.566)      | 33.766259 (exact)               |
| two_linking     | 48.672037   | 136.385939 (+87.714)     | 48.672037 (exact)               |

In every case the first stage over-estimates (Bonferroni, odd stage count) and
the inclusion-exclusion correction recovers the exact constant.

## 8. Notes on compatibility

* The solver call contract, the `Hypotheses` interface
  (`tracks() / t_idxs() / reindex_tracks() / hypothesis_probabilites()` and
  iteration to objects with `tracks() / probability()`), the
  `[-1, *gating_clusters]` per-measurement enumeration, and the
  `(marginals, theta_posterior, likelihood)` return triple all mirror
  `odinase/dfg_da`.
* `ConditionedCluster` captures each cluster's **global** track row indices
  *before* `reindex_tracks()` relabels them to local `1..n_c`, matching the
  upstream ordering subtlety.
* `EHM2Solver` uses `pyehm` (`pip install pyehm`). The packaged `pyehm` (≥2.2)
  exposes EHM2 as `EHM2().run(validation, likelihood) -> probabilities` and
  returns only normalized association probabilities; the per-hypothesis
  normalization constant is therefore obtained from the exact permanent
  (identical to EHM2's internal value). If `pyehm` is absent, the exact and LBP
  paths are unaffected and the EHM2 tests skip.
