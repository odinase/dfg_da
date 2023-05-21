import numpy as np
from scipy.io import loadmat
from typing import Dict, Any, List, TypeVar, FrozenSet, Optional, Union
from dataclasses import dataclass
from scipy.special import logsumexp
from .prior_hypothesis import PriorHypothesis, PriorHypotheses
from .marginals_computers import MarginalsComputer, ExactMarginals, MulticlusterExactOutput
import pickle
import pickletools
import asyncio


def merge_clusters(assocLocal, prior_hypotheses_per_cluster):
    num_posterior_clusters = np.sum(assocLocal[1])
    # First build master array
    prior_hypotheses_per_cluster_posterior: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        h for k, h in enumerate(prior_hypotheses_per_cluster) if assocLocal[1, k]
    ])
    master_idxs = np.cumsum(assocLocal[1]) - 1

    assert len(prior_hypotheses_per_cluster_posterior) == num_posterior_clusters
    
    for c, (master, is_master) in enumerate(assocLocal.T):
        if is_master:
            continue
            
        # We already have the masters, merge clusters
        hs = prior_hypotheses_per_cluster[c]
        prior_hypotheses_per_cluster_posterior[master_idxs[master]] = prior_hypotheses_per_cluster_posterior[master_idxs[master]].combine(hs)

    return prior_hypotheses_per_cluster_posterior

from collections import defaultdict

import py_dfg_da


@dataclass
class TrackEstimate:
    state: np.ndarray
    covariance: np.ndarray

@dataclass
class PredictedMeasurement:
    measurement: np.ndarray
    covariance: np.ndarray


@dataclass
class MatFileParser:
    ws: Dict[str, Any]

    reward_matrix_edmund: np.ndarray
    reward_matrix_lc: np.ndarray

    prior_hypotheses_per_cluster: List[PriorHypotheses]
    clusters_sorted: np.ndarray

    num_tracks: int
    num_measurements: int

    def __init__(self, filename: str, compute_hypotheses: bool = True, use_cpp: bool = False):
        ws = loadmat(filename)
        self.ws = ws
        R_wrapping = ws["gainMatPostC"] # This R has a strange shape... We convert it to order='F' for use with Eigen
        track_file = ws["trackFile"]
        measurements = ws["measurements"]
        
        n = track_file.shape[1]
        m = measurements.shape[1]

        self.num_tracks = n
        self.num_measurements = m

        npm = n + m
        self.reward_matrix_edmund = R_wrapping[:n, :npm]
        self.reward_matrix_lc = np.hstack((np.diag(self.reward_matrix_edmund[:,m:])[:,None], self.reward_matrix_edmund[:,:m]))

        self.using_cpp = use_cpp

        if compute_hypotheses and not use_cpp:
            self.prior_hypotheses_per_cluster, self.clusters_sorted = self.ws_to_prior_hypotheses(ws)

        if compute_hypotheses and use_cpp:
            self.prior_hypotheses_per_cluster, self.clusters_sorted = self.ws_to_prior_hypotheses_cpp(ws)

    def prior_hypotheses_per_cluster_posterior(self):
        if not self.using_cpp:
            raise NotImplementedError("We cannot do merging of prior hypotheses on Python implemenation")
        
        assocLocal = self.ws["assocLocal"]
        self.prior_hypotheses_per_cluster_posterior_ = merge_clusters(assocLocal, self.prior_hypotheses_per_cluster)
        return self.prior_hypotheses_per_cluster_posterior_

    def track_distribution(self, prior_hypotheses: py_dfg_da.hypothesis.Hypotheses, normalized: bool = False):
        tracks = defaultdict(lambda: 0)

        for ph in prior_hypotheses:
            for t in ph.tracks():
                tracks[t] += 1

        distr = np.array(list(tracks.items()))

        sorted_tracks = np.argsort(distr[:,0])
        distr = distr[sorted_tracks]

        if normalized:
            x = np.linspace(0, 1, distr.shape[0])
            y = distr[:, 1].astype(float)
            integral = np.trapz(y, x)
            y /= integral
            distr = np.vstack((x, y)).T

        return distr


    def ws_to_prior_hypotheses_cpp(self, ws):
        hypos = ws["hypos"].ravel().astype(int)
        hyposCard = ws["hyposCard"].ravel().astype(int)
        probLogHypos = ws["probLogHypos"].ravel()
        clustersCard = ws["clustersCard"].ravel().astype(int)
        clusters = ws["clusters"].ravel().astype(int) - 1 # We negate one here to make hypotheses 0-indexed, which is more convenient. Tracks we keep 1-indexed
        assert (clusters >= 0).all()

        num_clusters = len(clustersCard)

        # Let's actually first figure out the clusters we are working with and find the probabilities
        log_hypo_probs_per_cluster = []
        hypotheses_per_cluster = []

        i = 0
        for clusterC in clustersCard:
            # These hypotheses are in the cluster
            stop = i + clusterC
            hypotheses_in_cluster = clusters[i:stop]

            # hyposCard is as long as probLogCard, so pick out the elements that correspond to the "indices" in hypothesis_in_clutter
            probLogHyposInCluster = probLogHypos[hypotheses_in_cluster]

            log_hypo_probs_per_cluster.append(probLogHyposInCluster)
            hypotheses_per_cluster.append(hypotheses_in_cluster)

            i += clusterC

        # Before we start looping over clusters, let's do this the simple way of making a list of lists, containing tracks contained in each hypothesis, then we sort it afterwards
        tracks_in_hypotheses = [None] * hyposCard.shape[0]
        endInd = np.cumsum(hyposCard)
        beginInd = endInd - hyposCard
        for hypos_in_cluster in hypotheses_per_cluster:
            for h in hypos_in_cluster:
                start_idx = beginInd[h]
                stop_idx = endInd[h]
                tracks_in_hypothesis = hypos[start_idx:stop_idx]
                tracks_in_hypotheses[h] = tracks_in_hypothesis

        track_set = [set(h) for h in tracks_in_hypotheses]
        assert (np.array([sum([kk == hh for hh in track_set if len(hh) > 0]) for kk in track_set if len(kk) > 0]) == 1).all(), f"{track_set}"

        assert all(h is not None for h in tracks_in_hypotheses)

        # We now have all we need to return proper prior hypotheses

        # Compute the clusters we consider
        clusters_to_use = np.arange(num_clusters)#np.argsort(clustersCard)[::-1][:num_clusters]

        # prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        #     py_dfg_da.hypothesis.Hypotheses([
        #         py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
        #         py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
        #     ]),
        #     py_dfg_da.hypothesis.Hypotheses([
        #         py_dfg_da.hypothesis.Hypothesis([4], np.log(0.5)),
        #         py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5))
        #     ])
        # ])

        prior_hypotheses_per_cluster = py_dfg_da.hypothesis.HypothesesList()
        for c in clusters_to_use:
            hypotheses_in_cluster = hypotheses_per_cluster[c]
            log_hypo_probs_in_cluster = log_hypo_probs_per_cluster[c]
            prior_hypotheses_in_cluster = py_dfg_da.hypothesis.Hypotheses([py_dfg_da.hypothesis.Hypothesis(tracks_in_hypotheses[h], p) for h, p in zip(hypotheses_in_cluster, log_hypo_probs_in_cluster)])
            prior_hypotheses_per_cluster.append(prior_hypotheses_in_cluster)

        return prior_hypotheses_per_cluster, clusters_to_use


    def ws_to_prior_hypotheses(self, ws):
        hypos = ws["hypos"].ravel().astype(int)
        hyposCard = ws["hyposCard"].ravel().astype(int)
        probLogHypos = ws["probLogHypos"].ravel()
        clustersCard = ws["clustersCard"].ravel().astype(int)
        clusters = ws["clusters"].ravel().astype(int) - 1 # We negate one here to make hypotheses 0-indexed, which is more convenient. Tracks we keep 1-indexed
        assert (clusters >= 0).all()

        num_clusters = len(clustersCard)

        # Let's actually first figure out the clusters we are working with and find the probabilities
        log_hypo_probs_per_cluster = []
        hypotheses_per_cluster = []

        i = 0
        for clusterC in clustersCard:
            # These hypotheses are in the cluster
            stop = i + clusterC
            hypotheses_in_cluster = clusters[i:stop]

            # hyposCard is as long as probLogCard, so pick out the elements that correspond to the "indices" in hypothesis_in_clutter
            probLogHyposInCluster = probLogHypos[hypotheses_in_cluster]

            log_hypo_probs_per_cluster.append(probLogHyposInCluster)
            hypotheses_per_cluster.append(hypotheses_in_cluster)

            i += clusterC

        # Before we start looping over clusters, let's do this the simple way of making a list of lists, containing tracks contained in each hypothesis, then we sort it afterwards
        tracks_in_hypotheses = [None] * hyposCard.shape[0]
        endInd = np.cumsum(hyposCard)
        beginInd = endInd - hyposCard
        for hypos_in_cluster in hypotheses_per_cluster:
            for h in hypos_in_cluster:
                start_idx = beginInd[h]
                stop_idx = endInd[h]
                tracks_in_hypothesis = hypos[start_idx:stop_idx]
                tracks_in_hypotheses[h] = tracks_in_hypothesis

        track_set = [set(h) for h in tracks_in_hypotheses]
        assert (np.array([sum([kk == hh for hh in track_set if len(hh) > 0]) for kk in track_set if len(kk) > 0]) == 1).all(), f"{track_set}"

        assert all(h is not None for h in tracks_in_hypotheses)

        # We now have all we need to return proper prior hypotheses

        # Compute the clusters we consider
        clusters_to_use = np.arange(num_clusters)#np.argsort(clustersCard)[::-1][:num_clusters]

        prior_hypotheses_per_cluster = []
        for c in clusters_to_use:
            hypotheses_in_cluster = hypotheses_per_cluster[c]
            log_hypo_probs_in_cluster = log_hypo_probs_per_cluster[c]
            prior_hypotheses_in_cluster = PriorHypotheses([PriorHypothesis(tracks_in_hypotheses[h], p) for h, p in zip(hypotheses_in_cluster, log_hypo_probs_in_cluster)])
            prior_hypotheses_per_cluster.append(prior_hypotheses_in_cluster)

        return prior_hypotheses_per_cluster, clusters_to_use
    
    def track_states_covariances(self) -> List[TrackEstimate]:
        estimates: List[TrackEstimate] = []

        predX = self.ws["predX"].T
        predP = self.ws["predP"].transpose((-1, 0, 1))
        for x, P in zip(predX, predP):
            estimates.append(TrackEstimate(state=x, covariance=P))

        return estimates
    
    def predicted_track_measurement(self) -> List[PredictedMeasurement]:
        measurements: List[PredictedMeasurement] = []

        predZ = self.ws["predZ"].T
        predS = self.ws["predS"].transpose((-1, 0, 1))
        for z, S in zip(predZ, predS):
            measurements.append(PredictedMeasurement(measurement=z, covariance=S))

        return measurements
    
    def clusters(self) -> List[np.ndarray]:
        tracks_per_cluster: List[np.ndarray] = []

        for ph in self.prior_hypotheses_per_cluster:
            t = np.fromiter(ph.tracks(), dtype=int)
            tracks_per_cluster.append(t)

        return tracks_per_cluster


SelfMarginals = TypeVar("SelfMarginals", bound="StatsLogger.Marginals")

class Marginals:
    def __init__(self, marginals: np.ndarray):
        self.misdetection_marginals = marginals[:, 0]
        self.detection_marginals = marginals[:, 1:-1].ravel()
        self.nonexistence_marginals = marginals[:, -1]
        self.marginals = marginals.ravel()
        self.marginals_raw = marginals

    @classmethod
    def concatenate(cls, marginals_list: List[SelfMarginals]) -> SelfMarginals:
        this = cls(np.ones((1,1)))
        
        this.marginals = np.hstack([obj.marginals for obj in marginals_list])
        this.misdetection_marginals = np.hstack([obj.misdetection_marginals for obj in marginals_list])
        this.detection_marginals = np.hstack([obj.detection_marginals for obj in marginals_list])
        this.nonexistence_marginals = np.hstack([obj.nonexistence_marginals for obj in marginals_list])

        return this

    @classmethod
    def from_path(cls, path: str) -> SelfMarginals:
        marginal_files = [
            "marginals",
            "misdetection_marginals",
            "detection_marginals",
            "nonexistence_marginals"
        ]

        out = cls(np.ones((1,1)))
        for marginal_file in marginal_files:
            setattr(out, marginal_file, np.fromfile(f"{path}/{marginal_file}.bin"))

        return out

    def nonexistence_removed(self) -> SelfMarginals:
        """
        Sets P(nonexistence) = 0.0 and renormalizes the other probabilities
        """
        marginals = self.marginals_raw.copy()
        marginals[:, -1] = 0.0
        marginals = marginals / marginals.sum(axis=1, keepdims=True)

        return Marginals(marginals)


SelfMarginalsErrors = TypeVar("SelfMarginalsErrors", bound="StatsLogger.MarginalsErrors")

class MarginalsErrors:
    max_errors: np.ndarray
    abs_errors: np.ndarray
    raw_errors: np.ndarray
    misdetection_errors: np.ndarray
    detection_errors: np.ndarray
    nonexistence_errors: np.ndarray

    def __init__(self, exact_marginals: Marginals, approx_marginals: Marginals):
        raw_error_marginals = exact_marginals.marginals_raw - approx_marginals.marginals_raw
        abs_error_marginals = np.abs(raw_error_marginals)
        assert ((0 <= abs_error_marginals) & (abs_error_marginals <= 1.0)).all()
        self.max_errors = abs_error_marginals.max(axis=1)
        self.abs_errors = abs_error_marginals.ravel()
        self.raw_errors = raw_error_marginals.ravel()
        self.misdetection_errors = abs_error_marginals[:, 0]
        self.detection_errors = abs_error_marginals[:, 1:-1].ravel()
        self.nonexistence_errors  = abs_error_marginals[:, -1]

    @classmethod
    def concatenate(cls, marginal_errors: List[SelfMarginalsErrors]) -> SelfMarginalsErrors:
        max_errors = []
        abs_errors = []
        raw_errors = []
        misdetection_errors = []
        detection_errors = []
        nonexistence_errors = []

        for marginal_error in marginal_errors:
            max_errors.append(marginal_error.max_errors)
            abs_errors.append(marginal_error.abs_errors)
            raw_errors.append(marginal_error.raw_errors)
            misdetection_errors.append(marginal_error.misdetection_errors)
            detection_errors.append(marginal_error.detection_errors)
            nonexistence_errors.append(marginal_error.nonexistence_errors)

        max_errors = np.hstack(max_errors)
        abs_errors = np.hstack(abs_errors)
        raw_errors = np.hstack(raw_errors)
        misdetection_errors = np.hstack(misdetection_errors)
        detection_errors = np.hstack(detection_errors)
        nonexistence_errors = np.hstack(nonexistence_errors)

        e = cls(Marginals(np.ones((1,1))), Marginals(np.ones((1,1))))
        e.max_errors = max_errors
        e.abs_errors = abs_errors
        e.raw_errors = raw_errors
        e.misdetection_errors = misdetection_errors
        e.detection_errors = detection_errors
        e.nonexistence_errors = nonexistence_errors

        return e

    
    @classmethod
    def from_path(cls, path: str) -> SelfMarginalsErrors:
        error_names = [
            "max_errors",
            "abs_errors",
            "raw_errors",
            "misdetection_errors",
            "detection_errors",
            "nonexistence_errors",
        ]

        out = cls(Marginals(np.ones((1,1))), Marginals(np.ones((1,1))))
        for error_name in error_names:
            setattr(out, error_name, np.fromfile(f"{path}/{error_name}.bin"))

        return out

@dataclass
class StatsLogger:

    def save_errors(self, path: str,  errors: MarginalsErrors) -> None:
        error_names = [
            "max_errors",
            "abs_errors",
            "raw_errors",
            "misdetection_errors",
            "detection_errors",
            "nonexistence_errors",
        ]

        for error_name in error_names:
            error = getattr(errors, error_name)
            filepath = f"{path}/{error_name}.bin"
            error.tofile(filepath)

    def save_marginals(self, path: str,  marginals: Marginals) -> None:
        marginals_names = [
            "marginals",
            "misdetection_marginals",
            "detection_marginals",
            "nonexistence_marginals"
        ]

        for marginals_name in marginals_names:
            marginals_data = getattr(marginals, marginals_name)
            filepath = f"{path}/{marginals_name}.bin"
            marginals_data.tofile(filepath)

@dataclass
class LBPStats:
    num_iters_msg: int
    num_iters: int
    marginals: Marginals
    converged: bool

@dataclass
class WilliamsStats:
    lbp_iters: np.ndarray
    marginals: Marginals
    marginals_exact_normalization_constant: Optional[Marginals]
    normalization_constants: List[float]
    converged_list: np.ndarray

@dataclass
class BetheStats:
    marginals: Marginals
    normalization_constants_odin: List[float]
    normalization_constants_lc: List[float]

@dataclass
class ExactStats:
    marginals: Marginals
    normalization_constants: List[float]

# @dataclass
# class MulticlusterData:
#     exact_computation_error: bool

#     exact_marginals: Marginals
#     exact_normalization_constant: float

#     mhlbp_marginals: Marginals
#     bethe_normalization_constant: float

#     def save_data(self, path):
#         with open(path, "wb") as f:
#             pickle.dump(self, f)

#     @classmethod
#     def from_data(cls, path):
#         with open(path, "rb") as f:
#             return pickle.load(f)

@dataclass
class MulticlusterConditionendLBPOutput:
    marginals: np.ndarray
    likelihood: float
    theta_posteriors: Optional[Dict[int, np.ndarray]] = None
    raised_warning: bool = False


@dataclass(frozen=True)
class MulticlusterApproximateOutput:
    full_output: Union[MulticlusterConditionendLBPOutput, py_dfg_da.lbp.MHLBPMulticlusterOutput]
    approx_marginals: np.ndarray
    approx_normalization_constant: float
    approx_theta_posteriors: List[np.ndarray]
    runtime: float


@dataclass
class MulticlusterData:
    mcmhlbp_output: Optional[MulticlusterApproximateOutput] = None
    mc_bethe_output: Optional[MulticlusterConditionendLBPOutput] = None
    mc_phd_output: Optional[MulticlusterConditionendLBPOutput] = None
    mc_mhlbp_output: Optional[MulticlusterConditionendLBPOutput] = None
    exact_output: Optional[MulticlusterExactOutput] = None
    
    explicit_hypothesis_enumeration_error: bool = False

    def save_data(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_data(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    async def save_data_async(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    async def from_data_async(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class ClusterData:
    explicit_hypothesis_enumeration_error: bool = False

    cardinality: Optional[int] = None
    tracks: Optional[FrozenSet[int]] = None
    num_hypotheses: Optional[int] = None

    lbp_stats: Optional[LBPStats] = None
    williams_stats: Optional[WilliamsStats] = None
    bethe_stats: Optional[BetheStats] = None
    exact_stats: Optional[ExactStats] = None

    def save_data(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @classmethod
    def from_data(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
