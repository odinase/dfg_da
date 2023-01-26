import numpy as np
from scipy.io import loadmat
from typing import Dict, Any, List, TypeVar, FrozenSet, Optional
from dataclasses import dataclass
from scipy.special import logsumexp
from .prior_hypothesis import PriorHypothesis, PriorHypotheses
from .marginals_computers import MarginalsComputer, ExactMarginals
import pickle


@dataclass
class MatFileParser:
    ws: Dict[str, Any]

    reward_matrix_edmund: np.ndarray
    reward_matrix_lc: np.ndarray

    prior_hypotheses_per_cluster: List[PriorHypotheses]
    clusters_sorted: np.ndarray

    num_tracks: int
    num_measurements: int

    def __init__(self, filename: str, compute_hypotheses: bool = True):
        ws = loadmat(filename)
        self.ws = ws
        R_wrapping = ws["gainMatPostC"] # This R has a strange shape...
        track_file = ws["trackFile"]
        measurements = ws["measurements"]
        
        n = track_file.shape[1]
        m = measurements.shape[1]

        self.num_tracks = n
        self.num_measurements = m

        npm = n + m
        self.reward_matrix_edmund = R_wrapping[:n, :npm]
        self.reward_matrix_lc = np.hstack((np.diag(self.reward_matrix_edmund[:,m:])[:,None], self.reward_matrix_edmund[:,:m]))

        if compute_hypotheses:
            self.prior_hypotheses_per_cluster, self.clusters_sorted = self.ws_to_prior_hypotheses(ws)


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
