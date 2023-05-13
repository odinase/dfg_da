import numpy as np
from .marginal_association_Odin import lbp_marginal, exact_marginal, lbp_marginal_nonexistence, lbp_marginal_nonexistence_alternative
from abc import ABC, abstractmethod
from .prior_hypothesis import PriorHypotheses, PriorHypothesis
from typing import Tuple, Optional, Union, List, Dict
import py_dfg_da as pdd
from dataclasses import dataclass
from collections import Counter, defaultdict
from pyehm.core import EHM2
from cluster_data_asso import lc_to_edmund


class MarginalsComputer(ABC):
    def __call__(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> np.ndarray:
        return self.compute_marginals(R_LC, prior_hypotheses, **kwargs)

    @abstractmethod
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        return None


class LBPMarginalsByTotalProb(MarginalsComputer):
    def PHD_normalizing_constant_approximation(self, R_sub: np.ndarray) -> float:
        loglikelihoods = R_sub[:, 1:]
        gated_measurements = np.any(np.isfinite(loglikelihoods), axis=0)
        loglikelihoods = loglikelihoods[:, gated_measurements]
        
        mu = (1 - np.exp(R_sub[:, 0])).sum()
        normalizing_constant = np.exp(-mu) * (np.exp(loglikelihoods).sum(0) + 1).prod()

        return normalizing_constant

    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        n, mp1 = R_LC.shape
        m = mp1 - 1
        all_tracks_idx = np.arange(n)

        lbp_marginal_total = np.zeros((n, m + 1 + 1))

        conditioned_marginals = np.empty((n, m + 2))

        own_normalizing_constants = None
        if "own_normalizing_constants" in kwargs:
            own_normalizing_constants = kwargs["own_normalizing_constants"]
        if own_normalizing_constants is not None:
            lbp_marginal_total_exact_norm_const = np.zeros((n, m + 1 + 1))

        normalizing_constants = np.empty(len(prior_hypotheses))

        iters_list = np.empty(len(prior_hypotheses), dtype=int)
        converged_list = np.empty(len(prior_hypotheses), dtype=bool)

        for k, (tracks, hypo_prob) in enumerate(prior_hypotheses):

            R_sub = R_LC[tracks-1, :]

            if len(tracks) > 0:
                lbp_probs, it_from_lbp, converged, _ = lbp_marginal(R_sub)
            else:
                # Williams LBP returns wonky stuff for empty hypotheses, set sepcific values
                lbp_probs = np.empty((0, R_LC.shape[1]))
                it_from_lbp = 0
                converged = True

            iters_list[k] = it_from_lbp
            converged_list[k] = converged

            # We need to concatenate the JPDAprobs with all tracks and existence probs
            existing_tracks_idx = tracks - 1
            non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

            existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
            nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

            conditioned_marginals[existing_tracks_idx] = existing_probs
            conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs
            normalizing_constant = self.PHD_normalizing_constant_approximation(R_sub)

            if own_normalizing_constants is not None:
                exact_normalizing_constant = own_normalizing_constants[k]

            normalizing_constants[k] = normalizing_constant

            lbp_marginal_total += conditioned_marginals * normalizing_constant * hypo_prob

            if own_normalizing_constants is not None:
                lbp_marginal_total_exact_norm_const += conditioned_marginals * exact_normalizing_constant * hypo_prob

        lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)

        assert (np.abs(lbp_marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= lbp_marginal_total) & (lbp_marginal_total <= 1.0)).all()

        if own_normalizing_constants is not None:
            lbp_marginal_total_exact_norm_const = lbp_marginal_total_exact_norm_const / lbp_marginal_total_exact_norm_const.sum(axis=1, keepdims=True)
            assert (np.abs(lbp_marginal_total_exact_norm_const.sum(axis=1) - 1.0) < 1e-6).all()
            assert ((0 <= lbp_marginal_total_exact_norm_const) & (lbp_marginal_total_exact_norm_const <= 1.0)).all()

        if own_normalizing_constants is None:
            out = lbp_marginal_total, (normalizing_constants, iters_list, converged_list, None)
        else:
            out = lbp_marginal_total, (normalizing_constants, iters_list, converged_list, lbp_marginal_total_exact_norm_const)

        return out


class LBPMarginalsByTotalProbBethe(MarginalsComputer):
    def track_normalizing_constant(self, w_nmd: np.ndarray, nu: np.ndarray) -> np.ndarray:
        return (w_nmd*nu).sum(axis=1) + 1

    def meas_normalizing_constant(self, mu: np.ndarray) -> np.ndarray:
        return mu.sum(axis=0) + 1

    def edge_normalizing_constant(self, w_nmd: np.ndarray, mu: np.ndarray, nu: np.ndarray) -> np.ndarray:
        w_times_msg = w_nmd * nu
        Ztj = (1 + (mu.sum(axis=0, keepdims=True) - mu)) * (1 + (w_times_msg.sum(axis=1, keepdims=True) - w_times_msg)) + w_nmd
        return Ztj

    def bethe_constant(self, w_nmd: np.ndarray, mu: np.ndarray, nu: np.ndarray) -> float:
        Zt = self.track_normalizing_constant(w_nmd, nu)
        Zj = self.meas_normalizing_constant(mu)
        Ztj = self.edge_normalizing_constant(w_nmd, mu, nu)

        n, m = w_nmd.shape
        F_B_pseudo = (m - 1)*np.log(Zt).sum() + (n - 1)*np.log(Zj).sum() - np.log(Ztj).sum()

        return F_B_pseudo

    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        n, mp1 = R_LC.shape
        m = mp1 - 1
        all_tracks_idx = np.arange(n)
        
        lbp_marginal_total = np.zeros((n, m + 1 + 1))

        conditioned_marginals = np.empty((n, m + 2))
        conditioned_theta_posterior = np.empty(len(prior_hypotheses))

        normalizing_constants = np.empty(len(prior_hypotheses))
    
        likelihood = 0.0

        for k, ph in enumerate(prior_hypotheses):
            tracks = np.sort(np.array(ph.tracks())).astype(int)
            hypo_prob = ph.probability()

            R_sub = R_LC[tracks-1, :]
            # Williams LBP returns wonky stuff for empty hypotheses, set sepcific values
            lbp_probs = np.empty((0, R_LC.shape[1]))
            bethe_loglikelihood = 0

            if len(tracks) > 0:
                lbp_probs, it_from_lbp, converged, bethe_log_lc, mu, nu, w_nmd = lbp_marginal(R_sub, return_mu_nu_w_nmd=True)
                F_b_psuedo = self.bethe_constant(w_nmd, mu, nu)
                bethe_loglikelihood = -F_b_psuedo
            
            # We need to concatenate the JPDAprobs with all tracks and existence probs
            existing_tracks_idx = tracks - 1
            non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

            existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
            nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

            conditioned_marginals[existing_tracks_idx] = existing_probs
            conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

            normalizing_constant = np.exp(bethe_loglikelihood) # self.bethe_constant(mu, nu, w_nmd)
            normalizing_constants[k] = normalizing_constant
            conditioned_theta_posterior[k] = hypo_prob * normalizing_constant

            likelihood += normalizing_constant * hypo_prob
            lbp_marginal_total += conditioned_marginals * normalizing_constant * hypo_prob


        conditioned_theta_posterior = conditioned_theta_posterior / conditioned_theta_posterior.sum()
        lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)

        assert (np.abs(lbp_marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= lbp_marginal_total) & (lbp_marginal_total <= 1.0)).all()

        return lbp_marginal_total, conditioned_theta_posterior, likelihood



class LBPMarginalsByTotalProbPHD(MarginalsComputer):
    def PHD_normalizing_constant_approximation(self, R_sub: np.ndarray) -> float:
        loglikelihoods = R_sub[:, 1:]
        gated_measurements = np.any(np.isfinite(loglikelihoods), axis=0)
        loglikelihoods = loglikelihoods[:, gated_measurements]
        
        mu = (1 - np.exp(R_sub[:, 0])).sum()
        normalizing_constant = np.exp(-mu) * (np.exp(loglikelihoods).sum(0) + 1).prod()

        return normalizing_constant

    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        n, mp1 = R_LC.shape
        m = mp1 - 1
        all_tracks_idx = np.arange(n)
        
        lbp_marginal_total = np.zeros((n, m + 1 + 1))

        conditioned_marginals = np.empty((n, m + 2))
        conditioned_theta_posterior = np.empty(len(prior_hypotheses))

        normalizing_constants = np.empty(len(prior_hypotheses))
    
        likelihood = 0.0

        for k, ph in enumerate(prior_hypotheses):
            tracks = np.sort(np.array(ph.tracks())).astype(int)
            hypo_prob = ph.probability()

            R_sub = R_LC[tracks-1, :]
            lbp_probs = np.empty((0, R_LC.shape[1]))
            phd_constant = 1.0

            if len(tracks) > 0:
                lbp_probs, it_from_lbp, converged, bethe_log_lc, mu, nu, w_nmd = lbp_marginal(R_sub, return_mu_nu_w_nmd=True)
                phd_constant = self.PHD_normalizing_constant_approximation(R_sub)
            
            # We need to concatenate the JPDAprobs with all tracks and existence probs
            existing_tracks_idx = tracks - 1
            non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

            existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
            nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

            conditioned_marginals[existing_tracks_idx] = existing_probs
            conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

            normalizing_constant = phd_constant
            normalizing_constants[k] = normalizing_constant

            lbp_marginal_total += conditioned_marginals * normalizing_constant * hypo_prob
            likelihood += normalizing_constant * hypo_prob

            conditioned_theta_posterior[k] = hypo_prob * normalizing_constant

        conditioned_theta_posterior = conditioned_theta_posterior / conditioned_theta_posterior.sum()
        lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)

        assert (np.abs(lbp_marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= lbp_marginal_total) & (lbp_marginal_total <= 1.0)).all()

        return lbp_marginal_total, conditioned_theta_posterior, likelihood



class ExactMarginals(MarginalsComputer):
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        n, mp1 = R_LC.shape
        m = mp1 - 1
        all_tracks_idx = np.arange(n)

        normalizing_constants = np.empty(len(prior_hypotheses))
        marginal_total = np.zeros((n, m + 1 + 1))
        conditioned_marginals = np.empty((n, m + 2))

        _0 = np.zeros((n, 1))
        for k, hypothesis in enumerate(prior_hypotheses):
            hypo_prob = hypothesis.probability()
            tracks = (np.array(hypothesis.tracks())).astype(int)
            R_sub = R_LC[tracks-1, :]
            JPDAprobs, _, loglikelihood = exact_marginal(R_sub, False)

            # We need to concatenate the JPDAprobs with all tracks and existence probs
            existing_tracks_idx = tracks - 1
            non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

            existing_probs = np.hstack((JPDAprobs, _0[:len(tracks)]))
            nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

            conditioned_marginals[existing_tracks_idx] = existing_probs
            conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

            normalizing_constant = np.exp(loglikelihood)

            normalizing_constants[k] = normalizing_constant

            marginal_total += conditioned_marginals * normalizing_constant * hypo_prob

        marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)

        assert (np.abs(marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= marginal_total) & (marginal_total <= 1.0)).all()

        return marginal_total, (normalizing_constants,)


class LBPMarginalsFullAssociation(MarginalsComputer):
    def bethe_loglikelihood_cpp(self, lbp_single_cluster_output: pdd.lbp.MHLBPSingleClusterOutput) -> float:
        n, m = lbp_single_cluster_output.w_nmd.shape

        w_nmd = lbp_single_cluster_output.w_nmd
        w_0 = lbp_single_cluster_output.w_0
        mu = lbp_single_cluster_output.mu
        nu = lbp_single_cluster_output.nu
        rho = lbp_single_cluster_output.rho
        sigma = lbp_single_cluster_output.sigma
        phi = lbp_single_cluster_output.phi

        t2h = lbp_single_cluster_output.t2h
        t2h_not = lbp_single_cluster_output.t2h_not

        rho_prods = (t2h.T * rho + t2h_not.T).prod(1)

        phi_rho_prods = phi * rho_prods

        Z_theta = phi_rho_prods.sum()

        w_times_msg_sum = (w_nmd * nu).sum(1)

        Z_t = w_0 + w_times_msg_sum + sigma

        Z_j = 1.0 + mu.sum(0)

        Z_tth = ((w_0 + w_times_msg_sum) / rho) * (phi_rho_prods * t2h).sum(1) + (phi_rho_prods * t2h_not).sum(1)

        Z_tj = (1.0 + (mu.sum(axis=0, keepdims=True) - mu)) * ((w_0[:,None] + (w_times_msg_sum[:,None] - (w_nmd * nu))) + sigma[:,None]) + w_nmd

        F = (
            (n-1)*np.log(Z_theta)
            + m*np.log(Z_t).sum()
            + (n-1)*np.log(Z_j).sum()
            - np.log(Z_tth).sum()
            - np.log(Z_tj).sum()
        )

        return -F

    def bethe_loglikelihood(self, w_nmd, w_0, mu, nu, rho, sigma, phi, t2h, t2h_not) -> float:
        n, m = w_nmd.shape

        w_nmd = w_nmd
        mu = mu
        nu = nu
        w_0 = w_0.ravel()
        rho = rho
        sigma = sigma
        phi = phi

        t2h = t2h
        t2h_not = t2h_not

        rho_prods = (t2h.T * rho + t2h_not.T).prod(1)

        phi_rho_prods = phi * rho_prods

        Z_theta = phi_rho_prods.sum()

        w_times_msg_sum = (w_nmd * nu).sum(1)

        Z_t = w_0 + w_times_msg_sum + sigma

        Z_j = 1.0 + mu.sum(0)

        Z_tth = ((w_0 + w_times_msg_sum) / rho) * (phi_rho_prods * t2h).sum(1) + (phi_rho_prods * t2h_not).sum(1)

        Z_tj = (1.0 + (mu.sum(axis=0, keepdims=True) - mu)) * ((w_0[:,None] + (w_times_msg_sum[:,None] - (w_nmd * nu))) + sigma[:,None]) + w_nmd

        F = (
            (n-1)*np.log(Z_theta)
            + m*np.log(Z_t).sum()
            + (n-1)*np.log(Z_j).sum()
            - np.log(Z_tth).sum()
            - np.log(Z_tj).sum()
        )

        return -F
    
    def theta_posteriors(self, phi, rho, t2h_idx, t2noth_idx) -> np.ndarray:
        rho_prods = (t2h_idx.T * rho + t2noth_idx.T).prod(1)
        p = rho_prods * phi
        p = p / p.sum()
        return p

    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, **kwargs) -> Tuple[np.ndarray, np.ndarray, float]:
# MHLBPSingleClusterOutput lbp_single_cluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const hypothesis::Hypotheses &prior_hypotheses, size_t max_num_iters = 300);

    # py::class_<lbp::MHLBPSingleClusterOutput>(lbp, "MHLBPSingleClusterOutput")
    # .def("track_association_marginals",  &lbp::MHLBPSingleClusterOutput::track_association_marginals)
    # .def("bethe_pseudodual_loglikelihood",  &lbp::MHLBPSingleClusterOutput::bethe_pseudodual_loglikelihood)
    # .def("bethe_pseudodual_normalization_constant",  &lbp::MHLBPSingleClusterOutput::bethe_pseudodual_normalization_constant);
        # R = np.asfortranarray(lc_to_edmund(R_LC))
        # lbp_single_cluster_output: pdd.lbp.MHLBPSingleClusterOutput = pdd.lbp.lbp_single_cluster(R, prior_hypotheses, max_num_iters = 10_000)
        w_nmd, w_0, mu, nu, rho, sigma, phi, t2h_idx, t2noth_idx, marginals = lbp_marginal_nonexistence(R_LC, prior_hypotheses, max_iter=10_000)
        # bethe_loglikelihood_cpp = self.bethe_loglikelihood_cpp(lbp_single_cluster_output)
        bethe_loglikelihood = self.bethe_loglikelihood(w_nmd, w_0, mu, nu, rho, sigma, phi, t2h_idx, t2noth_idx)
        likelihood = np.exp(bethe_loglikelihood)
        conditioned_theta_posteriors = self.theta_posteriors(phi, rho, t2h_idx, t2noth_idx)
        # cpp_bethe_loglikelihood = lbp_single_cluster_output.bethe_pseudodual_loglikelihood()
        # assert abs(bethe_loglikelihood - cpp_bethe_loglikelihood) < 1e-4
        # assert abs(bethe_loglikelihood_cpp - bethe_loglikelihood) < 1e-4
        # marginals = lbp_single_cluster_output.track_association_marginals().T
        # likelihood = lbp_single_cluster_output.bethe_pseudodual_normalization_constant()

        return marginals, conditioned_theta_posteriors, likelihood


class LBPMarginalsFullAssociationCPP(MarginalsComputer):
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, **kwargs) -> Tuple[np.ndarray, np.ndarray, float]:
        R = np.asfortranarray(lc_to_edmund(R_LC))
        lbp_single_cluster_output: pdd.lbp.MHLBPSingleClusterOutput = pdd.lbp.lbp_single_cluster(R, prior_hypotheses, max_num_iters = 1000)
        conditioned_theta_posteriors = lbp_single_cluster_output.hypotheses_marginal()
        marginals = lbp_single_cluster_output.track_association_marginals().T
        likelihood = lbp_single_cluster_output.bethe_pseudodual_normalization_constant()

        return marginals, conditioned_theta_posteriors, likelihood


class LBPMarginalsFullAssociationAlternative(MarginalsComputer):
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        out = lbp_marginal_nonexistence_alternative(R_LC, prior_hypotheses, **kwargs)
        asso_prob = out[0]
        it, msg_it, converged = out[1:4]
        if len(out[4:]) > 0:
            extra = out[4:]
        else:
            extra = ()
        return (asso_prob, (it, msg_it, converged)) + extra


@dataclass(frozen=True)
class ClusterHypothesisLabel:
    cluster_idx: int
    hypo_idx: int

    def __str__(self) -> str:
        return f"(C{self.cluster_idx + 1}, H{self.hypo_idx + 1})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, rhs: object) -> bool:
        return (self.cluster_idx == rhs.cluster_idx) and (self.hypo_idx == rhs.hypo_idx)
    
    def __iter__(self):
        return iter(self.__dict__.values())

class ClusterHypothesesPosterior:
    """
    We need to keep track of what hypothesis in what cluster the merged hypotheses originally came from. Can this be achieved with a recursive list of labels that maps backwards?
    It would seem that every time we add a hypothesis of a normal cluster to the hypothesis of a super cluster, we add a label to a recursive map of pointers that tracks back what hypothesis is contained?
    """

    def __init__(self, assocLocal: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList):
        # Assuming assocLocal is straight from MATLAB, we need to shift the cluster idx to 0-index
        assocLocal[0] -= 1
        self.prior_hypotheses_per_cluster_posterior, self.hypothesis_index_map = self.merge_clusters_labled(assocLocal, prior_hypotheses_per_cluster)

    def create_master_mapping(self, assocLocal, prior_hypotheses_per_cluster):
        # Figure out cluster masters and initialize prior hypothesis list
        cluster_masters = np.where(assocLocal[1])[0]
        prior_hypotheses_per_cluster_posterior: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            prior_hypotheses_per_cluster[c] for c in cluster_masters
        ])

        # cluster hypothesis mapping is initialized as list of labels for each master
        hypothesis_index_map: List[List[List[ClusterHypothesisLabel]]] = []
        for k, (ph, c) in enumerate(zip(prior_hypotheses_per_cluster_posterior, cluster_masters)):
            hypothesis_index_map.append([])
            for h in range(ph.num_hypotheses()):
                hypothesis_index_map[k].append(
                    [ClusterHypothesisLabel(c, h)]
                )
        
        return prior_hypotheses_per_cluster_posterior, hypothesis_index_map

    def update_hypothesis_index_map_master(self, hypothesis_index_map_master: List[List[ClusterHypothesisLabel]], slave: int, slave_ph: pdd.hypothesis.Hypotheses):
        updated_hypothesis_index_map_master: List[ClusterHypothesisLabel] = []

        # The inner loop in the combine step is the RHS, i.e. the slave
        # We can do the outer loop over mappings in the master and inner loop over hypos in slave and append the number to the outer mapping
        for master_hypos in hypothesis_index_map_master:
            for n in range(slave_ph.num_hypotheses()):
                updated_hypothesis_index_map_master.append(
                    master_hypos + [ClusterHypothesisLabel(slave, n)]
                )

        return updated_hypothesis_index_map_master

    def merge_clusters_labled(self, assocLocal: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList):
        # Initialize
        prior_hypotheses_per_cluster_posterior, hypothesis_index_map = self.create_master_mapping(assocLocal, prior_hypotheses_per_cluster)

        # Loop over remaining slave clusters and incrementally build the prior hypotheses posteriors
        master_idxs = np.cumsum(assocLocal[1]) - 1
        for prior_cluster_slave_idx, (prior_cluster_master_idx, is_master) in enumerate(assocLocal.T):
            if is_master:
                continue

            # Map from prior cluster index to posterior cluster index
            posterior_cluster_idx = master_idxs[prior_cluster_master_idx]

            # We already have the masters, merge clusters
            slave_ph = prior_hypotheses_per_cluster[prior_cluster_slave_idx]
            prior_hypotheses_per_cluster_posterior[posterior_cluster_idx] = prior_hypotheses_per_cluster_posterior[posterior_cluster_idx].combine(slave_ph)
            
            # Find the updated mapping for master cluster
            updated_hypothesis_index_map_master = self.update_hypothesis_index_map_master(hypothesis_index_map[posterior_cluster_idx], prior_cluster_slave_idx, slave_ph)
            hypothesis_index_map[posterior_cluster_idx] = updated_hypothesis_index_map_master

        return prior_hypotheses_per_cluster_posterior, hypothesis_index_map
    
    def map_prior_to_posteriors(self, label_prior: ClusterHypothesisLabel):
        """
        We wish to, given a prior hypothesis of a prior cluster, find the prior hypotheses in the posterior clusters it appears in.
        Preferbly, it should be as easy as possible to then look up the hypothesis conditioned likelihoods corresponding to the prior hpyothesis
        """
        posterior_prior_hyps_idxs = []
        # With assocLocal, this first step would be way easier, but oh well. First, find the posterior cluster the prior cluster appears in
        for posterior_cluster_idx, cluster_hypothesis_map in enumerate(self.hypothesis_index_map):
            prior_clusters = set(label.cluster_idx for hypotheses_labels in cluster_hypothesis_map for label in hypotheses_labels)
            if label_prior.cluster_idx in prior_clusters:
                # Next step is to find all indices for hypotheses in the cluster that matches the prior label
                for k, hypotheses_labels in enumerate(cluster_hypothesis_map):
                    if label_prior in hypotheses_labels:
                        posterior_prior_hyps_idxs.append(k)

        # Should be all?
        return posterior_cluster_idx, posterior_prior_hyps_idxs
    
    def prior_hypos_in_posterior_clusters(self):
        ph_in_post_c: List[Dict[int, int]] = []
        for hypotheses_in_cluster in self.hypothesis_index_map:
            phs = defaultdict(lambda: 0)
            for prior_cluster_labels in hypotheses_in_cluster:
                for c_idx, h_idx in prior_cluster_labels:
                    phs[c_idx] = max(h_idx, phs[c_idx])

            for c_idx in phs:
                phs[c_idx] += 1 # Hypotheses are 0-indexed, so we need to add 1 to turn them into the number of hypos

            ph_in_post_c.append(phs)

        return ph_in_post_c



class MulticlusterMarginalsComputer(ABC):
    def __call__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, **kwargs) -> np.ndarray:
        return self.compute_marginals(R_LC, prior_hypotheses_per_cluster, **kwargs)

    @abstractmethod
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        return None
    

@dataclass(frozen=True)
class MulticlusterExactOutput:
    exact_marginals: np.ndarray
    hypo_cond_normalization_constants_per_cluster: List[np.ndarray]
    normalization_constant_per_cluster: np.ndarray
    exact_normalization_constant: float
    cluster_hypotheses_posterior: ClusterHypothesesPosterior

    def compute_theta_posteriors(self):
        """
        Given the method in 'map_prior_to_posteriors' in ClusterHypothesesPosterior, we should be able to compute all we need by looping over each prior cluster and then each hypothesis, collect the necessary hypothesis-conditioned likelihoods and then normalize in the end.
        """

        # From LC
        # merged hypotese: theta = (theta_1, …, theta_n), med vekter w^theta = prod_{i=1}^n w_i^{theta_i}
        # p((theta_1, …, theta_n) | Z) propto w^h * exp(JPDA_loglikelihood), fra merged cluster JPDA
        # p(theta_i = h | Z) = sum_{(theta_1, …, theta_n): theta_i = h} p((theta_1, …, theta_n) | Z)


        # Dictionary over prior clusters in each posterior cluster and their hypothesis cardinality
        phs_in_post_c = self.cluster_hypotheses_posterior.prior_hypos_in_posterior_clusters()
        # Initialize container for each prior clusters posterior
        theta_posterior_marginals = dict()
        for phs in phs_in_post_c:
            for prior_c_idxs, num_hypos in phs.items():
                theta_posterior_marginals[prior_c_idxs] = np.zeros(num_hypos)

        # We have the priors in the from of prior_hypotheses_per_cluster_posterior
        # For each cluster, form the joint posterior by multipying by the JPDA-likelihood and normalizing over the cluster in the end
        for ph_posterior, hypo_cond_norm_per_hypo, hypotheses_map_in_cluster in zip(
            self.cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior,
            self.hypo_cond_normalization_constants_per_cluster,
            self.cluster_hypotheses_posterior.hypothesis_index_map
            ):
            # ph posterior is the joint prior distribution. Get its probabilities
            prior_probs = np.array(ph_posterior.hypothesis_probabilites())

            # We have already calculated the hypothesis-conditioned likelihood. For each merged prior hypothesis probability, multiply by the corresponding normalization constant to get the joint posterior
            joint_theta_posterior_scores = prior_probs * hypo_cond_norm_per_hypo

            # Now that we have the joint posterior, look up what hypotheses in this joint branched from each prior hypothesis in the unmerged clusters and add that to the marginal
            for joint_theta_score, hypothesis in zip(joint_theta_posterior_scores, hypotheses_map_in_cluster):
                for prior_cluster_idx, hypo_idx in hypothesis:
                    theta_posterior_marginals[prior_cluster_idx][hypo_idx] += joint_theta_score

        theta_posteriors = [None]*len(theta_posterior_marginals)
        for prior_c in theta_posterior_marginals:
            theta_posteriors[prior_c] = theta_posterior_marginals[prior_c] / theta_posterior_marginals[prior_c].sum()

        return theta_posteriors 


class MulticlusterExact(MulticlusterMarginalsComputer):
    def __call__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, **kwargs) -> np.ndarray:
        return self.compute_marginals(R_LC, prior_hypotheses_per_cluster, **kwargs)


    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        if not "assocLocal" in kwargs:
            raise ValueError("assocLocal is required as input parameter because of cluster merging!")

        # Start by merging clusters if necessary
        assocLocal = kwargs["assocLocal"]
        cluster_hypotheses_posterior = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster = prior_hypotheses_per_cluster)
        prior_hypotheses_per_cluster_posterior = cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior
        n, mp1 = R_LC.shape
        m = mp1 - 1

        # Preallocate variables
        hypo_cond_normalization_constants_per_cluster = []
        normalization_constants_per_cluster = np.empty(len(prior_hypotheses_per_cluster_posterior))
        marginal_total = np.zeros((n, m + 1 + 1))
        conditioned_marginals = np.empty((n, m + 2))
        _0 = np.zeros((n, m + 1))
        _1 = np.ones((n, 1))

        # Loop over clusters
        for c, prior_hypotheses in enumerate(prior_hypotheses_per_cluster_posterior):
            all_tracks_idx = np.sort(np.fromiter(prior_hypotheses.tracks(), dtype=int)) - 1
            conditioned_marginals[...] = 0.0
            hypo_cond_normalizing_constants = np.empty(len(prior_hypotheses))

            normalizing_constant_cluster = 0.0
            for k, hypothesis in enumerate(prior_hypotheses):
                log_prob = hypothesis.log_prob()
                existing_tracks_idx = (np.array(hypothesis.tracks()) - 1).astype(int)
                if existing_tracks_idx.shape[0] > 0:
                    R_sub = R_LC[existing_tracks_idx, :]
                    JPDAprobs, (hypoProbs, JPDAhypmat), loglikelihood = exact_marginal(R_sub, False)
                else:
                    JPDAprobs = np.empty((0, m + 1))
                    hyp_prob_log = np.empty((0,))
                    loglikelihood = 0.0

                # We need to concatenate the JPDAprobs with all tracks and existence probs
                non_existing_tracks_idx = np.setdiff1d(all_tracks_idx, existing_tracks_idx, assume_unique=True)

                existing_probs = np.hstack((JPDAprobs, _0[:len(existing_tracks_idx), 0, None]))
                nonexisting_probs = np.hstack((_0[:len(non_existing_tracks_idx)], _1[:len(non_existing_tracks_idx), 0, None]))

                conditioned_marginals[existing_tracks_idx] = existing_probs
                conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

                hypo_cond_normalization_constant = np.exp(loglikelihood)

                hypo_cond_normalizing_constants[k] = hypo_cond_normalization_constant

                marginal_total += conditioned_marginals * np.exp(loglikelihood + log_prob)

                normalizing_constant_cluster += np.exp(loglikelihood + log_prob)


            normalization_constants_per_cluster[c] = normalizing_constant_cluster

            hypo_cond_normalization_constants_per_cluster.append(hypo_cond_normalizing_constants)


        exact_normalization_constant = np.prod(normalization_constants_per_cluster)

        marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)

        assert (np.abs(marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= marginal_total) & (marginal_total <= 1.0)).all()

        output = MulticlusterExactOutput(
            exact_marginals=marginal_total,
            hypo_cond_normalization_constants_per_cluster=hypo_cond_normalization_constants_per_cluster,
            normalization_constant_per_cluster=normalization_constants_per_cluster,
            exact_normalization_constant=exact_normalization_constant,
            cluster_hypotheses_posterior=cluster_hypotheses_posterior
        )

        return output


class MulticlusterExactEHM2(MulticlusterMarginalsComputer):
    def __call__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, **kwargs) -> np.ndarray:
        return self.compute_marginals(R_LC, prior_hypotheses_per_cluster, **kwargs)

    def R_LC_to_validation_likelihood_matrix(self, R_LC: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        likelihood_matrix = np.asfortranarray(np.exp(R_LC))
        validation_matrix = np.asfortranarray((likelihood_matrix > 0.0).astype(np.int32))

        return validation_matrix, likelihood_matrix


    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, **kwargs) -> MulticlusterExactOutput:
        if not "assocLocal" in kwargs:
            raise ValueError("assocLocal is required as input parameter because of cluster merging!")

        # Start by merging clusters if necessary
        assocLocal = kwargs["assocLocal"]
        cluster_hypotheses_posterior = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster = prior_hypotheses_per_cluster)
        prior_hypotheses_per_cluster_posterior = cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior
        n, mp1 = R_LC.shape
        m = mp1 - 1

        # Preallocate variables
        hypo_cond_normalization_constants_per_cluster = []
        normalization_constants_per_cluster = np.empty(len(prior_hypotheses_per_cluster_posterior))
        marginal_total = np.zeros((n, m + 1 + 1))
        conditioned_marginals = np.empty((n, m + 2))
        _0 = np.zeros((n, m + 1))
        _1 = np.ones((n, 1))

        # Loop over clusters
        for c, prior_hypotheses in enumerate(prior_hypotheses_per_cluster_posterior):
            all_tracks_idx = np.sort(np.fromiter(prior_hypotheses.tracks(), dtype=int)) - 1
            conditioned_marginals[...] = 0.0
            hypo_cond_normalizing_constants = np.empty(len(prior_hypotheses))

            normalizing_constant_cluster = 0.0
            for k, hypothesis in enumerate(prior_hypotheses):
                prob = hypothesis.probability()
                existing_tracks_idx = (np.array(hypothesis.tracks()) - 1).astype(int)
                if existing_tracks_idx.shape[0] > 0:
                    R_sub = R_LC[existing_tracks_idx, :]
                    validation_matrix, likelihood_matrix = self.R_LC_to_validation_likelihood_matrix(R_sub)
                    JPDAprobs, likelihood = EHM2.run_and_likelihood(validation_matrix, likelihood_matrix)
                else:
                    JPDAprobs = np.empty((0, m + 1))
                    likelihood = 1.0

                # We need to concatenate the JPDAprobs with all tracks and existence probs
                non_existing_tracks_idx = np.setdiff1d(all_tracks_idx, existing_tracks_idx, assume_unique=True)

                existing_probs = np.hstack((JPDAprobs, _0[:len(existing_tracks_idx), 0, None]))
                nonexisting_probs = np.hstack((_0[:len(non_existing_tracks_idx)], _1[:len(non_existing_tracks_idx), 0, None]))

                conditioned_marginals[existing_tracks_idx] = existing_probs
                conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

                hypo_cond_normalizing_constants[k] = likelihood

                marginal_total += conditioned_marginals * likelihood * prob

                normalizing_constant_cluster += likelihood * prob


            normalization_constants_per_cluster[c] = normalizing_constant_cluster

            hypo_cond_normalization_constants_per_cluster.append(hypo_cond_normalizing_constants)


        exact_normalization_constant = np.prod(normalization_constants_per_cluster)

        marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)

        assert (np.abs(marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= marginal_total) & (marginal_total <= 1.0)).all()

        output = MulticlusterExactOutput(
            exact_marginals=marginal_total,
            hypo_cond_normalization_constants_per_cluster=hypo_cond_normalization_constants_per_cluster,
            normalization_constant_per_cluster=normalization_constants_per_cluster,
            exact_normalization_constant=exact_normalization_constant,
            cluster_hypotheses_posterior=cluster_hypotheses_posterior
        )

        return output
