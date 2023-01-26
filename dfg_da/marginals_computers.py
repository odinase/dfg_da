import numpy as np
from .marginal_association_Odin import lbp_marginal, exact_marginal, lbp_marginal_nonexistence, lbp_marginal_nonexistence_alternative
from abc import ABC, abstractmethod
from .prior_hypothesis import PriorHypotheses, PriorHypothesis
from typing import Tuple, Optional, Union


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

        n, m = mu.shape
        F_B_pseudo = (m - 1)*np.log(Zt).sum() + (n - 1)*np.log(Zj).sum() - np.log(Ztj).sum()

        return F_B_pseudo

    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        n, mp1 = R_LC.shape
        m = mp1 - 1
        all_tracks_idx = np.arange(n)
        
        lbp_marginal_total = np.zeros((n, m + 1 + 1))

        conditioned_marginals = np.empty((n, m + 2))

        normalizing_constants_odin = np.empty(len(prior_hypotheses))
        normalizing_constants_lc = np.empty(len(prior_hypotheses))
    
        for k, (tracks, hypo_prob) in enumerate(prior_hypotheses):

            R_sub = R_LC[tracks-1, :]

            if len(tracks) > 0:
                lbp_probs, it_from_lbp, converged, bethe_log_lc, mu, nu, w_nmd = lbp_marginal(R_sub, return_mu_nu_w_nmd=True)
                F_b_psuedo = self.bethe_constant(w_nmd, mu, nu)
                bethe_log_odin = -F_b_psuedo
            else:
                # Williams LBP returns wonky stuff for empty hypotheses, set sepcific values
                lbp_probs = np.empty((0, R_LC.shape[1]))
                it_from_lbp = 0
                converged = True
                bethe_log_lc = 0
                bethe_log_odin = 0
            
            # We need to concatenate the JPDAprobs with all tracks and existence probs
            existing_tracks_idx = tracks - 1
            non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

            existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
            nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

            conditioned_marginals[existing_tracks_idx] = existing_probs
            conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

            normalizing_constant = np.exp(bethe_log_odin) # self.bethe_constant(mu, nu, w_nmd)
            normalizing_constants_odin[k] = normalizing_constant
            normalizing_constants_lc[k] = np.exp(bethe_log_lc)

            lbp_marginal_total += conditioned_marginals * normalizing_constant * hypo_prob


        lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)

        assert (np.abs(lbp_marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
        assert ((0 <= lbp_marginal_total) & (lbp_marginal_total <= 1.0)).all()

        return lbp_marginal_total, normalizing_constants_odin, normalizing_constants_lc



class ExactMarginals(MarginalsComputer):
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        n, mp1 = R_LC.shape
        m = mp1 - 1
        all_tracks_idx = np.arange(n)

        normalizing_constants = np.empty(len(prior_hypotheses))
        marginal_total = np.zeros((n, m + 1 + 1))
        conditioned_marginals = np.empty((n, m + 2))

        _0 = np.zeros((n, 1))
        for k, (tracks, hypo_prob) in enumerate(prior_hypotheses):
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
    def compute_marginals(self, R_LC: np.ndarray, prior_hypotheses: PriorHypotheses, **kwargs) -> Tuple[np.ndarray, Optional[Tuple]]:
        out = lbp_marginal_nonexistence(R_LC, prior_hypotheses, **kwargs)
        asso_prob = out[0]
        it, msg_it, converged = out[1:4]
        if len(out[4:]) > 0:
            extra = out[4:]
        else:
            extra = ()
        return (asso_prob, (it, msg_it, converged)) + extra


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