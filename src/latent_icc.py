"""
Latent-scale intraclass correlation for a binary outcome.

A random-intercept logit  logit P(y_ij = 1) = b0 + offset_ij + u_j,  u_j ~ N(0, s^2),  is fitted by maximum
likelihood with Gauss-Hermite quadrature over u_j. The latent-scale ICC is  s^2 / (s^2 + pi^2 / 3), the share of
variance on the logistic latent scale that lies between groups. Unlike the observed-scale (ANOVA) ICC of the 0/1
outcome, it does not depend on prevalence.

With an offset (the linear predictor of a covariate model fitted without groups), the ICC is conditional on
those covariates; the covariate coefficients are held at their pooled values rather than re-estimated jointly.

Without an offset, groups enter only through their (successes, trials) counts, so a fit costs milliseconds and
the ICC can be computed for thousands of reference partitions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_expit

_X, _W = np.polynomial.hermite.hermgauss(40)
_W = _W / np.sqrt(np.pi)
_LOG_S_BOUNDS = (-7.0, 3.0)


def _fit(nll, b0_start):
    best = None
    for log_s0 in (-1.0, 0.0):
        r = minimize(nll, [b0_start, log_s0], method="L-BFGS-B", bounds=[(-15, 15), _LOG_S_BOUNDS])
        if best is None or r.fun < best.fun:
            best = r
    return best


def latent_icc(y, groups, offset=None) -> float:
    """Latent-scale ICC of binary y within groups (optionally conditional on a fixed offset)."""
    y = np.asarray(y, float)
    _, gi = np.unique(np.asarray(groups), return_inverse=True)
    p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    b0_start = np.log(p / (1 - p))
    if offset is None:
        k = np.bincount(gi, weights=y)
        n = np.bincount(gi).astype(float)

        def nll(theta):
            eta = theta[0] + np.sqrt(2) * np.exp(theta[1]) * _X[None, :]
            ll = k[:, None] * log_expit(eta) + (n - k)[:, None] * log_expit(-eta)
            m = ll.max(1, keepdims=True)
            return -np.sum(m[:, 0] + np.log(np.exp(ll - m) @ _W))
    else:
        order = np.argsort(gi, kind="stable")
        off, ys = np.asarray(offset, float)[order], y[order]
        starts = np.flatnonzero(np.r_[True, np.diff(gi[order]) != 0])

        def nll(theta):
            eta = theta[0] + off[:, None] + np.sqrt(2) * np.exp(theta[1]) * _X[None, :]
            ll_i = np.where(ys[:, None] == 1, log_expit(eta), log_expit(-eta))
            ll = np.add.reduceat(ll_i, starts, axis=0)
            m = ll.max(1, keepdims=True)
            return -np.sum(m[:, 0] + np.log(np.exp(ll - m) @ _W))
        b0_start = 0.0
    s2 = np.exp(2 * _fit(nll, b0_start).x[1])
    return float(s2 / (s2 + np.pi ** 2 / 3))
