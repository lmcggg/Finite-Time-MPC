import numpy as np

from .utils import Polytope, dlqr_gain
from .single_input import SingleInputFiniteTimeMPC


def build_augmented_system(A, B, alpha1, C1):
    """Build the augmented system (31)-(32):

        sigma_1(k+1) = alpha1 * sigma_1(k) + C1 * x(k)
        x(k+1)       = A x(k) + B u(k)

        xi = [sigma_1; x],   xi(k+1) = A_aug xi(k) + B_aug u(k)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    C1 = np.atleast_2d(np.asarray(C1, dtype=float))
    p1 = C1.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    A_aug = np.block([
        [alpha1 * np.eye(p1), C1],
        [np.zeros((n, p1)), A],
    ])
    B_aug = np.block([
        [np.zeros((p1, m))],
        [B],
    ])
    return A_aug, B_aug, p1


def lift_state_polytope(X: Polytope, p1):
    """Lift a polytope on x to the augmented state xi = [sigma_1; x]."""
    H = np.hstack([np.zeros((X.n_ineq, p1)), X.H])
    return Polytope(H, X.b)


class AugmentedFiniteTimeMPC:
    """Algorithm 3: Constrained finite-time MPC with integral action.

    Augments the original linear plant with an integral-like state and applies
    Algorithm 1 to the augmented system. The control horizon becomes
    N = n + p_1, which improves the initial feasibility region.
    """

    def __init__(self, A, B, X: Polytope, U: Polytope, Q_aug, R, K_aug,
                 alpha1, C1, sigma1_0=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        if self.B.ndim == 1:
            self.B = self.B.reshape(-1, 1)
        self.alpha1 = float(alpha1)
        self.C1 = np.atleast_2d(np.asarray(C1, dtype=float))
        self.A_aug, self.B_aug, self.p1 = build_augmented_system(
            self.A, self.B, self.alpha1, self.C1
        )
        self.n_orig = self.A.shape[0]
        self.n_aug = self.A_aug.shape[0]
        self.X_orig = X
        self.X_aug = lift_state_polytope(X, self.p1)
        self.U = U
        self.Q_aug = np.asarray(Q_aug, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.K_aug = np.asarray(K_aug, dtype=float).reshape(self.B_aug.shape[1], self.n_aug)
        self.sigma1_0 = (
            np.zeros(self.p1) if sigma1_0 is None else np.asarray(sigma1_0, dtype=float).flatten()
        )
        self.mpc = SingleInputFiniteTimeMPC(
            self.A_aug, self.B_aug, self.X_aug, self.U,
            self.Q_aug, self.R, self.K_aug,
        )
        self.P = self.mpc.P
        self.eps = self.mpc.eps
        self.N = self.mpc.N

    def _augment_state(self, x, sigma1):
        return np.concatenate([sigma1.flatten(), x.flatten()])

    def step(self, x, sigma1):
        xi = self._augment_state(x, sigma1)
        return self.mpc.step(xi)

    def simulate(self, x0, T):
        n = self.n_orig
        m = self.B.shape[1]
        xs = np.zeros((T + 1, n))
        sigmas = np.zeros((T + 1, self.p1))
        us = np.zeros((T, m))
        xs[0] = np.asarray(x0).flatten()
        sigmas[0] = self.sigma1_0
        for k in range(T):
            u = self.step(xs[k], sigmas[k])
            if u is None:
                raise RuntimeError(f"Optimization infeasible at step k={k}.")
            us[k] = u
            xs[k + 1] = self.A @ xs[k] + self.B @ u
            sigmas[k + 1] = self.alpha1 * sigmas[k] + self.C1 @ xs[k]
        return xs, us, sigmas
