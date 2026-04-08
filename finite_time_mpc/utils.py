import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are


class Polytope:
    """Convex polytope defined by H x <= b."""

    def __init__(self, H, b):
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.b = np.atleast_1d(np.asarray(b, dtype=float)).flatten()
        if self.H.shape[0] != self.b.size:
            raise ValueError("Row count of H must match length of b.")

    @property
    def n_ineq(self):
        return self.H.shape[0]

    @property
    def dim(self):
        return self.H.shape[1]

    @classmethod
    def from_box(cls, lb, ub):
        lb = np.atleast_1d(np.asarray(lb, dtype=float)).flatten()
        ub = np.atleast_1d(np.asarray(ub, dtype=float)).flatten()
        n = lb.size
        H = np.vstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([ub, -lb])
        return cls(H, b)

    def contains(self, x, tol=1e-8):
        x = np.asarray(x).flatten()
        return np.all(self.H @ x - self.b <= tol)

    def transform_state(self, T):
        """Return polytope describing {z | H T^{-1} z <= b}, i.e., z = T x."""
        Tinv = np.linalg.inv(T)
        return Polytope(self.H @ Tinv, self.b)


def lyapunov_P(A, B, K, Q, R):
    """Solve P from A_K^T P A_K - P = -(Q + K^T R K), where A_K = A - B K."""
    A_K = A - B @ K
    Qbar = Q + K.T @ R @ K
    P = solve_discrete_lyapunov(A_K.T, Qbar)
    return P


def dlqr_gain(A, B, Q, R):
    """Discrete LQR gain K such that u = -K x stabilizes (A, B)."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K


def terminal_radius(P, K, X: Polytope, U: Polytope):
    """Compute eps in X_f = {x | x^T P x <= eps} from formula (14):

        eps = min[ min_{-K z in dU} z^T P z ,  min_{z in dX} z^T P z ]

    The minimum of x^T P x subject to h^T x = b has analytical value
    b^2 / (h^T P^{-1} h).
    """
    Pinv = np.linalg.inv(P)
    candidates = []
    for h, b in zip(X.H, X.b):
        denom = float(h @ Pinv @ h)
        if denom > 0:
            candidates.append(b * b / denom)
    for h, b in zip(U.H, U.b):
        h_eff = -K.T @ h
        denom = float(h_eff @ Pinv @ h_eff)
        if denom > 0:
            candidates.append(b * b / denom)
    if not candidates:
        raise ValueError("No active boundary inequalities for terminal set.")
    return float(min(candidates))


def lyapunov_initial_feasible_set(A, B, P, eps, X: Polytope, U: Polytope, N):
    """Initial feasible region (30): X intersect {x | A^N x in -S U_bar (+) X_f}.

    Returned as a sample-based check function rather than an explicit polytope.
    Used for plotting feasible region intersections.
    """
    n = A.shape[0]
    m = B.shape[1]
    # S = [A^{N-1} B, A^{N-2} B, ..., B]
    blocks = [np.linalg.matrix_power(A, N - 1 - i) @ B for i in range(N)]
    S = np.hstack(blocks)

    def feasible(x):
        x = np.asarray(x).flatten()
        if not X.contains(x):
            return False
        # Check existence of U_bar in U^N and xN in X_f such that
        # A^N x + S u_bar = xN AND xN^T P xN <= eps.
        # Solve a small LP/QP.
        import cvxpy as cp
        u_bar = cp.Variable(N * m)
        xN = cp.Variable(n)
        cons = [np.linalg.matrix_power(A, N) @ x + S @ u_bar == xN]
        for i in range(N):
            cons.append(U.H @ u_bar[i * m:(i + 1) * m] <= U.b)
        cons.append(cp.quad_form(xN, cp.psd_wrap(P)) <= eps)
        prob = cp.Problem(cp.Minimize(0), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                return False
        return prob.status in ("optimal", "optimal_inaccurate")

    return feasible
