import numpy as np
import cvxpy as cp

from .utils import Polytope, lyapunov_P, terminal_radius


class SingleInputFiniteTimeMPC:
    """Algorithm 1: Constrained finite-time MPC for single-input linear systems.

        x(k+1) = A x(k) + B u(k),  x in X,  u in U

    Key settings:
      * Control horizon N = n (system dimension).
      * Stage cost is zero; only terminal cost x(N|k)^T P x(N|k) is penalized.
      * P is solved from A_K^T P A_K - P = -(Q + K^T R K), A_K = A - B K.
      * Terminal constraint X_f = { x : x^T P x <= eps } from formula (14).
    """

    def __init__(self, A, B, X: Polytope, U: Polytope, Q, R, K,
                 solver=cp.CLARABEL):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if B.shape[1] != 1:
            raise ValueError("SingleInputFiniteTimeMPC requires single input.")
        self.A = A
        self.B = B
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.N = self.n
        self.X = X
        self.U = U
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float).reshape(self.m, self.m)
        self.K = np.asarray(K, dtype=float).reshape(self.m, self.n)
        self.A_K = A - B @ self.K
        eig = np.linalg.eigvals(self.A_K)
        if np.max(np.abs(eig)) >= 1.0 - 1e-12:
            raise ValueError(
                f"A - B K is not Schur stable, max |eig| = {np.max(np.abs(eig))}."
            )
        self.P = lyapunov_P(A, B, self.K, self.Q, self.R)
        self.eps = terminal_radius(self.P, self.K, X, U)
        self.solver = solver
        self._build_problem()

    def _build_problem(self):
        n, m, N = self.n, self.m, self.N
        self._x0 = cp.Parameter(n)
        self._x = cp.Variable((N + 1, n))
        self._u = cp.Variable((N, m))
        cons = [self._x[0] == self._x0]
        for i in range(N):
            cons.append(self._x[i + 1] == self.A @ self._x[i] + self.B @ self._u[i])
            cons.append(self.X.H @ self._x[i + 1] <= self.X.b)
            cons.append(self.U.H @ self._u[i] <= self.U.b)
        cons.append(cp.quad_form(self._x[N], cp.psd_wrap(self.P)) <= self.eps)
        obj = cp.Minimize(cp.quad_form(self._x[N], cp.psd_wrap(self.P)))
        self._prob = cp.Problem(obj, cons)

    def solve(self, x0):
        self._x0.value = np.asarray(x0, dtype=float).flatten()
        self._prob.solve(solver=self.solver)
        if self._prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None
        return self._u.value.copy(), self._x.value.copy()

    def step(self, x):
        U_seq, _ = self.solve(x)
        if U_seq is None:
            return None
        return U_seq[0]

    def simulate(self, x0, T):
        n, m = self.n, self.m
        xs = np.zeros((T + 1, n))
        us = np.zeros((T, m))
        xs[0] = np.asarray(x0).flatten()
        for k in range(T):
            u = self.step(xs[k])
            if u is None:
                raise RuntimeError(f"Optimization infeasible at step k={k}.")
            us[k] = u
            xs[k + 1] = self.A @ xs[k] + self.B @ u
        return xs, us
