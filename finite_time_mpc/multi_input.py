import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_lyapunov

from .utils import Polytope


def wonham_transform(A, B, F, G):
    """Compute square M such that F = M A M^{-1} and G = M B.

    Solves the linear system  M A - F M = 0,  M B - G = 0  for vec(M).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    n = A.shape[0]
    m = B.shape[1]
    I_n = np.eye(n)
    # vec(MA) = (A^T kron I_n) vec(M);  vec(FM) = (I_n kron F) vec(M)
    L1 = np.kron(A.T, I_n) - np.kron(I_n, F)
    L2 = np.kron(B.T, I_n)
    L = np.vstack([L1, L2])
    rhs = np.concatenate([np.zeros(n * n), G.flatten(order="F")])
    vec_M, *_ = np.linalg.lstsq(L, rhs, rcond=None)
    M = vec_M.reshape((n, n), order="F")
    return M


class MultiInputFiniteTimeMPC:
    """Algorithm 2: Constrained finite-time MPC for multi-input linear systems
    via Wonham canonical decoupling.

        x(k+1) = A x(k) + B u(k),     z = M x
        z(k+1) = F z(k) + G u(k)

    F is block upper triangular with diagonal blocks (F_ii) of dimension n_i,
    and (F_ii, g_i) is a controllable single-input pair, where g_i is the
    leading column of the i-th column block of G. The system is decoupled into
    q sub-systems with control horizons N_i = n_i and overall horizon
    N_p = max_i N_i.

    The optimization (24) is built directly in z-coordinates:

        min   sum_{i=1}^q  z_i(N_i|k)^T P_i z_i(N_i|k)
        s.t.  z(i+1|k) = F z(i|k) + G u(i|k),     i = 0, ..., N_p-1
              x(i+1|k) in X,                       i = 0, ..., N_p-1
              u(i|k)   in U,                        i = 0, ..., N_p-1
              z(0|k)  = M x(k)
              z_i(N_i|k) in Z_{f,i},                i = 1, ..., q

    where Z_{f,i} is given as a polytope on the i-th subsystem state.
    """

    def __init__(self, A, B, F, G, n_list, X, U, Q_list, R_list, K_list,
                 Zf_list, M=None, q=None, solver=cp.CLARABEL):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        self.A = A
        self.B = B
        self.F = F
        self.G = G
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.n_list = list(n_list)
        if sum(self.n_list) != self.n:
            raise ValueError("Sum of subsystem dimensions must equal n.")
        self.q = len(self.n_list) if q is None else q
        if self.q > self.m:
            raise ValueError("Number of subsystems q must be <= m.")
        self.X = X  # state polytope on x (original coordinates), or None
        self.U = U  # control polytope on u
        self.Q_list = [np.asarray(Q, dtype=float) for Q in Q_list]
        self.R_list = [np.asarray(R, dtype=float).reshape(1, 1) for R in R_list]
        self.K_list = [np.asarray(K, dtype=float).reshape(1, -1) for K in K_list]
        self.Zf_list = list(Zf_list)
        self.M = wonham_transform(A, B, F, G) if M is None else np.asarray(M, dtype=float)
        self.Minv = np.linalg.inv(self.M)

        # Subsystem index slices in z
        self.slices = []
        offset = 0
        for ni in self.n_list:
            self.slices.append(slice(offset, offset + ni))
            offset += ni

        # Per-subsystem F_ii, g_i and Lyapunov P_i from (23)
        self.F_blocks = []
        self.g_blocks = []
        self.P_list = []
        for i in range(self.q):
            sl = self.slices[i]
            F_ii = F[sl, sl]
            g_i = G[sl, i:i + 1]
            self.F_blocks.append(F_ii)
            self.g_blocks.append(g_i)
            K_i = self.K_list[i]
            A_K_i = F_ii - g_i @ K_i
            eig = np.linalg.eigvals(A_K_i)
            if np.max(np.abs(eig)) >= 1.0 - 1e-12:
                raise ValueError(
                    f"Subsystem {i+1}: F_ii - g_i K_i not Schur, max |eig| = {np.max(np.abs(eig))}."
                )
            Qbar = self.Q_list[i] + K_i.T @ self.R_list[i] @ K_i
            P_i = solve_discrete_lyapunov(A_K_i.T, Qbar)
            self.P_list.append(P_i)

        self.N_list = list(self.n_list)
        self.Np = max(self.N_list)
        self.solver = solver
        self._build_problem()

    def _build_problem(self):
        n, m, q, Np = self.n, self.m, self.q, self.Np
        self._x0 = cp.Parameter(n)
        self._z = cp.Variable((Np + 1, n))
        self._u = cp.Variable((Np, m))
        cons = [self._z[0] == self.M @ self._x0]
        for i in range(Np):
            cons.append(self._z[i + 1] == self.F @ self._z[i] + self.G @ self._u[i])
            if self.X is not None:
                # x = M^{-1} z
                cons.append(self.X.H @ (self.Minv @ self._z[i + 1]) <= self.X.b)
            cons.append(self.U.H @ self._u[i] <= self.U.b)
            # Inputs that are not subsystem leaders are pinned to zero
            for j in range(q, m):
                cons.append(self._u[i, j] == 0)
        # Subsystem terminal constraints at their own horizon N_i
        for i in range(q):
            sl = self.slices[i]
            Ni = self.N_list[i]
            zi_term = self._z[Ni, sl]
            Zf = self.Zf_list[i]
            cons.append(Zf.H @ zi_term <= Zf.b)
        # Cost is sum of subsystem terminal quadratic forms
        obj_expr = 0
        for i in range(q):
            sl = self.slices[i]
            Ni = self.N_list[i]
            zi_term = self._z[Ni, sl]
            obj_expr = obj_expr + cp.quad_form(zi_term, cp.psd_wrap(self.P_list[i]))
        self._prob = cp.Problem(cp.Minimize(obj_expr), cons)

    def solve(self, x0):
        self._x0.value = np.asarray(x0, dtype=float).flatten()
        self._prob.solve(solver=self.solver)
        if self._prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None
        return self._u.value.copy(), self._z.value.copy()

    def step(self, x):
        U_seq, _ = self.solve(x)
        if U_seq is None:
            return None
        return U_seq[0]

    def simulate(self, x0, T):
        xs = np.zeros((T + 1, self.n))
        us = np.zeros((T, self.m))
        xs[0] = np.asarray(x0).flatten()
        for k in range(T):
            u = self.step(xs[k])
            if u is None:
                raise RuntimeError(f"Optimization infeasible at step k={k}.")
            us[k] = u
            xs[k + 1] = self.A @ xs[k] + self.B @ u
        return xs, us
