"""Infinite-horizon finite-time MPC framework
(Zhu, Yuan, Zheng, Zuo, "Constrained finite-time stabilization by model
predictive control: an infinite control horizon framework", 2026).

This module extends the finite-time MPC of Zhu et al. (2024) [23] by:

  1. Replacing the pure terminal cost ``x(N|k)^T P x(N|k)`` with the
     infinite-horizon stage-cost sum ``sum_{i=n}^{infty} (||x||_Q^2 + ||u||_R^2)``,
     equivalently realised by the finite-horizon cost
     ``sum_{i=n}^{N-1} (||x||_Q^2 + ||u||_R^2) + ||x(N|k)||_P^2``.
  2. Allowing the control horizon ``N`` to be chosen larger than the system
     dimension ``n``. The terminal weight ``P`` is still the solution of the
     same Lyapunov equation
     ``A_K^T P A_K - P = -(Q + K^T R K)``  with ``A_K = A - B K``,
     and the terminal set is the same Lyapunov sub-level set
     ``X_f = { x : x^T P x <= eps }``.

The key effect of (1)-(2) is that the *first* ``n`` predicted steps incur no
running cost, so the optimiser can choose them aggressively to drive the
state into the terminal invariant set; once inside, the local linear law
``-K x`` (which the cost from ``i = n`` onwards favours) keeps the state at
the origin.  The longer ``N`` significantly enlarges the initial feasibility
region compared with the ``N = n`` scheme.
"""
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_lyapunov

from .utils import Polytope, lyapunov_P, terminal_radius
from .multi_input import wonham_transform


class InfiniteHorizonSingleInputFTMPC:
    """Theorem 2: single-input constrained finite-time MPC with N >= n and
    infinite-horizon-equivalent cost."""

    def __init__(self, A, B, X: Polytope, U: Polytope, Q, R, K, N,
                 solver=cp.CLARABEL):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if B.shape[1] != 1:
            raise ValueError("InfiniteHorizonSingleInputFTMPC requires single input.")
        self.A = A
        self.B = B
        self.n = A.shape[0]
        self.m = B.shape[1]
        if int(N) < self.n:
            raise ValueError(f"Control horizon N={N} must be >= n={self.n}.")
        self.N = int(N)
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
        # Terminal Lyapunov sub-level set
        cons.append(cp.quad_form(self._x[N], cp.psd_wrap(self.P)) <= self.eps)
        # Cost: stage cost from i = n to N-1, plus terminal cost ||x(N)||_P^2
        cost = 0
        for i in range(n, N):
            cost = cost + cp.quad_form(self._x[i], cp.psd_wrap(self.Q))
            cost = cost + cp.quad_form(self._u[i], cp.psd_wrap(self.R))
        cost = cost + cp.quad_form(self._x[N], cp.psd_wrap(self.P))
        self._prob = cp.Problem(cp.Minimize(cost), cons)

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

    def simulate(self, x0, T, disturbance=None):
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
            if disturbance is not None:
                xs[k + 1] = xs[k + 1] + np.asarray(disturbance(k)).flatten()
        return xs, us


class InfiniteHorizonMultiInputFTMPC:
    """Theorem 3: multi-input extension via Wonham canonical decoupling.

    The system

        x(k+1) = A x(k) + sum_j b_j u_j(k)

    is transformed by ``z = M x`` into the decoupled form
    ``z(k+1) = F z(k) + G u(k)``, where ``F`` is block upper triangular with
    diagonal blocks ``F_{jj}`` of dimensions ``n_j``, and ``g_j`` is the
    leading column of the j-th column block of ``G``. The closed-form cost is

        J_m = sum_{j=1}^q [ sum_{i=n_j}^{N-1} (||z_j||_{Q_j}^2 + ||u_j||_{R_j}^2)
                            + ||z_j(N|k)||_{P_j}^2 ]

    with the same control horizon ``N >= max_j n_j`` shared by all sub-
    systems. Inputs ``u_l`` for ``l > q`` are pinned to zero.
    """

    def __init__(self, A, B, F, G, n_list, X, U, Q_list, R_list, K_list,
                 Zf_list, N, M=None, solver=cp.CLARABEL):
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
        self.q = len(self.n_list)
        if self.q > self.m:
            raise ValueError("Number of subsystems q must be <= m.")
        if int(N) < max(self.n_list):
            raise ValueError(
                f"Control horizon N={N} must be >= max(n_list)={max(self.n_list)}."
            )
        self.N = int(N)
        self.X = X
        self.U = U
        self.Q_list = [np.asarray(Q, dtype=float) for Q in Q_list]
        self.R_list = [np.asarray(R, dtype=float).reshape(1, 1) for R in R_list]
        self.K_list = [np.asarray(K, dtype=float).reshape(1, -1) for K in K_list]
        self.Zf_list = list(Zf_list)
        self.M = wonham_transform(A, B, F, G) if M is None else np.asarray(M, dtype=float)
        self.Minv = np.linalg.inv(self.M)

        self.slices = []
        offset = 0
        for nj in self.n_list:
            self.slices.append(slice(offset, offset + nj))
            offset += nj

        self.F_blocks = []
        self.g_blocks = []
        self.P_list = []
        for j in range(self.q):
            sl = self.slices[j]
            F_jj = F[sl, sl]
            g_j = G[sl, j:j + 1]
            self.F_blocks.append(F_jj)
            self.g_blocks.append(g_j)
            K_j = self.K_list[j]
            A_K_j = F_jj - g_j @ K_j
            eig = np.linalg.eigvals(A_K_j)
            if np.max(np.abs(eig)) >= 1.0 - 1e-12:
                raise ValueError(
                    f"Subsystem {j+1}: F_jj - g_j K_j not Schur, "
                    f"max |eig| = {np.max(np.abs(eig))}."
                )
            Qbar = self.Q_list[j] + K_j.T @ self.R_list[j] @ K_j
            P_j = solve_discrete_lyapunov(A_K_j.T, Qbar)
            self.P_list.append(P_j)

        self.solver = solver
        self._build_problem()

    def _build_problem(self):
        n, m, q, N = self.n, self.m, self.q, self.N
        self._x0 = cp.Parameter(n)
        self._z = cp.Variable((N + 1, n))
        self._u = cp.Variable((N, m))
        cons = [self._z[0] == self.M @ self._x0]
        for i in range(N):
            cons.append(self._z[i + 1] == self.F @ self._z[i] + self.G @ self._u[i])
            if self.X is not None:
                cons.append(self.X.H @ (self.Minv @ self._z[i + 1]) <= self.X.b)
            cons.append(self.U.H @ self._u[i] <= self.U.b)
            for l in range(q, m):
                cons.append(self._u[i, l] == 0)
        # Subsystem-wise terminal Lyapunov sub-level sets at the *common* horizon N
        for j in range(q):
            sl = self.slices[j]
            zj_term = self._z[N, sl]
            Zf = self.Zf_list[j]
            cons.append(Zf.H @ zj_term <= Zf.b)
        # Cost: per-subsystem stage cost from i = n_j (subsystem dimension) to N-1
        cost = 0
        for j in range(q):
            sl = self.slices[j]
            n_j = self.n_list[j]
            for i in range(n_j, N):
                zj_i = self._z[i, sl]
                uj_i = self._u[i, j:j + 1]
                cost = cost + cp.quad_form(zj_i, cp.psd_wrap(self.Q_list[j]))
                cost = cost + cp.quad_form(uj_i, cp.psd_wrap(self.R_list[j]))
            zj_term = self._z[N, sl]
            cost = cost + cp.quad_form(zj_term, cp.psd_wrap(self.P_list[j]))
        self._prob = cp.Problem(cp.Minimize(cost), cons)

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

    def simulate(self, x0, T, disturbance=None):
        xs = np.zeros((T + 1, self.n))
        us = np.zeros((T, self.m))
        xs[0] = np.asarray(x0).flatten()
        for k in range(T):
            u = self.step(xs[k])
            if u is None:
                raise RuntimeError(f"Optimization infeasible at step k={k}.")
            us[k] = u
            xs[k + 1] = self.A @ xs[k] + self.B @ u
            if disturbance is not None:
                xs[k + 1] = xs[k + 1] + np.asarray(disturbance(k)).flatten()
        return xs, us


class InfiniteHorizonNonlinearFTMPC:
    """Theorem 4: nonlinear extension. The terminal weight ``P`` is computed
    from the Jacobian linearisation of ``f`` at the origin (matrices ``A``,
    ``b`` are the partial derivatives at ``x = 0, u = 0``), then ``P`` solves
    the same discrete Lyapunov equation as in the linear case. The
    optimisation is the nonlinear analogue of (24) with the same infinite-
    horizon-equivalent cost."""

    def __init__(self, f_dyn, n, m, A_lin, B_lin, Q, R, K, U_box,
                 terminal_box, N, state_ineqs=None, domain_box=None):
        import casadi as ca
        self.ca = ca
        self.f = f_dyn
        self.n = int(n)
        self.m = int(m)
        if int(N) < self.n:
            raise ValueError(f"N={N} must be >= n={self.n}.")
        self.N = int(N)
        self.A_lin = np.asarray(A_lin, dtype=float)
        self.B_lin = np.asarray(B_lin, dtype=float)
        if self.B_lin.ndim == 1:
            self.B_lin = self.B_lin.reshape(-1, 1)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float).reshape(self.m, self.m)
        self.K = np.asarray(K, dtype=float).reshape(self.m, self.n)
        self.A_K = self.A_lin - self.B_lin @ self.K
        eig = np.linalg.eigvals(self.A_K)
        if np.max(np.abs(eig)) >= 1.0 - 1e-12:
            raise ValueError(
                f"A_lin - B_lin K is not Schur, max |eig| = {np.max(np.abs(eig))}."
            )
        self.P = lyapunov_P(self.A_lin, self.B_lin, self.K, self.Q, self.R)
        self.U_lb = np.asarray(U_box[0], dtype=float).flatten()
        self.U_ub = np.asarray(U_box[1], dtype=float).flatten()
        self.term_lb = np.asarray(terminal_box[0], dtype=float).flatten()
        self.term_ub = np.asarray(terminal_box[1], dtype=float).flatten()
        self.state_ineqs = state_ineqs
        if domain_box is not None:
            self.dom_lb = np.asarray(domain_box[0], dtype=float).flatten()
            self.dom_ub = np.asarray(domain_box[1], dtype=float).flatten()
        else:
            self.dom_lb = None
            self.dom_ub = None
        self._build_solver()

    def _build_solver(self):
        ca = self.ca
        N, n, m = self.N, self.n, self.m
        opti = ca.Opti()
        x = opti.variable(n, N + 1)
        u = opti.variable(m, N)
        x0 = opti.parameter(n)
        opti.subject_to(x[:, 0] == x0)
        for i in range(N):
            opti.subject_to(x[:, i + 1] == self.f(x[:, i], u[:, i]))
            opti.subject_to(u[:, i] >= self.U_lb)
            opti.subject_to(u[:, i] <= self.U_ub)
            xi1 = x[:, i + 1]
            if self.dom_lb is not None:
                opti.subject_to(xi1 >= self.dom_lb)
                opti.subject_to(xi1 <= self.dom_ub)
            if self.state_ineqs is not None:
                for g in self.state_ineqs(xi1):
                    opti.subject_to(g <= 0)
        # Terminal box
        xN = x[:, N]
        opti.subject_to(xN >= self.term_lb)
        opti.subject_to(xN <= self.term_ub)
        # Infinite-horizon-equivalent cost: stage from i=n to N-1, plus terminal
        cost = 0
        for i in range(n, N):
            xi = x[:, i]
            ui = u[:, i]
            cost = cost + ca.mtimes([xi.T, self.Q, xi]) + ca.mtimes([ui.T, self.R, ui])
        cost = cost + ca.mtimes([xN.T, self.P, xN])
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes"},
            {"max_iter": 500, "tol": 1e-9},
        )
        self._opti = opti
        self._x = x
        self._u = u
        self._x0_par = x0
        self._u_init = None
        self._x_init = None

    def step(self, x_meas):
        ca = self.ca
        x_meas = np.asarray(x_meas, dtype=float).flatten()
        self._opti.set_value(self._x0_par, x_meas)
        if self._u_init is not None:
            self._opti.set_initial(self._u, self._u_init)
            self._opti.set_initial(self._x, self._x_init)
        try:
            sol = self._opti.solve()
        except RuntimeError:
            return None
        u_val = np.atleast_2d(sol.value(self._u))
        x_val = np.atleast_2d(sol.value(self._x))
        if u_val.shape[0] != self.m:
            u_val = u_val.reshape(self.m, self.N)
        if x_val.shape[0] != self.n:
            x_val = x_val.reshape(self.n, self.N + 1)
        u_shift = np.hstack([u_val[:, 1:], u_val[:, -1:]])
        x_shift = np.hstack([x_val[:, 1:], x_val[:, -1:]])
        self._u_init = u_shift
        self._x_init = x_shift
        return u_val[:, 0]

    def simulate(self, x0, T, disturbance=None):
        ca = self.ca
        n, m = self.n, self.m
        xs = np.zeros((T + 1, n))
        us = np.zeros((T, m))
        xs[0] = np.asarray(x0).flatten()
        x_sym = ca.SX.sym("x", n)
        u_sym = ca.SX.sym("u", m)
        f_sx = ca.Function("f_eval", [x_sym, u_sym], [self.f(x_sym, u_sym)])
        for k in range(T):
            u = self.step(xs[k])
            if u is None:
                raise RuntimeError(f"Optimization infeasible at step k={k}.")
            us[k] = u
            x_next = np.array(f_sx(xs[k], u)).flatten()
            if disturbance is not None:
                x_next = x_next + np.asarray(disturbance(k)).flatten()
            xs[k + 1] = x_next
        return xs, us
