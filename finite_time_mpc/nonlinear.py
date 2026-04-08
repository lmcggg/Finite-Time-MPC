import numpy as np
import casadi as ca


class NonlinearFiniteTimeMPC:
    """Algorithm 4: Constrained finite-time MPC for discrete-time nonlinear
    systems.

        x(k+1) = f(x(k), u(k)),    x in X cap D,    u in U

    The control horizon is set to the system dimension N = n. Only the
    terminal cost x(N|k)^T P x(N|k) is penalised. The terminal constraint
    is a (small) control invariant set near the origin, supplied as a box.

    The diffeomorphism z = T(x) used in the proof of Theorem 5 is *not*
    required for implementation, in line with Remark 9: it is only invoked
    theoretically.
    """

    def __init__(self, f_dyn, n, m, P, U_box, terminal_box,
                 state_ineqs=None, domain_box=None, N=None):
        """
        Parameters
        ----------
        f_dyn : callable
            ``f_dyn(x_sym, u_sym)`` -> CasADi expression for x(k+1).
        n, m : int
            State and input dimensions.
        P : (n, n) array
            Positive definite terminal weighting matrix.
        U_box : (lb, ub)
            Element-wise input bounds (1D arrays of length m).
        terminal_box : (lb, ub)
            Element-wise terminal state bounds (control invariant proxy).
        state_ineqs : callable, optional
            ``state_ineqs(x_sym)`` -> list of CasADi expressions ``g(x) <= 0``.
        domain_box : (lb, ub), optional
            Element-wise state bounds describing X cap D.
        N : int, optional
            Control horizon (defaults to n).
        """
        self.f = f_dyn
        self.n = int(n)
        self.m = int(m)
        self.P = np.asarray(P, dtype=float)
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
        self.N = self.n if N is None else int(N)
        self._build_solver()

    def _build_solver(self):
        N, n, m = self.N, self.n, self.m
        opti = ca.Opti()
        x = opti.variable(n, N + 1)
        u = opti.variable(m, N)
        x0 = opti.parameter(n)
        opti.subject_to(x[:, 0] == x0)
        for i in range(N):
            opti.subject_to(x[:, i + 1] == self.f(x[:, i], u[:, i]))
            # Input box constraints
            opti.subject_to(u[:, i] >= self.U_lb)
            opti.subject_to(u[:, i] <= self.U_ub)
            # State constraints x(i+1|k) in X cap D
            xi1 = x[:, i + 1]
            if self.dom_lb is not None:
                opti.subject_to(xi1 >= self.dom_lb)
                opti.subject_to(xi1 <= self.dom_ub)
            if self.state_ineqs is not None:
                for g in self.state_ineqs(xi1):
                    opti.subject_to(g <= 0)
        # Terminal constraint x(N|k) in T^{-1}(Z_f)  ~ small box near origin
        xN = x[:, N]
        opti.subject_to(xN >= self.term_lb)
        opti.subject_to(xN <= self.term_ub)
        # Terminal cost
        opti.minimize(ca.mtimes([xN.T, self.P, xN]))
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
        # Warm start: shift one step
        u_shift = np.hstack([u_val[:, 1:], u_val[:, -1:]])
        x_shift = np.hstack([x_val[:, 1:], x_val[:, -1:]])
        self._u_init = u_shift
        self._x_init = x_shift
        return u_val[:, 0]

    def simulate(self, x0, T, f_real=None, disturbance=None):
        """Closed-loop simulation. ``f_real`` defaults to the model ``self.f``.

        ``disturbance`` is an optional callable returning ``w`` added to the
        next state at each step (used for the disturbance test in Example D).
        """
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
