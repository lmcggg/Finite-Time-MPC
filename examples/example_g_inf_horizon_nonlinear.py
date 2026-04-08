"""Section IV-C of the infinite-horizon paper:
Nonlinear constrained finite-time MPC with N >> n.

Plant:
    x1(k+1) = -1.1 x1(k) + 2 sin(x2(k))
    x2(k+1) = 0.2 x1(k) x2(k) + 0.79 u(k)
Constraints: |x2| < pi/2, |u| <= 2

Reproduces Fig. 4 (settles in 5 steps).
"""
import os
import sys

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import InfiniteHorizonNonlinearFTMPC, dlqr_gain


def f_dyn(x, u):
    x1 = x[0]
    x2 = x[1]
    return ca.vertcat(
        -1.1 * x1 + 2 * ca.sin(x2),
        0.2 * x1 * x2 + 0.79 * u[0],
    )


def main():
    # Jacobian linearisation at x = 0, u = 0
    A_lin = np.array([
        [-1.1, 2.0],
        [0.0, 0.0],
    ])
    B_lin = np.array([[0.0], [0.79]])
    Q = np.eye(2)
    R = np.array([[0.1]])
    K = dlqr_gain(A_lin, B_lin, Q, R)

    # Constraints: |x_2| < pi/2 (use a slightly tighter box for the solver),
    # |u| <= 2
    eps_x2 = 1e-3
    dom_lb = np.array([-1e3, -np.pi / 2 + eps_x2])
    dom_ub = np.array([1e3, np.pi / 2 - eps_x2])
    U_box = (np.array([-2.0]), np.array([2.0]))
    # Conservative terminal box near origin (control invariant proxy)
    term_box = (np.array([-0.05, -0.05]), np.array([0.05, 0.05]))

    N = 8
    mpc = InfiniteHorizonNonlinearFTMPC(
        f_dyn=f_dyn, n=2, m=1,
        A_lin=A_lin, B_lin=B_lin, Q=Q, R=R, K=K,
        U_box=U_box, terminal_box=term_box,
        N=N, domain_box=(dom_lb, dom_ub),
    )

    print("=" * 60)
    print("Section IV-C: infinite-horizon nonlinear finite-time MPC")
    print("=" * 60)
    print(f"N = {mpc.N}, n = {mpc.n}")
    print(f"K (LQR on linearisation) = {K.flatten()}")
    print("Eigenvalues of A_lin - B_lin K:", np.linalg.eigvals(mpc.A_K))
    print("P (Lyapunov) =")
    print(np.array2string(mpc.P, precision=4, suppress_small=True))

    x0 = np.array([1.5, -1.0])
    T = 12
    xs, us = mpc.simulate(x0, T)
    print("\nClosed-loop trajectory:")
    for k in range(xs.shape[0]):
        print(f"  k={k:2d}  x={xs[k]}  |x|={np.linalg.norm(xs[k]):.6f}")
    settle = next((k for k, nv in enumerate(np.linalg.norm(xs, axis=1)) if nv < 1e-4), None)
    print(f"Settling step (|x|<1e-4): {settle}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    t = np.arange(T + 1)
    ax1.plot(t, xs[:, 0], "-", label=r"$x_1$")
    ax1.plot(t, xs[:, 1], "--", label=r"$x_2$")
    ax1.axhline(np.pi / 2, color="r", lw=0.5, ls=":")
    ax1.axhline(-np.pi / 2, color="r", lw=0.5, ls=":")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("x")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.step(np.arange(T), us[:, 0], where="post", label="u")
    ax2.axhline(2.0, color="r", lw=0.5, ls=":")
    ax2.axhline(-2.0, color="r", lw=0.5, ls=":")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("u")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Section IV-C: infinite-horizon nonlinear MPC, N={N}")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "example_g.png")
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
