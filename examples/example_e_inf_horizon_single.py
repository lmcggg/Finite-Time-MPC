"""Section IV-A of the infinite-horizon paper:
Single-input constrained finite-time MPC with N >> n.

Reproduces Fig. 1 (closed-loop trajectory, ~7-step settling) and Fig. 2
(initial feasibility comparison: short-horizon strategy vs prolonged
horizon).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import (
    Polytope,
    InfiniteHorizonSingleInputFTMPC,
    SingleInputFiniteTimeMPC,
)


def build_plant():
    A = np.array([
        [1.1, 2.0],
        [0.0, 0.95],
    ])
    B = np.array([[0.0], [0.079]])
    # No state constraint, only |u| <= 5
    H_x = np.zeros((0, 2))
    b_x = np.zeros(0)
    X = Polytope(np.vstack([np.eye(2), -np.eye(2)]),
                 np.array([1e6, 1e6, 1e6, 1e6]))  # box[-1e6, 1e6]^2
    U = Polytope.from_box([-5.0], [5.0])
    Q = np.eye(2)
    R = np.array([[0.1]])
    K = np.array([[4.3, 24.7]])
    return A, B, X, U, Q, R, K


def main():
    A, B, X, U, Q, R, K = build_plant()
    N_long = 8

    mpc_long = InfiniteHorizonSingleInputFTMPC(A, B, X, U, Q, R, K, N=N_long)
    mpc_short = SingleInputFiniteTimeMPC(A, B, X, U, Q, R, K)  # N = n = 2

    print("=" * 60)
    print("Section IV-A: infinite-horizon single-input finite-time MPC")
    print("=" * 60)
    print(f"Control horizon (extended) N = {mpc_long.N}, n = {mpc_long.n}")
    print("Lyapunov terminal weighting matrix P =")
    print(np.array2string(mpc_long.P, precision=4, suppress_small=True))
    print(f"Terminal radius eps = {mpc_long.eps:.4f}")
    eig_AK = np.linalg.eigvals(mpc_long.A_K)
    print(f"Eigenvalues of A - B K: {eig_AK}")

    # ---- Closed-loop simulation (Fig. 1) ----
    x0 = np.array([2.0, -1.0])
    T = 14
    xs, us = mpc_long.simulate(x0, T)
    print("\nClosed-loop trajectory:")
    for k in range(xs.shape[0]):
        print(f"  k={k:2d}  x={xs[k]}  |x|={np.linalg.norm(xs[k]):.6f}")
    settle = next((k for k, nv in enumerate(np.linalg.norm(xs, axis=1)) if nv < 1e-6), None)
    print(f"Settling step (|x|<1e-6): {settle}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    t = np.arange(T + 1)
    ax1.plot(t, xs[:, 0], "-", label=r"$x_1$")
    ax1.plot(t, xs[:, 1], "--", label=r"$x_2$")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("x")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.step(np.arange(T), us[:, 0], where="post", label="u")
    ax2.axhline(5.0, color="r", lw=0.5, ls=":")
    ax2.axhline(-5.0, color="r", lw=0.5, ls=":")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("u")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Section IV-A: infinite-horizon single-input MPC, N={N_long}")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "example_e_trajectory.png")
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")

    # ---- Initial feasibility comparison (Fig. 2) ----
    print("\nComputing initial feasibility regions ...")
    x1_grid = np.linspace(-12, 12, 81)
    x2_grid = np.linspace(-5, 5, 51)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    feas_long = np.zeros_like(X1, dtype=bool)
    feas_short = np.zeros_like(X1, dtype=bool)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_test = np.array([X1[i, j], X2[i, j]])
            U_l, _ = mpc_long.solve(x_test)
            feas_long[i, j] = U_l is not None
            U_s, _ = mpc_short.solve(x_test)
            feas_short[i, j] = U_s is not None

    fig2, ax = plt.subplots(figsize=(6, 5))
    ax.contour(X1, X2, feas_long.astype(float), levels=[0.5], colors="C0",
               linestyles="-", linewidths=2)
    ax.contour(X1, X2, feas_short.astype(float), levels=[0.5], colors="black",
               linestyles="--", linewidths=1.5)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="C0", lw=2, ls="-",
               label=f"Infinite-horizon strategy (N={N_long})"),
        Line2D([0], [0], color="black", lw=1.5, ls="--",
               label="Previous terminal-cost strategy (N=n=2)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Initial feasibility regions (Section IV-A, Fig. 2)")
    ax.grid(True, alpha=0.3)
    out2 = os.path.join(os.path.dirname(__file__), "..", "figures",
                        "example_e_feasibility.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150)
    print(f"Saved figure to {os.path.abspath(out2)}")


if __name__ == "__main__":
    main()
