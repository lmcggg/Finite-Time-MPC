"""Section VI-A: Constrained finite-time MPC for a single-input linear system.

Reproduces Fig. 1 of the paper.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import Polytope, SingleInputFiniteTimeMPC


def main():
    A = np.array([
        [1.1, 2.0, 0.0],
        [0.0, 0.95, 1.0],
        [0.0, 0.0, 1.2],
    ])
    B = np.array([[0.0], [0.079], [0.1]])

    # State constraint: -2 <= x_2 + x_3 <= 2  -> two half-planes
    H_x = np.array([[0.0, 1.0, 1.0], [0.0, -1.0, -1.0]])
    b_x = np.array([2.0, 2.0])
    X = Polytope(H_x, b_x)

    # Control constraint: -6 <= u <= 6
    U = Polytope.from_box([-6.0], [6.0])

    Q = np.eye(3)
    R = np.array([[0.1]])
    K = np.array([[2.2150, 15.0471, 14.6128]])

    mpc = SingleInputFiniteTimeMPC(A, B, X, U, Q, R, K)

    print("=" * 60)
    print("Example A: single-input constrained finite-time MPC")
    print("=" * 60)
    print(f"Control horizon N = {mpc.N}")
    print("Lyapunov-based terminal weighting matrix P:")
    print(np.array2string(mpc.P, precision=4, suppress_small=True))
    print(f"Terminal radius eps = {mpc.eps:.4f}")
    eig_AK = np.linalg.eigvals(mpc.A_K)
    print(f"Eigenvalues of A - B K: {eig_AK}")

    x0 = np.array([1.9, -1.1, 0.8])
    print(f"Initial state x(0) = {x0}")
    T = 10
    xs, us = mpc.simulate(x0, T)

    # Verify finite-time convergence: state reaches origin within ~6 steps
    norms = np.linalg.norm(xs, axis=1)
    print("\nState norm trajectory:")
    for k, nv in enumerate(norms):
        print(f"  k={k:2d}  |x|={nv:10.6f}")
    settle = next((k for k, nv in enumerate(norms) if nv < 1e-6), None)
    print(f"Settling step (|x| < 1e-6): {settle}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    t = np.arange(T + 1)
    ax1.plot(t, xs[:, 0], "-", label=r"$x_1$")
    ax1.plot(t, xs[:, 1], "--", label=r"$x_2$")
    ax1.plot(t, xs[:, 2], "-.", label=r"$x_3$")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("x")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.step(np.arange(T), us[:, 0], where="post", label="u")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("Time (step)")
    ax2.set_ylabel("u")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Example A: Single-input constrained finite-time MPC")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "example_a.png")
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
