"""Section IV-D of the infinite-horizon paper:
Robustness of the infinite-horizon finite-time MPC to bounded disturbance.

Revisits the single-input plant of Section IV-A under

    x(k+1) = A x(k) + b ( u(k) + w(k) ),    |w(k)| <= 1,

i.e. an additive matched disturbance equal to 20% of the control bound.
The simulation is run multiple times with different random seeds, mirroring
Fig. 5 of the paper (10 trials, ultimate boundedness).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import (
    Polytope,
    InfiniteHorizonSingleInputFTMPC,
)


def main():
    A = np.array([
        [1.1, 2.0],
        [0.0, 0.95],
    ])
    B = np.array([[0.0], [0.079]])
    X = Polytope(np.vstack([np.eye(2), -np.eye(2)]),
                 np.array([1e6, 1e6, 1e6, 1e6]))
    U = Polytope.from_box([-5.0], [5.0])
    Q = np.eye(2)
    R = np.array([[0.1]])
    K = np.array([[4.3, 24.7]])
    N = 8
    mpc = InfiniteHorizonSingleInputFTMPC(A, B, X, U, Q, R, K, N=N)

    print("=" * 60)
    print("Section IV-D: robustness to bounded matched disturbance")
    print("=" * 60)

    x0 = np.array([2.0, -1.0])
    T = 14
    n_trials = 10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    t = np.arange(T + 1)
    final_norms = []
    for trial in range(n_trials):
        rng = np.random.default_rng(100 + trial)

        def disturbance(k):
            w_scalar = rng.uniform(-1.0, 1.0)
            return (B.flatten() * w_scalar)

        xs, us = mpc.simulate(x0, T, disturbance=disturbance)
        ax1.plot(t, xs[:, 0], "-", color="C0", alpha=0.6)
        ax1.plot(t, xs[:, 1], "--", color="C3", alpha=0.6)
        ax2.step(np.arange(T), us[:, 0], where="post", color="C0", alpha=0.6)
        final_norms.append(np.linalg.norm(xs[-1]))

    print(f"Final-state norms over {n_trials} trials:")
    for k, fn in enumerate(final_norms):
        print(f"  trial {k}: |x(T)| = {fn:.4f}")
    print(f"Mean   = {np.mean(final_norms):.4f}")
    print(f"Max    = {np.max(final_norms):.4f}")

    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("x")
    ax1.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0], [0], color="C0", lw=1, label="$x_1$"),
        Line2D([0], [0], color="C3", lw=1, ls="--", label="$x_2$"),
    ], loc="upper right")
    ax2.axhline(5.0, color="r", lw=0.5, ls=":")
    ax2.axhline(-5.0, color="r", lw=0.5, ls=":")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("u")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Section IV-D: bounded disturbance, {n_trials} trials")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "example_h.png")
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
