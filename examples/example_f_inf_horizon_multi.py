"""Section IV-B of the infinite-horizon paper:
Multi-input constrained finite-time MPC with N >> n.

The plant has m = 2 inputs but the controllability index gives q = 1, i.e.
(A, b_1) is fully controllable, so the system is treated as a single
sub-system of full dimension n = 3 with N = 8, and u_2 is pinned to zero
by Algorithm step (40).

Reproduces Fig. 3 (settles in 11 steps).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import (
    Polytope,
    InfiniteHorizonMultiInputFTMPC,
    dlqr_gain,
)


def main():
    A = np.array([
        [1.1, 2.0, -0.4],
        [0.0, 0.95, -0.8],
        [0.0, 0.1, 1.0],
    ])
    B = np.array([
        [0.0, 0.0],
        [0.079, 0.0],
        [-0.1, 0.1],
    ])
    b1 = B[:, 0:1]

    # |u_1| <= 5, |u_2| <= 5
    U = Polytope.from_box([-5.0, -5.0], [5.0, 5.0])

    # Verify (A, b_1) is fully controllable -> q = 1, n_1 = 3
    Cmat = np.hstack([b1, A @ b1, A @ A @ b1])
    print(f"Controllability matrix [b1, A b1, A^2 b1] rank = {np.linalg.matrix_rank(Cmat)}")

    # Use the multi-input class with q = 1, n_list = [3]: F = A, G = B, M = I
    F = A
    G = B
    M = np.eye(3)

    n_list = [3]
    Q_list = [np.eye(3)]   # paper writes "Q_1 = I_{2x2}", inconsistent with q=1, n_1=3 -> use I_3
    R_list = [0.1 * np.eye(1)]
    K1 = dlqr_gain(A, b1, np.eye(3), 0.1 * np.eye(1))
    K_list = [K1]

    # Conservative terminal box on z_1 = x (since M = I)
    Zf_list = [Polytope.from_box(-0.2 * np.ones(3), 0.2 * np.ones(3))]

    N = 8
    mpc = InfiniteHorizonMultiInputFTMPC(
        A=A, B=B, F=F, G=G, n_list=n_list, X=None, U=U,
        Q_list=Q_list, R_list=R_list, K_list=K_list, Zf_list=Zf_list,
        N=N, M=M,
    )

    print("=" * 60)
    print("Section IV-B: infinite-horizon multi-input finite-time MPC")
    print("=" * 60)
    print(f"q = {mpc.q}, n_list = {mpc.n_list}, N = {mpc.N}")
    print(f"K_1 (LQR) = {K1.flatten()}")
    print("Eigenvalues of A - b_1 K_1:",
          np.linalg.eigvals(A - b1 @ K1))
    print("P_1 =")
    print(np.array2string(mpc.P_list[0], precision=4, suppress_small=True))

    x0 = np.array([3.8203, -3.4125, -1.2889])
    T = 14
    xs, us = mpc.simulate(x0, T)

    print("\nClosed-loop trajectory:")
    for k in range(xs.shape[0]):
        print(f"  k={k:2d}  x={xs[k]}  |x|={np.linalg.norm(xs[k]):.6f}")
    settle = next((k for k, nv in enumerate(np.linalg.norm(xs, axis=1)) if nv < 1e-6), None)
    print(f"Settling step (|x|<1e-6): {settle}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    t = np.arange(T + 1)
    ax1.plot(t, xs[:, 0], "-", label=r"$x_1$")
    ax1.plot(t, xs[:, 1], "--", label=r"$x_2$")
    ax1.plot(t, xs[:, 2], "-.", label=r"$x_3$")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("x")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    tu = np.arange(T)
    ax2.step(tu, us[:, 0], where="post", label="$u_1$")
    ax2.step(tu, us[:, 1], where="post", label="$u_2$")
    ax2.axhline(5.0, color="r", lw=0.5, ls=":")
    ax2.axhline(-5.0, color="r", lw=0.5, ls=":")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("u")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Section IV-B: infinite-horizon multi-input MPC, N={N}")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "example_f.png")
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
