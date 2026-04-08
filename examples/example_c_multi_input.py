"""Section VI-C: constrained finite-time MPC for a multi-input linear system.

Reproduces Fig. 3 of the paper.

Note: the value K_1 = [2.2150, 15.0471, 14.6128] printed in the paper does
not actually stabilise F_11 - g_1 K_1 (it appears to have been copy-pasted
from Example A whose A matrix is different). Here K_1 is recomputed via
discrete LQR on (F_11, g_1), which is consistent with Algorithm 2 step 3
("find K_i such that all eigenvalues of F_ii - g_i K_i are inside the unit
circle"). K_2 from the paper is correct and is used as-is.

Because the paper's F, G are given directly while M is not, and the eigen-
values of A do not match those of F (suggesting a typographical mismatch),
the simulation is carried out in the decoupled coordinates z = M x.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import Polytope, MultiInputFiniteTimeMPC, dlqr_gain


def main():
    # Decoupled (Wonham) form supplied directly by the paper
    F11 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.25, -3.5, 3.25],
    ])
    F12 = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    F22 = np.array([
        [0.0, 1.0],
        [0.99, -0.2],
    ])
    g1 = np.array([[0.0], [0.0], [1.0]])
    g2 = np.array([[0.0], [1.0]])
    F = np.block([
        [F11, F12],
        [np.zeros((2, 3)), F22],
    ])
    G = np.block([
        [g1, np.zeros((3, 1))],
        [np.zeros((2, 1)), g2],
    ])

    # Treat F, G as the "open-loop" system in z coordinates: A := F, B := G,
    # and identity transformation M = I.
    A = F
    B = G
    M = np.eye(5)

    # Coupled control constraints (51): -6<=u1<=6, -2<=u2<=2, -4<=0.5u1+u2<=4
    H_u = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.5, 1.0],
        [-0.5, -1.0],
    ])
    b_u = np.array([6.0, 6.0, 2.0, 2.0, 4.0, 4.0])
    U = Polytope(H_u, b_u)

    n_list = [3, 2]
    Q_list = [np.eye(3), np.eye(2)]
    R_list = [np.array([[0.1]]), np.array([[0.1]])]

    # K_1 via LQR on (F_11, g_1); K_2 taken from the paper
    K1 = dlqr_gain(F11, g1, np.eye(3), 0.1 * np.eye(1))
    K2 = np.array([[0.9750, -0.0120]])
    K_list = [K1, K2]

    # Conservative terminal box per subsystem: |z_j| <= 0.1
    Zf_list = [
        Polytope.from_box(-0.1 * np.ones(3), 0.1 * np.ones(3)),
        Polytope.from_box(-0.1 * np.ones(2), 0.1 * np.ones(2)),
    ]

    mpc = MultiInputFiniteTimeMPC(
        A=A, B=B, F=F, G=G, n_list=n_list, X=None, U=U,
        Q_list=Q_list, R_list=R_list, K_list=K_list, Zf_list=Zf_list, M=M,
    )

    print("=" * 60)
    print("Example C: multi-input constrained finite-time MPC")
    print("=" * 60)
    print(f"Subsystem dims n_i = {n_list}, overall horizon Np = {mpc.Np}")
    print("K_1 (recomputed via LQR) =", K1.flatten())
    print("K_2 (from paper)         =", K2.flatten())
    for i, P_i in enumerate(mpc.P_list, start=1):
        print(f"P_{i} =")
        print(np.array2string(P_i, precision=4, suppress_small=True))

    # Initial state mirroring Fig. 3 (states roughly in [-3, 2]).
    # Picked inside the feasibility region of the small terminal box.
    x0 = np.array([-2.9089, 1.1297, 2.9381, -2.4504, -2.0086])
    T = 10
    xs, us = mpc.simulate(x0, T)

    print("\nClosed-loop |x|:")
    for k in range(xs.shape[0]):
        print(f"  k={k:2d}  |x|={np.linalg.norm(xs[k]):10.6f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    t = np.arange(T + 1)
    for j in range(5):
        ax1.plot(t, xs[:, j], label=f"$x_{j+1}$")
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_ylabel("x")
    ax1.legend(loc="upper right", ncol=5, fontsize=8)
    ax1.grid(True, alpha=0.3)

    tu = np.arange(T)
    ax2.step(tu, us[:, 0], where="post", label="$u_1$")
    ax2.step(tu, us[:, 1], where="post", label="$u_2$")
    ax2.step(tu, 0.5 * us[:, 0] + us[:, 1], where="post", label="$0.5 u_1 + u_2$")
    ax2.axhline(6.0, color="r", lw=0.5, ls=":")
    ax2.axhline(-6.0, color="r", lw=0.5, ls=":")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_xlabel("Time (step)")
    ax2.set_ylabel("u")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Example C: Multi-input constrained finite-time MPC")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "example_c.png")
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
