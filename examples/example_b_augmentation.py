"""Section VI-B: improving initial feasibility by augmentation.

Reproduces Fig. 2 (initial feasibility regions) of the paper.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import (
    Polytope,
    SingleInputFiniteTimeMPC,
    AugmentedFiniteTimeMPC,
    dlqr_gain,
)
from finite_time_mpc.augmentation import build_augmented_system, lift_state_polytope
from finite_time_mpc.utils import lyapunov_initial_feasible_set


def build_plain_mpc():
    A = np.array([
        [1.1, 2.0, 0.0],
        [0.0, 0.95, 1.0],
        [0.0, 0.0, 1.2],
    ])
    B = np.array([[0.0], [0.079], [0.1]])
    H_x = np.array([[0.0, 1.0, 1.0], [0.0, -1.0, -1.0]])
    b_x = np.array([2.0, 2.0])
    X = Polytope(H_x, b_x)
    U = Polytope.from_box([-6.0], [6.0])
    Q = np.eye(3)
    R = np.array([[0.1]])
    K = np.array([[2.2150, 15.0471, 14.6128]])
    return SingleInputFiniteTimeMPC(A, B, X, U, Q, R, K)


def build_augmented_mpc():
    A = np.array([
        [1.1, 2.0, 0.0],
        [0.0, 0.95, 1.0],
        [0.0, 0.0, 1.2],
    ])
    B = np.array([[0.0], [0.079], [0.1]])
    H_x = np.array([[0.0, 1.0, 1.0], [0.0, -1.0, -1.0]])
    b_x = np.array([2.0, 2.0])
    X = Polytope(H_x, b_x)
    U = Polytope.from_box([-6.0], [6.0])
    alpha1 = -0.5
    C1 = np.array([[0.5, 0.0, 0.0]])
    A_aug, B_aug, p1 = build_augmented_system(A, B, alpha1, C1)
    # Choose K_aug via discrete LQR on augmented system
    Q_aug = np.eye(A_aug.shape[0])
    R_aug = np.array([[0.1]])
    K_aug = dlqr_gain(A_aug, B_aug, Q_aug, R_aug)
    return AugmentedFiniteTimeMPC(
        A=A, B=B, X=X, U=U,
        Q_aug=Q_aug, R=R_aug, K_aug=K_aug,
        alpha1=alpha1, C1=C1, sigma1_0=np.zeros(p1),
    ), A, B, X, U


def lexicographic_feasible_set(A, B, X: Polytope, U: Polytope, N=3):
    """Initial feasible region of lexicographic / equality terminal MPC:
    we require x(N|k) = 0 with admissible control and state sequence."""
    n = A.shape[0]
    m = B.shape[1]
    blocks = [np.linalg.matrix_power(A, N - 1 - i) @ B for i in range(N)]
    S = np.hstack(blocks)
    A_N = np.linalg.matrix_power(A, N)

    def feasible(x):
        import cvxpy as cp
        x = np.asarray(x).flatten()
        if not X.contains(x):
            return False
        u_bar = cp.Variable(N * m)
        cons = [A_N @ x + S @ u_bar == 0]
        for i in range(N):
            cons.append(U.H @ u_bar[i * m:(i + 1) * m] <= U.b)
            xi1 = np.linalg.matrix_power(A, i + 1) @ x + np.hstack(
                [np.linalg.matrix_power(A, i - j) @ B for j in range(i + 1)]
            ) @ u_bar[: (i + 1) * m]
            cons.append(X.H @ xi1 <= X.b)
        prob = cp.Problem(cp.Minimize(0), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            return False
        return prob.status in ("optimal", "optimal_inaccurate")

    return feasible


def main():
    plain = build_plain_mpc()
    aug, A, B, X, U = build_augmented_mpc()

    print("=" * 60)
    print("Example B: improving initial feasibility by augmentation")
    print("=" * 60)
    print("Augmented system size:", aug.n_aug, "control horizon:", aug.N)
    print("Augmented K_aug =", aug.K_aug)
    print("eps_aug =", aug.eps)

    # Build initial-feasibility checks (slice on x_1-x_2 plane with x_3 = 0)
    plain_feas = lyapunov_initial_feasible_set(
        plain.A, plain.B, plain.P, plain.eps, plain.X, plain.U, plain.N
    )
    aug_X = lift_state_polytope(X, aug.p1)
    aug_feas_xi = lyapunov_initial_feasible_set(
        aug.A_aug, aug.B_aug, aug.P, aug.eps, aug_X, aug.U, aug.N
    )

    def aug_feas(x):
        # initial sigma1 = 0
        xi = np.concatenate([np.zeros(aug.p1), x])
        return aug_feas_xi(xi)

    lex_feas = lexicographic_feasible_set(plain.A, plain.B, plain.X, plain.U, N=plain.N)

    # Sample x_1-x_2 plane with x_3 = 0 (matches "Initial feasibility regions
    # intersected by x1-x2 plane" caption of Fig. 2)
    x1_grid = np.linspace(-22, 22, 71)
    x2_grid = np.linspace(-4.5, 4.5, 51)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    plain_mask = np.zeros_like(X1, dtype=bool)
    aug_mask = np.zeros_like(X1, dtype=bool)
    lex_mask = np.zeros_like(X1, dtype=bool)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_test = np.array([X1[i, j], X2[i, j], 0.0])
            plain_mask[i, j] = plain_feas(x_test)
            aug_mask[i, j] = aug_feas(x_test)
            lex_mask[i, j] = lex_feas(x_test)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contour(X1, X2, aug_mask.astype(float), levels=[0.5], colors="red",
               linestyles="-", linewidths=2)
    ax.contour(X1, X2, plain_mask.astype(float), levels=[0.5], colors="black",
               linestyles="--", linewidths=1.6)
    ax.contour(X1, X2, lex_mask.astype(float), levels=[0.5], colors="grey",
               linestyles=":", linewidths=1.4)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="red", lw=2, ls="-",
               label="Augmented finite-time MPC"),
        Line2D([0], [0], color="black", lw=1.6, ls="--",
               label="Finite-time MPC"),
        Line2D([0], [0], color="grey", lw=1.4, ls=":",
               label="Lexicographic MPC"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Initial feasibility regions intersected by $x_1$-$x_2$ plane")
    ax.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "example_b.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")

    # Sanity check: simulate the augmented MPC from the same x0 as Example A
    x0 = np.array([1.9, -1.1, 0.8])
    xs, us, sigmas = aug.simulate(x0, T=10)
    print("\nClosed-loop |x| with augmented MPC:")
    for k in range(xs.shape[0]):
        print(f"  k={k:2d}  |x|={np.linalg.norm(xs[k]):10.6f}  sigma1={sigmas[k]}")


if __name__ == "__main__":
    main()
