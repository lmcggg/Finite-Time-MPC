"""Section VI-D: constrained finite-time MPC for a nonlinear system.

Reproduces Fig. 4 (no disturbance) and Fig. 5 (with bounded disturbance).

The plant is

    x1(k+1) = -1.1 x1(k) + 2 sin(x2(k)) + w1(k)
    x2(k+1) = 0.12 x1(k) x2(k) + 0.79 x3(k)
    x3(k+1) = x3(k) + u(k) + w2(k)

with constraints  -2 <= x2 + x3 <= 2,  -2 <= u <= 2,  feedback linearisable
in D = { x : -pi/2 <= x2 <= pi/2 }.
"""
import os
import sys

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finite_time_mpc import NonlinearFiniteTimeMPC, dlqr_gain
from finite_time_mpc.utils import lyapunov_P


def f_dyn(x, u):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return ca.vertcat(
        -1.1 * x1 + 2 * ca.sin(x2),
        0.12 * x1 * x2 + 0.79 * x3,
        x3 + u[0],
    )


def state_ineqs(x):
    # -2 <= x2 + x3 <= 2  ->  (x2 + x3) - 2 <= 0  and  -(x2 + x3) - 2 <= 0
    return [x[1] + x[2] - 2.0, -(x[1] + x[2]) - 2.0]


def make_mpc():
    # Linearise at origin to get a stabilising K and Lyapunov P
    A_lin = np.array([
        [-1.1, 2.0, 0.0],
        [0.0, 0.0, 0.79],
        [0.0, 0.0, 1.0],
    ])
    B_lin = np.array([[0.0], [0.0], [1.0]])
    Q = np.eye(3)
    R = np.array([[0.1]])
    K = dlqr_gain(A_lin, B_lin, Q, R)
    P = lyapunov_P(A_lin, B_lin, K, Q, R)

    # Small terminal box near origin (control invariant proxy as in Remark 9)
    term_lb = np.array([-0.2, -0.1, -0.1])
    term_ub = np.array([0.2, 0.1, 0.1])
    # Domain box D = { -pi/2 <= x2 <= pi/2 } and a generous box on x1, x3
    dom_lb = np.array([-10.0, -np.pi / 2, -10.0])
    dom_ub = np.array([10.0, np.pi / 2, 10.0])
    U_lb = np.array([-2.0])
    U_ub = np.array([2.0])
    return NonlinearFiniteTimeMPC(
        f_dyn=f_dyn,
        n=3, m=1,
        P=P,
        U_box=(U_lb, U_ub),
        terminal_box=(term_lb, term_ub),
        state_ineqs=state_ineqs,
        domain_box=(dom_lb, dom_ub),
        N=3,
    ), K, P


def run_no_disturbance(mpc):
    x0 = np.array([1.0, 0.3, -0.5])
    T = 10
    xs, us = mpc.simulate(x0, T)
    return xs, us


def run_with_disturbance(mpc, seed=0):
    rng = np.random.default_rng(seed)

    def disturbance(k):
        return np.array([rng.uniform(-0.2, 0.2), 0.0, rng.uniform(-0.1, 0.1)])

    x0 = np.array([1.0, 0.3, -0.5])
    T = 10
    xs, us = mpc.simulate(x0, T, disturbance=disturbance)
    return xs, us


def plot_trajectory(xs, us, title, fname):
    T = us.shape[0]
    t = np.arange(T + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
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
    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", fname)
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {os.path.abspath(out)}")


def main():
    mpc, K, P = make_mpc()
    print("=" * 60)
    print("Example D: nonlinear constrained finite-time MPC")
    print("=" * 60)
    print("Linearisation-based K =", K)
    print("Linearisation-based P =")
    print(np.array2string(P, precision=4, suppress_small=True))

    print("\n--- No disturbance ---")
    xs, us = run_no_disturbance(mpc)
    for k in range(xs.shape[0]):
        print(f"  k={k:2d}  x={xs[k]}  |x|={np.linalg.norm(xs[k]):.6f}")
    plot_trajectory(
        xs, us,
        "Example D: nonlinear MPC, no disturbance",
        "example_d_no_disturbance.png",
    )

    print("\n--- With disturbance |w1|<=0.2, |w2|<=0.1 ---")
    mpc2, _, _ = make_mpc()  # fresh mpc to reset warm start
    xs_d, us_d = run_with_disturbance(mpc2)
    for k in range(xs_d.shape[0]):
        print(f"  k={k:2d}  x={xs_d[k]}  |x|={np.linalg.norm(xs_d[k]):.6f}")
    plot_trajectory(
        xs_d, us_d,
        "Example D: nonlinear MPC with non-vanishing disturbance",
        "example_d_disturbance.png",
    )


if __name__ == "__main__":
    main()
