# Finite-Time MPC 


1. **Zhu, Yuan, Dai, Qiang. "Finite-Time Stabilization for Constrained Discrete-time Systems by Using Model Predictive Control."** *IEEE/CAA Journal of Automatica Sinica*, vol. 11, no. 7, pp. 1656–1666, 2024.
2. **Zhu, Yuan, Zheng, Zuo. "Constrained Finite-Time Stabilization by Model Predictive Control: an Infinite Control Horizon Framework."** arXiv:2603.09617, 2026.

The second paper is a direct extension of the first: it replaces the
"terminal-cost-only with N=n" formulation by "infinite-horizon stage cost
starting from i=n with N≥n", which significantly enlarges the initial
feasibility region while preserving finite-time convergence.

## Dependencies

```
python ≥ 3.9
numpy
scipy
cvxpy        # convex optimization (linear MPC)
casadi       # nonlinear NLP (IPOPT)
matplotlib   # plotting
```

Tested in the `moh` conda environment, where all dependencies are pre-installed.

## How to Run

```bash
conda activate moh
cd finite_time_mpc_repro

# Run all 8 examples in one go
python run_all.py

# Or run them individually
python examples/example_a_single_input.py        # Paper 1, §VI-A
python examples/example_b_augmentation.py        # Paper 1, §VI-B
python examples/example_c_multi_input.py         # Paper 1, §VI-C
python examples/example_d_nonlinear.py           # Paper 1, §VI-D
python examples/example_e_inf_horizon_single.py  # Paper 2, §IV-A + Fig. 2
python examples/example_f_inf_horizon_multi.py   # Paper 2, §IV-B
python examples/example_g_inf_horizon_nonlinear.py # Paper 2, §IV-C
python examples/example_h_inf_horizon_disturbance.py # Paper 2, §IV-D
```

All figures are saved to [`figures/`](figures/).

## Project Layout

```
finite_time_mpc_repro/
├── finite_time_mpc/
│   ├── utils.py            # Polytope, Lyapunov, terminal radius, DLQR
│   ├── single_input.py     # Paper 1, Algorithm 1
│   ├── multi_input.py      # Paper 1, Algorithm 2 + Wonham transform
│   ├── augmentation.py     # Paper 1, Algorithm 3
│   ├── nonlinear.py        # Paper 1, Algorithm 4
│   └── infinite_horizon.py # Paper 2, Theorems 2 / 3 / 4
├── examples/               # 8 numerical examples
├── figures/                # generated figures
└── run_all.py
```

## Reproduced Results

### Paper 1 [Zhu et al. 2024]

| Example | Quantity | Paper value | Reproduced |
|---|---|---|---|
| §VI-A single-input | P (1,1), (2,2), (3,3) | 6.1590 / 96.8173 / 29.9407 | **6.159 / 96.817 / 29.9412** |
| §VI-A single-input | terminal radius ε | 4.6151 | **4.6151** |
| §VI-A single-input | settling steps | 6 | 5 |
| §VI-B augmentation | control horizon N | 4 | 4 |
| §VI-B augmentation | feasibility regions | augmented > plain > lexicographic | three nested regions reproduced |
| §VI-C multi-input | settling steps | ≤ 6 | 3 |
| §VI-D nonlinear (no disturbance) | settling steps | ≤ 5 | 3 |
| §VI-D nonlinear (with disturbance) | ultimate boundedness | ✓ | ✓ |

### Paper 2 [Zhu et al. 2026]

| Example | Quantity | Paper value | Reproduced |
|---|---|---|---|
| §IV-A single-input N=8 | P (1,1), (2,2) | 6.7 / 106.8 | **6.7397 / 107.36** |
| §IV-A single-input N=8 | terminal radius ε | 4.15 | **4.1625** |
| §IV-A single-input N=8 | settling steps | 7 | 6 |
| §IV-A Fig. 2 feasibility | new ≫ old | drastically enlarged | **fully reproduced** |
| §IV-B multi-input N=8 | settling steps | **11** | **11** |
| §IV-B multi-input N=8 | u₂ | identically zero | identically zero |
| §IV-C nonlinear N=8 | settling steps | 5 | 3 |
| §IV-D bounded disturbance (10 trials) | ultimate boundedness | ✓ | mean \|x(T)\| ≈ 0.097 |

## Typos / Inconsistencies Found in the Papers

* **Paper 1, §VI-C**: K₁ = [2.2150, 15.0471, 14.6128] is copy-pasted verbatim
  from §VI-A, but it does **not** stabilize the decoupled subsystem
  (F₁₁, g₁) (eigenvalues of F₁₁ − g₁K₁ are roughly {−9.4, −1.9, −0.05}).
  The reproduction recomputes K₁ via discrete LQR on (F₁₁, g₁), as required
  by step 3 of Algorithm 2.
* **Paper 2, §IV-B**: "Q₁ = I_{2×2}" is inconsistent with q = 1, n₁ = 3 and
  appears to be a typo for I_{3×3}. The reproduction uses I_{3×3}.
* **Paper 2, equation (29)**: literally written as ε = max xᵀPx subject to
  x ∈ X, Kx ∈ U, which is unbounded for any unbounded constraint set.
  The intended meaning is the largest sub-level set of P that fits inside
  the constraint set, which numerically equals the formula (14) of Paper 1.
  The reproduction reuses [`utils.terminal_radius`](finite_time_mpc/utils.py).
