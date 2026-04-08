"""Run every example and report a brief summary."""
import importlib
import os
import sys

EXAMPLES = [
    # Original paper [Zhu et al. 2024]
    "examples.example_a_single_input",
    "examples.example_b_augmentation",
    "examples.example_c_multi_input",
    "examples.example_d_nonlinear",
    # Infinite-horizon extension paper [Zhu et al. 2026]
    "examples.example_e_inf_horizon_single",
    "examples.example_f_inf_horizon_multi",
    "examples.example_g_inf_horizon_nonlinear",
    "examples.example_h_inf_horizon_disturbance",
]


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for name in EXAMPLES:
        print("\n" + "#" * 70)
        print(f"# Running {name}")
        print("#" * 70)
        mod = importlib.import_module(name)
        mod.main()


if __name__ == "__main__":
    main()
