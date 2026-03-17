#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.sensitivity_p_lethal import run_sensitivity_p_lethal
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

ORGAN_MODEL = {"brain": "II", "heart": "II", "liver": "III", "lungs": "III"}


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: sample p_lethal from a log-Uniform distribution."
    )

    parser.add_argument(
        "--organ", nargs="+", required=True,
        choices=["brain", "heart", "liver", "lungs"],
        help="One or more organs to analyse."
    )
    parser.add_argument(
        "--organ_s", action="store_const", const="LPC", default=None,
        help="Include LPC stem-cell compartment (liver / Model IIIB only)."
    )

    parser.add_argument("--n_samples",  type=int,   default=50,
                        help="Number of (p_SNV, p_indels) pairs drawn per organ.")
    parser.add_argument("--p_min",      type=float, default=1e-7,
                        help="Lower bound of the organ log-uniform sampling interval.")
    parser.add_argument("--p_max",      type=float, default=1e-3,
                        help="Upper bound of the organ log-uniform sampling interval.")

    parser.add_argument("--p_lpc_min",  type=float, default=None,
                        help="Lower bound for LPC p sampling (defaults to --p_min).")
    parser.add_argument("--p_lpc_max",  type=float, default=None,
                        help="Upper bound for LPC p sampling (defaults to --p_max).")

    parser.add_argument("--n_mc",             type=int,   default=5_000,
                        help="MC runs per p_lethal sample.")
    parser.add_argument("--t_max",            type=int,   default=100_000)
    parser.add_argument("--n_common_points",  type=int,   default=5_000,
                        help="Time-grid resolution.")
    parser.add_argument("--N",                type=float, default=8e9)
    parser.add_argument("--seed",             type=int,   default=42)

    parser.add_argument("--n_workers", type=int, default=4,
                        help="Parallel workers (Model III / ODE organs only).")

    parser.add_argument(
        "--outdir", type=str, default="results/sensitivity_p_lethal",
        help="Root output directory."
    )

    args = parser.parse_args()

    organs = list(dict.fromkeys(args.organ))
    for organ in organs:
        organ_s = args.organ_s if (organ == "liver") else None

        run_sensitivity_p_lethal(
            organ=organ,
            organ_s=organ_s,
            n_samples=args.n_samples,
            p_min=args.p_min,
            p_max=args.p_max,
            p_lpc_min=args.p_lpc_min,
            p_lpc_max=args.p_lpc_max,
            n_mc=args.n_mc,
            n_workers=args.n_workers,
            t_max=args.t_max,
            n_common_points=args.n_common_points,
            seed=args.seed,
            outdir=args.outdir,
            N=args.N,
        )

    print(f"\nAll {len(organs)} sensitivity analyses completed!")


if __name__ == "__main__":
    main()
