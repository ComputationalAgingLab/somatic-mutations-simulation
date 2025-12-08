#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.run_model_ii import run_model_ii
    from src.run_model_iii import run_model_iii
    from src.model_ii_hazard import compute_hazard
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

ORGAN_MODEL = {"brain": "II", "heart": "II", "liver": "III", "lungs": "III"}

def main():
    parser = argparse.ArgumentParser(description="Run models for one or more organs.")
    parser.add_argument("--organ", nargs='+', required=True,
                        choices=["brain", "heart", "liver", "lungs"],
                        help="Organs to simulate")
    parser.add_argument("--organ_s", choices=["LPC"], default=None,
                        help="Stem type for liver (applies to all liver runs)")


    parser.add_argument("--n_mc", type=int, default=10000,
                        help="MC samples for survival/KM estimation")
    parser.add_argument("--n_traj", type=int, default=5000,
                        help="Samples for trajectory output (Model II: subsample; Model III: ODE runs)")
    parser.add_argument("--t_max", type=int, default=100000)
    parser.add_argument("--n_common_points", type=int, default=10000)
    parser.add_argument("--N", type=float, default=8e9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="results")


    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--save_traces", type=int, default=100,
                        help="How many full traces to save (Model III only)")
    
    parser.add_argument(
        "--compute_hazard",
        action="store_true",
        help="Also compute and save hazard rates for Model II organs (brain/heart)"
    )

    parser.add_argument(
        "--sparse_hazard",
        action="store_true",
        help="Use sparse time grid for hazards in Model II or not. Recommended True for heart"
    )

    args = parser.parse_args()

    organs = list(dict.fromkeys(args.organ))
    for organ in organs:
        model = ORGAN_MODEL[organ]
        organ_s = args.organ_s if (organ == "liver") else None

        if model == "II":
            run_model_ii(
                organ=organ,
                n_mc=args.n_mc,
                n_traj=args.n_traj,
                t_max=args.t_max,
                n_common_points=args.n_common_points,
                seed=args.seed,
                outdir=os.path.join(args.outdir, "model_ii"),
                N=args.N
            )
            if args.compute_hazard:
                    compute_hazard(
                        organ=organ,
                        n_mc=args.n_mc,
                        t_max=args.t_max,
                        time_points=args.n_common_points,
                        sparse=False,
                        outdir=os.path.join(args.outdir, "hazard_model_ii"),
                        seed=args.seed
                    )
        else:
            if organ_s and organ != "liver":
                parser.error("--organ_s only valid for liver")
            run_model_iii(
                organ_x=organ,
                organ_s=organ_s,
                n_mc=args.n_mc,
                n_traj=args.n_traj,
                t_max=args.t_max,
                n_workers=args.n_workers,
                save_traces=args.save_traces,
                seed=args.seed,
                outdir=os.path.join(args.outdir, "model_iii"),
                n_common_points=args.n_common_points,
                N=args.N
            )

    print(f"\n All {len(organs)} simulations completed!")

if __name__ == "__main__":
    main()