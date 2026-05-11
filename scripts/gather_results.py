"""Gather and display all experiment results."""
import json
from pathlib import Path

import numpy as np


def main():
    runs_dir = Path("./runs")
    if not runs_dir.exists():
        print("No runs/ directory found.")
        return

    results = {}
    for result_file in sorted(runs_dir.rglob("results.json")):
        with open(result_file) as f:
            data = json.load(f)
        run_name = result_file.parent.name
        results[run_name] = data

    # Group by model (strip _seedN suffix)
    from collections import defaultdict
    grouped = defaultdict(list)
    for name, data in results.items():
        # e.g. "model_a_seed42" -> "model_a"
        parts = name.rsplit("_seed", 1)
        base = parts[0]
        grouped[base].append(data)

    print("=" * 80)
    print(f"{'Model':<25} {'Probe Acc (mean±std)':<25} {'Eff. Rank':<15} {'Uniformity':<15}")
    print("=" * 80)

    for model_name in ["model_a", "model_b", "model_c",
                       "confounder_lr2x", "confounder_lr4x", "confounder_normed"]:
        runs = grouped.get(model_name, [])
        if not runs:
            continue

        accs = [r["probe_euclidean"] * 100 for r in runs if "probe_euclidean" in r]
        ranks = [r["final_collapse_metrics"]["effective_rank"] for r in runs
                 if "final_collapse_metrics" in r]
        uniforms = [r["final_collapse_metrics"]["uniformity"] for r in runs
                    if "final_collapse_metrics" in r]

        acc_str = f"{np.mean(accs):.2f} ± {np.std(accs):.2f}" if accs else "N/A"
        rank_str = f"{np.mean(ranks):.1f}" if ranks else "N/A"
        uni_str = f"{np.mean(uniforms):.4f}" if uniforms else "N/A"

        n_runs = len(accs)
        print(f"{model_name:<25} {acc_str:<25} {rank_str:<15} {uni_str:<15} (n={n_runs})")

        # Show tangent probe for hyperbolic models
        tang_accs = [r["probe_tangent"] * 100 for r in runs if "probe_tangent" in r]
        if tang_accs:
            tang_str = f"{np.mean(tang_accs):.2f} ± {np.std(tang_accs):.2f}"
            print(f"  {'(tangent probe)':<23} {tang_str:<25}")

    print("=" * 80)
    print("\nConfounder Analysis:")
    print("-" * 80)

    a_accs = [r["probe_euclidean"] * 100 for r in grouped.get("model_a", [])
              if "probe_euclidean" in r]
    b_accs = [r["probe_euclidean"] * 100 for r in grouped.get("model_b", [])
              if "probe_euclidean" in r]

    if a_accs and b_accs:
        gap = np.mean(b_accs) - np.mean(a_accs)
        print(f"  B - A gap: {gap:+.2f} pp")

    for conf_name in ["confounder_lr2x", "confounder_lr4x", "confounder_normed"]:
        runs = grouped.get(conf_name, [])
        if runs:
            conf_accs = [r["probe_euclidean"] * 100 for r in runs if "probe_euclidean" in r]
            if conf_accs and a_accs:
                conf_gap = np.mean(conf_accs) - np.mean(a_accs)
                closes_gap = "YES - confounder!" if conf_gap >= gap * 0.7 else "no"
                print(f"  {conf_name}: {np.mean(conf_accs):.2f}% "
                      f"(+{conf_gap:.2f} pp vs A) — closes B's gap? {closes_gap}")

    # Collapse comparison
    print("\nCollapse Analysis:")
    print("-" * 80)
    for model_name in ["model_a", "model_b", "model_c"]:
        runs = grouped.get(model_name, [])
        if runs:
            ranks = [r["final_collapse_metrics"]["effective_rank"] for r in runs
                     if "final_collapse_metrics" in r]
            vars_ = [r["final_collapse_metrics"]["mean_var"] for r in runs
                     if "final_collapse_metrics" in r]
            if ranks:
                print(f"  {model_name}: eff_rank={np.mean(ranks):.1f}  mean_var={np.mean(vars_):.4f}")


if __name__ == "__main__":
    main()
