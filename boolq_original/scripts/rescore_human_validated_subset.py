#!/usr/bin/env python3
"""Recompute the archived polarity audit on the human-validated pair subset.

This wrapper does not change the metric implementation in score_contrast_set.py.
It filters the archived raw predictions to items marked usable in all conditions,
then calls the same compute_metrics/compare/decision_audit functions and repeats
the prespecified 100,000-replicate clustered bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from score_contrast_set import (
    compare,
    compute_metrics,
    decision_audit,
    mean_metrics,
    strip,
)


SEEDS = (13, 42, 71)
MODELS = ("head", "mlp", "lcca")
ARMS = ("o", "a", "b")
N_BOOT = 100_000
BOOTSTRAP_SEED = 20260831


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def load_valid_uids(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 81 or len({row["uid"] for row in rows}) != 81:
        raise ValueError("Expected 81 unique human-validation rows")
    valid = {row["uid"] for row in rows if row["final_item_usable"] == "yes"}
    if len(valid) != 58:
        raise ValueError(f"Expected 58 all-condition-valid items, found {len(valid)}")
    return valid


def load_predictions(path: Path, valid_uids: set[str]):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    preds = []
    for row in rows:
        if row["uid"] not in valid_uids:
            continue
        preds.append(
            {
                "uid": row["uid"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "arm": row["arm"],
                "pred": row["pred"],
                "p_yes": float(row["p_yes"]),
                "p_no": float(row["p_no"]),
            }
        )
    expected = len(valid_uids) * len(MODELS) * len(SEEDS) * len(ARMS)
    keys = {(p["uid"], p["model"], p["seed"], p["arm"]) for p in preds}
    if len(preds) != expected or len(keys) != expected:
        raise ValueError(f"Expected {expected} unique filtered predictions, found {len(keys)}")
    return preds


def manuscript_table(per_model):
    lines = [
        "| Model | bal. acc O | bal. acc A | bal. acc B | edit cost | polarity cost | contrast consistency | flip rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display = {"head": "Head-only", "mlp": "Matched MLP", "lcca": "LRAA"}
    for name in MODELS:
        m = per_model[name]
        b = m["balanced_accuracy"]
        lines.append(
            f"| {display[name]} | {b['o']:.2f} | {b['a']:.2f} | {b['b']:.2f} | "
            f"{m['edit_cost']:+.2f} | {m['polarity_cost_balanced']:+.2f} | "
            f"{m['contrast_consistency']:.2f} | {m['flip_rate']:.2f} |"
        )
    return "\n".join(lines)


def bootstrap_from_raw(raw_path: Path, valid_uids: set[str]):
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["uid"] in valid_uids]
    hits = defaultdict(dict)
    for row in rows:
        hits[(row["uid"], row["model"], int(row["seed"]))][row["arm"]] = int(row["correct"])

    uids = np.array(sorted(valid_uids))
    lraa = np.empty((len(uids), len(SEEDS)), dtype=float)
    mlp = np.empty_like(lraa)
    for i, uid in enumerate(uids):
        for j, seed in enumerate(SEEDS):
            for model, target in (("lcca", lraa), ("mlp", mlp)):
                arm_hits = hits[(uid, model, seed)]
                if set(arm_hits) != set(ARMS):
                    raise ValueError(f"Incomplete raw predictions for {(uid, model, seed)}")
                target[i, j] = int(arm_hits["a"] == 1 and arm_hits["b"] == 1)

    delta = (lraa - mlp).mean(axis=1) * 100
    point = float(delta.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    item_draws = np.empty(N_BOOT)
    two_way_draws = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 5000):
        count = min(5000, N_BOOT - start)
        item_idx = rng.integers(0, len(uids), size=(count, len(uids)))
        la, ma = lraa[item_idx], mlp[item_idx]
        item_draws[start : start + count] = (la - ma).mean(axis=(1, 2)) * 100
        seed_idx = rng.integers(0, len(SEEDS), size=(count, len(SEEDS)))[:, None, :]
        two_way_draws[start : start + count] = (
            np.take_along_axis(la, seed_idx, axis=2)
            - np.take_along_axis(ma, seed_idx, axis=2)
        ).mean(axis=(1, 2)) * 100

    lraa_majority = (lraa.sum(axis=1) >= 2).astype(int)
    mlp_majority = (mlp.sum(axis=1) >= 2).astype(int)
    lraa_only = int(((lraa_majority == 1) & (mlp_majority == 0)).sum())
    mlp_only = int(((lraa_majority == 0) & (mlp_majority == 1)).sum())

    def ci(draws):
        return [round(float(x), 6) for x in np.quantile(draws, [0.025, 0.975])]

    cluster_sd = float(np.std(delta, ddof=1))
    z_sum = 1.959964 + 0.841621
    discordant = lraa_only + mlp_only
    power_sensitivity = {
        "purpose": "Post-result sensitivity calculation for the human-validated clustered contrast comparison; not a prospective sample-size calculation.",
        "analysis_unit": f"{len(uids)} unique human-validated contrast UID clusters",
        "alpha_two_sided": 0.05,
        "target_power": 0.8,
        "observed_gain_percentage_points": point,
        "cluster_mean_difference_sd_percentage_points": cluster_sd,
        "normal_approximation_mde_percentage_points": z_sum * cluster_sd / math.sqrt(len(uids)),
        "item_majority_discordant_clusters": discordant,
        "mcnemar_normal_approximation_mde_percentage_points": z_sum * math.sqrt(discordant) / len(uids) * 100,
        "interpretation": "The 58-item validated audit is underpowered for the observed approximately 6.9-point gain. Q1 failure is inconclusive and is not evidence of equivalence.",
    }

    result = {
        "analysis_unit": "58 human-validated contrast UID clusters; three paired seed repeats retained",
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "point_gain_percentage_points": round(point, 6),
        "per_seed_gain_percentage_points": {
            str(seed): round(float((lraa[:, i] - mlp[:, i]).mean() * 100), 6)
            for i, seed in enumerate(SEEDS)
        },
        "item_cluster_bootstrap": {
            "ci_95_percentage_points": ci(item_draws),
            "probability_gain_le_zero": round(float((item_draws <= 0).mean()), 6),
        },
        "two_way_item_seed_bootstrap_sensitivity": {
            "ci_95_percentage_points": ci(two_way_draws),
            "probability_gain_le_zero": round(float((two_way_draws <= 0).mean()), 6),
            "caveat": "Only three seeds are available; seed-population uncertainty is coarsely estimated.",
        },
        "item_majority_mcnemar": {
            "lraa_only_correct_clusters": lraa_only,
            "mlp_only_correct_clusters": mlp_only,
            "exact_two_sided_p": round(exact_mcnemar_p(lraa_only, mlp_only), 8),
            "lraa_majority_consistency_percent": round(float(lraa_majority.mean() * 100), 6),
            "mlp_majority_consistency_percent": round(float(mlp_majority.mean() * 100), 6),
        },
    }
    clusters = []
    for i, uid in enumerate(uids):
        clusters.append(
            {
                "uid": uid,
                "lraa_consistency_seed_mean": float(lraa[i].mean()),
                "mlp_consistency_seed_mean": float(mlp[i].mean()),
                "gain_percentage_points": float(delta[i]),
                "lraa_majority_consistent": int(lraa_majority[i]),
                "mlp_majority_consistent": int(mlp_majority[i]),
            }
        )
    return result, clusters, power_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--contrast-set", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    valid_uids = load_valid_uids(args.validation)
    all_rows = load_jsonl(args.contrast_set)
    rows = [row for row in all_rows if row["uid"] in valid_uids]
    if len(rows) != 58:
        raise ValueError(f"Expected 58 validated contrast rows, found {len(rows)}")
    preds = load_predictions(args.raw_predictions, valid_uids)

    per_run = {}
    by_model = defaultdict(list)
    for model in MODELS:
        for seed in SEEDS:
            run_preds = [p for p in preds if p["model"] == model and p["seed"] == seed]
            metric = compute_metrics(rows, run_preds)
            key = f"{model}/seed{seed}"
            per_run[key] = metric
            by_model[model].append(metric)

    per_model = {model: mean_metrics(by_model[model]) for model in MODELS}
    per_seed = {}
    positive = 0
    pooled_lraa = {"_uids": [], "_consistency": []}
    pooled_mlp = {"_uids": [], "_consistency": []}
    for i, seed in enumerate(SEEDS):
        lraa_metric = per_run[f"lcca/seed{seed}"]
        mlp_metric = per_run[f"mlp/seed{seed}"]
        comparison = compare(lraa_metric, mlp_metric, seed=seed)
        per_seed[f"seed{seed}"] = comparison
        positive += comparison["gain_points"] > 0
        pooled_lraa["_uids"] += [f"{uid}#{i}" for uid in lraa_metric["_uids"]]
        pooled_lraa["_consistency"] += lraa_metric["_consistency"]
        pooled_mlp["_uids"] += [f"{uid}#{i}" for uid in mlp_metric["_uids"]]
        pooled_mlp["_consistency"] += mlp_metric["_consistency"]
    pooled = compare(pooled_lraa, pooled_mlp)
    decision = decision_audit(per_model["lcca"], pooled, positive, len(SEEDS))
    clustered, cluster_rows, power_sensitivity = bootstrap_from_raw(args.raw_predictions, valid_uids)

    result = {
        "n_pairs": 58,
        "subset_rule": "final_item_usable == yes in human_validation_final_81.csv",
        "metric_implementation": "unchanged functions imported from score_contrast_set.py",
        "per_run": {key: strip(value) for key, value in per_run.items()},
        "per_model_mean": per_model,
        "per_seed_comparison": per_seed,
        "pooled_comparison": pooled,
        "decision_audit": decision,
        "clustered_bootstrap": clustered,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "human_validated_58_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table = manuscript_table(per_model)
    (args.out_dir / "human_validated_58_markdown_table.md").write_text(table + "\n", encoding="utf-8")
    (args.out_dir / "human_validated_58_power_sensitivity.json").write_text(
        json.dumps(power_sensitivity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "human_validated_58_item_clusters.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
        writer.writeheader()
        writer.writerows(cluster_rows)
    with (args.out_dir / "human_validated_58_contrast_set.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(table)
    print(json.dumps(clustered, indent=2, ensure_ascii=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
