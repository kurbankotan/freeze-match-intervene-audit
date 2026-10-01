#!/usr/bin/env python3
"""Score trained checkpoints on the polarity contrast set.

No API calls, no network. Runs wherever the checkpoints run.

    python score_contrast_set.py contrast_set.jsonl --out metrics.json

Three arms per pair, sharing one unmodified question:

    O   original passage                    gold = g
    A   evidence sentence paraphrased       gold = g       (edit-distribution control)
    B   evidence sentence negated           gold = not g   (polarity probe)

Comparing O to B alone confounds polarity with the shift caused by rewriting at all.
The A arm separates the two:

    edit cost     = acc(O) - acc(A)     attributable to rewriting
    polarity cost = acc(A) - acc(B)     attributable to polarity, editing held constant

------------------------------------------------------------------------------
PLUG YOUR MODEL IN HERE
------------------------------------------------------------------------------
Implement one function per model/seed and register it in MODELS:

    def score_fn(passage: str, question: str, option: str) -> float

It must build [CLS; passage; SEP; question; SEP; option; SEP], run the frozen
encoder, apply the adapter under test, and return the scalar scorer output for
that one candidate. The two candidate scores are softmaxed here, so the margin
|p_yes - p_no| is comparable across models.

A worked skeleton for the frozen-backbone setup is at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

OPTIONS = ("yes", "no")
ARMS = ("o", "a", "b")
CHANCE_CONSISTENCY = 25.0        # each pair is one yes + one no by construction
CONSTANT_LABEL_CONSISTENCY = 0.0  # a model answering identically to both arms


# --------------------------------------------------------------------- scoring


def score_arms(rows, score_fn, model, seed):
    preds = []
    for row in rows:
        for arm in ARMS:
            raw = [score_fn(row[f"passage_{arm}"], row["question"], o) for o in OPTIONS]
            top = max(raw)
            e = [math.exp(s - top) for s in raw]
            tot = sum(e)
            p_yes, p_no = e[0] / tot, e[1] / tot
            preds.append({"uid": row["uid"], "model": model, "seed": seed, "arm": arm,
                          "pred": "yes" if p_yes >= p_no else "no",
                          "p_yes": round(p_yes, 6), "p_no": round(p_no, 6)})
    return preds


def _gold(row, arm):
    return row["gold_b"] if arm == "b" else row["gold_o"]


def compute_metrics(rows, preds):
    """Metrics for one model/seed."""
    by_uid = {r["uid"]: r for r in rows}
    table = defaultdict(dict)
    for p in preds:
        if p["uid"] in by_uid:
            table[p["uid"]][p["arm"]] = p

    correct = {a: [] for a in ARMS}
    # per-arm, per-gold-class hits. The set is skewed toward yes-gold sources, so a
    # constant "yes" predictor scores well on arms O and A; balanced accuracy removes
    # that artefact and pre-empts the obvious reviewer question.
    by_class = {a: defaultdict(list) for a in ARMS}
    consistency, flips, cond = [], [], []
    margins = {"correct": [], "incorrect": []}
    by_cue, by_direction = defaultdict(list), defaultdict(list)
    pair_uids = []

    for uid, ap in sorted(table.items()):
        row = by_uid[uid]
        if not {"a", "b"} <= ap.keys():
            continue
        hit = {}
        for arm in ARMS:
            if arm in ap:
                g = _gold(row, arm)
                hit[arm] = int(ap[arm]["pred"] == g)
                correct[arm].append(hit[arm])
                by_class[arm][g].append(hit[arm])

        consistency.append(int(hit.get("a") == 1 and hit.get("b") == 1))
        pair_uids.append(uid)
        flips.append(int(ap["a"]["pred"] != ap["b"]["pred"]))
        if hit.get("a") == 1:
            cond.append(hit["b"])

        m = abs(ap["b"]["p_yes"] - ap["b"]["p_no"])
        margins["correct" if hit.get("b") else "incorrect"].append(m)
        by_cue[row.get("cue_family", "?")].append(hit.get("b", 0))
        # adding a negation vs removing one are different operations; the set is
        # skewed toward the first, so report them separately
        by_direction[f"{row['gold_o']}->{row['gold_b']}"].append(hit.get("b", 0))

    def rate(xs):
        return round(100 * sum(xs) / len(xs), 2) if xs else None

    def msum(xs):
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean_margin": round(statistics.mean(xs), 4),
                "frac_above_0.9": round(sum(x > 0.9 for x in xs) / len(xs), 4)}

    acc = {a: rate(correct[a]) for a in ARMS}

    def balanced(arm):
        per = [rate(v) for v in by_class[arm].values() if v]
        return round(statistics.mean(per), 2) if per else None

    bacc = {a: balanced(a) for a in ARMS}
    sub = lambda x, y: round(x - y, 2) if None not in (x, y) else None

    return {
        "n_pairs": len(consistency),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "predicted_yes_rate": {
            a: round(100 * sum(1 for p in preds if p["arm"] == a and p["pred"] == "yes")
                     / max(1, sum(1 for p in preds if p["arm"] == a)), 2)
            for a in ARMS},
        "edit_cost": sub(acc["o"], acc["a"]),
        "polarity_cost": sub(acc["a"], acc["b"]),
        "polarity_cost_balanced": sub(bacc["a"], bacc["b"]),
        "contrast_consistency": rate(consistency),
        "chance_consistency": CHANCE_CONSISTENCY,
        "constant_label_consistency": CONSTANT_LABEL_CONSISTENCY,
        "flip_rate": rate(flips),
        "conditional_negation_accuracy": rate(cond),
        "margin_profile_arm_b": {k: msum(v) for k, v in margins.items()},
        "arm_b_by_cue_family": {k: {"n": len(v), "acc": rate(v)}
                                for k, v in sorted(by_cue.items())},
        "arm_b_by_direction": {k: {"n": len(v), "acc": rate(v)}
                               for k, v in sorted(by_direction.items())},
        "_uids": pair_uids,
        "_consistency": consistency,
    }


# ------------------------------------------------------------------ statistics


def mcnemar_exact(a_only, b_only):
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(x, y, n_boot=10000, seed=0, alpha=0.05):
    if len(x) != len(y) or not x:
        raise ValueError("need equal, non-empty vectors")
    rng = random.Random(seed)
    n = len(x)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] for i in idx) / n - sum(y[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(100 * (sum(x) / n - sum(y) / n), 2), round(100 * lo, 2), round(100 * hi, 2))


def min_detectable_effect(n_pairs, discordant_rate, power=0.80, alpha=0.05):
    """Smallest consistency difference McNemar can detect at this n.

    Reporting it keeps a null result honest: 'no difference' and 'too few pairs
    to see one' are not the same claim.
    """
    d = n_pairs * discordant_rate
    if d < 1:
        return None
    z_a, z_b = 1.959964, 0.841621          # two-sided 5%, 80% power
    # normal approximation to the sign test on discordant pairs
    return round(100 * (z_a + z_b) * math.sqrt(d) / n_pairs, 1)


def compare(m_a, m_b, seed=0):
    map_a = dict(zip(m_a["_uids"], m_a["_consistency"]))
    map_b = dict(zip(m_b["_uids"], m_b["_consistency"]))
    shared = [u for u in m_a["_uids"] if u in map_b]
    x = [map_a[u] for u in shared]
    y = [map_b[u] for u in shared]
    a_only = sum(1 for i in range(len(shared)) if x[i] and not y[i])
    b_only = sum(1 for i in range(len(shared)) if y[i] and not x[i])
    point, lo, hi = paired_bootstrap(x, y, seed=seed)
    disc = (a_only + b_only) / len(shared) if shared else 0
    return {"n_pairs": len(shared), "gain_points": point, "bootstrap_95_ci": [lo, hi],
            "discordant_a_only": a_only, "discordant_b_only": b_only,
            "discordant_rate": round(disc, 3),
            "mcnemar_exact_p": round(mcnemar_exact(a_only, b_only), 6),
            "min_detectable_effect_pts": min_detectable_effect(len(shared), disc)}


def decision_audit(treatment_mean, comparison, seeds_positive, n_seeds):
    q1 = [("mean gain in contrast consistency >= 5.0 pts",
           comparison["gain_points"] >= 5.0),
          (f"treatment above control in all {n_seeds} seeds", seeds_positive == n_seeds),
          ("bootstrap 95% CI excludes zero", comparison["bootstrap_95_ci"][0] > 0)]
    q2 = [("contrast consistency >= 45%", (treatment_mean.get("contrast_consistency") or 0) >= 45.0),
          ("flip rate >= 50%", (treatment_mean.get("flip_rate") or 0) >= 50.0)]
    mde = comparison.get("min_detectable_effect_pts")
    return {
        "Q1_topology_buys_polarity": {
            "criteria": [{"rule": r, "pass": bool(p)} for r, p in q1],
            "decision": "PASS" if all(p for _, p in q1) else "FAIL",
            "min_detectable_effect_pts": mde,
            "caveat": (f"At n={comparison['n_pairs']} the smallest difference detectable at "
                       f"80% power is about {mde} points. A FAIL below that is inconclusive, "
                       f"not evidence of no difference.") if mde else None,
        },
        "Q2_any_configuration_handles_polarity": {
            "criteria": [{"rule": r, "pass": bool(p)} for r, p in q2],
            "decision": "PASS" if all(p for _, p in q2) else "FAIL",
        },
        "note": ("Q1 PASS with Q2 FAIL is coherent: the topology helps but no configuration "
                 "represents polarity. Both thresholds were fixed before any run."),
    }


# --------------------------------------------------------------------- reports


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def markdown_table(per_model):
    """Manuscript-ready table. Balanced accuracy is the one to read: the source
    items skew toward yes-gold, so raw accuracy on arms O and A flatters a
    constant-yes predictor."""
    lines = ["| Model | bal. acc O | bal. acc A | bal. acc B | edit cost | polarity cost | "
             "contrast consistency | flip rate |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in per_model.items():
        b = m["balanced_accuracy"]
        lines.append(f"| {name} | {b['o']} | {b['a']} | {b['b']} | {m['edit_cost']} | "
                     f"{m['polarity_cost_balanced']} | {m['contrast_consistency']} | "
                     f"{m['flip_rate']} |")
    lines += ["",
              f"Chance contrast consistency {CHANCE_CONSISTENCY}; a model answering "
              f"identically to both arms scores {CONSTANT_LABEL_CONSISTENCY}. "
              "Polarity cost is balanced acc(A) - balanced acc(B)."]
    return "\n".join(lines)


def mean_metrics(ms):
    ms = list(ms)
    keys = ("contrast_consistency", "flip_rate", "conditional_negation_accuracy",
            "polarity_cost", "polarity_cost_balanced", "edit_cost")
    out = {}
    for k in keys:
        vals = [m[k] for m in ms if m.get(k) is not None]
        out[k] = round(statistics.mean(vals), 2) if vals else None
    for field in ("accuracy", "balanced_accuracy"):
        per = {a: [m[field][a] for m in ms if m[field].get(a) is not None] for a in ARMS}
        out[field] = {a: round(statistics.mean(v), 2) if v else None for a, v in per.items()}
    return out


def run(rows, models, treatment="lcca", control="mlp", out_path="metrics.json"):
    """models: {(name, seed): score_fn}"""
    per_run, preds_all = {}, []
    for (name, seed), fn in sorted(models.items()):
        preds = score_arms(rows, fn, name, seed)
        preds_all += preds
        per_run[f"{name}/seed{seed}"] = compute_metrics(rows, preds)
        m = per_run[f"{name}/seed{seed}"]
        print(f"{name}/seed{seed}: consistency {m['contrast_consistency']}  "
              f"flip {m['flip_rate']}  polarity cost {m['polarity_cost']}")

    by_model = defaultdict(list)
    for key, m in per_run.items():
        by_model[key.split("/")[0]].append(m)

    result = {"n_pairs": len(rows),
              "per_run": {k: strip(v) for k, v in per_run.items()},
              "per_model_mean": {k: mean_metrics(v) for k, v in by_model.items()}}

    seeds = sorted({s for (_, s) in models})
    if treatment in by_model and control in by_model:
        per_seed, positive = {}, 0
        for s in seeds:
            a, b = per_run.get(f"{treatment}/seed{s}"), per_run.get(f"{control}/seed{s}")
            if a and b:
                c = compare(a, b, seed=s)
                per_seed[f"seed{s}"] = c
                positive += c["gain_points"] > 0
        pooled_a = {"_uids": [], "_consistency": []}
        pooled_b = {"_uids": [], "_consistency": []}
        for i, s in enumerate(seeds):
            a, b = per_run.get(f"{treatment}/seed{s}"), per_run.get(f"{control}/seed{s}")
            if a and b:
                pooled_a["_uids"] += [f"{u}#{i}" for u in a["_uids"]]
                pooled_a["_consistency"] += a["_consistency"]
                pooled_b["_uids"] += [f"{u}#{i}" for u in b["_uids"]]
                pooled_b["_consistency"] += b["_consistency"]
        pooled = compare(pooled_a, pooled_b)
        result["per_seed_comparison"] = per_seed
        result["pooled_comparison"] = pooled
        result["decision_audit"] = decision_audit(
            mean_metrics(by_model[treatment]), pooled, positive, len(per_seed))

    result["markdown_table"] = markdown_table(
        {k: mean_metrics(v) for k, v in by_model.items()})

    Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print("\n" + result["markdown_table"])
    if "decision_audit" in result:
        print("\n" + json.dumps(result["decision_audit"], indent=2, ensure_ascii=False))
    print(f"\nwritten to {out_path}")
    return result


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ------------------------------------------------------------------- skeleton
# Replace this with the real checkpoints. The contract is the only thing that
# matters: score_fn(passage, question, option) -> float.
#
#   import torch
#   from transformers import AutoTokenizer, AutoModel
#
#   tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
#   enc = AutoModel.from_pretrained("microsoft/deberta-v3-large").eval().cuda()
#   for p in enc.parameters():
#       p.requires_grad = False
#
#   def make_score_fn(adapter, scorer):
#       @torch.no_grad()
#       def score_fn(passage, question, option):
#           ids = tok(f"{passage} [SEP] {question} [SEP] {option}",
#                     truncation=True, max_length=512, return_tensors="pt").to("cuda")
#           H = enc(**ids).last_hidden_state
#           if adapter is not None:
#               H = adapter(H)                 # LCCA or the matched MLP
#           return scorer(H[:, 0]).item()      # the 1,025-parameter scorer on [CLS]
#       return score_fn
#
#   MODELS = {}
#   for seed in (13, 42, 71):
#       ckpt = torch.load(f"lcca_seed{seed}.pt")
#       MODELS[("lcca", seed)] = make_score_fn(ckpt["adapter"], ckpt["scorer"])
#       ckpt = torch.load(f"mlp_seed{seed}.pt")
#       MODELS[("mlp", seed)] = make_score_fn(ckpt["adapter"], ckpt["scorer"])
#       ckpt = torch.load(f"head_seed{seed}.pt")
#       MODELS[("head", seed)] = make_score_fn(None, ckpt["scorer"])
#
#   run(load("contrast_set.jsonl"), MODELS)


def lexical_overlap_score_fn(passage, question, option):
    """Floor baseline. This is the shortcut the contrast set exists to expose:
    it never looks at polarity, so it should score 0 contrast consistency."""
    overlap = len(set(passage.lower().split()) & set(question.lower().split()))
    return overlap + (0.5 if option == "yes" else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("contrast_set")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--baseline-only", action="store_true",
                    help="run only the lexical floor, no checkpoints needed")
    args = ap.parse_args()

    rows = load(args.contrast_set)
    print(f"{len(rows)} pairs, {3 * len(rows)} evaluation items\n")

    if args.baseline_only:
        run(rows, {("lexical", 0): lexical_overlap_score_fn}, out_path=args.out)
        print("\nThis is the floor, not a result. Register the real checkpoints in MODELS.")
        return

    print("No checkpoints registered. Edit MODELS at the bottom of this file, or pass "
          "--baseline-only to see the lexical floor.")


if __name__ == "__main__":
    main()
