"""Five-fold one-factor parameter comparisons under each fixed primary strategy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold

import cda_experiment as cda


LR_VALUES = [0.01, 0.03, 0.05, 0.06, 0.07]
HEAD_VALUES = [2, 4, 8, 16]
LAYER_VALUES = list(range(1, 11))


def load_config(path: Path) -> cda.Config:
    row = pd.read_csv(path).iloc[0]
    return cda.Config(
        lr=float(row.lr), epochs=int(row.epochs), hidden_dim=int(row.hidden_dim),
        heads=int(row.heads), layers=int(row.layers), dropout=float(row.dropout),
        weight_decay=float(row.weight_decay), patience=int(row.patience),
        contrastive_weight=float(row.contrastive_weight), scales=int(row.scales), seed=int(row.seed),
    )


def parameter_variants(best: cda.Config, dataset_name: str):
    variants = {"Best": best}
    for value in LR_VALUES:
        if abs(value - best.lr) > 1e-12:
            variants[f"lr={value:g}"] = replace(best, lr=value)
    for value in HEAD_VALUES:
        if value == best.heads:
            continue
        variants[f"heads={value}"] = replace(best, heads=value)
    for value in LAYER_VALUES:
        if value == best.layers:
            continue
        variants[f"layers={value}"] = replace(best, layers=value)
    return variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--results", type=Path, default=Path("_experiment_work"))
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--incremental", action="store_true",
        help="Reuse complete five-fold variants from parameter_comparison_fold_metrics.csv.",
    )
    args = p.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dirs = sorted(x for x in args.dataset_root.iterdir() if x.is_dir())
    if args.datasets:
        dirs = [x for x in dirs if x.name in set(args.datasets)]
    for dataset_dir in dirs:
        name = dataset_dir.name
        positives, _ = cda.read_dataset(dataset_dir)
        circs, diseases = sorted(positives.circRNA.unique()), sorted(positives.disease.unique())
        c2i, d2i = {x: i for i, x in enumerate(circs)}, {x: i for i, x in enumerate(diseases)}
        features = cda.load_sequence_features(dataset_dir, circs)
        a_all = cda.interaction_matrix(positives, c2i, d2i)
        best = load_config(args.results / name / "selected_config.csv")
        strategy = pd.read_csv(args.results / name / "primary_protocol.csv").iloc[0, 0]
        splits = list(KFold(5, shuffle=True, random_state=best.seed).split(positives))
        previous_path = args.results / name / "parameter_comparison_fold_metrics.csv"
        previous = pd.read_csv(previous_path) if args.incremental and previous_path.exists() else pd.DataFrame()
        all_rows, summaries = [], []
        for variant, cfg in parameter_variants(best, name).items():
            safe_variant = variant.replace("=", "_").replace(".", "p")
            cached = previous.loc[previous.variant == variant].sort_values("fold") if not previous.empty else pd.DataFrame()
            config_matches = False
            if len(cached) == 5:
                config_matches = all(
                    all(abs(float(row[key]) - float(value)) < 1e-12 for key, value in asdict(cfg).items())
                    for _, row in cached.iterrows()
                )
            if len(cached) == 5 and cached.fold.tolist() == [1, 2, 3, 4, 5] and config_matches:
                print(f"[{name}] parameter={variant}: reuse existing 5 folds")
                rows = cached[cda.METRICS + ["fold"]].to_dict("records")
                all_rows.extend(cached.to_dict("records"))
            else:
                rows = []
                for fold, (tr, te) in enumerate(splits, 1):
                    print(f"[{name}] parameter={variant} fold={fold}/5")
                    row = cda.run_fold(
                        name, fold, positives.iloc[tr].reset_index(drop=True), positives.iloc[te].reset_index(drop=True),
                        positives, circs, diseases, features, c2i, d2i, a_all,
                        cfg, strategy, device, args.results, f"param_{safe_variant}",
                    )
                    rows.append(row)
                    all_rows.append({"variant": variant, **asdict(cfg), **row})
            summaries.append(cda.summarize(rows, {"experiment": "parameter_comparison", "variant": variant}))
            cda.save_csv(cda.five_fold_table(rows), args.results / name / "parameter_tables" / f"{safe_variant}.csv")
        cda.save_csv(pd.DataFrame(all_rows), args.results / name / "parameter_comparison_fold_metrics.csv")
        cda.save_csv(pd.concat(summaries, ignore_index=True), args.results / name / "parameter_comparison_summary.csv")


if __name__ == "__main__":
    main()
