"""Leakage-safe five-fold comparison of graph hops and corresponding scales.

Hop counting starts at one.  The unpropagated attribute representation is not
reported as a zero-hop experiment, but it is retained as one model scale.
Therefore ``scale_count = max_hop + 1`` (for example, 2 hops = 3 scales).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold

import cda_experiment as cda


def load_protocol(result_root: Path, name: str, seed: int) -> tuple[cda.Config, str]:
    config_row = pd.read_csv(result_root / name / "selected_config.csv").iloc[0]
    cfg = cda.Config(
        lr=float(config_row.lr), epochs=int(config_row.epochs),
        hidden_dim=int(config_row.hidden_dim), heads=int(config_row.heads),
        layers=int(config_row.layers), dropout=float(config_row.dropout),
        weight_decay=float(config_row.weight_decay), patience=int(config_row.patience),
        contrastive_weight=float(config_row.contrastive_weight),
        scales=int(config_row.scales), seed=seed,
    )
    strategy = pd.read_csv(result_root / name / "primary_protocol.csv").iloc[0, 0]
    return cfg, strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--max-hops", nargs="*", type=int, default=list(range(1, 9)))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if any(hop < 1 for hop in args.max_hops):
        raise ValueError("Hop count starts at 1; zero-hop is not an experiment candidate")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset_dirs = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    if args.datasets:
        selected = set(args.datasets)
        dataset_dirs = [path for path in dataset_dirs if path.name in selected]

    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        base_cfg, strategy = load_protocol(args.output, name, args.seed)
        positives, _ = cda.read_dataset(dataset_dir)
        circ_ids = sorted(positives.circRNA.unique())
        disease_ids = sorted(positives.disease.unique())
        c2i = {value: index for index, value in enumerate(circ_ids)}
        d2i = {value: index for index, value in enumerate(disease_ids)}
        circ_features = cda.load_sequence_features(dataset_dir, circ_ids)
        a_all = cda.interaction_matrix(positives, c2i, d2i)
        splits = list(KFold(5, shuffle=True, random_state=args.seed).split(positives))

        dataset_output = args.output / name
        cda.save_csv(pd.DataFrame([asdict(base_cfg)]), dataset_output / "selected_config.csv")
        cda.save_csv(
            pd.DataFrame([{"primary_negative_strategy": strategy}]),
            dataset_output / "primary_protocol.csv",
        )

        all_rows, summaries = [], []
        for max_hop in args.max_hops:
            scale_count = max_hop + 1
            cfg = replace(base_cfg, scales=scale_count)
            variant = f"scale_{scale_count}_hop_{max_hop}"
            rows = []
            for fold, (train_index, test_index) in enumerate(splits, 1):
                print(f"[{name}] max_hop={max_hop} fold={fold}/5 device={device}", flush=True)
                row = cda.run_fold(
                    name, fold,
                    positives.iloc[train_index].reset_index(drop=True),
                    positives.iloc[test_index].reset_index(drop=True),
                    positives, circ_ids, disease_ids, circ_features, c2i, d2i, a_all,
                    cfg, strategy, device, args.output, variant,
                )
                rows.append(row)
                all_rows.append({
                    "variant": variant, "max_hop": max_hop,
                    "scale_count": scale_count, **asdict(cfg), **row,
                })
            table = cda.five_fold_table(rows)
            cda.save_csv(
                table,
                dataset_output / "hop_comparison" / strategy / variant / "five_fold_table.csv",
            )
            summaries.append(cda.summarize(
                rows,
                {
                    "experiment": "hop_comparison", "variant": variant,
                    "max_hop": str(max_hop), "scale_count": str(scale_count),
                },
            ))

        cda.save_csv(pd.DataFrame(all_rows), dataset_output / "hop_comparison_fold_metrics.csv")
        cda.save_csv(pd.concat(summaries, ignore_index=True), dataset_output / "hop_comparison_summary.csv")


if __name__ == "__main__":
    main()
