"""Five-fold comparison of circular k-mer composition size."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

import cda_experiment as cda


def composition(seq: str, k: int) -> np.ndarray:
    seq = "".join(x for x in str(seq).upper().replace("U", "T") if x in "ACGT")
    out = np.zeros(4 ** k, dtype=np.float32)
    if len(seq) < k:
        return out
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    circular = seq + seq[: k - 1]
    for start in range(len(seq)):
        idx = 0
        for char in circular[start : start + k]:
            idx = idx * 4 + code[char]
        out[idx] += 1
    return out / max(float(out.sum()), 1.0)


def standardize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)).astype(np.float32)


def load_dnabert(dataset_dir: Path, circ_ids):
    df = pd.read_csv(dataset_dir / "circRNA_DNABERT.csv", dtype={"circRNA": str}).set_index("circRNA")
    cols = [x for x in df.columns if x.startswith("dnabert_")]
    return standardize(df.loc[list(circ_ids), cols].to_numpy(np.float32))


def config_from_csv(path: Path) -> cda.Config:
    row = pd.read_csv(path).iloc[0]
    return cda.Config(
        lr=float(row.lr), epochs=int(row.epochs), hidden_dim=int(row.hidden_dim),
        heads=int(row.heads), layers=int(row.layers), dropout=float(row.dropout),
        weight_decay=float(row.weight_decay), patience=int(row.patience),
        contrastive_weight=float(row.contrastive_weight), scales=int(row.scales),
        seed=int(row.seed),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--negative-strategy", default="hybrid_adaptive_0.82")
    parser.add_argument("--k-values", nargs="*", type=int, default=list(range(1, 9)))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dirs = [p for p in sorted(args.dataset_root.iterdir()) if p.is_dir()]
    if args.datasets:
        dirs = [p for p in dirs if p.name in set(args.datasets)]

    for dataset_dir in dirs:
        name = dataset_dir.name
        positives, seq_df = cda.read_dataset(dataset_dir)
        circ_ids, disease_ids = sorted(positives.circRNA.unique()), sorted(positives.disease.unique())
        c2i = {x: i for i, x in enumerate(circ_ids)}
        d2i = {x: i for i, x in enumerate(disease_ids)}
        a_all = cda.interaction_matrix(positives, c2i, d2i)
        config = config_from_csv(args.results / name / "selected_config.csv")
        dnabert = load_dnabert(dataset_dir, circ_ids)
        seq_map = seq_df.set_index("circRNA").sequence.to_dict()
        splits = list(KFold(5, shuffle=True, random_state=args.seed).split(positives))
        all_rows, all_summary = [], []
        variants = [("dnabert_only", dnabert)]
        for k in args.k_values:
            kmers = np.stack([composition(seq_map[x], k) for x in circ_ids])
            cda.save_csv(
                pd.concat([
                    pd.DataFrame({"circRNA": circ_ids}),
                    pd.DataFrame(kmers, columns=[f"kmer{k}_{i}" for i in range(kmers.shape[1])]),
                ], axis=1),
                args.results / name / "k_comparison" / f"kmer_{k}_composition.csv",
            )
            variants.append((f"kmer_{k}", np.concatenate([dnabert, standardize(kmers)], 1)))

        for variant, features in variants:
            rows = []
            for fold, (tr, te) in enumerate(splits, 1):
                print(f"[{name}] {variant} fold={fold}/5")
                row = cda.run_fold(
                    name, fold, positives.iloc[tr].reset_index(drop=True), positives.iloc[te].reset_index(drop=True),
                    positives, circ_ids, disease_ids, features, c2i, d2i, a_all,
                    config, args.negative_strategy, device, args.results, variant,
                )
                rows.append(row)
                all_rows.append({"variant": variant, **row})
            all_summary.append(cda.summarize(rows, {"experiment": "k_comparison", "variant": variant}))
            cda.save_csv(
                cda.five_fold_table(rows),
                args.results / name / "k_comparison" / args.negative_strategy / variant / "five_fold_table.csv",
            )
        cda.save_csv(pd.DataFrame(all_rows), args.results / name / "k_comparison_fold_metrics.csv")
        cda.save_csv(pd.concat(all_summary, ignore_index=True), args.results / name / "k_comparison_summary_mean_std.csv")


if __name__ == "__main__":
    main()
