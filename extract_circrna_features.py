"""Extract circular-aware DNABERT + k-mer composition features for CDA."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def read_sequences(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    records, current_id, current_seq = [], None, ""
    for row in raw.itertuples(index=False, name=None):
        left = str(row[0]).strip()
        right = str(row[1]).strip() if len(row) > 1 else ""
        if left and right:
            if current_id is not None:
                records.append((current_id, current_seq.upper()))
            current_id, current_seq = left, right
        elif left and current_id is not None:
            current_seq += left
    if current_id is not None:
        records.append((current_id, current_seq.upper()))
    return pd.DataFrame(records, columns=["circRNA", "sequence"]).drop_duplicates("circRNA")


def clean(seq: str) -> str:
    return "".join(x for x in seq.upper().replace("U", "T") if x in "ACGT")


def circular_windows(seq: str, size: int = 507, max_windows: int = 8):
    seq = clean(seq)
    if not seq:
        return ["N" * 6]
    if len(seq) <= size:
        return [seq]
    starts = np.linspace(0, len(seq) - 1, min(max_windows, math.ceil(len(seq) / size)), dtype=int)
    circular = seq + seq[:size]
    return [circular[s:s + size] for s in starts]


def to_kmers(seq: str, k: int = 6) -> str:
    if len(seq) < k:
        return seq
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def kmer_composition(seq: str, k: int = 4) -> np.ndarray:
    seq = clean(seq)
    dim = 4 ** k
    out = np.zeros(dim, dtype=np.float32)
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    if len(seq) < k:
        return out
    circular = seq + seq[:k - 1]
    for i in range(len(seq)):
        idx = 0
        for char in circular[i:i + k]:
            idx = idx * 4 + code[char]
        out[idx] += 1
    return out / max(out.sum(), 1.0)


@torch.inference_mode()
def extract_dataset(sequences, tokenizer, model, device, batch_size, max_windows):
    texts, owners = [], []
    for owner, seq in enumerate(sequences):
        windows = circular_windows(seq, max_windows=max_windows)
        texts.extend(to_kmers(x) for x in windows)
        owners.extend([owner] * len(windows))
    sums = torch.zeros((len(sequences), model.config.hidden_size), dtype=torch.float32)
    maxima = torch.full_like(sums, -torch.inf)
    counts = torch.zeros(len(sequences), dtype=torch.float32)
    for start in tqdm(range(0, len(texts), batch_size), desc="DNABERT windows", leave=False):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = pooled.float().cpu()
        batch_owners = torch.as_tensor(owners[start:start + len(batch_texts)], dtype=torch.long)
        sums.index_add_(0, batch_owners, pooled)
        counts.index_add_(0, batch_owners, torch.ones(len(batch_owners)))
        maxima.scatter_reduce_(0, batch_owners[:, None].expand_as(pooled), pooled, reduce="amax", include_self=True)
    means = sums / counts.clamp_min(1).unsqueeze(1)
    return torch.cat([means, maxima], 1).numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--model", default="zhihan1996/DNA_bert_6")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    model = AutoModel.from_pretrained(args.model, local_files_only=not args.allow_download).to(device).eval()
    dirs = [p for p in sorted(args.dataset_root.iterdir()) if p.is_dir()]
    if args.datasets:
        dirs = [p for p in dirs if p.name in set(args.datasets)]
    for directory in dirs:
        df = read_sequences(directory / "Get_circRNAsequence.csv")
        bert = extract_dataset(df.sequence.tolist(), tokenizer, model, device, args.batch_size, args.max_windows)
        k = int(os.environ.get("CDA_FEATURE_KMER", "4"))
        comp = np.stack([kmer_composition(x, k=k) for x in tqdm(df.sequence, desc=f"{k}-mer")])
        x = np.concatenate([bert, comp], axis=1)
        columns = [f"dnabert_{i}" for i in range(model.config.hidden_size * 2)] + [f"kmer{k}_{i}" for i in range(4 ** k)]
        out = pd.concat([df[["circRNA"]].reset_index(drop=True), pd.DataFrame(x, columns=columns)], axis=1)
        out.to_csv(directory / "circRNA_DNABERT.csv", index=False, float_format="%.6f")
        print(f"Saved {directory / 'circRNA_DNABERT.csv'}: {x.shape}")


if __name__ == "__main__":
    main()
