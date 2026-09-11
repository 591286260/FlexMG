"""Leakage-safe five-fold circRNA-disease association experiments.

The graph, disease GIP kernel, and topology-dependent negative scores are rebuilt
from the training positives of every fold.  Known positives from all folds are
only used as an exclusion list so that they are never mislabeled as negatives.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split


METRICS = ["acc", "prec", "f1", "rec", "auc", "aupr"]
TABLE_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "AUPR"]


@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    epochs: int = 60
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.20
    weight_decay: float = 1e-4
    patience: int = 15
    contrastive_weight: float = 0.05
    scales: int = 3
    seed: int = 2026


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")


def read_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(
        dataset_dir / "interaction.csv", header=None,
        names=["circRNA", "disease"], dtype=str,
    ).dropna().drop_duplicates().reset_index(drop=True)
    pairs["circRNA"] = pairs["circRNA"].str.strip()
    pairs["disease"] = pairs["disease"].str.strip()
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    seq = pd.read_csv(
        dataset_dir / "Get_circRNAsequence.csv", header=None,
        names=["circRNA", "sequence"], dtype=str, keep_default_na=False,
    )
    # Some source files may wrap a sequence onto continuation lines.
    records: List[Tuple[str, str]] = []
    current_id, current_seq = None, ""
    for row in seq.itertuples(index=False):
        left, right = str(row.circRNA).strip(), str(row.sequence).strip()
        if left and right:
            if current_id is not None:
                records.append((current_id, current_seq.upper()))
            current_id, current_seq = left, right
        elif left and current_id is not None:
            current_seq += left
    if current_id is not None:
        records.append((current_id, current_seq.upper()))
    seq_df = pd.DataFrame(records, columns=["circRNA", "sequence"]).drop_duplicates("circRNA")
    missing = sorted(set(pairs.circRNA) - set(seq_df.circRNA))
    if missing:
        raise ValueError(f"{dataset_dir.name}: {len(missing)} circRNAs have no sequence")
    return pairs, seq_df


def load_sequence_features(dataset_dir: Path, circs: Sequence[str]) -> np.ndarray:
    feature_file = dataset_dir / "circRNA_DNABERT.csv"
    if not feature_file.exists():
        raise FileNotFoundError(
            f"Missing {feature_file}. Run ncRNA_DNABERT.py --dataset-dir \"{dataset_dir}\" first."
        )
    df = pd.read_csv(feature_file, dtype={"circRNA": str})
    if "circRNA" not in df.columns:
        raw = pd.read_csv(feature_file, header=None)
        raw.columns = ["circRNA"] + [f"f{i}" for i in range(raw.shape[1] - 1)]
        df = raw
    df["circRNA"] = df["circRNA"].astype(str)
    table = df.set_index("circRNA")
    missing = [x for x in circs if x not in table.index]
    if missing:
        raise ValueError(f"Sequence features missing for {len(missing)} circRNAs")
    requested_k = os.environ.get("CDA_FEATURE_KMER")
    if requested_k:
        k = int(requested_k)
        dnabert_cols = [col for col in table.columns if str(col).startswith("dnabert_")]
        if not dnabert_cols:
            dnabert_cols = list(table.columns)
        dnabert = table.loc[list(circs), dnabert_cols].to_numpy(dtype=np.float32)
        _, seq_df = read_dataset(dataset_dir)
        seq_map = seq_df.set_index("circRNA").sequence.to_dict()
        code = {"A": 0, "C": 1, "G": 2, "T": 3}
        kmers = []
        for circ in circs:
            seq = "".join(ch for ch in str(seq_map[circ]).upper().replace("U", "T") if ch in code)
            comp = np.zeros(4 ** k, dtype=np.float32)
            if len(seq) >= k:
                circular = seq + seq[: k - 1]
                for start in range(len(seq)):
                    idx = 0
                    for char in circular[start : start + k]:
                        idx = idx * 4 + code[char]
                    comp[idx] += 1
                comp /= max(float(comp.sum()), 1.0)
            kmers.append(comp)
        kmer_x = np.stack(kmers)
        kmer_x = np.nan_to_num((kmer_x - kmer_x.mean(0, keepdims=True)) / (kmer_x.std(0, keepdims=True) + 1e-6))
        x = np.concatenate([dnabert, kmer_x], axis=1).astype(np.float32)
    else:
        x = table.loc[list(circs)].to_numpy(dtype=np.float32)
    # Fixed, label-independent normalization is safe to share across folds.
    mean, std = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    return np.nan_to_num((x - mean) / (std + 1e-6), copy=False)


def interaction_matrix(
    pairs: pd.DataFrame, circ2idx: Dict[str, int], disease2idx: Dict[str, int]
) -> np.ndarray:
    a = np.zeros((len(circ2idx), len(disease2idx)), dtype=np.float32)
    for c, d in pairs[["circRNA", "disease"]].itertuples(index=False, name=None):
        a[circ2idx[c], disease2idx[d]] = 1.0
    return a


def disease_gip(a_train: np.ndarray) -> Tuple[np.ndarray, float]:
    profiles = a_train.T
    sq_norm = np.sum(profiles * profiles, axis=1)
    nonzero = sq_norm[sq_norm > 0]
    gamma = 1.0 / float(nonzero.mean()) if len(nonzero) else 1.0
    dist2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * profiles @ profiles.T
    gip = np.exp(-gamma * np.maximum(dist2, 0.0)).astype(np.float32)
    return gip, gamma


def edge_index_from_matrix(a: np.ndarray, device: torch.device) -> torch.Tensor:
    c, d = np.nonzero(a)
    n_c = a.shape[0]
    src = np.concatenate([c, n_c + d, np.arange(a.shape[0] + a.shape[1])])
    dst = np.concatenate([n_c + d, c, np.arange(a.shape[0] + a.shape[1])])
    return torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=device)


def all_unknown_pairs(a_all: np.ndarray) -> np.ndarray:
    return np.argwhere(a_all == 0).astype(np.int64)


def sample_negatives(
    strategy: str,
    count: int,
    a_train: np.ndarray,
    a_all: np.ndarray,
    rng: np.random.Generator,
    forbidden: Iterable[Tuple[int, int]] = (),
) -> np.ndarray:
    candidates = all_unknown_pairs(a_all)
    forbidden_set = set(forbidden)
    if forbidden_set:
        keep = np.array([tuple(x) not in forbidden_set for x in candidates])
        candidates = candidates[keep]
    if count > len(candidates):
        raise ValueError("Not enough unknown pairs for balanced negative sampling")
    if strategy == "random":
        chosen = rng.choice(len(candidates), count, replace=False)
        return candidates[chosen]

    c_deg = a_train.sum(1)
    d_deg = a_train.sum(0)
    c_profiles = a_train
    d_profiles = a_train.T
    if strategy == "profile_dissimilar":
        # Faithful CDA adaptation of the supplied script: prefer endpoints with
        # weak interaction-profile evidence, with a tiny jitter for stable ties.
        score = c_deg[candidates[:, 0]] + d_deg[candidates[:, 1]]
        score += rng.random(len(score)) * 1e-6
        chosen = np.argpartition(score, count - 1)[:count]
        return candidates[chosen]
    if strategy.startswith("hybrid_"):
        try:
            if strategy.startswith("hybrid_adaptive_"):
                base_ratio = float(strategy.rsplit("_", 1)[1])
                mean_edges_per_node = float(a_train.sum()) / float(a_train.shape[0] + a_train.shape[1])
                # Normalize negative difficulty for graph density. Sparse graphs
                # retain the base ratio; denser graphs receive fewer easy pairs.
                reference_density = 0.80
                easy_ratio = min(
                    base_ratio,
                    base_ratio * math.sqrt(reference_density / max(mean_edges_per_node, reference_density)),
                )
            else:
                easy_ratio = float(strategy.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid hybrid strategy: {strategy}") from exc
        if not 0.0 <= easy_ratio <= 1.0:
            raise ValueError("Hybrid easy ratio must be in [0, 1]")
        score = c_deg[candidates[:, 0]] + d_deg[candidates[:, 1]]
        score = score + rng.random(len(score)) * 1e-6
        easy_count = int(round(count * easy_ratio))
        easy_idx = np.argpartition(score, max(easy_count - 1, 0))[:easy_count] if easy_count else np.empty(0, dtype=int)
        remaining_mask = np.ones(len(candidates), dtype=bool)
        remaining_mask[easy_idx] = False
        remaining_idx = np.flatnonzero(remaining_mask)
        random_count = count - easy_count
        random_idx = rng.choice(remaining_idx, random_count, replace=False) if random_count else np.empty(0, dtype=int)
        selected = np.concatenate([easy_idx, random_idx])
        rng.shuffle(selected)
        return candidates[selected]
    if strategy == "degree_matched":
        # A less biased alternative: match positive endpoint popularity instead
        # of filling the negative class with trivially isolated endpoints.
        score = (c_deg[candidates[:, 0]] + 1.0) * (d_deg[candidates[:, 1]] + 1.0)
        prob = score / score.sum()
        chosen = rng.choice(len(candidates), count, replace=False, p=prob)
        return candidates[chosen]
    raise ValueError(f"Unknown negative strategy: {strategy}")


class SparseGraphTransformerLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads, self.head_dim = heads, hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n, h, dh = x.shape[0], self.heads, self.head_dim
        q, k, v = self.qkv(x).view(n, 3, h, dh).unbind(1)
        src, dst = edge_index
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(dh)
        dst_h = dst[:, None].expand(-1, h)
        max_per_dst = torch.full((n, h), -torch.inf, device=x.device)
        max_per_dst.scatter_reduce_(0, dst_h, score, reduce="amax", include_self=True)
        exp_score = torch.exp(score - max_per_dst[dst])
        denom = torch.zeros((n, h), device=x.device)
        denom.scatter_add_(0, dst_h, exp_score)
        alpha = self.drop(exp_score / (denom[dst] + 1e-9))
        msg = v[src] * alpha.unsqueeze(-1)
        out = torch.zeros((n, h, dh), device=x.device)
        out.scatter_add_(0, dst_h.unsqueeze(-1).expand(-1, h, dh), msg)
        x = self.norm1(x + self.drop(self.out(out.flatten(1))))
        return self.norm2(x + self.drop(self.ffn(x)))


class MeanGraphLayer(nn.Module):
    """Attention-free control layer for the Transformer ablation.

    It preserves the number of graph layers, residual normalization and FFN,
    while replacing learned multi-head edge attention with uniform neighbor
    averaging. This isolates the contribution of Transformer attention.
    """
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.out = nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = torch.zeros_like(x)
        msg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        msg = msg / deg.clamp_min(1).unsqueeze(1)
        x = self.norm1(x + self.drop(self.out(msg)))
        return self.norm2(x + self.drop(self.ffn(x)))


class CDAEncoder(nn.Module):
    """Sequence/GIP type projections + adaptive multi-hop diffusion + GT."""

    def __init__(self, circ_dim: int, disease_dim: int, cfg: Config):
        super().__init__()
        h = cfg.hidden_dim
        self.circ_proj = nn.Sequential(nn.Linear(circ_dim, h, bias=False), nn.GELU(), nn.LayerNorm(h))
        self.dis_proj = nn.Sequential(nn.Linear(disease_dim, h), nn.GELU(), nn.LayerNorm(h))
        self.type_embedding = nn.Parameter(torch.empty(2, h))
        self.circ_base = nn.Parameter(torch.zeros(1, h))
        # Start conservatively: sequence evidence must earn its contribution.
        self.sequence_gate = nn.Parameter(torch.full((1, h), -2.0))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.scale_gate = nn.Sequential(nn.Linear(h, h // 2), nn.Tanh(), nn.Linear(h // 2, 1))
        self.scale_prior = nn.Parameter(torch.linspace(4.0, -2.0, cfg.scales))
        self.layers = nn.ModuleList(
            [SparseGraphTransformerLayer(h, cfg.heads, cfg.dropout) for _ in range(cfg.layers)]
        )
        self.layer_gates = nn.Parameter(torch.full((cfg.layers,), -2.0))
        self.scales = cfg.scales

    @staticmethod
    def propagate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return out / deg.clamp_min(1).unsqueeze(1)

    def forward(self, circ_x: torch.Tensor, dis_x: torch.Tensor, edge_index: torch.Tensor):
        c = self.circ_base + self.type_embedding[0] + torch.sigmoid(self.sequence_gate) * self.circ_proj(circ_x)
        d = self.dis_proj(dis_x) + self.type_embedding[1]
        x = torch.cat([c, d], 0)
        scales = [x]
        for _ in range(self.scales - 1):
            scales.append(self.propagate(scales[-1], edge_index))
        stack = torch.stack(scales, 1)
        weights = torch.softmax(self.scale_gate(stack) + self.scale_prior.view(1, -1, 1), 1)
        x = (stack * weights).sum(1)
        for gate, layer in zip(self.layer_gates, self.layers):
            updated = layer(x, edge_index)
            x = x + torch.sigmoid(gate) * (updated - x)
        return x, weights.squeeze(-1)


class CDAModel(nn.Module):
    def __init__(self, circ_dim: int, disease_dim: int, cfg: Config):
        super().__init__()
        self.encoder = CDAEncoder(circ_dim, disease_dim, cfg)
        h = cfg.hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(h * 4, h * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h * 2, h // 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h // 2, 1),
        )

    def pair_logits(self, z: torch.Tensor, pairs: torch.Tensor, n_c: int) -> torch.Tensor:
        zc, zd = z[pairs[:, 0]], z[n_c + pairs[:, 1]]
        pair_x = torch.cat([zc, zd, torch.abs(zc - zd), zc * zd], 1)
        return self.predictor(pair_x).squeeze(1)

    def forward(self, circ_x, dis_x, edge_index, pairs):
        z, weights = self.encoder(circ_x, dis_x, edge_index)
        return self.pair_logits(z, pairs, circ_x.shape[0]), z, weights


def metric_dict(
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
    require_two_predicted_classes: bool = False,
) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    if require_two_predicted_classes and np.unique(pred).size != 2:
        raise RuntimeError(
            f"Degenerate test predictions at validation-derived threshold={threshold:.6f}: "
            f"predicted_positive_rate={pred.mean():.6f}"
        )
    return {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
        "aupr": average_precision_score(y, prob),
    }


def select_validation_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Select a leakage-safe decision threshold on the validation split only.

    The objective gives Accuracy, Precision, Recall, and F1 equal weight. A
    candidate must predict both classes on the balanced validation set. AUC and
    AUPR are threshold-free and therefore are not part of threshold selection.
    """
    unique = np.unique(prob.astype(float))
    if len(unique) < 2:
        raise RuntimeError("Validation probabilities are constant; threshold calibration is impossible")
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2.0,
        np.nextafter(unique[-1], np.inf),
    ]
    best = None
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        positive_rate = float(pred.mean())
        if positive_rate <= 0.05 or positive_rate >= 0.95:
            continue
        values = (
            accuracy_score(y, pred),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0),
        )
        score = float(np.mean(values))
        candidate = (score, -abs(positive_rate - 0.5), -abs(float(threshold) - 0.5), float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No non-degenerate validation threshold candidate was found")
    return best[-1]


def pairs_and_labels(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator):
    pairs = np.vstack([pos, neg]).astype(np.int64)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    order = rng.permutation(len(labels))
    return pairs[order], labels[order]


def train_one(
    circ_features: np.ndarray,
    gip: np.ndarray,
    a_train: np.ndarray,
    train_pairs: np.ndarray,
    train_y: np.ndarray,
    val_pairs: np.ndarray,
    val_y: np.ndarray,
    cfg: Config,
    device: torch.device,
    ablation: str = "full", save_models: bool = False,
    exact_epochs: bool = False,
):
    seed_everything(cfg.seed)
    circ_x = torch.as_tensor(circ_features, dtype=torch.float32, device=device)
    dis_x = torch.as_tensor(gip, dtype=torch.float32, device=device)
    if ablation == "no_sequence":
        circ_x = torch.zeros_like(circ_x)
    if ablation == "no_gip":
        # Remove Gaussian interaction-profile features entirely.  An identity
        # matrix would leak a unique disease ID and make this an unfair
        # transductive feature replacement rather than a GIP ablation.
        dis_x = torch.zeros_like(dis_x)
    if ablation == "no_gip_structural":
        # GIP-free control with one training-fold-only structural signal:
        # normalized disease degree. This is intentionally reported as a
        # separate variant, not as the pure no_gip ablation.
        deg = torch.as_tensor(a_train.sum(0), dtype=torch.float32, device=device)
        deg = deg / deg.max().clamp_min(1.0)
        dis_x = torch.zeros_like(dis_x)
        dis_x[:, 0] = deg
    edge_index = edge_index_from_matrix(a_train, device)
    if ablation == "no_multiscale":
        model_cfg = replace(cfg, scales=1)
    elif ablation == "no_multiscale_no_graph_transformer":
        # Joint ablation of the two graph-structure enhancement stages:
        # retain node projections and the predictor, but remove diffusion
        # scales and all graph Transformer layers.
        model_cfg = replace(cfg, scales=1, layers=0)
    elif ablation in ("no_graph_transformer", "no_transformer_bypass", "no_graph_transformer_no_consistency"):
        # Legacy whole-layer removal is retained for audit compatibility.
        model_cfg = replace(cfg, layers=0)
    else:
        model_cfg = cfg
    model = CDAModel(circ_x.shape[1], dis_x.shape[1], model_cfg).to(device)
    if ablation == "no_transformer_attention":
        # Fair control: retain the configured depth and FFNs, but replace
        # multi-head attention by uniform neighbor aggregation.
        model.encoder.layers = nn.ModuleList(
            [MeanGraphLayer(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.layers)]
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_p = torch.as_tensor(train_pairs, dtype=torch.long, device=device)
    train_t = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    val_p = torch.as_tensor(val_pairs, dtype=torch.long, device=device)
    history, best_state, best_score, stale = [], None, -np.inf, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        logits, z, _ = model(circ_x, dis_x, edge_index, train_p)
        loss = F.binary_cross_entropy_with_logits(logits, train_t)
        if cfg.contrastive_weight > 0 and ablation not in ("no_consistency", "no_graph_transformer_no_consistency"):
            noisy_c = F.dropout(circ_x, p=0.1, training=True)
            noisy_d = F.dropout(dis_x, p=0.1, training=True)
            z2, _ = model.encoder(noisy_c, noisy_d, edge_index)
            consistency = 1.0 - F.cosine_similarity(z, z2, dim=1).mean()
            loss = loss + cfg.contrastive_weight * consistency
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(circ_x, dis_x, edge_index, val_p)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(val_y, val_prob)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_auc": val_auc})
        if val_auc > best_score + 1e-5:
            best_score, stale = val_auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if not exact_epochs and stale >= cfg.patience:
            break
    if not exact_epochs:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), circ_x, dis_x, edge_index, best_score


@torch.no_grad()
def predict(model, circ_x, dis_x, edge_index, pairs: np.ndarray):
    model.eval()
    p = torch.as_tensor(pairs, dtype=torch.long, device=circ_x.device)
    logits, z, weights = model(circ_x, dis_x, edge_index, p)
    return torch.sigmoid(logits).cpu().numpy(), z.cpu().numpy(), weights.cpu().numpy()


def indexed_pairs(df: pd.DataFrame, c2i, d2i) -> np.ndarray:
    return np.array([(c2i[c], d2i[d]) for c, d in df.itertuples(index=False, name=None)], dtype=np.int64)


def choose_hyperparameters(
    positives: pd.DataFrame,
    circ_features: np.ndarray,
    c2i, d2i, a_all, device, base: Config, strategy: str,
) -> Tuple[Config, pd.DataFrame]:
    # One fixed inner split makes every candidate directly comparable. Each
    # candidate rebuilds GIP/graph from the inner-training edges only.
    tr_df, va_df = train_test_split(positives, test_size=0.2, random_state=base.seed)
    a_inner = interaction_matrix(tr_df, c2i, d2i)
    gip, _ = disease_gip(a_inner)
    rng = np.random.default_rng(base.seed)
    tr_pos, va_pos = indexed_pairs(tr_df, c2i, d2i), indexed_pairs(va_df, c2i, d2i)
    tr_neg = sample_negatives(strategy, len(tr_pos), a_inner, a_all, rng)
    va_neg = sample_negatives(strategy, len(va_pos), a_inner, a_all, rng, map(tuple, tr_neg))
    tr_p, tr_y = pairs_and_labels(tr_pos, tr_neg, rng)
    va_p, va_y = pairs_and_labels(va_pos, va_neg, rng)
    candidates = (
        [replace(base, lr=x) for x in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]] +
        [replace(base, epochs=x) for x in [20, 40, 60, 80, 100]] +
        [replace(base, heads=x) for x in [1, 2, 4, 8, 16]] +
        [replace(base, layers=x) for x in [0, 1, 2, 3, 4]] +
        [replace(base, hidden_dim=x) for x in [64, 128, 256]] +
        [replace(base, dropout=x) for x in [0.1, 0.3, 0.5]] +
        [replace(base, weight_decay=x) for x in [1e-5, 1e-4, 1e-3]] +
        [replace(base, contrastive_weight=x) for x in [0.0, 0.05, 0.1]] +
        [replace(base, scales=x) for x in [1, 2, 3, 4]]
    )
    # Deduplicate the repeated baseline configuration.
    unique = {json.dumps(asdict(x), sort_keys=True): x for x in candidates}
    rows = []
    for idx, cfg in enumerate(unique.values(), 1):
        # Keep initialization/augmentation RNG identical across candidates so
        # the ranking reflects hyperparameters rather than a lucky seed.
        cfg = replace(cfg, seed=base.seed)
        model, hist, cx, dx, edges, score = train_one(
            circ_features, gip, a_inner, tr_p, tr_y, va_p, va_y, cfg, device
        )
        prob, _, _ = predict(model, cx, dx, edges, va_p)
        row = asdict(cfg) | metric_dict(va_y, prob) | {"best_epoch": int(hist.val_auc.idxmax() + 1)}
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    metric_cols = ["acc", "prec", "rec", "f1", "auc", "aupr"]
    result["composite_score"] = result[metric_cols].mean(axis=1)
    result = result.sort_values(["composite_score", "auc", "aupr"], ascending=False).reset_index(drop=True)
    top = result.iloc[0]
    best = Config(
        lr=float(top.lr), epochs=int(top.epochs), hidden_dim=int(top.hidden_dim),
        heads=int(top.heads), layers=int(top.layers), dropout=float(top.dropout),
        weight_decay=float(top.weight_decay), patience=int(top.patience),
        contrastive_weight=float(top.contrastive_weight), scales=int(top.scales),
        seed=base.seed,
    )
    return best, result


def run_fold(
    dataset_name: str,
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    positives: pd.DataFrame,
    circ_ids: Sequence[str], disease_ids: Sequence[str],
    circ_features: np.ndarray,
    c2i, d2i, a_all,
    cfg: Config, strategy: str, device: torch.device,
    output_dir: Path,
    ablation: str = "full", save_models: bool = False,
) -> Dict[str, float]:
    section = (
        "main" if ablation == "full" else
        "k_comparison" if ablation.startswith("kmer_") or ablation == "dnabert_only" else
        "hop_comparison" if ablation.startswith("hop_") or ablation.startswith("scale_") else
        "ablation"
    )
    fold_dir = output_dir / dataset_name / section / strategy / ablation / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    inner_train, val_df = train_test_split(train_df, test_size=0.15, random_state=cfg.seed + fold)
    a_train = interaction_matrix(inner_train, c2i, d2i)
    gip, gamma = disease_gip(a_train)
    rng = np.random.default_rng(cfg.seed + fold * 101)
    tr_pos, va_pos, te_pos = (indexed_pairs(x, c2i, d2i) for x in (inner_train, val_df, test_df))
    # Optional fixed per-fold negative pool.  The pool is sampled once from
    # training-only graph statistics and then partitioned deterministically;
    # this removes extra sampling variance while keeping test labels hidden.
    if strategy.startswith("fixed_pool_"):
        pool_strategy = strategy[len("fixed_pool_"):]
        pool = sample_negatives(pool_strategy, len(tr_pos) + len(va_pos) + len(te_pos), a_train, a_all, rng)
        rng.shuffle(pool)
        ntr, nva = len(tr_pos), len(va_pos)
        tr_neg, va_neg, te_neg = pool[:ntr], pool[ntr:ntr+nva], pool[ntr+nva:]
    else:
        tr_neg = sample_negatives(strategy, len(tr_pos), a_train, a_all, rng)
        va_neg = sample_negatives(strategy, len(va_pos), a_train, a_all, rng, map(tuple, tr_neg))
        forbidden = set(map(tuple, tr_neg)) | set(map(tuple, va_neg))
        te_neg = sample_negatives(strategy, len(te_pos), a_train, a_all, rng, forbidden)
    tr_p, tr_y = pairs_and_labels(tr_pos, tr_neg, rng)
    va_p, va_y = pairs_and_labels(va_pos, va_neg, rng)
    te_p, te_y = pairs_and_labels(te_pos, te_neg, rng)

    model, history, cx, dx, edges, _ = train_one(
        circ_features, gip, a_train, tr_p, tr_y, va_p, va_y,
        replace(cfg, seed=cfg.seed + fold), device, ablation,
        exact_epochs=ablation.startswith("param_epochs_"),
    )
    if save_models:
        torch.save(model.state_dict(), fold_dir / "model_state.pt")
    train_prob, _, _ = predict(model, cx, dx, edges, tr_p)
    val_prob, _, _ = predict(model, cx, dx, edges, va_p)
    test_prob, embeddings, scale_weights = predict(model, cx, dx, edges, te_p)
    threshold = select_validation_threshold(va_y, val_prob)
    metrics = metric_dict(
        te_y, test_prob, threshold=threshold, require_two_predicted_classes=True
    )

    def readable(pairs, labels, probs=None):
        frame = pd.DataFrame({
            "circRNA": [circ_ids[i] for i in pairs[:, 0]],
            "disease": [disease_ids[j] for j in pairs[:, 1]],
            "label": labels.astype(int),
        })
        if probs is not None:
            frame["probability"] = probs
            frame["prediction"] = (probs >= threshold).astype(int)
            frame["decision_threshold"] = threshold
        return frame

    save_csv(readable(tr_p, tr_y, train_prob), fold_dir / "train_predictions.csv")
    save_csv(readable(va_p, va_y, val_prob), fold_dir / "validation_predictions.csv")
    save_csv(readable(te_p, te_y, test_prob), fold_dir / "test_predictions.csv")
    save_csv(history, fold_dir / "epoch_history.csv")
    save_csv(pd.DataFrame([asdict(cfg) | {"gip_gamma": gamma, "ablation": ablation, "decision_threshold": threshold}]), fold_dir / "config.csv")
    save_csv(pd.DataFrame(gip, index=disease_ids, columns=disease_ids).reset_index(names="disease"), fold_dir / "disease_gip.csv")
    node_ids = list(circ_ids) + list(disease_ids)
    node_types = ["circRNA"] * len(circ_ids) + ["disease"] * len(disease_ids)
    emb_df = pd.DataFrame(embeddings, columns=[f"z{i}" for i in range(embeddings.shape[1])])
    save_csv(pd.concat([pd.DataFrame({"node_id": node_ids, "node_type": node_types}), emb_df], axis=1), fold_dir / "node_embeddings.csv")
    # Report scales with one-based labels.  The original attribute view is
    # scale_1; a two-hop model therefore exposes scale_1..scale_3.
    sw = pd.DataFrame(scale_weights, columns=[f"scale_{i + 1}" for i in range(scale_weights.shape[1])])
    save_csv(pd.concat([pd.DataFrame({"node_id": node_ids, "node_type": node_types}), sw], axis=1), fold_dir / "scale_attention.csv")
    save_csv(pd.DataFrame([{"fold": fold, "decision_threshold": threshold, **metrics}]), fold_dir / "metrics.csv")
    save_csv(train_df.assign(split="outer_train"), fold_dir / "outer_train_positives.csv")
    save_csv(test_df.assign(split="outer_test"), fold_dir / "outer_test_positives.csv")
    return {"fold": fold, **metrics}


def summarize(rows: List[Dict[str, float]], extra: Dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    records = []
    for metric in METRICS:
        records.append(extra | {
            "metric": metric,
            "mean": frame[metric].mean(),
            "std": frame[metric].std(ddof=1),
        })
    return pd.DataFrame(records)


def five_fold_table(rows: List[Dict[str, float]]) -> pd.DataFrame:
    """Return the exact fold/Average/Std layout requested for reporting."""
    frame = pd.DataFrame(rows).sort_values("fold")
    mapping = {
        "Accuracy": "acc", "Precision": "prec", "Recall": "rec",
        "F1": "f1", "AUC": "auc", "AUPR": "aupr",
    }
    out = pd.DataFrame({"Fold": frame["fold"].astype(int).astype(str)})
    for display, source in mapping.items():
        out[display] = frame[source].to_numpy()
    average = {"Fold": "Average", **{display: frame[source].mean() for display, source in mapping.items()}}
    std = {"Fold": "Std", **{display: frame[source].std(ddof=1) for display, source in mapping.items()}}
    return pd.concat([out, pd.DataFrame([average, std])], ignore_index=True)


def run_dataset(dataset_dir: Path, out: Path, args, device: torch.device):
    name = dataset_dir.name
    positives, seq_df = read_dataset(dataset_dir)
    circ_ids = sorted(positives.circRNA.unique())
    disease_ids = sorted(positives.disease.unique())
    c2i = {x: i for i, x in enumerate(circ_ids)}
    d2i = {x: i for i, x in enumerate(disease_ids)}
    circ_features = load_sequence_features(dataset_dir, circ_ids)
    a_all = interaction_matrix(positives, c2i, d2i)
    base = Config(epochs=args.epochs, seed=args.seed)
    feature_df = pd.DataFrame(circ_features, columns=[f"x{i}" for i in range(circ_features.shape[1])])
    save_csv(pd.concat([pd.DataFrame({"circRNA": circ_ids}), feature_df], axis=1), out / name / "normalized_circRNA_features.csv")
    save_csv(pd.DataFrame([{
        "dataset": name, "positive_pairs": len(positives), "circRNAs": len(circ_ids),
        "diseases": len(disease_ids), "sequence_feature_dim": circ_features.shape[1],
    }]), out / name / "dataset_statistics.csv")
    print(f"[{name}] tuning on {len(positives)} associations; device={device}")
    config_path = out / name / "selected_config.csv"
    if args.skip_tuning and config_path.exists():
        row = pd.read_csv(config_path).iloc[0]
        best = Config(
            lr=float(row.lr), epochs=int(row.epochs), hidden_dim=int(row.hidden_dim),
            heads=int(row.heads), layers=int(row.layers), dropout=float(row.dropout),
            weight_decay=float(row.weight_decay), patience=int(row.patience),
            contrastive_weight=float(row.contrastive_weight), scales=int(row.scales), seed=args.seed,
        )
    else:
        best, tuning = choose_hyperparameters(
            positives, circ_features, c2i, d2i, a_all, device, base, args.primary_negative
        )
        save_csv(tuning, out / name / "tuning_results.csv")
    save_csv(pd.DataFrame([asdict(best)]), out / name / "selected_config.csv")
    save_csv(pd.DataFrame([{"primary_negative_strategy": args.primary_negative}]), out / name / "primary_protocol.csv")
    if args.tune_only:
        return

    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    split_indices = list(kfold.split(positives))
    all_summaries, all_fold_rows = [], []
    strategies = list(dict.fromkeys([args.primary_negative] if args.case_study_only else ["random", "degree_matched", args.primary_negative]))
    for strategy in strategies:
        rows = []
        for fold, (tr, te) in enumerate(split_indices, 1):
            print(f"[{name}] negative={strategy} fold={fold}/5")
            row = run_fold(
                name, fold, positives.iloc[tr].reset_index(drop=True), positives.iloc[te].reset_index(drop=True),
                positives, circ_ids, disease_ids, circ_features, c2i, d2i, a_all,
                best, strategy, device, out, save_models=args.save_models,
            )
            rows.append(row)
            all_fold_rows.append({"experiment": "negative_sampling", "variant": strategy, **row})
        all_summaries.append(summarize(rows, {"experiment": "negative_sampling", "variant": strategy}))
        save_csv(five_fold_table(rows), out / name / "main" / strategy / "full" / "five_fold_table.csv")

    ablation_variants = ["no_sequence", "no_gip", "no_gip_structural", "no_multiscale", "no_graph_transformer", "no_transformer_bypass", "no_transformer_attention", "no_multiscale_no_graph_transformer", "no_graph_transformer_no_consistency", "no_consistency"]
    if getattr(args, "only_ablation", None):
        ablation_variants = list(args.only_ablation)
    for ablation in ([] if args.case_study_only else ablation_variants):
        rows = []
        for fold, (tr, te) in enumerate(split_indices, 1):
            print(f"[{name}] ablation={ablation} fold={fold}/5")
            row = run_fold(
                name, fold, positives.iloc[tr].reset_index(drop=True), positives.iloc[te].reset_index(drop=True),
                positives, circ_ids, disease_ids, circ_features, c2i, d2i, a_all,
                best, args.primary_negative, device, out, ablation, args.save_models,
            )
            rows.append(row)
            all_fold_rows.append({"experiment": "ablation", "variant": ablation, **row})
        all_summaries.append(summarize(rows, {"experiment": "ablation", "variant": ablation}))
        save_csv(five_fold_table(rows), out / name / "ablation" / args.primary_negative / ablation / "five_fold_table.csv")
    save_csv(pd.DataFrame(all_fold_rows), out / name / "all_fold_metrics.csv")
    save_csv(pd.concat(all_summaries, ignore_index=True), out / name / "summary_mean_std.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--primary-negative", default="hybrid_adaptive_0.82")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tune-only", action="store_true")
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--case-study-only", action="store_true")
    p.add_argument("--only-ablation", nargs="+", default=None, choices=["no_sequence", "no_gip", "no_gip_structural", "no_multiscale", "no_graph_transformer", "no_transformer_bypass", "no_transformer_attention", "no_multiscale_no_graph_transformer", "no_graph_transformer_no_consistency", "no_consistency"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dirs = [p for p in sorted(args.dataset_root.iterdir()) if p.is_dir()]
    if args.datasets:
        dirs = [p for p in dirs if p.name in set(args.datasets)]
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset_dir in dirs:
        run_dataset(dataset_dir, args.output, args, device)


if __name__ == "__main__":
    main()
