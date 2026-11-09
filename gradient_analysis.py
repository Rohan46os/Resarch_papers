"""Gradient Interference Analysis for Standard and Shared-Q Causal Self-Attention.

Investigates whether sharing Query and Key projections introduces gradient interference
because both Query-role and Key-role gradients update the same Wq parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, learning_rate_at
from dataset import CharacterDataset
from model import DecoderLM, build_matched_model, build_paired_models


ANALYSIS_STEPS = (100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000)
ANALYSIS_SEEDS = (1337, 2024, 42)

CSV_COLUMNS = [
    "architecture",
    "seed",
    "training_step",
    "layer",
    "head",
    "query_gradient_norm",
    "key_gradient_norm",
    "total_gradient_norm",
    "cosine_similarity",
    "cancellation_ratio",
    "query_gradient_ratio",
    "key_gradient_ratio",
]

CHECKPOINT_CSV_COLUMNS = [
    "architecture",
    "seed",
    "checkpoint",
    "batch_idx",
    "training_step",
    "layer",
    "head",
    "query_gradient_norm",
    "key_gradient_norm",
    "total_gradient_norm",
    "cosine_similarity",
    "cancellation_ratio",
    "query_gradient_ratio",
    "key_gradient_ratio",
]


def compute_attention_gradients_manual(
    x_norm: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    U: torch.Tensor,
    head_dim: int,
    is_shared_q: bool = True,
    dropout_p: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Calculate exact analytical gradients for attention projections.

    Mathematical derivation:
      Q = X Wq, K = X Wk, V = X Wv (or K=Q in Shared-Q)
      S = Q K^T / sqrt(dk)
      P = softmax(S_masked)
      Y = P V
      U = dL/dY
      R = U V^T
      G = P * (R - sum(R * P, dim=-1, keepdim=True))  [masked positions zeroed]
      g_query = G K / sqrt(dk)
      g_key = G^T Q / sqrt(dk)
      g_v = P^T U
      grad_query_Wq = g_query_flat^T @ X_flat
      grad_key_Wq = g_key_flat^T @ X_flat
      grad_total_Wq = grad_query_Wq + grad_key_Wq (for Shared-Q)
      grad_Wv = g_v_flat^T @ X_flat

    Args:
        x_norm: (B, T, D) LayerNorm-normalized input tensor to attention
        q: (B, H, T, dk) Query tensor
        k: (B, H, T, dk) Key tensor
        v: (B, H, T, dv) Value tensor
        U: (B, H, T, dv) Gradient of loss with respect to attended output Y
        head_dim: Query head dimension dk
        is_shared_q: Whether K is exactly Q
        dropout_p: Attention dropout probability (0.0 during analysis/eval)

    Returns:
        Dictionary containing projection gradients:
          - grad_query_Wq: (H * dk, D)
          - grad_key_Wq: (H * dk, D)
          - grad_total_Wq: (H * dk, D)
          - grad_Wv: (H * dv, D)
          - grad_Wk: (H * dk, D) [for standard attention, equals grad_key_Wq]
          - intermediate tensors (P, G, R) for inspection
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    D = x_norm.shape[-1]

    # 1. R = U @ V^T -> (B, H, T, T)
    R = U @ v.transpose(-2, -1)

    # 2. Recompute P = softmax(masked_scores)
    raw_scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    future = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(diagonal=1)
    masked_scores = raw_scores.masked_fill(future, float("-inf"))
    P = F.softmax(masked_scores, dim=-1)

    # 3. Softmax gradient: G = P * (R - sum(R * P, dim=-1, keepdim=True))
    G = P * (R - (R * P).sum(dim=-1, keepdim=True))
    # Ensure masked positions have exact zero gradient
    G = torch.where(future, torch.zeros_like(G), G)

    # 4. Gradients on intermediate Q, K, V
    # Query role: g_query = G @ K / sqrt(dk) -> (B, H, T, dk)
    g_query = (G @ k) / math.sqrt(head_dim)
    # Key role: g_key = G^T @ Q / sqrt(dk) -> (B, H, T, dk)
    g_key = (G.transpose(-2, -1) @ q) / math.sqrt(head_dim)
    # Value role: g_v = P^T @ U -> (B, H, T, dv)
    g_v = P.transpose(-2, -1) @ U

    # 5. Projection weight gradients respecting PyTorch orientation [D_out, D_in]:
    # grad_W = g_flat^T @ x_flat
    x_flat = x_norm.reshape(B * T, D)
    g_query_flat = g_query.transpose(1, 2).contiguous().reshape(B * T, H * dk)
    g_key_flat = g_key.transpose(1, 2).contiguous().reshape(B * T, H * dk)
    g_v_flat = g_v.transpose(1, 2).contiguous().reshape(B * T, H * dv)

    grad_query_Wq = g_query_flat.T @ x_flat  # (H * dk, D)
    grad_key_Wq = g_key_flat.T @ x_flat      # (H * dk, D)
    grad_total_Wq = grad_query_Wq + grad_key_Wq
    grad_Wv = g_v_flat.T @ x_flat            # (H * dv, D)

    result = {
        "grad_query_Wq": grad_query_Wq,
        "grad_key_Wq": grad_key_Wq,
        "grad_total_Wq": grad_total_Wq,
        "grad_Wv": grad_Wv,
        "P": P,
        "G": G,
        "R": R,
    }
    if not is_shared_q:
        result["grad_Wk"] = grad_key_Wq
    return result


def compute_gradient_interference_metrics(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_total: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Calculate gradient cosine similarity, norms, cancellation, and ratios.

    Args:
        grad_query: Query-role weight gradient tensor
        grad_key: Key-role weight gradient tensor
        grad_total: Optional combined gradient (defaults to grad_query + grad_key)
        eps: Epsilon for numerical stability

    Returns:
        Dictionary of scalar metrics.
    """
    if grad_total is None:
        grad_total = grad_query + grad_key

    norm_query = float(torch.linalg.norm(grad_query).item())
    norm_key = float(torch.linalg.norm(grad_key).item())
    norm_total = float(torch.linalg.norm(grad_total).item())

    dot_prod = float(torch.sum(grad_query * grad_key).item())
    denom_cos = norm_query * norm_key + eps
    cosine_sim = dot_prod / denom_cos
    cosine_sim = max(-1.0, min(1.0, cosine_sim))

    denom_cancel = norm_query + norm_key + eps
    cancellation_ratio = 1.0 - (norm_total / denom_cancel)
    # Numerical bounds: cancellation ratio in [0, 1]
    cancellation_ratio = max(0.0, min(1.0, cancellation_ratio))

    query_gradient_ratio = norm_query / denom_cancel
    key_gradient_ratio = norm_key / denom_cancel

    return {
        "query_gradient_norm": norm_query,
        "key_gradient_norm": norm_key,
        "total_gradient_norm": norm_total,
        "cosine_similarity": cosine_sim,
        "cancellation_ratio": cancellation_ratio,
        "query_gradient_ratio": query_gradient_ratio,
        "key_gradient_ratio": key_gradient_ratio,
    }


class LayerGradientHook:
    """Hooks into a single DecoderBlock to capture inputs and gradients."""

    def __init__(self, block_index: int, block: nn.Module):
        self.block_index = block_index
        self.block = block
        self.x_norm: Optional[torch.Tensor] = None
        self.out_proj_inp: Optional[torch.Tensor] = None
        self.U_flat: Optional[torch.Tensor] = None
        self.handles = []
        self._register()

    def _register(self):
        def attn_fwd_hook(module, inp, out):
            self.x_norm = inp[0].detach()

        def out_proj_fwd_hook(module, inp, out):
            tensor = inp[0]
            self.out_proj_inp = tensor

            def bwd_hook(grad):
                self.U_flat = grad.detach()

            tensor.register_hook(bwd_hook)

        self.handles.append(self.block.attention.register_forward_hook(attn_fwd_hook))
        self.handles.append(self.block.attention.out_proj.register_forward_hook(out_proj_fwd_hook))

    def clear(self):
        self.x_norm = None
        self.out_proj_inp = None
        self.U_flat = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class ModelGradientAnalyzer:
    """Manages gradient analysis hooks across all layers of a DecoderLM."""

    def __init__(self, model: DecoderLM):
        self.model = model
        self.layer_hooks = [LayerGradientHook(i, blk) for i, blk in enumerate(model.blocks)]

    def clear(self):
        for hook in self.layer_hooks:
            hook.clear()

    def remove(self):
        for hook in self.layer_hooks:
            hook.remove()

    def analyze(self) -> Dict[str, Any]:
        """Compute layer-wise and head-wise gradient interference metrics from the last backward pass."""
        results = {
            "layers": {},
            "model_wide": {},
        }
        is_shared_q = self.model.attention_type != "standard"

        all_query_grads = []
        all_key_grads = []
        all_total_grads = []
        all_v_grads = []

        for l, hook in enumerate(self.layer_hooks):
            if hook.x_norm is None or hook.U_flat is None:
                raise RuntimeError(f"Layer {l} did not capture activations or gradients. Run backward() first.")

            blk = self.model.blocks[l]
            attn = blk.attention
            x_norm = hook.x_norm
            B, T, D = x_norm.shape
            H = attn.n_heads
            dk = attn.head_dim
            dv = getattr(attn, "value_head_dim", dk)

            # Recompute q, k, v using current weights
            Wq = attn.q_proj.weight
            Wv = attn.v_proj.weight
            q = (x_norm @ Wq.T).reshape(B, T, H, dk).transpose(1, 2)
            if is_shared_q:
                k = q
            else:
                Wk = attn.k_proj.weight
                k = (x_norm @ Wk.T).reshape(B, T, H, dk).transpose(1, 2)
            v = (x_norm @ Wv.T).reshape(B, T, H, dv).transpose(1, 2)

            U = hook.U_flat.reshape(B, T, H, dv).transpose(1, 2)

            grads = compute_attention_gradients_manual(
                x_norm=x_norm,
                q=q,
                k=k,
                v=v,
                U=U,
                head_dim=dk,
                is_shared_q=is_shared_q,
            )

            grad_query_Wq = grads["grad_query_Wq"]
            grad_key_Wq = grads["grad_key_Wq"]
            grad_total_Wq = grads["grad_total_Wq"]
            grad_Wv = grads["grad_Wv"]

            # Layer-level metrics across the complete Wq matrix
            layer_metrics = compute_gradient_interference_metrics(
                grad_query_Wq, grad_key_Wq, grad_total_Wq
            )
            layer_metrics["value_gradient_norm"] = float(torch.linalg.norm(grad_Wv).item())
            if not is_shared_q:
                layer_metrics["key_projection_gradient_norm"] = float(torch.linalg.norm(grads["grad_Wk"]).item())

            # Per-head metrics
            heads_metrics = {}
            for h in range(H):
                q_head = grad_query_Wq[h * dk : (h + 1) * dk, :]
                k_head = grad_key_Wq[h * dk : (h + 1) * dk, :]
                tot_head = grad_total_Wq[h * dk : (h + 1) * dk, :]
                head_m = compute_gradient_interference_metrics(q_head, k_head, tot_head)
                heads_metrics[h] = head_m

            results["layers"][l] = {
                "layer_metrics": layer_metrics,
                "heads": heads_metrics,
            }

            all_query_grads.append(grad_query_Wq.reshape(-1))
            all_key_grads.append(grad_key_Wq.reshape(-1))
            all_total_grads.append(grad_total_Wq.reshape(-1))
            all_v_grads.append(grad_Wv.reshape(-1))

        # Model-wide aggregated metrics across all layers
        cat_q = torch.cat(all_query_grads)
        cat_k = torch.cat(all_key_grads)
        cat_tot = torch.cat(all_total_grads)
        cat_v = torch.cat(all_v_grads)
        model_metrics = compute_gradient_interference_metrics(cat_q, cat_k, cat_tot)
        model_metrics["value_gradient_norm"] = float(torch.linalg.norm(cat_v).item())
        results["model_wide"] = model_metrics

        return results


def run_single_batch_analysis(
    model: DecoderLM,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute a single gradient analysis pass in eval mode on a given batch."""
    was_training = model.training
    model.eval()
    analyzer = ModelGradientAnalyzer(model)

    try:
        model.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        loss.backward()
        analysis = analyzer.analyze()
        analysis["loss"] = float(loss.item())
        model.zero_grad(set_to_none=True)
        return analysis
    finally:
        analyzer.remove()
        model.train(was_training)


def load_model_for_analysis(
    architecture: str,
    seed: int,
    config: Config,
    checkpoint_path: Path,
    vocab_size: int,
    device: torch.device,
) -> DecoderLM:
    """Instantiate model and load checkpoint weights strictly."""
    if architecture == "shared_q_query_heavy":
        model = build_matched_model(config, vocab_size, seed, "shared_q_query_heavy")
    elif architecture == "shared_q":
        models = build_paired_models(config, vocab_size, seed)
        model = models["shared_q"]
    elif architecture == "standard":
        models = build_paired_models(config, vocab_size, seed)
        model = models["standard"]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def run_checkpoint_analysis(
    dataset: CharacterDataset,
    config: Config,
    device: torch.device,
    output_dir: Path,
    num_batches: int = 20,
    batch_seed: int = 918273,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate trained checkpoints on fixed batches and record conflict frequencies."""
    print("=== Running Checkpoint Gradient Analysis ===", flush=True)
    checkpoint_rows: List[Dict[str, Any]] = []
    conflict_stats: Dict[str, Any] = {}

    architectures = [
        ("standard", "outputs/full"),
        ("shared_q", "outputs/full"),
        ("shared_q_query_heavy", "outputs/model_c3"),
    ]
    checkpoints = ["final", "best"]

    # Fixed batches for deterministic comparison
    eval_batches = list(dataset.evaluation_batches("validation", batch_seed))[:num_batches]
    vocab_size = len(dataset.vocab)

    for arch, base_dir in architectures:
        for seed in ANALYSIS_SEEDS:
            for ckpt_name in checkpoints:
                ckpt_path = Path(base_dir) / f"seed_{seed}" / arch / f"{ckpt_name}.pt"
                if not ckpt_path.exists():
                    print(f"Skipping missing checkpoint: {ckpt_path}", flush=True)
                    continue

                model = load_model_for_analysis(arch, seed, config, ckpt_path, vocab_size, device)
                print(f"Evaluating checkpoint: {arch} seed={seed} ckpt={ckpt_name}", flush=True)

                # Store cosine similarities for frequency statistics
                key_prefix = f"{arch}_seed{seed}_{ckpt_name}"
                cos_per_layer: Dict[int, List[float]] = {l: [] for l in range(config.n_layers)}
                cos_per_head: Dict[Tuple[int, int], List[float]] = {
                    (l, h): [] for l in range(config.n_layers) for h in range(config.n_heads)
                }

                for b_idx, (x, y) in enumerate(eval_batches):
                    analysis = run_single_batch_analysis(model, x, y, device)

                    # Model-wide row
                    mw = analysis["model_wide"]
                    checkpoint_rows.append({
                        "architecture": arch,
                        "seed": seed,
                        "checkpoint": ckpt_name,
                        "batch_idx": b_idx,
                        "training_step": "checkpoint",
                        "layer": "all",
                        "head": "all",
                        "query_gradient_norm": mw["query_gradient_norm"],
                        "key_gradient_norm": mw["key_gradient_norm"],
                        "total_gradient_norm": mw["total_gradient_norm"],
                        "cosine_similarity": mw["cosine_similarity"],
                        "cancellation_ratio": mw["cancellation_ratio"],
                        "query_gradient_ratio": mw["query_gradient_ratio"],
                        "key_gradient_ratio": mw["key_gradient_ratio"],
                    })

                    # Layer and head rows
                    for l, l_data in analysis["layers"].items():
                        lm = l_data["layer_metrics"]
                        cos_per_layer[l].append(lm["cosine_similarity"])
                        checkpoint_rows.append({
                            "architecture": arch,
                            "seed": seed,
                            "checkpoint": ckpt_name,
                            "batch_idx": b_idx,
                            "training_step": "checkpoint",
                            "layer": l,
                            "head": "all",
                            "query_gradient_norm": lm["query_gradient_norm"],
                            "key_gradient_norm": lm["key_gradient_norm"],
                            "total_gradient_norm": lm["total_gradient_norm"],
                            "cosine_similarity": lm["cosine_similarity"],
                            "cancellation_ratio": lm["cancellation_ratio"],
                            "query_gradient_ratio": lm["query_gradient_ratio"],
                            "key_gradient_ratio": lm["key_gradient_ratio"],
                        })

                        for h, hm in l_data["heads"].items():
                            cos_per_head[(l, h)].append(hm["cosine_similarity"])
                            checkpoint_rows.append({
                                "architecture": arch,
                                "seed": seed,
                                "checkpoint": ckpt_name,
                                "batch_idx": b_idx,
                                "training_step": "checkpoint",
                                "layer": l,
                                "head": h,
                                "query_gradient_norm": hm["query_gradient_norm"],
                                "key_gradient_norm": hm["key_gradient_norm"],
                                "total_gradient_norm": hm["total_gradient_norm"],
                                "cosine_similarity": hm["cosine_similarity"],
                                "cancellation_ratio": hm["cancellation_ratio"],
                                "query_gradient_ratio": hm["query_gradient_ratio"],
                                "key_gradient_ratio": hm["key_gradient_ratio"],
                            })

                # Compute frequency statistics across analyzed batches
                layer_stats = {}
                for l, vals in cos_per_layer.items():
                    arr = np.array(vals)
                    layer_stats[l] = {
                        "pct_negative": float(np.mean(arr < -1e-4) * 100),
                        "pct_positive": float(np.mean(arr > 1e-4) * 100),
                        "pct_zero": float(np.mean(np.abs(arr) <= 1e-4) * 100),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
                    }

                head_stats = {}
                for (l, h), vals in cos_per_head.items():
                    arr = np.array(vals)
                    head_stats[f"L{l}_H{h}"] = {
                        "layer": l,
                        "head": h,
                        "pct_negative": float(np.mean(arr < -1e-4) * 100),
                        "pct_positive": float(np.mean(arr > 1e-4) * 100),
                        "pct_zero": float(np.mean(np.abs(arr) <= 1e-4) * 100),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
                    }

                conflict_stats[key_prefix] = {
                    "architecture": arch,
                    "seed": seed,
                    "checkpoint": ckpt_name,
                    "layer_stats": layer_stats,
                    "head_stats": head_stats,
                }
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Save checkpoint CSV
    ckpt_csv_path = output_dir / "checkpoint_gradient_metrics.csv"
    with ckpt_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    print(f"Saved: {ckpt_csv_path}", flush=True)

    conflict_json_path = output_dir / "conflict_frequency_metrics.json"
    conflict_json_path.write_text(json.dumps(conflict_stats, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {conflict_json_path}", flush=True)

    return checkpoint_rows, conflict_stats


def run_training_trajectory_analysis(
    dataset: CharacterDataset,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replay exact training trajectories and collect gradient statistics at requested update intervals."""
    print("=== Running Training Trajectory Gradient Analysis ===", flush=True)
    gradient_metrics_rows: List[Dict[str, Any]] = []
    layerwise_rows: List[Dict[str, Any]] = []
    headwise_rows: List[Dict[str, Any]] = []

    vocab_size = len(dataset.vocab)
    architectures = ["standard", "shared_q", "shared_q_query_heavy"]

    for arch in architectures:
        for seed in ANALYSIS_SEEDS:
            print(f"\n--- Running Trajectory: {arch} (seed={seed}) ---", flush=True)
            seed_config = config
            random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            if arch == "shared_q_query_heavy":
                model = build_matched_model(seed_config, vocab_size, seed, "shared_q_query_heavy")
            elif arch == "shared_q":
                models = build_paired_models(seed_config, vocab_size, seed)
                model = models["shared_q"]
            else:
                models = build_paired_models(seed_config, vocab_size, seed)
                model = models["standard"]

            model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=seed_config.learning_rate,
                betas=tuple(seed_config.adam_betas),
                eps=seed_config.adam_eps,
                weight_decay=seed_config.weight_decay,
                foreach=False,
            )
            sampler = torch.Generator().manual_seed(seed)
            start_time = time.perf_counter()

            for step in range(1, seed_config.max_steps + 1):
                x, y = dataset.batch("train", sampler)
                lr = learning_rate_at(step, seed_config)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                if step in ANALYSIS_STEPS:
                    # Execute separate analysis pass without changing parameters or optimizer state
                    analysis = run_single_batch_analysis(model, x, y, device)

                    # Model-wide row
                    mw = analysis["model_wide"]
                    gradient_metrics_rows.append({
                        "architecture": arch,
                        "seed": seed,
                        "training_step": step,
                        "layer": "all",
                        "head": "all",
                        "query_gradient_norm": mw["query_gradient_norm"],
                        "key_gradient_norm": mw["key_gradient_norm"],
                        "total_gradient_norm": mw["total_gradient_norm"],
                        "cosine_similarity": mw["cosine_similarity"],
                        "cancellation_ratio": mw["cancellation_ratio"],
                        "query_gradient_ratio": mw["query_gradient_ratio"],
                        "key_gradient_ratio": mw["key_gradient_ratio"],
                    })

                    # Layer-wise rows
                    for l, l_data in analysis["layers"].items():
                        lm = l_data["layer_metrics"]
                        layerwise_rows.append({
                            "architecture": arch,
                            "seed": seed,
                            "training_step": step,
                            "layer": l,
                            "head": "all",
                            "query_gradient_norm": lm["query_gradient_norm"],
                            "key_gradient_norm": lm["key_gradient_norm"],
                            "total_gradient_norm": lm["total_gradient_norm"],
                            "cosine_similarity": lm["cosine_similarity"],
                            "cancellation_ratio": lm["cancellation_ratio"],
                            "query_gradient_ratio": lm["query_gradient_ratio"],
                            "key_gradient_ratio": lm["key_gradient_ratio"],
                        })

                        # Head-wise rows
                        for h, hm in l_data["heads"].items():
                            headwise_rows.append({
                                "architecture": arch,
                                "seed": seed,
                                "training_step": step,
                                "layer": l,
                                "head": h,
                                "query_gradient_norm": hm["query_gradient_norm"],
                                "key_gradient_norm": hm["key_gradient_norm"],
                                "total_gradient_norm": hm["total_gradient_norm"],
                                "cosine_similarity": hm["cosine_similarity"],
                                "cancellation_ratio": hm["cancellation_ratio"],
                                "query_gradient_ratio": hm["query_gradient_ratio"],
                                "key_gradient_ratio": hm["key_gradient_ratio"],
                            })

                    print(f"step={step:4d} cos={mw['cosine_similarity']:+.4f} "
                          f"cancel={mw['cancellation_ratio']:.4f} "
                          f"q_norm={mw['query_gradient_norm']:.2f} k_norm={mw['key_gradient_norm']:.2f}",
                          flush=True)

                # Standard training update step
                model.train()
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x.to(device), y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), seed_config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()

            elapsed = time.perf_counter() - start_time
            print(f"Trajectory {arch} seed={seed} completed in {elapsed:.1f}s", flush=True)
            optimizer.zero_grad(set_to_none=True)
            del optimizer, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save CSV outputs
    g_path = output_dir / "gradient_metrics.csv"
    with g_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(gradient_metrics_rows)
    print(f"Saved: {g_path}", flush=True)

    l_path = output_dir / "layerwise_gradient_metrics.csv"
    with l_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(layerwise_rows)
    print(f"Saved: {l_path}", flush=True)

    h_path = output_dir / "headwise_gradient_metrics.csv"
    with h_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(headwise_rows)
    print(f"Saved: {h_path}", flush=True)

    return gradient_metrics_rows, layerwise_rows, headwise_rows


def generate_gradient_plots(output_dir: Path):
    """Generate all 10 requested publication-quality research figures."""
    print("=== Generating Gradient Analysis Figures ===", flush=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV data
    gm_path = output_dir / "gradient_metrics.csv"
    lm_path = output_dir / "layerwise_gradient_metrics.csv"
    hm_path = output_dir / "headwise_gradient_metrics.csv"
    cm_path = output_dir / "checkpoint_gradient_metrics.csv"

    def read_csv_rows(p):
        with open(p, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    gm_rows = read_csv_rows(gm_path) if gm_path.exists() else []
    lm_rows = read_csv_rows(lm_path) if lm_path.exists() else []
    hm_rows = read_csv_rows(hm_path) if hm_path.exists() else []
    cm_rows = read_csv_rows(cm_path) if cm_path.exists() else []

    # Styling helper
    colors = {
        "standard": "#2b5c8f",
        "shared_q": "#d95f02",
        "shared_q_query_heavy": "#7570b3",
    }
    labels = {
        "standard": "Model A (Standard)",
        "shared_q": "Model B (Shared-Q)",
        "shared_q_query_heavy": "Model C3 (Query-Heavy Shared-Q)",
    }

    # 1. Gradient cosine similarity versus training steps
    plt.figure(figsize=(8, 5), dpi=150)
    for arch in ("shared_q", "shared_q_query_heavy"):
        steps = sorted(list(set(int(r["training_step"]) for r in gm_rows if r["architecture"] == arch)))
        if not steps:
            continue
        means = []
        sems = []
        for s in steps:
            vals = [float(r["cosine_similarity"]) for r in gm_rows if r["architecture"] == arch and int(r["training_step"]) == s]
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        means = np.array(means)
        sems = np.array(sems)
        plt.plot(steps, means, marker="o", label=labels[arch], color=colors[arch], linewidth=2)
        plt.fill_between(steps, means - sems, means + sems, color=colors[arch], alpha=0.2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Gradient Cosine Similarity cos(theta)")
    plt.title("Figure 1: Gradient Cosine Similarity vs. Training Steps (Model-Wide)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "gradient_cosine_similarity_vs_steps.png")
    plt.close()

    # 2. Layer-wise mean gradient cosine similarity
    plt.figure(figsize=(8, 5), dpi=150)
    layers = sorted(list(set(int(r["layer"]) for r in lm_rows if r["layer"] != "all")))
    x = np.arange(len(layers))
    width = 0.35
    for i, arch in enumerate(("shared_q", "shared_q_query_heavy")):
        means = []
        sems = []
        for l in layers:
            vals = [float(r["cosine_similarity"]) for r in lm_rows if r["architecture"] == arch and int(r["layer"]) == l]
            means.append(np.mean(vals) if vals else 0.0)
            sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        plt.bar(x + (i - 0.5) * width, means, width, yerr=sems, label=labels[arch], color=colors[arch], capsize=4, alpha=0.9)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Transformer Layer")
    plt.ylabel("Mean Cosine Similarity cos(theta)")
    plt.title("Figure 2: Layer-wise Mean Gradient Cosine Similarity")
    plt.xticks(x, [f"Layer {l}" for l in layers])
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "layerwise_mean_cosine_similarity.png")
    plt.close()

    # 3. Head-wise gradient cosine similarity heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for idx, arch in enumerate(("shared_q", "shared_q_query_heavy")):
        grid = np.zeros((4, 4))
        for l in range(4):
            for h in range(4):
                vals = [float(r["cosine_similarity"]) for r in hm_rows if r["architecture"] == arch and int(r["layer"]) == l and int(r["head"]) == h]
                grid[l, h] = np.mean(vals) if vals else 0.0
        im = axes[idx].imshow(grid, cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto")
        axes[idx].set_title(f"{labels[arch]}")
        axes[idx].set_xlabel("Head Index")
        axes[idx].set_ylabel("Layer Index")
        axes[idx].set_xticks(range(4))
        axes[idx].set_yticks(range(4))
        for l in range(4):
            for h in range(4):
                axes[idx].text(h, l, f"{grid[l, h]:.2f}", ha="center", va="center", color="black" if abs(grid[l, h]) < 0.3 else "white", fontsize=9)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    fig.suptitle("Figure 3: Head-wise Gradient Cosine Similarity Heatmap (Across Trajectory)")
    plt.savefig(plots_dir / "headwise_cosine_similarity_heatmap.png")
    plt.close()

    # 4. Query-role versus Key-role gradient norms
    plt.figure(figsize=(8, 5), dpi=150)
    for arch in ("shared_q", "shared_q_query_heavy"):
        steps = sorted(list(set(int(r["training_step"]) for r in gm_rows if r["architecture"] == arch)))
        q_norms = []
        k_norms = []
        for s in steps:
            q_v = [float(r["query_gradient_norm"]) for r in gm_rows if r["architecture"] == arch and int(r["training_step"]) == s]
            k_v = [float(r["key_gradient_norm"]) for r in gm_rows if r["architecture"] == arch and int(r["training_step"]) == s]
            q_norms.append(np.mean(q_v))
            k_norms.append(np.mean(k_v))
        plt.plot(steps, q_norms, marker="o", label=f"{labels[arch]} Query Role", color=colors[arch], linewidth=2)
        plt.plot(steps, k_norms, marker="s", linestyle="--", label=f"{labels[arch]} Key Role", color=colors[arch], linewidth=2, alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Gradient Norm ||grad_Wq||")
    plt.title("Figure 4: Query-Role vs. Key-Role Gradient Norms")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "query_vs_key_gradient_norms.png")
    plt.close()

    # 5. Gradient cancellation ratio versus training steps
    plt.figure(figsize=(8, 5), dpi=150)
    for arch in ("shared_q", "shared_q_query_heavy"):
        steps = sorted(list(set(int(r["training_step"]) for r in gm_rows if r["architecture"] == arch)))
        means = []
        sems = []
        for s in steps:
            vals = [float(r["cancellation_ratio"]) for r in gm_rows if r["architecture"] == arch and int(r["training_step"]) == s]
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        means = np.array(means)
        sems = np.array(sems)
        plt.plot(steps, means, marker="o", label=labels[arch], color=colors[arch], linewidth=2)
        plt.fill_between(steps, means - sems, means + sems, color=colors[arch], alpha=0.2)
    plt.xlabel("Training Steps")
    plt.ylabel("Cancellation Ratio: 1 - ||g_total|| / (||g_q|| + ||g_k||)")
    plt.title("Figure 5: Gradient Cancellation Ratio vs. Training Steps")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "gradient_cancellation_ratio_vs_steps.png")
    plt.close()

    # 6. Query and Key gradient magnitude ratios
    plt.figure(figsize=(8, 5), dpi=150)
    for arch in ("shared_q", "shared_q_query_heavy"):
        steps = sorted(list(set(int(r["training_step"]) for r in gm_rows if r["architecture"] == arch)))
        q_ratios = []
        for s in steps:
            vals = [float(r["query_gradient_ratio"]) for r in gm_rows if r["architecture"] == arch and int(r["training_step"]) == s]
            q_ratios.append(np.mean(vals))
        plt.plot(steps, q_ratios, marker="o", label=f"{labels[arch]} Query Ratio", color=colors[arch], linewidth=2)
    plt.axhline(0.5, color="black", linestyle=":", linewidth=1.5, label="Balanced (50/50)")
    plt.xlabel("Training Steps")
    plt.ylabel("Query Gradient Ratio ||g_q|| / (||g_q|| + ||g_k||)")
    plt.title("Figure 6: Query vs. Key Gradient Magnitude Ratios Across Training")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "query_key_magnitude_ratios.png")
    plt.close()

    # 7. Negative gradient conflict frequency across layers
    plt.figure(figsize=(8, 5), dpi=150)
    layers = sorted(list(set(int(r["layer"]) for r in lm_rows if r["layer"] != "all")))
    x = np.arange(len(layers))
    width = 0.35
    for i, arch in enumerate(("shared_q", "shared_q_query_heavy")):
        neg_pcts = []
        for l in layers:
            vals = [float(r["cosine_similarity"]) for r in lm_rows if r["architecture"] == arch and int(r["layer"]) == l]
            pct = np.mean(np.array(vals) < 0.0) * 100.0 if vals else 0.0
            neg_pcts.append(pct)
        plt.bar(x + (i - 0.5) * width, neg_pcts, width, label=labels[arch], color=colors[arch], alpha=0.9)
    plt.xlabel("Transformer Layer")
    plt.ylabel("Frequency of Negative Cosine Similarity (%)")
    plt.title("Figure 7: Frequency of Negative Gradient Conflict Across Layers")
    plt.xticks(x, [f"Layer {l}" for l in layers])
    plt.ylim(0, 100)
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "negative_conflict_frequency_across_layers.png")
    plt.close()

    # 8. Standard versus Shared-Q gradient norm comparison
    plt.figure(figsize=(9, 5), dpi=150)
    steps = sorted(list(set(int(r["training_step"]) for r in gm_rows if r["architecture"] == "standard")))
    if steps:
        std_q = [np.mean([float(r["query_gradient_norm"]) for r in gm_rows if r["architecture"] == "standard" and int(r["training_step"]) == s]) for s in steps]
        std_k = [np.mean([float(r["key_gradient_norm"]) for r in gm_rows if r["architecture"] == "standard" and int(r["training_step"]) == s]) for s in steps]
        std_tot = [np.mean([float(r["total_gradient_norm"]) for r in gm_rows if r["architecture"] == "shared_q" and int(r["training_step"]) == s]) for s in steps]
        c3_tot = [np.mean([float(r["total_gradient_norm"]) for r in gm_rows if r["architecture"] == "shared_q_query_heavy" and int(r["training_step"]) == s]) for s in steps]

        plt.plot(steps, std_q, marker="o", label="Model A Standard Wq", color="#2b5c8f", linewidth=2)
        plt.plot(steps, std_k, marker="v", linestyle="--", label="Model A Standard Wk", color="#41b6c4", linewidth=2)
        plt.plot(steps, std_tot, marker="s", label="Model B Shared-Q Wq (Combined)", color="#d95f02", linewidth=2)
        plt.plot(steps, c3_tot, marker="^", label="Model C3 Query-Heavy Wq (Combined)", color="#7570b3", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("Gradient Norm")
    plt.title("Figure 8: Gradient Norms: Standard (Independent) vs. Shared-Q (Shared)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "standard_vs_shared_q_gradient_norms.png")
    plt.close()

    # 9. Early, middle and late training gradient comparison
    plt.figure(figsize=(9, 5), dpi=150)
    phases = {
        "Early\n(100-500)": [100, 200, 300, 500],
        "Middle\n(1000-2000)": [1000, 1500, 2000],
        "Late\n(2500-3000)": [2500, 3000],
    }
    x = np.arange(len(phases))
    width = 0.35
    for i, arch in enumerate(("shared_q", "shared_q_query_heavy")):
        means = []
        sems = []
        for p_steps in phases.values():
            vals = [float(r["cosine_similarity"]) for r in gm_rows if r["architecture"] == arch and int(r["training_step"]) in p_steps]
            means.append(np.mean(vals) if vals else 0.0)
            sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        plt.bar(x + (i - 0.5) * width, means, width, yerr=sems, label=labels[arch], color=colors[arch], capsize=4, alpha=0.9)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Training Phase")
    plt.ylabel("Mean Cosine Similarity cos(theta)")
    plt.title("Figure 9: Cosine Similarity Across Early, Middle, and Late Training")
    plt.xticks(x, list(phases.keys()))
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "early_middle_late_gradient_comparison.png")
    plt.close()

    # 10. B versus C3 gradient interference comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    steps = sorted(list(set(int(r["training_step"]) for r in gm_rows if r["architecture"] == "shared_q")))

    # Subplot 1: Cosine similarity
    b_cos = [np.mean([float(r["cosine_similarity"]) for r in gm_rows if r["architecture"] == "shared_q" and int(r["training_step"]) == s]) for s in steps]
    c3_cos = [np.mean([float(r["cosine_similarity"]) for r in gm_rows if r["architecture"] == "shared_q_query_heavy" and int(r["training_step"]) == s]) for s in steps]
    ax1.plot(steps, b_cos, marker="o", label=labels["shared_q"], color=colors["shared_q"], linewidth=2)
    ax1.plot(steps, c3_cos, marker="s", label=labels["shared_q_query_heavy"], color=colors["shared_q_query_heavy"], linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Cosine Similarity cos(theta)")
    ax1.set_title("Cosine Similarity: B vs. C3")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Subplot 2: Cancellation ratio
    b_can = [np.mean([float(r["cancellation_ratio"]) for r in gm_rows if r["architecture"] == "shared_q" and int(r["training_step"]) == s]) for s in steps]
    c3_can = [np.mean([float(r["cancellation_ratio"]) for r in gm_rows if r["architecture"] == "shared_q_query_heavy" and int(r["training_step"]) == s]) for s in steps]
    ax2.plot(steps, b_can, marker="o", label=labels["shared_q"], color=colors["shared_q"], linewidth=2)
    ax2.plot(steps, c3_can, marker="s", label=labels["shared_q_query_heavy"], color=colors["shared_q_query_heavy"], linewidth=2)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Cancellation Ratio")
    ax2.set_title("Cancellation Ratio: B vs. C3")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.suptitle("Figure 10: Direct Interference Comparison: Model B vs. Model C3")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_b_vs_c3_interference_comparison.png")
    plt.close()

    print(f"All 10 figures generated successfully in: {plots_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "checkpoint", "trajectory", "plots"), default="all")
    parser.add_argument("--output-dir", default="outputs/gradient_analysis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-eval-batches", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Gradient Analysis Runner | Device: {device} | Output: {output_dir}", flush=True)

    dataset = CharacterDataset(config)

    if args.mode in ("all", "checkpoint"):
        run_checkpoint_analysis(dataset, config, device, output_dir, num_batches=args.num_eval_batches)

    if args.mode in ("all", "trajectory"):
        run_training_trajectory_analysis(dataset, config, device, output_dir)

    if args.mode in ("all", "plots"):
        generate_gradient_plots(output_dir)


if __name__ == "__main__":
    main()
