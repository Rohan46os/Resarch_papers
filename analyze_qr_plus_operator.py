"""Comprehensive Analysis & Diagnostics for QR Plus (QR + RQ) vs QR Minus (QR - RQ).

Performs:
1. Full verification and re-evaluation of all 8 architectures in the Section 10 comparison table
   under exact identical Protocol A (20 fixed batches, 81,920 tokens) and Protocol B (full validation
   corpus, 111,539 tokens), cross-checked against original historical summary.json / metrics.csv.
2. Per-layer and per-head Section 9 diagnostics for the 4 focal architectures
   (Antisymmetric-Only Q28, Plus-Only Q28, Full QR Minus Q112, Full QR Plus Q112) across all 3 seeds:
   - 9A: Raw pre-mask score matrix geometry (Frobenius/spectral norm, stable rank, numerical rank,
         singular value distribution & pairing, eigenvalue distribution & sign structure,
         diagonal mean/std/Frobenius, off-diagonal Frobenius, total score energy, symmetry/skew error).
   - 9B: Post-softmax causal attention probabilities (entropy, normalized entropy, max prob,
         diagonal vs off-diagonal probability mass, mean attention distance, top-3 & top-5 mass).
   - 9C: Projection & optimization diagnostics (Q/R/V projection RMS & norm, W_Q/W_R/W_V/W_O norms,
         Q & R gradient norms from training logs + validation pass, attention output RMS,
         raw logit std, gradient clipping frequency, training/validation loss trajectories).
3. Exports structured CSVs, JSONs, and compact .npz arrays for 300 DPI publication plots.

Outputs saved strictly under:
  outputs/qr_plus_operator/analysis/csv/
  outputs/qr_plus_operator/analysis/json/
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM, parameter_counts
from train_qr_lambda_extended_q112 import safe_exp, write_json
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_OUTPUT_ROOT = Path("outputs/qr_plus_operator")

COMPARISON_ARCHS = [
    {
        "key": "standard_q112",
        "label": "Original Standard Q112",
        "model_type": "standard_q112",
        "qr_lambda": None,
        "dir": Path("outputs/standard_attention_q112_10k"),
        "counterpart": None,
    },
    {
        "key": "diagonal_free_standard_q112",
        "label": "Diagonal-Free Standard Q112",
        "model_type": "diagonal_free_standard_q112",
        "qr_lambda": None,
        "dir": Path("outputs/standard_attention_diagonal_removal/trained_diagonal_free"),
        "counterpart": "standard_q112",
    },
    {
        "key": "full_qr_minus_q112",
        "label": "Full QR Minus Q112",
        "model_type": "qr_antisymmetric_q112",
        "qr_lambda": 20.0,
        "dir": Path("outputs/qr_lambda_upper_probe_q112/lambda_20.0"),
        "counterpart": "standard_q112",
    },
    {
        "key": "full_qr_plus_q112",
        "label": "Full QR Plus Q112",
        "model_type": "qr_full_plus_q112",
        "qr_lambda": 20.0,
        "dir": Path("outputs/qr_plus_operator/full_qr_plus_q112"),
        "counterpart": "full_qr_minus_q112",
    },
    {
        "key": "antisymmetric_only_q28",
        "label": "Antisymmetric-Only Q28",
        "model_type": "qr_antisymmetric_only_q28",
        "qr_lambda": 20.0,
        "dir": Path("outputs/qr_parameter_matched_components/antisymmetric_only_q28"),
        "counterpart": "standard_q28",
    },
    {
        "key": "plus_only_q28",
        "label": "Plus-Only QR Q28",
        "model_type": "qr_plus_only_q28",
        "qr_lambda": 20.0,
        "dir": Path("outputs/qr_plus_operator/plus_only_q28"),
        "counterpart": "antisymmetric_only_q28",
    },
    {
        "key": "standard_q28",
        "label": "Standard Q28",
        "model_type": "standard_q28",
        "qr_lambda": None,
        "dir": Path("outputs/reduced_attention_70_percent/standard_q28"),
        "counterpart": None,
    },
    {
        "key": "symmetric_only_q56",
        "label": "Symmetric-Only Q56",
        "model_type": "symmetric_only_q56",
        "qr_lambda": None,
        "dir": Path("outputs/qr_parameter_matched_components/symmetric_only_q56"),
        "counterpart": "antisymmetric_only_q28",
    },
]

FOCAL_ARCH_KEYS = [
    "antisymmetric_only_q28",
    "plus_only_q28",
    "full_qr_minus_q112",
    "full_qr_plus_q112",
]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _read_metrics_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if v in ("", None, "True", "False"):
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    pass
            rows.append(parsed)
    return rows


def _extract_training_log_summary(run_dir: Path) -> Dict[str, Any]:
    s_path = run_dir / "summary.json"
    m_path = run_dir / "metrics.csv"
    sdata = json.loads(s_path.read_text(encoding="utf-8")) if s_path.exists() else {}
    mrows = _read_metrics_csv(m_path)

    # Extract final train loss & grad/clipping stats from metrics.csv
    nonzero_rows = [r for r in mrows if r.get("step", 0) > 0]
    final_train_loss = None
    if nonzero_rows:
        final_train_loss = nonzero_rows[-1].get("train_loss")
    if final_train_loss is None:
        final_train_loss = sdata.get("final_train_loss")

    mean_wq_grad = float(np.mean([r["wq_grad_norm_mean"] for r in nonzero_rows if "wq_grad_norm_mean" in r])) if any("wq_grad_norm_mean" in r for r in nonzero_rows) else None
    wr_key = "wr_grad_norm_mean" if any("wr_grad_norm_mean" in r for r in nonzero_rows) else (
        "wk_or_wr_grad_norm_mean" if any("wk_or_wr_grad_norm_mean" in r for r in nonzero_rows) else (
            "wk_grad_norm_mean" if any("wk_grad_norm_mean" in r for r in nonzero_rows) else None
        )
    )
    mean_wr_grad = float(np.mean([r[wr_key] for r in nonzero_rows if wr_key and wr_key in r])) if wr_key else None
    final_wq_grad = nonzero_rows[-1].get("wq_grad_norm_mean") if nonzero_rows else None
    final_wr_grad = nonzero_rows[-1].get(wr_key) if (nonzero_rows and wr_key) else None

    clip_freq = None
    if nonzero_rows and "cum_grad_clip_freq" in nonzero_rows[-1]:
        clip_freq = nonzero_rows[-1]["cum_grad_clip_freq"]
    elif "cum_grad_clip_freq" in sdata:
        clip_freq = sdata["cum_grad_clip_freq"]
    elif "overall_grad_clip_freq" in sdata:
        clip_freq = sdata["overall_grad_clip_freq"]

    hist_proto_a = sdata.get("final_validation_loss", sdata.get("final_loss"))
    best_step = sdata.get("best_step")
    best_val_hist = sdata.get("best_validation_loss", sdata.get("best_loss"))
    return {
        "final_train_loss": final_train_loss,
        "historical_proto_a_loss": hist_proto_a,
        "best_step": int(best_step) if best_step is not None else -1,
        "historical_best_val_loss": best_val_hist,
        "mean_wq_grad_norm": mean_wq_grad,
        "mean_wr_grad_norm": mean_wr_grad,
        "final_wq_grad_norm": final_wq_grad,
        "final_wr_grad_norm": final_wr_grad,
        "cum_grad_clip_freq": clip_freq,
        "init_state_sha256": sdata.get("init_state_sha256"),
        "training_batch_sha256": sdata.get("training_batch_sha256"),
        "training_seconds": sdata.get("training_seconds"),
        "training_tokens_per_second": sdata.get("training_tokens_per_second"),
    }


def _load_model(spec: Dict[str, Any], ckpt_path: Path, config: Config, vocab_size: int, device: torch.device) -> DecoderLM:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DecoderLM(config, vocab_size, spec["model_type"])
    if spec["qr_lambda"] is not None:
        model.set_qr_lambda(float(spec["qr_lambda"]))
    model.load_state_dict(ckpt["model"], strict=True)
    if model.lm_head.weight is not model.token_embedding.weight:
        model.lm_head.weight = model.token_embedding.weight
    model.to(device).eval()
    return model


def compute_eval_grad_norms(
    model: DecoderLM,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int = 4,
) -> Dict[int, Dict[str, float]]:
    """Compute exact per-layer W_Q and W_R gradient norms on validation batches (eval mode)."""
    model.eval()
    model.zero_grad(set_to_none=True)
    n_use = min(max_batches, len(batches))
    total_loss = torch.zeros((), device=device)
    for xb, yb in batches[:n_use]:
        _, loss = model(xb.to(device), yb.to(device))
        total_loss = total_loss + loss / float(n_use)
    total_loss.backward()

    out: Dict[int, Dict[str, float]] = {}
    for idx, block in enumerate(model.blocks):
        attn = block.attention
        out[idx] = {
            "eval_wq_grad_norm": float(torch.linalg.vector_norm(attn.q_proj.weight.grad.detach()).item()),
            "eval_wr_grad_norm": float(torch.linalg.vector_norm(attn.r_proj.weight.grad.detach()).item()),
            "eval_wv_grad_norm": float(torch.linalg.vector_norm(attn.v_proj.weight.grad.detach()).item()),
            "eval_wo_grad_norm": float(torch.linalg.vector_norm(attn.out_proj.weight.grad.detach()).item()),
        }
    model.zero_grad(set_to_none=True)
    return out


@torch.inference_mode()
def compute_head_and_layer_diagnostics(
    model: DecoderLM,
    arch_key: str,
    seed: int,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    collect_samples: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Compute full Section 9A/9B/9C metrics for every layer (0..3) and head (0..3)."""
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attention.n_heads
    is_plus = arch_key in ("plus_only_q28", "full_qr_plus_q112")
    is_full = arch_key in ("full_qr_minus_q112", "full_qr_plus_q112")

    # Accumulators keyed by (layer, head)
    acc: Dict[Tuple[int, int], Dict[str, float]] = {(l, h): {} for l in range(n_layers) for h in range(n_heads)}
    sv_acc: Dict[Tuple[int, int], List[np.ndarray]] = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}
    eig_real_samples: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}
    eig_imag_samples: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}
    diag_samples: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}
    saved_arrays: Dict[str, np.ndarray] = {}

    n_batches = 0
    n_spec_batches = 0

    for b_idx, (x_cpu, _) in enumerate(batches):
        x = model._embed(x_cpu.to(device))
        B, T, _ = x.shape
        pos_idx = torch.arange(T, device=device, dtype=torch.float32)
        dist_matrix = (pos_idx.unsqueeze(1) - pos_idx.unsqueeze(0)).clamp(min=0.0)
        max_ent = torch.log(torch.arange(1, T + 1, device=device, dtype=torch.float32)).view(1, T)
        tril_mask = torch.ones(T, T, dtype=torch.bool, device=device).tril(diagonal=0)

        for l_idx, block in enumerate(model.blocks):
            norm_x = block.attention_norm(x)
            attn = block.attention
            q, r, v = attn._project_qrv(norm_x)  # (B, H, T, d)
            maps = attn._scores(q, r, return_components=True)
            raw = maps["raw_scores"].float()        # (B, H, T, T)
            masked = maps["masked_scores"].float()
            probs = maps["probabilities"].float()   # (B, H, T, T)
            attn_out = attn(norm_x).float()         # (B, T, D)
            attn_out_rms = float(attn_out.pow(2).mean().sqrt().item())

            scale = math.sqrt(attn.q_head_dim)
            sym_scaled = maps["sym_scores"].float() / scale
            cross_scaled = (attn.lambda_param * maps["cross_scores"].float()) / scale

            if collect_samples and b_idx == 0:
                saved_arrays[f"{arch_key}_L{l_idx}_raw_scores"] = raw[0].cpu().numpy()
                saved_arrays[f"{arch_key}_L{l_idx}_masked_scores"] = masked[0].cpu().numpy()
                saved_arrays[f"{arch_key}_L{l_idx}_probs"] = probs[0].cpu().numpy()

            # Compute SVD & eigenvalues on first 4 sequences of first 5 batches (20 sequences/head = accurate & fast)
            do_spectral = b_idx < 5
            if do_spectral:
                sub = raw[: min(4, B)]  # (b, H, T, T)
                sv_all = torch.linalg.svdvals(sub).cpu().numpy()  # (b, H, T)
                if is_plus:
                    # Symmetrize tiny float32 rounding (1e-7) for exact eigvalsh
                    sym_sub = 0.5 * (sub + sub.transpose(-2, -1))
                    eig_r = torch.linalg.eigvalsh(sym_sub).cpu().numpy()  # (b, H, T)
                    eig_i = np.zeros_like(eig_r)
                else:
                    eig_c = torch.linalg.eigvals(sub).cpu().numpy()       # (b, H, T) complex
                    eig_r = eig_c.real
                    eig_i = eig_c.imag

            for h in range(n_heads):
                raw_h = raw[:, h]       # (B, T, T)
                probs_h = probs[:, h]   # (B, T, T)
                q_h = q[:, h].float()   # (B, T, dq)
                r_h = r[:, h].float()
                v_h = v[:, h].float()

                diag_h = torch.diagonal(raw_h, dim1=-2, dim2=-1)  # (B, T)
                if b_idx < 5:
                    diag_samples[l_idx].append(diag_h[:4].cpu().numpy().ravel())

                total_energy = raw_h.pow(2).sum(dim=(-2, -1))  # (B,)
                diag_energy = diag_h.pow(2).sum(dim=-1)         # (B,)
                off_energy = (total_energy - diag_energy).clamp(min=0.0)

                frob_h = total_energy.sqrt().mean().item()
                diag_frob_h = diag_energy.sqrt().mean().item()
                off_frob_h = off_energy.sqrt().mean().item()
                diag_energy_frac = float((diag_energy / (total_energy + 1e-12)).mean().item())

                sym_err = float(
                    (
                        torch.linalg.matrix_norm(raw_h - raw_h.transpose(-2, -1), ord="fro")
                        / (torch.linalg.matrix_norm(raw_h, ord="fro") + 1e-12)
                    )
                    .mean()
                    .item()
                )
                skew_err = float(
                    (
                        torch.linalg.matrix_norm(raw_h + raw_h.transpose(-2, -1), ord="fro")
                        / (torch.linalg.matrix_norm(raw_h, ord="fro") + 1e-12)
                    )
                    .mean()
                    .item()
                )

                # Attention probability stats
                ent = -(probs_h * torch.log(probs_h.clamp(min=1e-12))).sum(dim=-1)  # (B, T)
                norm_ent = torch.where(max_ent > 1e-6, ent / max_ent, torch.zeros_like(ent))
                diag_p = torch.diagonal(probs_h, dim1=-2, dim2=-1).mean().item()
                mean_dist = float((probs_h * dist_matrix.unsqueeze(0)).sum(dim=-1).mean().item())
                top3 = float(probs_h.topk(k=3, dim=-1).values.sum(dim=-1).mean().item())
                top5 = float(probs_h.topk(k=5, dim=-1).values.sum(dim=-1).mean().item())

                # Projection stats
                q_rms = float(q_h.pow(2).mean().sqrt().item())
                r_rms = float(r_h.pow(2).mean().sqrt().item())
                v_rms = float(v_h.pow(2).mean().sqrt().item())
                q_norm = float(torch.linalg.matrix_norm(q_h, ord="fro").mean().item())
                r_norm = float(torch.linalg.matrix_norm(r_h, ord="fro").mean().item())
                v_norm = float(torch.linalg.matrix_norm(v_h, ord="fro").mean().item())

                sym_frob_h = float(torch.linalg.matrix_norm(sym_scaled[:, h], ord="fro").mean().item())
                cross_frob_h = float(torch.linalg.matrix_norm(cross_scaled[:, h], ord="fro").mean().item())

                metrics = {
                    "raw_frob_norm": float(frob_h),
                    "raw_diag_mean": float(diag_h.mean().item()),
                    "raw_diag_std": float(diag_h.std(unbiased=False).item()),
                    "raw_diag_frob": float(diag_frob_h),
                    "raw_off_diag_frob": float(off_frob_h),
                    "raw_total_energy": float(total_energy.mean().item()),
                    "raw_diag_energy_fraction": diag_energy_frac,
                    "raw_score_std": float(raw_h.std(unbiased=False).item()),
                    "raw_causal_score_std": float(raw_h[:, tril_mask].std(unbiased=False).item()),
                    "raw_symmetry_error": sym_err,
                    "raw_skew_error": skew_err,
                    "sym_component_frob": sym_frob_h,
                    "cross_component_frob": cross_frob_h,
                    "attention_entropy": float(ent.mean().item()),
                    "normalized_attention_entropy": float(norm_ent[:, 1:].mean().item()),
                    "mean_max_attention_prob": float(probs_h.max(dim=-1).values.mean().item()),
                    "diagonal_attention_mass": float(diag_p),
                    "off_diagonal_attention_mass": float(1.0 - diag_p),
                    "mean_attention_distance": mean_dist,
                    "top3_attention_mass": top3,
                    "top5_attention_mass": top5,
                    "q_proj_rms": q_rms,
                    "r_proj_rms": r_rms,
                    "v_proj_rms": v_rms,
                    "q_proj_norm": q_norm,
                    "r_proj_norm": r_norm,
                    "v_proj_norm": v_norm,
                    "attn_output_rms": attn_out_rms,
                }
                for k, val in metrics.items():
                    acc[(l_idx, h)][k] = acc[(l_idx, h)].get(k, 0.0) + val

                if do_spectral:
                    sv_h = sv_all[:, h, :]  # (b, T)
                    sv_acc[(l_idx, h)].append(sv_h)
                    spec_norm = float(sv_h[:, 0].mean())
                    s_energy = (sv_h ** 2).sum(axis=-1)
                    st_rank = float((s_energy / (sv_h[:, 0] ** 2 + 1e-12)).mean())
                    num_rank = float((sv_h > (1e-5 * sv_h[:, :1])).sum(axis=-1).mean())
                    # Pairing gap across top 2*dq singular values
                    top_k = min(2 * attn.q_head_dim, T - (T % 2))
                    pair_gaps = np.abs(sv_h[:, 0:top_k:2] - sv_h[:, 1:top_k:2]) / (sv_h[:, :1] + 1e-12)
                    sv_pair_gap = float(pair_gaps.mean())

                    er_h = eig_r[:, h, :]
                    ei_h = eig_i[:, h, :]
                    if collect_samples:
                        eig_real_samples[l_idx].append(er_h.ravel())
                        eig_imag_samples[l_idx].append(ei_h.ravel())

                    # Significant eigenvalues (magnitude > 1e-3 * max_mag)
                    mag = np.sqrt(er_h ** 2 + ei_h ** 2)
                    sig = mag > (1e-3 * np.maximum(mag.max(axis=-1, keepdims=True), 1e-8))
                    if is_plus:
                        pos_frac = float(((er_h > 1e-3) & sig).sum() / max(1, sig.sum()))
                        neg_frac = float(((er_h < -1e-3) & sig).sum() / max(1, sig.sum()))
                        pos_e = float((er_h[er_h > 0] ** 2).sum())
                        neg_e = float((er_h[er_h < 0] ** 2).sum())
                        tot_e = pos_e + neg_e + 1e-12
                        pos_energy_frac = pos_e / tot_e
                        neg_energy_frac = neg_e / tot_e
                    else:
                        pos_frac = float(((er_h > 1e-3) & sig).sum() / max(1, sig.sum()))
                        neg_frac = float(((er_h < -1e-3) & sig).sum() / max(1, sig.sum()))
                        pos_energy_frac = float("nan")
                        neg_energy_frac = float("nan")

                    spec_metrics = {
                        "raw_spectral_norm": spec_norm,
                        "raw_stable_rank": st_rank,
                        "raw_numerical_rank": num_rank,
                        "sv_pairing_gap": sv_pair_gap,
                        "eig_real_min": float(er_h.min()),
                        "eig_real_max": float(er_h.max()),
                        "eig_real_mean": float(er_h.mean()),
                        "eig_real_abs_max": float(np.abs(er_h).max()),
                        "eig_imag_abs_max": float(np.abs(ei_h).max()),
                        "eig_pos_fraction_sig": pos_frac,
                        "eig_neg_fraction_sig": neg_frac,
                        "eig_pos_energy_fraction": pos_energy_frac if is_plus else 0.0,
                        "eig_neg_energy_fraction": neg_energy_frac if is_plus else 0.0,
                    }
                    for k, val in spec_metrics.items():
                        acc[(l_idx, h)][k] = acc[(l_idx, h)].get(k, 0.0) + val

            x = block(x)
        n_batches += 1
        if do_spectral:
            n_spec_batches += 1

    # Weight norms per layer
    layer_weight_norms = {}
    for l_idx, block in enumerate(model.blocks):
        attn = block.attention
        layer_weight_norms[l_idx] = {
            "wq_weight_norm": float(torch.linalg.vector_norm(attn.q_proj.weight.detach()).item()),
            "wr_weight_norm": float(torch.linalg.vector_norm(attn.r_proj.weight.detach()).item()),
            "wv_weight_norm": float(torch.linalg.vector_norm(attn.v_proj.weight.detach()).item()),
            "wo_weight_norm": float(torch.linalg.vector_norm(attn.out_proj.weight.detach()).item()),
        }

    spec_keys = {
        "raw_spectral_norm",
        "raw_stable_rank",
        "raw_numerical_rank",
        "sv_pairing_gap",
        "eig_real_min",
        "eig_real_max",
        "eig_real_mean",
        "eig_real_abs_max",
        "eig_imag_abs_max",
        "eig_pos_fraction_sig",
        "eig_neg_fraction_sig",
        "eig_pos_energy_fraction",
        "eig_neg_energy_fraction",
    }

    rows: List[Dict[str, Any]] = []
    for l_idx in range(n_layers):
        for h in range(n_heads):
            row: Dict[str, Any] = {
                "arch_key": arch_key,
                "seed": int(seed),
                "layer": l_idx,
                "head": h,
                "is_plus": is_plus,
                "is_full_qr": is_full,
            }
            for k, v in acc[(l_idx, h)].items():
                denom = n_spec_batches if k in spec_keys else n_batches
                row[k] = v / max(1, denom)
            row.update(layer_weight_norms[l_idx])
            rows.append(row)

            if collect_samples and sv_acc[(l_idx, h)]:
                sv_cat = np.concatenate(sv_acc[(l_idx, h)], axis=0)  # (N, T)
                saved_arrays[f"{arch_key}_L{l_idx}_H{h}_sv_mean"] = sv_cat.mean(axis=0)
                saved_arrays[f"{arch_key}_L{l_idx}_H{h}_sv_std"] = sv_cat.std(axis=0)

        if collect_samples:
            if eig_real_samples[l_idx]:
                saved_arrays[f"{arch_key}_L{l_idx}_eig_real"] = np.concatenate(eig_real_samples[l_idx])
                saved_arrays[f"{arch_key}_L{l_idx}_eig_imag"] = np.concatenate(eig_imag_samples[l_idx])
            if diag_samples[l_idx]:
                saved_arrays[f"{arch_key}_L{l_idx}_diag_scores"] = np.concatenate(diag_samples[l_idx])

    return rows, saved_arrays


def run_analysis(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seeds: Sequence[int] = SEEDS,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_str)
    csv_dir = Path(output_root) / "analysis" / "csv"
    json_dir = Path(output_root) / "analysis" / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        n_layers=4,
        d_model=128,
        n_heads=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
        batch_size=32,
        max_steps=10000,
        eval_interval=250,
    )
    dataset = CharacterDataset(config)
    vocab_size = len(dataset.vocab)
    proto_a_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    proto_b_batches = full_corpus_validation_batches(
        dataset.validation, config.context_length, config.batch_size
    )

    # ------------------------------------------------------------------
    # 1. Verify & evaluate all 8 comparison architectures across 3 seeds
    # ------------------------------------------------------------------
    per_seed_rows: List[Dict[str, Any]] = []
    loss_curve_rows: List[Dict[str, Any]] = []

    print("=== Auditing & Evaluating All 8 Architectures (Protocol A & B) ===", flush=True)
    for spec in COMPARISON_ARCHS:
        arch_key = spec["key"]
        for seed in seeds:
            run_dir = spec["dir"] / f"seed_{seed}"
            final_ckpt = run_dir / "final.pt"
            best_ckpt = run_dir / "best.pt"
            if not final_ckpt.exists():
                raise FileNotFoundError(f"Missing final.pt for {arch_key} seed {seed}: {final_ckpt}")

            log_meta = _extract_training_log_summary(run_dir)
            model_final = _load_model(spec, final_ckpt, config, vocab_size, device)
            counts = parameter_counts(model_final)
            pa_final = evaluate_loss(model_final, proto_a_batches, device)
            pb_final = evaluate_loss(model_final, proto_b_batches, device)

            if best_ckpt.exists():
                model_best = _load_model(spec, best_ckpt, config, vocab_size, device)
                pa_best = evaluate_loss(model_best, proto_a_batches, device)
                pb_best = evaluate_loss(model_best, proto_b_batches, device)
                del model_best
            else:
                pa_best, pb_best = pa_final, pb_final
            del model_final

            hist_pa = log_meta["historical_proto_a_loss"]
            pa_disc = abs(pa_final - hist_pa) if hist_pa is not None else 0.0
            if pa_disc > 1e-4:
                print(
                    f"  WARNING [{arch_key} s={seed}]: re-evaluated Proto A ({pa_final:.6f}) differs from historical ({hist_pa:.6f}) by {pa_disc:.6f}",
                    flush=True,
                )

            per_seed_rows.append(
                {
                    "arch_key": arch_key,
                    "label": spec["label"],
                    "model_type": spec["model_type"],
                    "seed": int(seed),
                    "attention_parameters": counts["attention"],
                    "non_attention_parameters": counts["total"] - counts["attention"],
                    "total_parameters": counts["total"],
                    "final_train_loss": log_meta["final_train_loss"],
                    "proto_a_final_loss": float(pa_final),
                    "proto_a_final_ppl": float(safe_exp(pa_final)),
                    "proto_a_best_loss": float(pa_best),
                    "proto_a_best_ppl": float(safe_exp(pa_best)),
                    "best_step": log_meta["best_step"],
                    "proto_b_final_loss": float(pb_final),
                    "proto_b_final_ppl": float(safe_exp(pb_final)),
                    "proto_b_best_loss": float(pb_best),
                    "proto_b_best_ppl": float(safe_exp(pb_best)),
                    "historical_proto_a_loss": hist_pa,
                    "proto_a_verification_diff": float(pa_disc),
                    "mean_wq_grad_norm": log_meta["mean_wq_grad_norm"],
                    "mean_wr_grad_norm": log_meta["mean_wr_grad_norm"],
                    "final_wq_grad_norm": log_meta["final_wq_grad_norm"],
                    "final_wr_grad_norm": log_meta["final_wr_grad_norm"],
                    "cum_grad_clip_freq": log_meta["cum_grad_clip_freq"],
                    "init_state_sha256": log_meta["init_state_sha256"],
                    "training_batch_sha256": log_meta["training_batch_sha256"],
                    "training_seconds": log_meta["training_seconds"],
                    "training_tokens_per_second": log_meta["training_tokens_per_second"],
                }
            )

            # Collect validation-loss curves for focal architectures
            if arch_key in FOCAL_ARCH_KEYS:
                mrows = _read_metrics_csv(run_dir / "metrics.csv")
                for mr in mrows:
                    loss_curve_rows.append(
                        {
                            "arch_key": arch_key,
                            "label": spec["label"],
                            "seed": int(seed),
                            "step": int(mr.get("step", 0)),
                            "train_loss": mr.get("train_loss", ""),
                            "validation_loss": mr.get("validation_loss", ""),
                            "validation_perplexity": mr.get("validation_perplexity", ""),
                            "global_grad_norm_mean": mr.get("global_grad_norm_mean", ""),
                            "wq_grad_norm_mean": mr.get("wq_grad_norm_mean", ""),
                            "wr_grad_norm_mean": mr.get("wr_grad_norm_mean", mr.get("wk_or_wr_grad_norm_mean", "")),
                            "grad_clip_freq": mr.get("grad_clip_freq", ""),
                            "cum_grad_clip_freq": mr.get("cum_grad_clip_freq", ""),
                        }
                    )

        arch_rows = [r for r in per_seed_rows if r["arch_key"] == arch_key]
        pa_m, pa_s = _mean_std([r["proto_a_final_loss"] for r in arch_rows])
        pb_m, pb_s = _mean_std([r["proto_b_final_loss"] for r in arch_rows])
        print(f"  {spec['label']:30s} | Proto A: {pa_m:.4f} +/- {pa_s:.4f} | Proto B: {pb_m:.4f} +/- {pb_s:.4f}", flush=True)

    _write_csv(csv_dir / "all_architectures_per_seed.csv", per_seed_rows)
    _write_csv(csv_dir / "focal_training_curves.csv", loss_curve_rows)

    # Build lookup by (arch_key, seed) for paired differences
    by_arch_seed = {(r["arch_key"], r["seed"]): r for r in per_seed_rows}

    summary_table_rows: List[Dict[str, Any]] = []
    for spec in COMPARISON_ARCHS:
        ak = spec["key"]
        cp = spec["counterpart"]
        ar = [by_arch_seed[(ak, s)] for s in seeds]
        pa_m, pa_s = _mean_std([r["proto_a_final_loss"] for r in ar])
        pa_ppl_m, pa_ppl_s = _mean_std([r["proto_a_final_ppl"] for r in ar])
        pb_m, pb_s = _mean_std([r["proto_b_final_loss"] for r in ar])
        pb_ppl_m, pb_ppl_s = _mean_std([r["proto_b_final_ppl"] for r in ar])
        pa_best_m, pa_best_s = _mean_std([r["proto_a_best_loss"] for r in ar])
        pb_best_m, pb_best_s = _mean_std([r["proto_b_best_loss"] for r in ar])
        tr_vals = [r["final_train_loss"] for r in ar if r["final_train_loss"] is not None]
        tr_m, tr_s = _mean_std(tr_vals) if tr_vals else (float("nan"), float("nan"))

        if cp is not None:
            diff_a = [by_arch_seed[(ak, s)]["proto_a_final_loss"] - by_arch_seed[(cp, s)]["proto_a_final_loss"] for s in seeds]
            diff_b = [by_arch_seed[(ak, s)]["proto_b_final_loss"] - by_arch_seed[(cp, s)]["proto_b_final_loss"] for s in seeds]
            da_m, da_s = _mean_std(diff_a)
            db_m, db_s = _mean_std(diff_b)
        else:
            da_m, da_s, db_m, db_s = None, None, None, None

        summary_table_rows.append(
            {
                "arch_key": ak,
                "label": spec["label"],
                "attention_parameters": ar[0]["attention_parameters"],
                "non_attention_parameters": ar[0]["non_attention_parameters"],
                "total_parameters": ar[0]["total_parameters"],
                "counterpart": cp or "",
                "final_train_loss_mean": tr_m,
                "final_train_loss_std": tr_s,
                "proto_a_final_loss_mean": pa_m,
                "proto_a_final_loss_std": pa_s,
                "proto_a_final_ppl_mean": pa_ppl_m,
                "proto_a_final_ppl_std": pa_ppl_s,
                "proto_a_best_loss_mean": pa_best_m,
                "proto_a_best_loss_std": pa_best_s,
                "proto_b_final_loss_mean": pb_m,
                "proto_b_final_loss_std": pb_s,
                "proto_b_final_ppl_mean": pb_ppl_m,
                "proto_b_final_ppl_std": pb_ppl_s,
                "proto_b_best_loss_mean": pb_best_m,
                "proto_b_best_loss_std": pb_best_s,
                "paired_diff_proto_a_mean": da_m,
                "paired_diff_proto_a_std": da_s,
                "paired_diff_proto_b_mean": db_m,
                "paired_diff_proto_b_std": db_s,
            }
        )
    _write_csv(csv_dir / "comparison_table_8_architectures.csv", summary_table_rows)

    # Explicit paired comparisons table
    paired_specs = [
        ("plus_only_q28", "antisymmetric_only_q28", "Plus-Only Q28 - Antisymmetric-Only Q28"),
        ("full_qr_plus_q112", "full_qr_minus_q112", "Full QR Plus Q112 - Full QR Minus Q112"),
        ("plus_only_q28", "standard_q28", "Plus-Only Q28 - Standard Q28"),
        ("plus_only_q28", "symmetric_only_q56", "Plus-Only Q28 - Symmetric-Only Q56"),
        ("antisymmetric_only_q28", "standard_q28", "Antisymmetric-Only Q28 - Standard Q28"),
        ("antisymmetric_only_q28", "symmetric_only_q56", "Antisymmetric-Only Q28 - Symmetric-Only Q56"),
        ("full_qr_plus_q112", "standard_q112", "Full QR Plus Q112 - Original Standard Q112"),
        ("full_qr_plus_q112", "diagonal_free_standard_q112", "Full QR Plus Q112 - Diagonal-Free Standard Q112"),
        ("full_qr_minus_q112", "standard_q112", "Full QR Minus Q112 - Original Standard Q112"),
        ("full_qr_minus_q112", "diagonal_free_standard_q112", "Full QR Minus Q112 - Diagonal-Free Standard Q112"),
        ("diagonal_free_standard_q112", "standard_q112", "Diagonal-Free Standard Q112 - Original Standard Q112"),
    ]
    paired_rows: List[Dict[str, Any]] = []
    for a_key, b_key, desc in paired_specs:
        diffs_a = [by_arch_seed[(a_key, s)]["proto_a_final_loss"] - by_arch_seed[(b_key, s)]["proto_a_final_loss"] for s in seeds]
        diffs_b = [by_arch_seed[(a_key, s)]["proto_b_final_loss"] - by_arch_seed[(b_key, s)]["proto_b_final_loss"] for s in seeds]
        da_m, da_s = _mean_std(diffs_a)
        db_m, db_s = _mean_std(diffs_b)
        paired_rows.append(
            {
                "comparison": desc,
                "arch_a": a_key,
                "arch_b": b_key,
                "seed_1337_delta_a": diffs_a[0],
                "seed_2024_delta_a": diffs_a[1],
                "seed_42_delta_a": diffs_a[2],
                "mean_delta_a": da_m,
                "std_delta_a": da_s,
                "seed_1337_delta_b": diffs_b[0],
                "seed_2024_delta_b": diffs_b[1],
                "seed_42_delta_b": diffs_b[2],
                "mean_delta_b": db_m,
                "std_delta_b": db_s,
                "consistent_direction_a": bool(all(d < 0 for d in diffs_a) or all(d > 0 for d in diffs_a)),
                "consistent_direction_b": bool(all(d < 0 for d in diffs_b) or all(d > 0 for d in diffs_b)),
            }
        )
    _write_csv(csv_dir / "paired_differences.csv", paired_rows)

    # ------------------------------------------------------------------
    # 2. Exhaustive Section 9 per-layer & per-head diagnostics
    # ------------------------------------------------------------------
    print("\n=== Computing Section 9 Per-Layer & Per-Head Spectral/Attention Diagnostics ===", flush=True)
    head_rows: List[Dict[str, Any]] = []
    eval_grad_rows: List[Dict[str, Any]] = []
    all_saved_arrays: Dict[str, np.ndarray] = {}
    spec_by_key = {s["key"]: s for s in COMPARISON_ARCHS}

    for ak in FOCAL_ARCH_KEYS:
        spec = spec_by_key[ak]
        for seed in seeds:
            ckpt_path = spec["dir"] / f"seed_{seed}" / "final.pt"
            model = _load_model(spec, ckpt_path, config, vocab_size, device)
            collect = (seed == 1337)
            rows, arrs = compute_head_and_layer_diagnostics(
                model, ak, seed, proto_a_batches, device, collect_samples=collect
            )
            head_rows.extend(rows)
            all_saved_arrays.update(arrs)

            # Eval-pass gradient norms for exact apple-to-apple Q/R gradient comparison
            gn = compute_eval_grad_norms(model, proto_a_batches, device, max_batches=4)
            for l_idx, gdict in gn.items():
                eval_grad_rows.append({"arch_key": ak, "seed": int(seed), "layer": l_idx, **gdict})
            del model
            print(f"  Diagnostics done: {ak} seed={seed}", flush=True)

    _write_csv(csv_dir / "diagnostics_per_layer_head_per_seed.csv", head_rows)
    _write_csv(csv_dir / "eval_gradient_norms_per_layer_seed.csv", eval_grad_rows)
    np.savez_compressed(json_dir / "heatmap_and_spectrum_samples.npz", **all_saved_arrays)

    # Aggregate per (arch_key, layer, head) across seeds
    numeric_cols = [
        k for k in head_rows[0].keys()
        if k not in ("arch_key", "seed", "layer", "head", "is_plus", "is_full_qr")
    ]
    by_lh_rows: List[Dict[str, Any]] = []
    for ak in FOCAL_ARCH_KEYS:
        for l_idx in range(4):
            for h_idx in range(4):
                sub = [r for r in head_rows if r["arch_key"] == ak and r["layer"] == l_idx and r["head"] == h_idx]
                row: Dict[str, Any] = {"arch_key": ak, "layer": l_idx, "head": h_idx}
                for col in numeric_cols:
                    m, s = _mean_std([r[col] for r in sub])
                    row[f"{col}_mean"] = m
                    row[f"{col}_std"] = s
                by_lh_rows.append(row)
    _write_csv(csv_dir / "diagnostics_by_layer_head_summary.csv", by_lh_rows)

    # Aggregate per (arch_key, layer) across seeds and heads
    by_layer_rows: List[Dict[str, Any]] = []
    for ak in FOCAL_ARCH_KEYS:
        for l_idx in range(4):
            # Average over heads within each seed first, then mean/std across seeds
            per_seed_layer = []
            for seed in seeds:
                sub = [r for r in head_rows if r["arch_key"] == ak and r["seed"] == seed and r["layer"] == l_idx]
                per_seed_layer.append({col: float(np.mean([r[col] for r in sub])) for col in numeric_cols})
            g_sub = [g for g in eval_grad_rows if g["arch_key"] == ak and g["layer"] == l_idx]
            row = {"arch_key": ak, "layer": l_idx}
            for col in numeric_cols:
                m, s = _mean_std([d[col] for d in per_seed_layer])
                row[f"{col}_mean"] = m
                row[f"{col}_std"] = s
            for gcol in ("eval_wq_grad_norm", "eval_wr_grad_norm", "eval_wv_grad_norm", "eval_wo_grad_norm"):
                m, s = _mean_std([g[gcol] for g in g_sub])
                row[f"{gcol}_mean"] = m
                row[f"{gcol}_std"] = s
            by_layer_rows.append(row)
    _write_csv(csv_dir / "diagnostics_by_layer_summary.csv", by_layer_rows)

    # Aggregate overall per arch_key across seeds
    overall_diag_rows: List[Dict[str, Any]] = []
    for ak in FOCAL_ARCH_KEYS:
        per_seed_all = []
        for seed in seeds:
            sub = [r for r in head_rows if r["arch_key"] == ak and r["seed"] == seed]
            per_seed_all.append({col: float(np.mean([r[col] for r in sub])) for col in numeric_cols})
        row = {"arch_key": ak}
        for col in numeric_cols:
            m, s = _mean_std([d[col] for d in per_seed_all])
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
        overall_diag_rows.append(row)
    _write_csv(csv_dir / "diagnostics_overall_summary.csv", overall_diag_rows)

    # Combine master JSON summary (including Stage A if already present)
    stage_a_path = Path(output_root) / "inference_intervention" / "stage_a_summary.json"
    stage_a_data = json.loads(stage_a_path.read_text(encoding="utf-8")) if stage_a_path.exists() else None

    master_summary = {
        "experiment": "qr_plus_operator",
        "seeds": list(seeds),
        "validation_seed": VALIDATION_SEED,
        "proto_a_tokens": 81920,
        "proto_b_tokens": 111539,
        "stage_a_inference_intervention": stage_a_data,
        "comparison_table_8_architectures": summary_table_rows,
        "paired_differences": paired_rows,
        "diagnostics_overall": overall_diag_rows,
        "diagnostics_by_layer": by_layer_rows,
    }
    write_json(json_dir / "master_analysis_summary.json", master_summary)
    print(f"\nAnalysis complete. Saved CSVs to {csv_dir} and JSONs to {json_dir}.", flush=True)
    return master_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze QR Plus vs QR Minus experiment")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_analysis(output_root=args.output_dir, device_str=args.device)


if __name__ == "__main__":
    main()
