"""Inference-Time Component Ablation of QR Antisymmetric Attention.

Evaluates trained Q28/V32 QR Antisymmetric Attention checkpoints (seeds 1337, 2024, 42)
under four global inference configurations:
  A. Full QR (original model): S = [QQ^T + 20(QR^T - RQ^T)] / sqrt(7)
  B. Symmetric-only: S = QQ^T / sqrt(7)
  C. Antisymmetric-only: S = 20(QR^T - RQ^T) / sqrt(7)
  D. Antisymmetric disabled after training: component-disable switch in QR attention

Additionally evaluates layer-wise ablation:
  Disables antisymmetric component for each individual layer (0..3) while keeping
  other layers unchanged.

Evaluates under both established protocols:
  Protocol A: 20 fixed validation batches (validation_seed=918273, 81,920 target tokens)
  Protocol B: Entire validation corpus (111,539 target tokens)

Collects per-layer and per-head score component norms and attention diagnostics.
Audits checkpoint integrity (SHA-256 before and after evaluation).
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches
from train_reduced_attention_70pct import (
    REDUCED_CONFIG_KEY,
    REDUCED_SEEDS,
    VALIDATION_SEED,
    verify_reduced_q28_model_contract,
)

OUTPUT_DIR = Path("outputs/qr_component_inference_ablation")
CHECKPOINT_ROOT = Path("outputs/reduced_attention_70_percent/qr_antisymmetric_q28")
PRIMARY_SEEDS = [1337, 2024, 42]
PRIMARY_CKPT_KINDS = ["final", "best"]


def compute_file_sha256(path: Path) -> str:
    """Compute standard SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoint_hashes(root: Path, seeds: List[int], kinds: List[str]) -> Dict[str, str]:
    """Audit SHA256 hashes of all target checkpoints."""
    hashes = {}
    for seed in seeds:
        for kind in kinds:
            ckpt_path = root / f"seed_{seed}" / f"{kind}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            hashes[str(ckpt_path)] = compute_file_sha256(ckpt_path)
    return hashes


@torch.inference_mode()
def collect_score_component_norms(
    model: DecoderLM,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    lambda_val: float = 20.0,
) -> List[Dict[str, Any]]:
    """Compute Frobenius norms and energy fractions of score components per layer and head before causal masking."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attention.n_heads

    # Accumulators per layer and head
    accum_sym = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    accum_anti = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    accum_anti_raw = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    accum_full = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    accum_sq_sym = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    accum_sq_anti = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    accum_asym = [0.0 for _ in range(n_layers)]
    n_batches = 0

    try:
        for x_cpu, _ in validation_batches:
            x = model._embed(x_cpu.to(device))
            for layer_idx, block in enumerate(model.blocks):
                norm_x = block.attention_norm(x)
                attn = block.attention
                q, r, _ = attn._project_qrv(norm_x)
                scale = math.sqrt(attn.q_head_dim)

                # Pre-mask components: (B, H, T, T)
                s_sym = (q @ q.transpose(-2, -1)) / scale
                s_anti_raw = ((q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1))) / scale
                s_anti = float(lambda_val) * s_anti_raw
                s_full = s_sym + s_anti

                # Frobenius norms per head: shape (B, H)
                fnorm_sym = torch.linalg.norm(s_sym, dim=(-2, -1))
                fnorm_anti = torch.linalg.norm(s_anti, dim=(-2, -1))
                fnorm_anti_raw = torch.linalg.norm(s_anti_raw, dim=(-2, -1))
                fnorm_full = torch.linalg.norm(s_full, dim=(-2, -1))

                # Asymmetry metric
                diff = s_full - s_full.transpose(-2, -1)
                plus = s_full + s_full.transpose(-2, -1)
                asym = float(
                    (torch.linalg.norm(diff, dim=(-2, -1)) / (torch.linalg.norm(plus, dim=(-2, -1)) + 1e-8)).mean().item()
                )
                accum_asym[layer_idx] += asym

                for h in range(n_heads):
                    accum_sym[layer_idx][h] += float(fnorm_sym[:, h].mean().item())
                    accum_anti[layer_idx][h] += float(fnorm_anti[:, h].mean().item())
                    accum_anti_raw[layer_idx][h] += float(fnorm_anti_raw[:, h].mean().item())
                    accum_full[layer_idx][h] += float(fnorm_full[:, h].mean().item())
                    accum_sq_sym[layer_idx][h] += float((fnorm_sym[:, h] ** 2).mean().item())
                    accum_sq_anti[layer_idx][h] += float((fnorm_anti[:, h] ** 2).mean().item())

                x = block(x)
            n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    records = []
    # Build per-layer and per-head rows
    all_sym_sum = 0.0
    all_anti_sum = 0.0
    all_anti_raw_sum = 0.0
    all_full_sum = 0.0
    all_sq_sym_sum = 0.0
    all_sq_anti_sum = 0.0
    total_heads = n_layers * n_heads

    for layer_idx in range(n_layers):
        layer_sym = 0.0
        layer_anti = 0.0
        layer_anti_raw = 0.0
        layer_full = 0.0
        layer_sq_sym = 0.0
        layer_sq_anti = 0.0
        layer_asym = accum_asym[layer_idx] / n_batches

        for h in range(n_heads):
            m_sym = accum_sym[layer_idx][h] / n_batches
            m_anti = accum_anti[layer_idx][h] / n_batches
            m_anti_raw = accum_anti_raw[layer_idx][h] / n_batches
            m_full = accum_full[layer_idx][h] / n_batches
            m_sq_sym = accum_sq_sym[layer_idx][h] / n_batches
            m_sq_anti = accum_sq_anti[layer_idx][h] / n_batches

            layer_sym += m_sym / n_heads
            layer_anti += m_anti / n_heads
            layer_anti_raw += m_anti_raw / n_heads
            layer_full += m_full / n_heads
            layer_sq_sym += m_sq_sym / n_heads
            layer_sq_anti += m_sq_anti / n_heads

            e_frac_anti = m_sq_anti / (m_sq_sym + m_sq_anti + 1e-8)
            e_frac_sym = m_sq_sym / (m_sq_sym + m_sq_anti + 1e-8)
            lin_frac_anti = m_anti / (m_sym + m_anti + 1e-8)
            lin_frac_sym = m_sym / (m_sym + m_anti + 1e-8)

            records.append({
                "layer": layer_idx,
                "head": h,
                "norm_symmetric": m_sym,
                "norm_antisymmetric": m_anti,
                "norm_antisymmetric_raw": m_anti_raw,
                "norm_full": m_full,
                "ratio_anti_to_sym": m_anti / (m_sym + 1e-8),
                "ratio_anti_raw_to_sym": m_anti_raw / (m_sym + 1e-8),
                "energy_fraction_antisymmetric": e_frac_anti,
                "energy_fraction_symmetric": e_frac_sym,
                "linear_fraction_antisymmetric": lin_frac_anti,
                "linear_fraction_symmetric": lin_frac_sym,
                "score_asymmetry": layer_asym,
            })

        # Layer average row
        e_layer_anti = layer_sq_anti / (layer_sq_sym + layer_sq_anti + 1e-8)
        e_layer_sym = layer_sq_sym / (layer_sq_sym + layer_sq_anti + 1e-8)
        lin_layer_anti = layer_anti / (layer_sym + layer_anti + 1e-8)
        lin_layer_sym = layer_sym / (layer_sym + layer_anti + 1e-8)

        records.append({
            "layer": layer_idx,
            "head": "all",
            "norm_symmetric": layer_sym,
            "norm_antisymmetric": layer_anti,
            "norm_antisymmetric_raw": layer_anti_raw,
            "norm_full": layer_full,
            "ratio_anti_to_sym": layer_anti / (layer_sym + 1e-8),
            "ratio_anti_raw_to_sym": layer_anti_raw / (layer_sym + 1e-8),
            "energy_fraction_antisymmetric": e_layer_anti,
            "energy_fraction_symmetric": e_layer_sym,
            "linear_fraction_antisymmetric": lin_layer_anti,
            "linear_fraction_symmetric": lin_layer_sym,
            "score_asymmetry": layer_asym,
        })

        all_sym_sum += layer_sym / n_layers
        all_anti_sum += layer_anti / n_layers
        all_anti_raw_sum += layer_anti_raw / n_layers
        all_full_sum += layer_full / n_layers
        all_sq_sym_sum += layer_sq_sym / n_layers
        all_sq_anti_sum += layer_sq_anti / n_layers

    # Network-wide "all" row
    e_all_anti = all_sq_anti_sum / (all_sq_sym_sum + all_sq_anti_sum + 1e-8)
    e_all_sym = all_sq_sym_sum / (all_sq_sym_sum + all_sq_anti_sum + 1e-8)
    lin_all_anti = all_anti_sum / (all_sym_sum + all_anti_sum + 1e-8)
    lin_all_sym = all_sym_sum / (all_sym_sum + all_anti_sum + 1e-8)
    mean_network_asym = sum(accum_asym) / (n_layers * n_batches)

    records.insert(0, {
        "layer": "all",
        "head": "all",
        "norm_symmetric": all_sym_sum,
        "norm_antisymmetric": all_anti_sum,
        "norm_antisymmetric_raw": all_anti_raw_sum,
        "norm_full": all_full_sum,
        "ratio_anti_to_sym": all_anti_sum / (all_sym_sum + 1e-8),
        "ratio_anti_raw_to_sym": all_anti_raw_sum / (all_sym_sum + 1e-8),
        "energy_fraction_antisymmetric": e_all_anti,
        "energy_fraction_symmetric": e_all_sym,
        "linear_fraction_antisymmetric": lin_all_anti,
        "linear_fraction_symmetric": lin_all_sym,
        "score_asymmetry": mean_network_asym,
    })

    return records


@torch.inference_mode()
def collect_attention_diagnostics_for_config(
    model: DecoderLM,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    config_name: str,
    lambda_val: float = 20.0,
) -> List[Dict[str, Any]]:
    """Compute attention entropy, max probability, diagonal mass, and logit stats for a given inference configuration."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)

    # Set up configuration mode and switch
    if config_name == "full_qr":
        model.set_qr_component_mode("full")
        model.set_qr_antisymmetric_enabled(True)
    elif config_name == "symmetric_only":
        model.set_qr_component_mode("symmetric_only")
        model.set_qr_antisymmetric_enabled(True)
    elif config_name == "antisymmetric_only":
        model.set_qr_component_mode("antisymmetric_only")
        model.set_qr_antisymmetric_enabled(True)
    elif config_name == "antisymmetric_disabled":
        model.set_qr_component_mode("full")
        model.set_qr_antisymmetric_enabled(False)
    else:
        raise ValueError(f"Unknown config_name {config_name}")

    accum: List[Dict[str, float]] = [{} for _ in range(n_layers)]
    n_batches = 0

    try:
        for x_cpu, _ in validation_batches:
            x = model._embed(x_cpu.to(device))
            for layer_idx, block in enumerate(model.blocks):
                norm_x = block.attention_norm(x)
                attn = block.attention
                maps = attn.attention_maps(norm_x, lambda_val=lambda_val)
                raw_scores = maps["raw_scores"]      # (B, H, T, T)
                probs = maps["probabilities"].double() # (B, H, T, T)
                T = raw_scores.size(-1)

                # Raw logits stats (pre-mask)
                r_mean = float(raw_scores.mean().item())
                r_std = float(raw_scores.std(unbiased=False).item())

                # Causal logits stats
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(diagonal=0)
                causal_scores = raw_scores[..., causal_mask]
                c_mean = float(causal_scores.mean().item())
                c_std = float(causal_scores.std(unbiased=False).item())

                # Attention entropy (nats)
                p_clamp = probs.clamp_min(1e-300)
                entropy = -(probs * p_clamp.log()).sum(dim=-1) # (B, H, T)
                m_entropy = float(entropy.mean().item())
                m_entropy_non_initial = float(entropy[..., 1:].mean().item()) if T > 1 else 0.0

                # Normalized entropy
                # Position-aware: H / ln(i+1) for i >= 1
                pos_denom = torch.log(torch.arange(1, T + 1, device=x.device, dtype=probs.dtype))
                norm_ent_pos = float((entropy[..., 1:] / pos_denom[1:]).mean().item()) if T > 1 else 0.0
                # Context-aware: H / ln(T)
                norm_ent_ctx = float((entropy / math.log(T)).mean().item())

                # Max probability & diagonal mass
                m_max_p = float(probs.amax(dim=-1).mean().item())
                m_diag = float(probs.diagonal(dim1=-2, dim2=-1).mean().item())

                layer_metrics = {
                    "mean_attention_entropy": m_entropy,
                    "mean_attention_entropy_non_initial": m_entropy_non_initial,
                    "normalized_attention_entropy_pos": norm_ent_pos,
                    "normalized_attention_entropy_ctx": norm_ent_ctx,
                    "mean_max_attention_prob": m_max_p,
                    "mean_diagonal_mass": m_diag,
                    "raw_logits_mean": r_mean,
                    "raw_logits_std": r_std,
                    "causal_logits_mean": c_mean,
                    "causal_logits_std": c_std,
                }
                for k, v in layer_metrics.items():
                    accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v

                x = block(x)
            n_batches += 1
    finally:
        # Reset model back to full
        model.set_qr_component_mode("full")
        model.set_qr_antisymmetric_enabled(True)
        model.train(was_training)

    if n_batches == 0:
        return []

    records = []
    all_layer_metrics: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        l_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        for k, v in l_avg.items():
            all_layer_metrics[k] = all_layer_metrics.get(k, 0.0) + v / n_layers
        records.append({
            "configuration": config_name,
            "layer": layer_idx,
            **l_avg,
        })

    records.insert(0, {
        "configuration": config_name,
        "layer": "all",
        **all_layer_metrics,
    })
    return records


def run_experiment() -> Dict[str, Any]:
    """Execute complete inference-time ablation across 3 seeds and 4 configurations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== QR Antisymmetric Attention Component Ablation ===", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"PyTorch version: {torch.__version__} | Python: {platform.python_version()}", flush=True)

    # 1. Audit checkpoint hashes BEFORE evaluation
    print("\n[Step 1/6] Auditing checkpoint hashes before evaluation...", flush=True)
    pre_hashes = audit_checkpoint_hashes(CHECKPOINT_ROOT, PRIMARY_SEEDS, PRIMARY_CKPT_KINDS)
    for p, h in pre_hashes.items():
        print(f"  {Path(p).parent.name}/{Path(p).name}: {h[:16]}...{h[-8:]}", flush=True)

    # Load dataset
    sample_cfg_path = CHECKPOINT_ROOT / "seed_1337" / "config.json"
    with sample_cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)
    dataset = CharacterDataset(cfg)
    print(f"Dataset fingerprint: {dataset.fingerprint}", flush=True)
    print(f"Train chars: {len(dataset.train)} | Val chars: {len(dataset.validation)} | Vocab: {len(dataset.vocab)}", flush=True)

    # Build validation protocols
    val_20_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    tokens_20 = sum(y.numel() for _, y in val_20_batches)
    assert tokens_20 == 81920, f"Expected 81920 Protocol A tokens, got {tokens_20}"

    full_val_batches = full_corpus_validation_batches(
        dataset.validation, context_length=cfg.context_length, batch_size=cfg.batch_size
    )
    tokens_full = sum(y.numel() for _, y in full_val_batches)
    assert tokens_full == 111539, f"Expected 111539 Protocol B tokens, got {tokens_full}"

    print(f"Protocol A: {len(val_20_batches)} batches, {tokens_20} target tokens", flush=True)
    print(f"Protocol B: {len(full_val_batches)} batches, {tokens_full} target tokens", flush=True)

    # Global configurations to evaluate
    configurations = [
        ("full_qr", "Full QR (Control)"),
        ("symmetric_only", "Symmetric-Only"),
        ("antisymmetric_only", "Antisymmetric-Only"),
        ("antisymmetric_disabled", "Antisymmetric Disabled"),
    ]

    protocols = [
        ("protocol_a_20batch", "Protocol A (20 batches, 81,920 tokens)", val_20_batches, tokens_20),
        ("protocol_b_full_corpus", "Protocol B (Full Corpus, 111,539 tokens)", full_val_batches, tokens_full),
    ]

    per_seed_records: List[Dict[str, Any]] = []
    layerwise_records: List[Dict[str, Any]] = []
    score_norm_records: List[Dict[str, Any]] = []
    attention_diag_records: List[Dict[str, Any]] = []

    print("\n[Step 2/6] Evaluating global configurations across seeds...", flush=True)
    for seed in PRIMARY_SEEDS:
        for ckpt_kind in PRIMARY_CKPT_KINDS:
            ckpt_path = CHECKPOINT_ROOT / f"seed_{seed}" / f"{ckpt_kind}.pt"
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            lam = float(payload.get("qr_lambda", 20.0))

            model = DecoderLM(cfg, len(dataset.vocab), payload["model_type"]).to(device)
            model.load_state_dict(payload["model"], strict=True)
            model.set_qr_lambda(lam)
            model.eval()

            # Verify parameter contract
            accounting = verify_reduced_q28_model_contract(model, arch_key="qr_antisymmetric", expected_lambda=lam)

            # Store Full QR baseline losses for computing delta
            full_qr_losses = {}

            # First pass: evaluate 4 configurations under both protocols
            for cfg_name, cfg_label in configurations:
                # Configure model
                if cfg_name == "full_qr":
                    model.set_qr_component_mode("full")
                    model.set_qr_antisymmetric_enabled(True)
                elif cfg_name == "symmetric_only":
                    model.set_qr_component_mode("symmetric_only")
                    model.set_qr_antisymmetric_enabled(True)
                elif cfg_name == "antisymmetric_only":
                    model.set_qr_component_mode("antisymmetric_only")
                    model.set_qr_antisymmetric_enabled(True)
                elif cfg_name == "antisymmetric_disabled":
                    model.set_qr_component_mode("full")
                    model.set_qr_antisymmetric_enabled(False)

                for prot_id, prot_label, batches, n_tokens in protocols:
                    t0 = time.perf_counter()
                    loss = evaluate_loss(model, batches, device)
                    eval_sec = time.perf_counter() - t0
                    ppl = math.exp(loss)

                    if cfg_name == "full_qr":
                        full_qr_losses[prot_id] = loss

                    loss_inc = loss - full_qr_losses[prot_id]
                    base_loss = full_qr_losses[prot_id]
                    rel_loss_pct = (loss_inc / base_loss) * 100.0
                    ppl_inc = ppl - math.exp(base_loss)
                    collapsed = bool(loss > 2.5 or rel_loss_pct > 50.0)

                    record = {
                        "seed": seed,
                        "checkpoint_kind": ckpt_kind,
                        "configuration": cfg_name,
                        "config_label": cfg_label,
                        "protocol": prot_id,
                        "protocol_label": prot_label,
                        "loss": loss,
                        "perplexity": ppl,
                        "tokens_evaluated": n_tokens,
                        "eval_seconds": eval_sec,
                        "loss_increase_vs_full": loss_inc,
                        "ppl_increase_vs_full": ppl_inc,
                        "relative_loss_pct": rel_loss_pct,
                        "performance_collapse": collapsed,
                        "checkpoint_path": str(ckpt_path),
                    }
                    per_seed_records.append(record)

                    print(
                        f"  seed={seed} [{ckpt_kind}] {cfg_name:24s} | {prot_id:22s} | "
                        f"loss={loss:.4f} ppl={ppl:.2f} (Δ={loss_inc:+.4f}, {rel_loss_pct:+.1f}%) in {eval_sec:.2f}s",
                        flush=True,
                    )

            # Reset model to full
            model.set_qr_component_mode("full")
            model.set_qr_antisymmetric_enabled(True)

            # Evaluate layer-wise ablation (disable each layer's antisymmetric component individually)
            print(f"  --> Running layer-wise ablation for seed={seed} [{ckpt_kind}]...", flush=True)
            for target_layer in range(len(model.blocks)):
                model.set_qr_antisymmetric_enabled(True)
                model.set_qr_antisymmetric_enabled(False, layer=target_layer)

                for prot_id, prot_label, batches, n_tokens in protocols:
                    t0 = time.perf_counter()
                    loss = evaluate_loss(model, batches, device)
                    eval_sec = time.perf_counter() - t0
                    ppl = math.exp(loss)
                    base_loss = full_qr_losses[prot_id]
                    loss_inc = loss - base_loss
                    rel_loss_pct = (loss_inc / base_loss) * 100.0
                    ppl_inc = ppl - math.exp(base_loss)

                    layerwise_records.append({
                        "seed": seed,
                        "checkpoint_kind": ckpt_kind,
                        "disabled_layer": target_layer,
                        "layer_label": f"Layer {target_layer} Disabled",
                        "protocol": prot_id,
                        "protocol_label": prot_label,
                        "loss": loss,
                        "perplexity": ppl,
                        "tokens_evaluated": n_tokens,
                        "eval_seconds": eval_sec,
                        "loss_increase_vs_full": loss_inc,
                        "ppl_increase_vs_full": ppl_inc,
                        "relative_loss_pct": rel_loss_pct,
                        "checkpoint_path": str(ckpt_path),
                    })
                    print(
                        f"    disabled layer {target_layer} | {prot_id:22s} | "
                        f"loss={loss:.4f} ppl={ppl:.2f} (Δ={loss_inc:+.4f}) in {eval_sec:.2f}s",
                        flush=True,
                    )

            # Reset model to full
            model.set_qr_antisymmetric_enabled(True)

            # Diagnostics: Collect score component norms (pre-mask)
            print(f"  --> Collecting score component norms for seed={seed} [{ckpt_kind}]...", flush=True)
            score_norms = collect_score_component_norms(model, val_20_batches, device, lambda_val=lam)
            for sn in score_norms:
                score_norm_records.append({
                    "seed": seed,
                    "checkpoint_kind": ckpt_kind,
                    **sn,
                })

            # Diagnostics: Collect attention entropy & distribution for all 4 configurations
            print(f"  --> Collecting attention diagnostics for seed={seed} [{ckpt_kind}]...", flush=True)
            for cfg_name, _ in configurations:
                attn_diags = collect_attention_diagnostics_for_config(
                    model, val_20_batches, device, config_name=cfg_name, lambda_val=lam
                )
                for ad in attn_diags:
                    attention_diag_records.append({
                        "seed": seed,
                        "checkpoint_kind": ckpt_kind,
                        **ad,
                    })

            model.cpu()
            del model
            torch.cuda.empty_cache()

    # 3. Audit checkpoint hashes AFTER evaluation
    print("\n[Step 3/6] Auditing checkpoint hashes after evaluation...", flush=True)
    post_hashes = audit_checkpoint_hashes(CHECKPOINT_ROOT, PRIMARY_SEEDS, PRIMARY_CKPT_KINDS)
    audit_passed = True
    integrity_rows = []
    for p, h_pre in pre_hashes.items():
        h_post = post_hashes[p]
        matches = bool(h_pre == h_post)
        if not matches:
            audit_passed = False
        integrity_rows.append({
            "checkpoint_path": p,
            "sha256_before": h_pre,
            "sha256_after": h_post,
            "verified_unchanged": matches,
        })
        print(f"  {Path(p).parent.name}/{Path(p).name}: {'VERIFIED MATCH' if matches else 'HASH MISMATCH ERROR'}", flush=True)

    if not audit_passed:
        raise RuntimeError("FATAL: Checkpoint hashes differed after evaluation! Checkpoint corruption detected!")

    # 4. Generate Summary Aggregations
    print("\n[Step 4/6] Aggregating results across seeds...", flush=True)
    summary_records: List[Dict[str, Any]] = []
    for ckpt_kind in PRIMARY_CKPT_KINDS:
        for cfg_name, cfg_label in configurations:
            for prot_id, prot_label, _, _ in protocols:
                sub = [
                    r for r in per_seed_records
                    if r["checkpoint_kind"] == ckpt_kind
                    and r["configuration"] == cfg_name
                    and r["protocol"] == prot_id
                ]
                losses = [r["loss"] for r in sub]
                ppls = [r["perplexity"] for r in sub]
                loss_incs = [r["loss_increase_vs_full"] for r in sub]
                ppl_incs = [r["ppl_increase_vs_full"] for r in sub]
                rel_losses = [r["relative_loss_pct"] for r in sub]

                n = len(losses)
                m_loss = sum(losses) / n
                s_loss = math.sqrt(sum((x - m_loss) ** 2 for x in losses) / (n - 1)) if n > 1 else 0.0
                m_ppl = sum(ppls) / n
                s_ppl = math.sqrt(sum((x - m_ppl) ** 2 for x in ppls) / (n - 1)) if n > 1 else 0.0
                m_inc = sum(loss_incs) / n
                s_inc = math.sqrt(sum((x - m_inc) ** 2 for x in loss_incs) / (n - 1)) if n > 1 else 0.0
                m_ppl_inc = sum(ppl_incs) / n
                s_ppl_inc = math.sqrt(sum((x - m_ppl_inc) ** 2 for x in ppl_incs) / (n - 1)) if n > 1 else 0.0
                m_rel = sum(rel_losses) / n
                s_rel = math.sqrt(sum((x - m_rel) ** 2 for x in rel_losses) / (n - 1)) if n > 1 else 0.0

                summary_records.append({
                    "checkpoint_kind": ckpt_kind,
                    "configuration": cfg_name,
                    "config_label": cfg_label,
                    "protocol": prot_id,
                    "protocol_label": prot_label,
                    "n_seeds": n,
                    "mean_loss": m_loss,
                    "std_loss": s_loss,
                    "mean_perplexity": m_ppl,
                    "std_perplexity": s_ppl,
                    "mean_loss_increase": m_inc,
                    "std_loss_increase": s_inc,
                    "mean_ppl_increase": m_ppl_inc,
                    "std_ppl_increase": s_ppl_inc,
                    "mean_relative_loss_pct": m_rel,
                    "std_relative_loss_pct": s_rel,
                })

    # 5. Consistency Check: Verify Config B and Config D produce identical values
    print("\n[Step 5/6] Verifying Config B and Config D consistency check...", flush=True)
    max_bd_diff = 0.0
    for r_b in per_seed_records:
        if r_b["configuration"] == "symmetric_only":
            match = next(
                r for r in per_seed_records
                if r["configuration"] == "antisymmetric_disabled"
                and r["seed"] == r_b["seed"]
                and r["checkpoint_kind"] == r_b["checkpoint_kind"]
                and r["protocol"] == r_b["protocol"]
            )
            diff = abs(r_b["loss"] - match["loss"])
            max_bd_diff = max(max_bd_diff, diff)

    print(f"Max difference between Symmetric-Only (B) and Antisymmetric Disabled (D): {max_bd_diff:.2e}", flush=True)
    if max_bd_diff > 1e-11:
        raise AssertionError(f"Consistency check failed: B and D differ by {max_bd_diff}")

    # 6. Save all CSVs and JSONs
    print("\n[Step 6/6] Writing output files...", flush=True)

    def write_csv(path: Path, data: List[Dict[str, Any]]) -> None:
        if not data:
            return
        keys = list(data[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"  Wrote {path.name} ({len(data)} rows)", flush=True)

    write_csv(OUTPUT_DIR / "qr_component_ablation_summary.csv", summary_records)
    write_csv(OUTPUT_DIR / "qr_component_ablation_per_seed.csv", per_seed_records)
    write_csv(OUTPUT_DIR / "qr_layerwise_ablation.csv", layerwise_records)
    write_csv(OUTPUT_DIR / "qr_score_component_norms.csv", score_norm_records)
    write_csv(OUTPUT_DIR / "qr_attention_diagnostics.csv", attention_diag_records)

    # Checkpoint integrity audit
    with (OUTPUT_DIR / "checkpoint_integrity_audit.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "status": "passed",
                "all_checkpoints_unchanged": audit_passed,
                "checkpoints_evaluated": integrity_rows,
            },
            f,
            indent=2,
        )
    print("  Wrote checkpoint_integrity_audit.json", flush=True)

    # Reproducibility info
    repro_info = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seeds": PRIMARY_SEEDS,
        "checkpoint_kinds": PRIMARY_CKPT_KINDS,
        "lambda_fixed": 20.0,
        "sqrt_scaling": "sqrt(7) = 2.6457513110645906",
        "protocol_a": {
            "name": "20-batch validation",
            "seed": VALIDATION_SEED,
            "target_tokens": tokens_20,
            "batches": len(val_20_batches),
        },
        "protocol_b": {
            "name": "full-corpus validation",
            "target_tokens": tokens_full,
            "batches": len(full_val_batches),
        },
        "consistency_check_max_bd_diff": max_bd_diff,
    }
    with (OUTPUT_DIR / "reproducibility_info.json").open("w", encoding="utf-8") as f:
        json.dump(repro_info, f, indent=2)
    print("  Wrote reproducibility_info.json", flush=True)

    print("\nAll evaluation stages complete!", flush=True)
    return {
        "summary": summary_records,
        "per_seed": per_seed_records,
        "layerwise": layerwise_records,
        "score_norms": score_norm_records,
        "attention_diags": attention_diag_records,
    }


if __name__ == "__main__":
    run_experiment()
