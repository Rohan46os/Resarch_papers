"""Gradient Surgery for Shared-Q Causal Self-Attention.

Implements Query-preserving and Key-preserving gradient projection strategies
to eliminate destructive interference between Query-role and Key-role gradients
on the shared Query projection weights (Wq), without modifying forward-pass
mathematical behaviour or non-Wq parameter updates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from model import DecoderLM


STRATEGIES = ("normal", "query_preserving", "key_preserving")


def compute_gradient_conflict_metrics(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_final: Optional[torch.Tensor] = None,
    grad_key_projected: Optional[torch.Tensor] = None,
    grad_query_projected: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Calculate gradient cosine similarity, norms, cancellation, and projection metrics.

    Args:
        grad_query: Query-role weight gradient tensor (H * dk, D)
        grad_key: Key-role weight gradient tensor (H * dk, D)
        grad_final: Optional final gradient applied to weights (defaults to g_q + g_k)
        grad_key_projected: Optional projected key gradient (for query-preserving)
        grad_query_projected: Optional projected query gradient (for key-preserving)
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary containing all gradient interference and surgery metrics.
    """
    g_total_raw = grad_query + grad_key
    if grad_final is None:
        grad_final = g_total_raw

    norm_query = float(torch.linalg.norm(grad_query).item())
    norm_key = float(torch.linalg.norm(grad_key).item())
    norm_total_raw = float(torch.linalg.norm(g_total_raw).item())
    norm_final = float(torch.linalg.norm(grad_final).item())

    # Raw dot product and cosine similarity before projection
    dot_prod = float(torch.sum(grad_query * grad_key).item())
    denom_cos = norm_query * norm_key + eps
    cosine_sim_before = max(-1.0, min(1.0, dot_prod / denom_cos))

    # Raw cancellation ratio: 1 - ||g_q + g_k|| / (||g_q|| + ||g_k||)
    denom_cancel = norm_query + norm_key + eps
    cancellation_ratio_before = max(0.0, min(1.0, 1.0 - (norm_total_raw / denom_cancel)))

    query_gradient_ratio = norm_query / denom_cancel
    key_gradient_ratio = norm_key / denom_cancel

    # Cosine similarity and cancellation after projection
    if grad_key_projected is not None:
        dot_after = float(torch.sum(grad_query * grad_key_projected).item())
        norm_kp = float(torch.linalg.norm(grad_key_projected).item())
        cosine_sim_after = max(-1.0, min(1.0, dot_after / (norm_query * norm_kp + eps)))
        denom_cancel_after = norm_query + norm_kp + eps
        cancellation_ratio_after = max(0.0, min(1.0, 1.0 - (norm_final / denom_cancel_after)))
    elif grad_query_projected is not None:
        dot_after = float(torch.sum(grad_query_projected * grad_key).item())
        norm_qp = float(torch.linalg.norm(grad_query_projected).item())
        cosine_sim_after = max(-1.0, min(1.0, dot_after / (norm_qp * norm_key + eps)))
        denom_cancel_after = norm_qp + norm_key + eps
        cancellation_ratio_after = max(0.0, min(1.0, 1.0 - (norm_final / denom_cancel_after)))
    else:
        cosine_sim_after = cosine_sim_before
        cancellation_ratio_after = cancellation_ratio_before

    return {
        "query_gradient_norm": norm_query,
        "key_gradient_norm": norm_key,
        "total_gradient_norm": norm_total_raw,
        "projected_gradient_norm": norm_final,
        "dot_product": dot_prod,
        "cosine_similarity": cosine_sim_before,
        "cosine_similarity_after": cosine_sim_after,
        "cancellation_ratio": cancellation_ratio_before,
        "cancellation_ratio_after": cancellation_ratio_after,
        "query_gradient_ratio": query_gradient_ratio,
        "key_gradient_ratio": key_gradient_ratio,
        "conflict_detected": 1.0 if dot_prod < 0.0 else 0.0,
    }


def project_gradient_query_preserving(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Project Key gradient onto the hyperplane orthogonal to Query gradient when conflicting.

    If dot(g_query, g_key) < 0:
        g_key_projected = g_key - (dot / (||g_query||^2 + eps)) * g_query
    else:
        g_key_projected = g_key
    g_final = g_query + g_key_projected

    Returns:
        (g_final, g_key_projected, dot)
    """
    dot = float(torch.sum(grad_query * grad_key).item())
    if dot < 0.0:
        norm_q_sq = torch.sum(grad_query * grad_query)
        coeff = dot / (norm_q_sq + eps)
        grad_key_proj = grad_key - coeff * grad_query
    else:
        grad_key_proj = grad_key
    grad_final = grad_query + grad_key_proj
    return grad_final, grad_key_proj, dot


def project_gradient_key_preserving(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Project Query gradient onto the hyperplane orthogonal to Key gradient when conflicting.

    If dot(g_query, g_key) < 0:
        g_query_projected = g_query - (dot / (||g_key||^2 + eps)) * g_key
    else:
        g_query_projected = g_query
    g_final = g_query_projected + g_key

    Returns:
        (g_final, g_query_projected, dot)
    """
    dot = float(torch.sum(grad_query * grad_key).item())
    if dot < 0.0:
        norm_k_sq = torch.sum(grad_key * grad_key)
        coeff = dot / (norm_k_sq + eps)
        grad_query_proj = grad_query - coeff * grad_key
    else:
        grad_query_proj = grad_query
    grad_final = grad_query_proj + grad_key
    return grad_final, grad_query_proj, dot


def apply_gradient_surgery(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    strategy: str,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Apply the chosen gradient surgery strategy to a single layer's Wq gradients.

    Args:
        grad_query: (H * dk, D) Query role gradient
        grad_key: (H * dk, D) Key role gradient
        strategy: 'normal', 'query_preserving', or 'key_preserving'
        eps: Numerical epsilon

    Returns:
        (grad_final, layer_metrics, grad_key_proj, grad_query_proj)
    """
    g_key_proj = None
    g_query_proj = None

    if strategy == "normal":
        g_final = grad_query + grad_key
    elif strategy == "query_preserving":
        g_final, g_key_proj, _ = project_gradient_query_preserving(grad_query, grad_key, eps=eps)
    elif strategy == "key_preserving":
        g_final, g_query_proj, _ = project_gradient_key_preserving(grad_query, grad_key, eps=eps)
    else:
        raise ValueError(f"Unknown gradient strategy: {strategy}. Must be one of {STRATEGIES}")

    metrics = compute_gradient_conflict_metrics(
        grad_query=grad_query,
        grad_key=grad_key,
        grad_final=g_final,
        grad_key_projected=g_key_proj,
        grad_query_projected=g_query_proj,
        eps=eps,
    )
    return g_final, metrics, g_key_proj, g_query_proj


def decompose_shared_q_gradients(
    model: DecoderLM,
    loss: torch.Tensor,
) -> List[Dict[str, torch.Tensor]]:
    """Extract separate Query and Key role gradients with respect to Wq across all layers.

    Uses autograd.grad on cached q_query and q_key projection outputs, and reconstructs
    grad_query_Wq = grad_q_flat^T @ x_flat and grad_key_Wq = grad_k_flat^T @ x_flat.
    """
    queries = []
    keys = []
    for blk in model.blocks:
        attn = blk.attention
        if attn.last_q_query is None or attn.last_q_key is None or attn.last_x is None:
            raise RuntimeError("Block attention did not cache Q/K or input x. Ensure set_separate_qk(True) was called.")
        queries.append(attn.last_q_query)
        keys.append(attn.last_q_key)

    grads = torch.autograd.grad(loss, tuple(queries + keys), retain_graph=True)
    num_layers = len(model.blocks)
    q_grads = grads[:num_layers]
    k_grads = grads[num_layers:]

    layer_decompositions = []
    for l, blk in enumerate(model.blocks):
        attn = blk.attention
        x_norm = attn.last_x
        B, T, D = x_norm.shape
        out_dim = attn.n_heads * attn.head_dim

        x_flat = x_norm.reshape(B * T, D)
        g_q_flat = q_grads[l].reshape(B * T, out_dim)
        g_k_flat = k_grads[l].reshape(B * T, out_dim)

        grad_query_Wq = g_q_flat.T @ x_flat  # (out_dim, D)
        grad_key_Wq = g_k_flat.T @ x_flat    # (out_dim, D)

        layer_decompositions.append({
            "grad_query_Wq": grad_query_Wq,
            "grad_key_Wq": grad_key_Wq,
            "out_dim": out_dim,
            "in_dim": D,
            "n_heads": attn.n_heads,
            "head_dim": attn.head_dim,
        })

    return layer_decompositions


def execute_training_step(
    model: DecoderLM,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    gradient_clip: float,
    strategy: str = "normal",
    record_metrics: bool = False,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Execute a single training update with the specified gradient surgery strategy.

    Processing order:
      1. Forward pass.
      2. Calculate ordinary language modelling loss.
      3. Calculate separate Query-role and Key-role gradients.
      4. Apply selected gradient projection to Wq.
      5. Assign modified Wq gradient.
      6. Assign normal gradients to all other parameters (via loss.backward()).
      7. Apply existing gradient clipping.
      8. Perform existing AdamW optimizer step.

    Returns:
        (loss_value, step_metrics_or_None)
    """
    is_shared_q = model.attention_type != "standard"
    surgery_active = is_shared_q and strategy in ("query_preserving", "key_preserving")

    # If surgery is active or metrics are requested on a shared-q model, separate pathways
    need_separation = is_shared_q and (surgery_active or record_metrics)
    model.set_separate_qk(need_separation)

    optimizer.zero_grad(set_to_none=True)

    # 1 & 2. Forward pass and loss
    _, loss = model(x, y)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite loss encountered during training: {loss.item()}")
    loss_val = float(loss.item())

    step_metrics: Optional[Dict[str, Any]] = None

    if not need_separation:
        # Standard backpropagation
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        model.clear_qk_cache()
        return loss_val, None

    # Decompose Query and Key role gradients via autograd
    decomps = decompose_shared_q_gradients(model, loss)

    projected_grads = []
    layer_stats = {}
    head_stats = {}

    for l, decomp in enumerate(decomps):
        g_q = decomp["grad_query_Wq"]
        g_k = decomp["grad_key_Wq"]
        H = decomp["n_heads"]
        dk = decomp["head_dim"]

        g_final, layer_m, g_kp, g_qp = apply_gradient_surgery(g_q, g_k, strategy=strategy)
        projected_grads.append(g_final)

        if record_metrics:
            layer_stats[l] = layer_m
            # Per-head metrics
            layer_head_m = {}
            for h in range(H):
                q_h = g_q[h * dk : (h + 1) * dk, :]
                k_h = g_k[h * dk : (h + 1) * dk, :]
                fin_h = g_final[h * dk : (h + 1) * dk, :]
                kp_h = g_kp[h * dk : (h + 1) * dk, :] if g_kp is not None else None
                qp_h = g_qp[h * dk : (h + 1) * dk, :] if g_qp is not None else None
                head_metric = compute_gradient_conflict_metrics(
                    q_h, k_h, grad_final=fin_h,
                    grad_key_projected=kp_h, grad_query_projected=qp_h,
                )
                layer_head_m[h] = head_metric
            head_stats[l] = layer_head_m

    # 6. Assign normal gradients to all other parameters
    loss.backward()

    # 5. Assign modified Wq gradient (overwriting normal combined Wq gradient)
    if surgery_active:
        for l, blk in enumerate(model.blocks):
            blk.attention.q_proj.weight.grad.copy_(projected_grads[l])

    # 7. Apply existing gradient clipping
    clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)

    # 8. Perform existing AdamW optimizer step
    optimizer.step()

    # Release cached activations
    model.clear_qk_cache()

    if record_metrics:
        step_metrics = {
            "strategy": strategy,
            "layers": layer_stats,
            "heads": head_stats,
        }

    return loss_val, step_metrics
