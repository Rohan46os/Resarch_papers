"""Mathematical definitions and vectorized routines for the attention Jacobian spectrum
and squared-singular-value effective rank analysis.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_scores_and_probabilities(
    q: torch.Tensor,
    k_or_r: Optional[torch.Tensor],
    arch_type: str,
    lambda_val: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute raw causal scores and softmax attention probabilities in float64.

    Args:
        q: (..., T, d_k)
        k_or_r: (..., T, d_k) or None
        arch_type: one of 'standard', 'diagonal_free_standard', 'symmetric_only',
                   'antisymmetric_only', 'full_qr_minus', 'plus_only', 'full_qr_plus'
        lambda_val: scalar multiplier for cross-terms

    Returns:
        raw_scores: (..., T, T)
        probabilities: (..., T, T) with causal mask applied
    """
    d_k = q.shape[-1]
    T = q.shape[-2]
    scale = math.sqrt(d_k)

    if arch_type == "standard":
        if k_or_r is None:
            raise ValueError("k_or_r must be provided for standard attention")
        raw = (q @ k_or_r.transpose(-2, -1)) / scale

    elif arch_type == "diagonal_free_standard":
        if k_or_r is None:
            raise ValueError("k_or_r must be provided for diagonal_free_standard attention")
        raw = (q @ k_or_r.transpose(-2, -1)) / scale
        diag = torch.diagonal(raw, dim1=-2, dim2=-1)
        raw = raw - torch.diag_embed(diag)

    elif arch_type == "symmetric_only":
        raw = (q @ q.transpose(-2, -1)) / scale

    elif arch_type == "antisymmetric_only":
        if k_or_r is None:
            raise ValueError("k_or_r must be provided for antisymmetric_only attention")
        qr = q @ k_or_r.transpose(-2, -1)
        rq = k_or_r @ q.transpose(-2, -1)
        raw = (lambda_val * (qr - rq)) / scale

    elif arch_type == "full_qr_minus":
        if k_or_r is None:
            raise ValueError("k_or_r must be provided for full_qr_minus attention")
        qq = q @ q.transpose(-2, -1)
        qr = q @ k_or_r.transpose(-2, -1)
        rq = k_or_r @ q.transpose(-2, -1)
        raw = (qq + lambda_val * (qr - rq)) / scale

    elif arch_type == "plus_only":
        if k_or_r is None:
            raise ValueError("k_or_r must be provided for plus_only attention")
        qr = q @ k_or_r.transpose(-2, -1)
        rq = k_or_r @ q.transpose(-2, -1)
        raw = (lambda_val * (qr + rq)) / scale

    elif arch_type == "full_qr_plus":
        if k_or_r is None:
            raise ValueError("k_or_r must be provided for full_qr_plus attention")
        qq = q @ q.transpose(-2, -1)
        qr = q @ k_or_r.transpose(-2, -1)
        rq = k_or_r @ q.transpose(-2, -1)
        raw = (qq + lambda_val * (qr + rq)) / scale

    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

    # Causal masking: mask future positions where j > i
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
    masked = raw.masked_fill(mask, float("-inf"))
    probs = F.softmax(masked, dim=-1)
    return raw, probs


def compute_analytical_jacobians(
    q: torch.Tensor,
    k_or_r: Optional[torch.Tensor],
    v: torch.Tensor,
    probs: torch.Tensor,
    arch_type: str,
    lambda_val: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """Compute exact analytical output-to-query Jacobians.

    For QR architectures: returns 'J_q', 'J_r', and 'J_qr'.
    For standard / symmetric-only architectures: returns 'J_q'.

    Shapes:
        q: (..., T, d_k)
        k_or_r: (..., T, d_k) or None
        v: (..., T, d_v)
        probs: (..., T, T)
    Returns:
        dict with:
            'J_q': (..., T, d_v, d_k)
            'J_r': (..., T, d_v, d_k) [if QR]
            'J_qr': (..., T, d_v, 2 * d_k) [if QR]
    """
    d_k = q.shape[-1]
    T = q.shape[-2]
    scale = math.sqrt(d_k)
    batch_shape = q.shape[:-2]

    # Precompute head output y = probs @ v: (..., T, d_v)
    y = probs @ v

    # Helper function to compute J given D: (..., T, T, d_u)
    def _accumulate_jacobian(D: torch.Tensor) -> torch.Tensor:
        # M: (..., T, T, d_u)
        M = probs.unsqueeze(-1) * D
        # term1: sum_{j} M_{..., i, j, d} * v_{..., j, c} -> (..., T, d_v, d_u)
        # Using tensordot/matmul for safety across arbitrary batch dimensions:
        # Flatten leading batch dimensions
        M_flat = M.reshape(-1, T, T, D.shape[-1])
        v_flat = v.reshape(-1, T, v.shape[-1])
        y_flat = y.reshape(-1, T, v.shape[-1])

        # term1 = einsum('bijd,bjc->bicd', M_flat, v_flat)
        term1_flat = torch.einsum("bijd,bjc->bicd", M_flat, v_flat)
        # D_bar = sum_{j} M_{b, i, j, d} -> (B, T, d_u)
        D_bar_flat = M_flat.sum(dim=2)
        # term2 = y_{b, i, c} * D_bar_{b, i, d} -> (B, T, d_v, d_u)
        term2_flat = y_flat.unsqueeze(-1) * D_bar_flat.unsqueeze(2)

        J_flat = term1_flat - term2_flat
        return J_flat.reshape(*batch_shape, T, v.shape[-1], D.shape[-1])

    # 1. Build D_q: shape (..., T, T, d_k)
    # Off-diagonal derivative w.r.t q_i depends on j:
    # Standard / Diag-Free: k_j / scale
    # Symmetric-only: q_j / scale
    # Antisymmetric-only: lambda * r_j / scale
    # Full QR Minus: (q_j + lambda * r_j) / scale
    # Plus-only: lambda * r_j / scale
    # Full QR Plus: (q_j + lambda * r_j) / scale
    if arch_type in ("standard", "diagonal_free_standard"):
        W_q = k_or_r / scale
    elif arch_type == "symmetric_only":
        W_q = q / scale
    elif arch_type == "antisymmetric_only":
        W_q = (lambda_val * k_or_r) / scale
    elif arch_type == "full_qr_minus":
        W_q = (q + lambda_val * k_or_r) / scale
    elif arch_type == "plus_only":
        W_q = (lambda_val * k_or_r) / scale
    elif arch_type == "full_qr_plus":
        W_q = (q + lambda_val * k_or_r) / scale
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

    # Expand W_q across rows i: D_q[..., i, j, :] = W_q[..., j, :]
    D_q = W_q.unsqueeze(-3).expand(*batch_shape, T, T, d_k).clone()

    # Diagonal corrections for j = i:
    if arch_type == "standard":
        # ds_ii / dq_i = k_i / scale (already set)
        pass
    elif arch_type == "diagonal_free_standard":
        # ds_ii / dq_i = 0
        for i in range(T):
            D_q[..., i, i, :] = 0.0
    elif arch_type == "symmetric_only":
        # ds_ii / dq_i = (2 / scale) * q_i
        diag_q = (2.0 / scale) * q
        for i in range(T):
            D_q[..., i, i, :] = diag_q[..., i, :]
    elif arch_type == "antisymmetric_only":
        # ds_ii / dq_i = 0
        for i in range(T):
            D_q[..., i, i, :] = 0.0
    elif arch_type == "full_qr_minus":
        # ds_ii / dq_i = (2 / scale) * q_i
        diag_q = (2.0 / scale) * q
        for i in range(T):
            D_q[..., i, i, :] = diag_q[..., i, :]
    elif arch_type == "plus_only":
        # ds_ii / dq_i = (2 * lambda / scale) * r_i
        diag_q = (2.0 * lambda_val / scale) * k_or_r
        for i in range(T):
            D_q[..., i, i, :] = diag_q[..., i, :]
    elif arch_type == "full_qr_plus":
        # ds_ii / dq_i = (2 / scale) * (q_i + lambda * r_i)
        diag_q = (2.0 / scale) * (q + lambda_val * k_or_r)
        for i in range(T):
            D_q[..., i, i, :] = diag_q[..., i, :]

    J_q = _accumulate_jacobian(D_q)

    # 2. Check if QR architecture
    is_qr = arch_type in ("antisymmetric_only", "full_qr_minus", "plus_only", "full_qr_plus")
    if not is_qr:
        return {"J_q": J_q}

    # Build D_r: shape (..., T, T, d_k)
    # Off-diagonal:
    # Antisymmetric-only: -lambda * q_j / scale
    # Full QR Minus: -lambda * q_j / scale
    # Plus-only: lambda * q_j / scale
    # Full QR Plus: lambda * q_j / scale
    if arch_type in ("antisymmetric_only", "full_qr_minus"):
        W_r = (-lambda_val * q) / scale
    elif arch_type in ("plus_only", "full_qr_plus"):
        W_r = (lambda_val * q) / scale
    else:
        raise ValueError(f"Not a QR arch: {arch_type}")

    D_r = W_r.unsqueeze(-3).expand(*batch_shape, T, T, d_k).clone()

    # Diagonal corrections for j = i:
    if arch_type in ("antisymmetric_only", "full_qr_minus"):
        # ds_ii / dr_i = 0
        for i in range(T):
            D_r[..., i, i, :] = 0.0
    elif arch_type in ("plus_only", "full_qr_plus"):
        # ds_ii / dr_i = (2 * lambda / scale) * q_i
        diag_r = (2.0 * lambda_val / scale) * q
        for i in range(T):
            D_r[..., i, i, :] = diag_r[..., i, :]

    J_r = _accumulate_jacobian(D_r)
    J_qr = torch.cat([J_q, J_r], dim=-1)

    return {
        "J_q": J_q,
        "J_r": J_r,
        "J_qr": J_qr,
    }


def compute_squared_spectrum_effective_rank(
    J: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute squared-spectrum effective rank via torch.linalg.svdvals in float64.

    Formula:
        p_r = sigma_r^2 / sum_s(sigma_s^2)
        r_eff = exp( -sum_r( p_r * log(p_r) ) )
        r_eff_norm = r_eff / min(J.shape[-2], J.shape[-1])

    Returns:
        r_eff: (...)
        r_eff_norm: (...)
        is_skipped: (...) bool tensor where sum(sigma_r^2) <= 1e-30
    """
    # Compute singular values in float64
    s = torch.linalg.svdvals(J)  # (..., K)
    s_sq = s.pow(2)
    sum_sq = s_sq.sum(dim=-1)  # (...)

    is_skipped = sum_sq <= 1e-30

    # Safe normalization
    safe_sum_sq = torch.where(is_skipped, torch.ones_like(sum_sq), sum_sq)
    p = s_sq / safe_sum_sq.unsqueeze(-1)  # (..., K)

    # 0 * log(0) = 0 via masking
    log_p = torch.where(p > 0.0, torch.log(p), torch.zeros_like(p))
    entropy = -(p * log_p).sum(dim=-1)
    r_eff = torch.exp(entropy)

    rank_ceiling = float(min(J.shape[-2], J.shape[-1]))
    r_eff_norm = r_eff / rank_ceiling

    return r_eff, r_eff_norm, is_skipped


def compute_attention_metrics(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token top-1 attention weight and normalized attention entropy.

    Args:
        probs: (..., T, T) attention probabilities with causal mask

    Returns:
        top1_attn: (..., T)
        norm_entropy: (..., T)
    """
    T = probs.shape[-1]
    top1_attn = probs.max(dim=-1).values

    # Row entropy
    safe_probs = probs.clamp(min=1e-30)
    ent = -(probs * torch.log(safe_probs)).sum(dim=-1)

    # Normalized entropy: H / log(i + 1)
    # At i = 0, log(1) = 0, so set norm_ent = 0.0
    indices = torch.arange(1, T + 1, dtype=probs.dtype, device=probs.device)
    max_ent = torch.log(indices)  # (T,)
    norm_entropy = torch.where(max_ent > 1e-12, ent / max_ent, torch.zeros_like(ent))

    return top1_attn, norm_entropy


def autograd_jacobian_reference(
    q: torch.Tensor,
    k_or_r: Optional[torch.Tensor],
    v: torch.Tensor,
    arch_type: str,
    lambda_val: float,
    token_idx: int,
) -> Dict[str, torch.Tensor]:
    """Reference implementation using torch.autograd.functional.jacobian for a single sequence."""
    d_k = q.shape[-1]
    d_v = v.shape[-1]
    scale = math.sqrt(d_k)
    tok = token_idx
    is_qr = arch_type in ("antisymmetric_only", "full_qr_minus", "plus_only", "full_qr_plus")

    # Helper function for scores given single token vectors
    def _compute_y_from_q(q_cur: torch.Tensor) -> torch.Tensor:
        s_list = []
        for j in range(tok + 1):
            q_j = q_cur if j == tok else q[j]
            if arch_type == "standard":
                s_ij = torch.dot(q_cur, k_or_r[j]) / scale
            elif arch_type == "diagonal_free_standard":
                s_ij = torch.dot(q_cur, k_or_r[j]) / scale if j < tok else torch.tensor(0.0, dtype=q.dtype)
            elif arch_type == "symmetric_only":
                s_ij = torch.dot(q_cur, q_j) / scale
            elif arch_type == "antisymmetric_only":
                s_ij = lambda_val * (torch.dot(q_cur, k_or_r[j]) - torch.dot(k_or_r[tok], q_j)) / scale
            elif arch_type == "full_qr_minus":
                s_ij = (
                    torch.dot(q_cur, q_j)
                    + lambda_val * (torch.dot(q_cur, k_or_r[j]) - torch.dot(k_or_r[tok], q_j))
                ) / scale
            elif arch_type == "plus_only":
                s_ij = lambda_val * (torch.dot(q_cur, k_or_r[j]) + torch.dot(k_or_r[tok], q_j)) / scale
            elif arch_type == "full_qr_plus":
                s_ij = (
                    torch.dot(q_cur, q_j)
                    + lambda_val * (torch.dot(q_cur, k_or_r[j]) + torch.dot(k_or_r[tok], q_j))
                ) / scale
            s_list.append(s_ij)
        s_vec = torch.stack(s_list)
        a_vec = F.softmax(s_vec, dim=-1)
        return sum(a_vec[j] * v[j] for j in range(tok + 1))

    J_ag_q = torch.autograd.functional.jacobian(_compute_y_from_q, q[tok])
    res = {"J_q": J_ag_q}

    if is_qr:
        def _compute_y_from_r(r_cur: torch.Tensor) -> torch.Tensor:
            s_list = []
            for j in range(tok + 1):
                r_j = r_cur if j == tok else k_or_r[j]
                if arch_type == "antisymmetric_only":
                    s_ij = lambda_val * (torch.dot(q[tok], r_j) - torch.dot(r_cur, q[j])) / scale
                elif arch_type == "full_qr_minus":
                    s_ij = (
                        torch.dot(q[tok], q[j])
                        + lambda_val * (torch.dot(q[tok], r_j) - torch.dot(r_cur, q[j]))
                    ) / scale
                elif arch_type == "plus_only":
                    s_ij = lambda_val * (torch.dot(q[tok], r_j) + torch.dot(r_cur, q[j])) / scale
                elif arch_type == "full_qr_plus":
                    s_ij = (
                        torch.dot(q[tok], q[j])
                        + lambda_val * (torch.dot(q[tok], r_j) + torch.dot(r_cur, q[j]))
                    ) / scale
                s_list.append(s_ij)
            s_vec = torch.stack(s_list)
            a_vec = F.softmax(s_vec, dim=-1)
            return sum(a_vec[j] * v[j] for j in range(tok + 1))

        J_ag_r = torch.autograd.functional.jacobian(_compute_y_from_r, k_or_r[tok])
        res["J_r"] = J_ag_r

        def _compute_y_from_qr(qr_cur: torch.Tensor) -> torch.Tensor:
            q_cur = qr_cur[:d_k]
            r_cur = qr_cur[d_k:]
            s_list = []
            for j in range(tok + 1):
                q_j = q_cur if j == tok else q[j]
                r_j = r_cur if j == tok else k_or_r[j]
                if arch_type == "antisymmetric_only":
                    s_ij = lambda_val * (torch.dot(q_cur, r_j) - torch.dot(r_cur, q_j)) / scale
                elif arch_type == "full_qr_minus":
                    s_ij = (
                        torch.dot(q_cur, q_j)
                        + lambda_val * (torch.dot(q_cur, r_j) - torch.dot(r_cur, q_j))
                    ) / scale
                elif arch_type == "plus_only":
                    s_ij = lambda_val * (torch.dot(q_cur, r_j) + torch.dot(r_cur, q_j)) / scale
                elif arch_type == "full_qr_plus":
                    s_ij = (
                        torch.dot(q_cur, q_j)
                        + lambda_val * (torch.dot(q_cur, r_j) + torch.dot(r_cur, q_j))
                    ) / scale
                s_list.append(s_ij)
            s_vec = torch.stack(s_list)
            a_vec = F.softmax(s_vec, dim=-1)
            return sum(a_vec[j] * v[j] for j in range(tok + 1))

        J_ag_qr = torch.autograd.functional.jacobian(_compute_y_from_qr, torch.cat([q[tok], k_or_r[tok]]))
        res["J_qr"] = J_ag_qr

    return res
