"""Unit tests verifying analytical attention Jacobians against torch.autograd.functional.jacobian in float64."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))
from jacobian_math import (
    autograd_jacobian_reference,
    compute_analytical_jacobians,
    compute_scores_and_probabilities,
)

ALL_ARCHITECTURES = [
    "standard",
    "diagonal_free_standard",
    "symmetric_only",
    "antisymmetric_only",
    "full_qr_minus",
    "plus_only",
    "full_qr_plus",
]


@pytest.mark.parametrize("arch_type", ALL_ARCHITECTURES)
def test_autograd_jacobian_verification(arch_type: str):
    """Verify analytical Jacobian against autograd for every architecture on a sequence."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(98765)

    T = 8
    d_k = 7
    d_v = 8
    lambda_val = 2.5  # Non-zero lambda

    # Moderate variance so softmax logits are healthy and do not underflow in float64
    q = torch.randn(T, d_k, dtype=torch.float64) * 0.3
    k_or_r = torch.randn(T, d_k, dtype=torch.float64) * 0.3
    v = torch.randn(T, d_v, dtype=torch.float64)

    # If symmetric_only, k_or_r is not used
    if arch_type == "symmetric_only":
        k_or_r = None

    # Compute analytical Jacobians
    raw, probs = compute_scores_and_probabilities(q, k_or_r, arch_type, lambda_val=lambda_val)
    jac_dict = compute_analytical_jacobians(q, k_or_r, v, probs, arch_type, lambda_val=lambda_val)

    # Check each token i >= 1 against autograd
    for tok in range(1, T):
        ag_dict = autograd_jacobian_reference(q, k_or_r, v, arch_type, lambda_val, token_idx=tok)

        for key in ag_dict.keys():
            J_ana = jac_dict[key][tok]
            J_ag = ag_dict[key]

            # Shape assertion
            assert J_ana.shape == J_ag.shape, f"{arch_type} {key} tok={tok} shape mismatch: {J_ana.shape} vs {J_ag.shape}"

            # Relative Frobenius error
            frob_diff = torch.linalg.norm(J_ana - J_ag, ord="fro").item()
            frob_ag = torch.linalg.norm(J_ag, ord="fro").item()
            rel_error = frob_diff / (frob_ag + 1e-30)

            # Assert error < 1e-12 (target ~1e-15)
            assert rel_error < 1e-12, (
                f"Autograd verification failed for {arch_type} {key} at tok={tok}: "
                f"rel_error={rel_error:.2e} (frob_diff={frob_diff:.2e}, frob_ag={frob_ag:.2e})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
