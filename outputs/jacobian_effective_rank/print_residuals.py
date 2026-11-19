import torch
from jacobian_math import (
    compute_scores_and_probabilities,
    compute_analytical_jacobians,
    autograd_jacobian_reference
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(98765)
T, d_k, d_v, lam = 8, 7, 8, 2.5
q = torch.randn(T, d_k) * 0.3
k_or_r = torch.randn(T, d_k) * 0.3
v = torch.randn(T, d_v)

archs = [
    "standard", "diagonal_free_standard", "symmetric_only",
    "antisymmetric_only", "full_qr_minus", "plus_only", "full_qr_plus"
]

header = f"{'Architecture':<28} | {'J_q max rel err':<16} | {'J_r max rel err':<16} | {'J_qr max rel err':<16}"
print(header)
print("-" * len(header))
for arch in archs:
    kr = None if arch == "symmetric_only" else k_or_r
    raw, probs = compute_scores_and_probabilities(q, kr, arch, lambda_val=lam)
    jacs = compute_analytical_jacobians(q, kr, v, probs, arch, lambda_val=lam)
    max_err = {"J_q": 0.0, "J_r": 0.0, "J_qr": 0.0}
    for tok in range(1, T):
        ag = autograd_jacobian_reference(q, kr, v, arch, lam, tok)
        for k in ag:
            err = (torch.linalg.norm(jacs[k][tok] - ag[k]) / (torch.linalg.norm(ag[k]) + 1e-30)).item()
            max_err[k] = max(max_err[k], err)
    err_q = f"{max_err['J_q']:.2e}"
    err_r = f"{max_err['J_r']:.2e}" if "J_r" in jacs else "N/A"
    err_qr = f"{max_err['J_qr']:.2e}" if "J_qr" in jacs else "N/A"
    print(f"{arch:<28} | {err_q:<16} | {err_r:<16} | {err_qr:<16}")
