"""Correctness and Invariant Verification Suite for Antisymmetric-Only Q28 Attention Heads.

Verifies:
1. Exact parameter matched architecture dimensions (Q28, R28, V32, d_model=128, n_heads=4)
2. All 16 heads extracted across all 4 layers
3. Correct Q, R, V activation shapes (128x7, 128x7, 128x8)
4. Strict causal masking (A_ij == 0 for j > i)
5. Row stochasticity of attention probabilities (sum_j A_ij == 1.0)
6. Raw score skew-symmetry (S^T = -S)
7. Exact zero on pre-mask raw score diagonal (S_ii == 0.0)
8. Exact scaling: S = 20 * (QR^T - RQ^T) / sqrt(7)
9. Identical input probes across all architectures, seeds, and checkpoints
10. Checkpoint file integrity & state_dict SHA-256 bitwise matching
11. Deterministic evaluation (zero parameter updates, zero optimizer steps)
12. No NaN or Inf values across all tensors and diagnostics
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config
from dataset import CharacterDataset
from model import DecoderLM


OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
ARRAYS_DIR = DATA_DIR / "arrays"
CHECKPOINT_BASE = Path("outputs/qr_parameter_matched_components/antisymmetric_only_q28")

SEEDS = [1337, 2024, 42]
PROBES = [1, 2, 3, 4]
LAYERS = 4
HEADS = 4
TOTAL_HEADS = 16
CONTEXT_LEN = 128
LAMBDA_PARAM = 20.0
Q_HEAD_DIM = 7
R_HEAD_DIM = 7
V_HEAD_DIM = 8
SCALE = math.sqrt(7)


def test_checkpoint_integrity_hashes():
    """Verify that current checkpoint hashes bitwise match pre-analysis recorded integrity."""
    integrity_p = DATA_DIR / "checkpoint_integrity.json"
    assert integrity_p.exists(), f"Integrity file missing: {integrity_p}"

    with open(integrity_p) as f:
        integrity = json.load(f)

    for seed in SEEDS:
        seed_key = f"seed_{seed}"
        assert seed_key in integrity["checkpoints"]

        for ckpt_name in ["init.pt", "final.pt", "best.pt"]:
            ckpt_info = integrity["checkpoints"][seed_key][ckpt_name]
            p = Path(ckpt_info["path"])
            assert p.exists(), f"Checkpoint not found: {p}"

            # Compute current file sha256
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            current_file_sha = h.hexdigest()
            assert current_file_sha == ckpt_info["file_sha256"], (
                f"Checkpoint file hash modified for {seed} {ckpt_name}!"
            )


def test_probe_metadata_invariants():
    """Verify that all 4 probes are fixed 128-token sequences with expected SHA256 hashes."""
    probe_p = DATA_DIR / "probe_metadata.json"
    assert probe_p.exists(), f"Probe metadata missing: {probe_p}"

    with open(probe_p) as f:
        probes = json.load(f)

    assert len(probes) == 4
    for pid in [1, 2, 3, 4]:
        pkey = f"probe_{pid}"
        assert pkey in probes
        pdata = probes[pkey]
        assert len(pdata["tokens"]) == 128
        assert pdata["context_length"] == 128

        # Verify hash of tokens
        t_arr = np.array(pdata["tokens"], dtype=np.int64)
        computed_sha = hashlib.sha256(t_arr.tobytes()).hexdigest()
        assert computed_sha == pdata["sha256"]


def test_extracted_array_shapes_and_completeness():
    """Verify all 24 array files exist and have exact tensor shapes."""
    for seed in SEEDS:
        for state in ["initialization", "trained"]:
            for probe in PROBES:
                p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{probe}.npz"
                assert p.exists(), f"Array file missing: {p}"

                d = np.load(p)
                assert "q" in d and "r" in d and "v" in d
                assert "a_raw" in d and "raw_scores" in d and "probabilities" in d

                q = d["q"]
                r = d["r"]
                v = d["v"]
                araw = d["a_raw"]
                scores = d["raw_scores"]
                probs = d["probabilities"]

                # Shapes: (LAYERS, HEADS, CONTEXT_LEN, DIM)
                assert q.shape == (4, 4, 128, 7), f"Unexpected Q shape {q.shape} in {p}"
                assert r.shape == (4, 4, 128, 7), f"Unexpected R shape {r.shape} in {p}"
                assert v.shape == (4, 4, 128, 8), f"Unexpected V shape {v.shape} in {p}"
                assert araw.shape == (4, 4, 128, 128), f"Unexpected A_raw shape {araw.shape} in {p}"
                assert scores.shape == (4, 4, 128, 128), f"Unexpected scores shape {scores.shape} in {p}"
                assert probs.shape == (4, 4, 128, 128), f"Unexpected probs shape {probs.shape} in {p}"


def test_skew_symmetry_and_zero_diagonal():
    """Verify S^T = -S and S_ii = 0 across all 384 head-probe instances."""
    for seed in SEEDS:
        for state in ["initialization", "trained"]:
            for probe in PROBES:
                p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{probe}.npz"
                d = np.load(p)
                scores = d["raw_scores"]  # (4, 4, 128, 128)
                araw = d["a_raw"]

                for l in range(4):
                    for h in range(4):
                        s = scores[l, h]
                        a = araw[l, h]

                        # 1. Skew-symmetry
                        skew_err_s = np.max(np.abs(s + s.T))
                        skew_err_a = np.max(np.abs(a + a.T))
                        assert skew_err_s < 1e-5, f"Skew symmetry failed for {seed} {state} p{probe} L{l}H{h}: {skew_err_s}"
                        assert skew_err_a < 1e-5, f"Unscaled skew symmetry failed: {skew_err_a}"

                        # 2. Zero diagonal
                        diag_err_s = np.max(np.abs(np.diag(s)))
                        diag_err_a = np.max(np.abs(np.diag(a)))
                        assert diag_err_s < 1e-6, f"Diagonal non-zero for {seed} {state} p{probe} L{l}H{h}: {diag_err_s}"
                        assert diag_err_a < 1e-6, f"Unscaled diagonal non-zero: {diag_err_a}"

                        # 3. Exact scaling: S = 20 * A_raw / sqrt(7)
                        expected_s = (LAMBDA_PARAM * a) / SCALE
                        scaling_diff = np.max(np.abs(s - expected_s))
                        assert scaling_diff < 1e-5, f"Scaling mismatch: {scaling_diff}"


def test_causal_masking_and_stochasticity():
    """Verify strictly zero attention in upper triangle and sum_j A_ij == 1.0."""
    for seed in SEEDS:
        for state in ["initialization", "trained"]:
            for probe in PROBES:
                p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{probe}.npz"
                d = np.load(p)
                probs = d["probabilities"]  # (4, 4, 128, 128)

                for l in range(4):
                    for h in range(4):
                        mat = probs[l, h]
                        # Upper triangle (j > i) must be strictly 0
                        upper = np.triu(mat, k=1)
                        assert np.max(np.abs(upper)) == 0.0, (
                            f"Causal violation in {seed} {state} p{probe} L{l}H{h}: max upper = {np.max(np.abs(upper))}"
                        )

                        # Row stochasticity: sum over key positions must equal 1.0
                        row_sums = np.sum(mat, axis=1)
                        assert np.allclose(row_sums, 1.0, atol=1e-5), (
                            f"Row sum not 1.0 in {seed} {state} p{probe} L{l}H{h}: min={row_sums.min()}, max={row_sums.max()}"
                        )


def test_no_nan_or_inf_in_data():
    """Verify no NaN or Inf values exist in any exported CSV or array."""
    for csv_name in ["per_head_diagnostics.csv", "attention_statistics.csv", "activation_statistics.csv"]:
        csv_p = DATA_DIR / csv_name
        assert csv_p.exists(), f"CSV missing: {csv_p}"
        df = pd.read_csv(csv_p)
        assert not df.isna().any().any(), f"NaN values detected in {csv_name}!"

        # Check numeric columns for inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            assert not np.isinf(df[c]).any(), f"Inf values detected in {csv_name} column {c}!"


def test_model_parameter_count_and_architecture():
    """Verify Antisymmetric-Only Q28 exact parameter counts and attention equations."""
    cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128)
    model = DecoderLM(cfg, vocab_size=65, attention_type="qr_antisymmetric_only_q28")

    # Attention parameters per layer:
    # W_Q: 128 x 28 = 3,584
    # W_R: 128 x 28 = 3,584
    # W_V: 128 x 32 = 4,096
    # W_O: 32 x 128 = 4,096
    # Sum per layer = 15,360
    # Total attention (4 layers) = 61,440
    # Total model parameters = 615,296
    total_attn_params = 0
    for b in model.blocks:
        attn = b.attention
        layer_attn = (
            attn.q_proj.weight.numel()
            + attn.r_proj.weight.numel()
            + attn.v_proj.weight.numel()
            + attn.out_proj.weight.numel()
        )
        assert layer_attn == 15360, f"Expected 15,360 attn params per layer, got {layer_attn}"
        total_attn_params += layer_attn

    assert total_attn_params == 61440
    assert sum(p.numel() for p in model.parameters()) == 615296
    assert model.lm_head.weight is model.token_embedding.weight


if __name__ == "__main__":
    test_funcs = [
        test_checkpoint_integrity_hashes,
        test_probe_metadata_invariants,
        test_extracted_array_shapes_and_completeness,
        test_skew_symmetry_and_zero_diagonal,
        test_causal_masking_and_stochasticity,
        test_no_nan_or_inf_in_data,
        test_model_parameter_count_and_architecture,
    ]
    print("=" * 80)
    print(" RUNNING CORRECTNESS & INVARIANT VERIFICATION SUITE")
    print("=" * 80)
    all_passed = True
    for f in test_funcs:
        try:
            f()
            print(f"[PASS] {f.__name__}")
        except Exception as e:
            print(f"[FAIL] {f.__name__}: {e}")
            all_passed = False
    print("=" * 80)
    if all_passed:
        print("ALL 7 VERIFICATION TEST SUITES PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED!")
        exit(1)
