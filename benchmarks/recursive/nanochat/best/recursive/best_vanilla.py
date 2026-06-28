# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""
Input-token-routed mixture of independent low-rank output projections.
Replace single weight-tied lm_head with K=3 output heads routed by input embedding.
Mean validation BPB: 0.9344 (10 seeds).
Usage: uv run train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

cap = torch.cuda.get_device_capability()

if cap[0] >= 10:
    # Blackwell (B200, SM100): wrap flash-attn-4 as a custom op so torch.compile
    # treats it as opaque (no tracing into cutlass DSL, no recompile-cache thrash,
    # no per-call Python kernel build).
    from flash_attn.cute import flash_attn_func as _fa4_raw
    from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

    @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
    def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (window_left, 0) if window_left > 0 else (None, None)
        out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
        return out, lse

    @_fa4_causal_op.register_fake
    def _fa4_causal_fake(q, k, v, window_left):
        B, T, H, D = q.shape
        return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    def _fa4_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl = window_left if window_left > 0 else None
        dq, dk, dv = _fa4_bwd_raw(
            q, k, v, out, grad_output, lse,
            causal=True, window_size_left=wl, window_size_right=0,
        )
        return dq, dk, dv

    @_fa4_bwd_op.register_fake
    def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_backward(ctx, grad_output, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
        return dq, dk, dv, None

    _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
        return out

    class _FA4Shim:
        flash_attn_func = staticmethod(flash_attn_func)

    fa3 = _FA4Shim()
    print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
else:
    # Hopper/Ampere (H100, A100): use flash-attn-3 via kernels package
    from kernels import get_kernel

    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    print(f"Using flash-attn-3 from {repo} (GPU capability {cap})")

from lib import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader  # noqa: E402

# ---------------------------------------------------------------------------
# Muon Optimizer
# ---------------------------------------------------------------------------


def newton_schulz_5(G, steps=5, row_normalize=True):
    """Transpose-optimal NS: for tall matrices (m>n), transpose to minimize
    inner matmul size from m^2 to n^2. Mathematically equivalent."""
    assert G.ndim == 2
    m, n = G.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    X_f = G.float()
    if row_normalize:
        row_norms = X_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
        X_f = X_f / row_norms
    X_f = X_f / (X_f.norm() + 1e-7)
    transposed = m > n
    if transposed:
        X_f = X_f.T
    X = X_f.bfloat16()
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def batched_newton_schulz_5(G_batch, steps=5, row_normalize=True):
    """Batched transpose-optimal NS."""
    assert G_batch.ndim == 3
    bs, m, n = G_batch.shape
    a, b, c = (3.4445, -4.7750, 2.0315)
    X_f = G_batch.float()
    if row_normalize:
        row_norms = X_f.norm(dim=2, keepdim=True).clamp(min=1e-8)
        X_f = X_f / row_norms
    norms = X_f.flatten(1).norm(dim=1, keepdim=True).unsqueeze(2) + 1e-7
    X_f = X_f / norms
    transposed = m > n
    if transposed:
        X_f = X_f.transpose(1, 2)
    X = X_f.bfloat16()
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if transposed:
        X = X.transpose(1, 2)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def reset_momentum(self, decay_factor=0.0):
        """Decay momentum buffers. decay_factor=0.0 is full reset."""
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "momentum_buffer" in self.state[p]:
                    self.state[p]["momentum_buffer"].mul_(decay_factor)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group.get("weight_decay", 0.0)

            shape_groups = {}

            for p in group["params"]:
                if p.grad is None:
                    continue

                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf.clone()

                if g.ndim >= 2:
                    orig_shape = g.shape
                    if g.ndim > 2:
                        g = g.reshape(g.shape[0], -1)
                    shape_key = g.shape
                    if shape_key not in shape_groups:
                        shape_groups[shape_key] = []
                    shape_groups[shape_key].append((p, g, orig_shape))
                else:
                    p.add_(g, alpha=-lr)

            for shape, items in shape_groups.items():
                if len(items) == 1:
                    p, g, orig_shape = items[0]
                    g = newton_schulz_5(g, steps=ns_steps)
                    p.add_(g.reshape(orig_shape), alpha=-lr)
                else:
                    grads = torch.stack([g for _, g, _ in items], dim=0)
                    orth_grads = batched_newton_schulz_5(grads, steps=ns_steps)
                    for i, (p, _, orig_shape) in enumerate(items):
                        p.add_(orth_grads[i].reshape(orig_shape), alpha=-lr)


# ---------------------------------------------------------------------------
# GPT Model with Hourglass Architecture
# ---------------------------------------------------------------------------


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def center(x):
    """Mean-center along last dimension -- parameter-free output centering."""
    return x - x.mean(dim=-1, keepdim=True)


class ContextNorm(nn.Module):
    """Causal context-conditioned normalization -- PER-CHANNEL ADDITIVE."""
    def __init__(self, dim, init_scale=0.12):
        super().__init__()
        self.ctx_weight = nn.Parameter(torch.zeros(dim))
        self.ctx_scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        h = F.rms_norm(x, (x.size(-1),))
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        return h + self.ctx_scale * (self.ctx_weight * x_prev)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, head_dim, layer_idx, n_layer, use_window, window_size):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.head_dim = head_dim
        self.c_q = nn.Linear(dim, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(dim, n_head * head_dim, bias=False)
        self.c_v = nn.Linear(dim, n_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_head * head_dim, dim, bias=False)
        nn.init.zeros_(self.c_proj.weight)
        self.use_window = use_window
        self.window_size = window_size
        # Per-head heterogeneous temperature for multi-scale attention
        self.attn_temperature = nn.Parameter(torch.linspace(0.7, 1.3, n_head))
        # V-Res lambda
        vres_init = layer_idx / max(n_layer - 1, 1)
        self.vres_lambda = nn.Parameter(torch.tensor(vres_init * 0.5))
        self.q_shift_beta = nn.Parameter(torch.zeros(dim) + 0.30)
        self.k_shift_beta = nn.Parameter(torch.zeros(dim) + 0.30)

    def forward(self, x, cos_sin, v_embed=None):
        B, T, C = x.size()
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        q = self.c_q(x + self.q_shift_beta * x_prev).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x + self.k_shift_beta * x_prev).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        if v_embed is not None:
            lam = torch.sigmoid(self.vres_lambda)
            v = (1.0 - lam) * v + lam * v_embed

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        temp = self.attn_temperature[None, None, :, None].to(q.dtype)
        q = q * temp

        if self.use_window:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=(self.window_size, 0))
        else:
            y = fa3.flash_attn_func(q, k, v, causal=True)

        # Sandwich norm: per-head RMSNorm on attention output
        y = norm(y)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SwiGLUMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(8 / 3 * dim)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))


class SquaredReLUMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 4 * dim
        self.c_fc = nn.Linear(dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        h = self.c_fc(x)
        h = F.relu(h)
        h = h * h
        return self.c_proj(h)


class WideSquaredReLUMLP(nn.Module):
    """SquaredReLU MLP with custom wider hidden dim for decoupled working dimension."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        h = self.c_fc(x)
        h = F.relu(h)
        h = h * h
        return self.c_proj(h)


class Block(nn.Module):
    def __init__(self, dim, n_head, head_dim, layer_idx, n_layer, use_window, window_size, use_skip=False, mlp_hidden_dim=None):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_head, head_dim, layer_idx, n_layer, use_window, window_size)
        if mlp_hidden_dim is not None:
            self.mlp = WideSquaredReLUMLP(dim, mlp_hidden_dim)
        else:
            self.mlp = SquaredReLUMLP(dim)
        self.attn_alpha = nn.Parameter(torch.tensor(1.0))
        self.mlp_alpha = nn.Parameter(torch.tensor(1.0))
        self.use_skip = use_skip
        if use_skip:
            self.skip_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, cos_sin, v_embed=None, skip_signal=None):
        if self.use_skip and skip_signal is not None:
            x = x + self.skip_scale * skip_signal
        x = x + self.attn_alpha * self.attn(norm(x), cos_sin, v_embed=v_embed)
        x = x + self.mlp_alpha * center(self.mlp(norm(x)))
        return x


class MLPOnlyBlock(nn.Module):
    """MLP-only block with multi-stride causal shift for token mixing."""
    def __init__(self, dim, layer_idx):
        super().__init__()
        self.shift_beta1 = nn.Parameter(torch.zeros(dim) + 0.25)
        self.shift_beta2 = nn.Parameter(torch.zeros(dim) + 0.15)
        self.shift_beta4 = nn.Parameter(torch.zeros(dim) + 0.10)
        self.mlp = SquaredReLUMLP(dim)
        self.mlp_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        T = x.size(1)
        x_s1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        blend = x + self.shift_beta1.unsqueeze(0).unsqueeze(0) * x_s1
        if T > 2:
            x_s2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))
            blend = blend + self.shift_beta2.unsqueeze(0).unsqueeze(0) * x_s2
        if T > 4:
            x_s4 = F.pad(x[:, :-4, :], (0, 0, 4, 0))
            blend = blend + self.shift_beta4.unsqueeze(0).unsqueeze(0) * x_s4
        x = x + self.mlp_alpha * self.mlp(norm(blend))
        return x


class NarrowSquaredReLUMLP(nn.Module):
    """SquaredReLU MLP with configurable expansion ratio for bottleneck."""
    def __init__(self, dim, expansion=6):
        super().__init__()
        hidden_dim = expansion * dim
        self.c_fc = nn.Linear(dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        h = self.c_fc(x)
        h = F.relu(h)
        h = h * h
        return self.c_proj(h)


class AttentionPrimaryBlock(nn.Module):
    """Attention-primary block for bottleneck with wide MLP and shift-blend."""
    def __init__(self, dim, n_head, head_dim, layer_idx, n_layer, use_window, window_size, mlp_expansion=6):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_head, head_dim, layer_idx, n_layer, use_window, window_size)
        self.mlp = NarrowSquaredReLUMLP(dim, expansion=mlp_expansion)
        self.attn_alpha = nn.Parameter(torch.tensor(1.0))
        self.mlp_alpha = nn.Parameter(torch.tensor(1.0))
        self.mlp_shift_gamma = nn.Parameter(torch.zeros(dim) + 0.15)

    def forward(self, x, cos_sin, v_embed=None):
        x = x + self.attn_alpha * self.attn(norm(x), cos_sin, v_embed=v_embed)
        x_normed = norm(x)
        x_s1 = F.pad(x_normed[:, :-1, :], (0, 0, 1, 0))
        x_normed = x_normed + self.mlp_shift_gamma.unsqueeze(0).unsqueeze(0) * x_s1
        x = x + self.mlp_alpha * center(self.mlp(x_normed))
        return x


class ByteFeatureEmbedding(nn.Module):
    """Byte-content-initialized auxiliary embedding. Zero runtime overhead."""
    def __init__(self, tokenizer_enc, vocab_size, embed_dim, max_bytes=16):
        super().__init__()
        combined = torch.zeros(vocab_size, 769)
        for token_id in range(vocab_size):
            try:
                raw_bytes = tokenizer_enc.decode_single_token_bytes(token_id)
                if len(raw_bytes) > 0:
                    for b in raw_bytes[:max_bytes]:
                        combined[token_id, b] += 1.0 / len(raw_bytes)
                    combined[token_id, 256 + raw_bytes[0]] = 1.0
                    combined[token_id, 512 + raw_bytes[-1]] = 1.0
                    combined[token_id, 768] = len(raw_bytes) / max_bytes
            except Exception:
                pass
        torch.manual_seed(1337)
        proj = torch.randn(769, embed_dim) * 0.01
        init_emb = combined @ proj
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.data.copy_(init_emb)
        self.mix_alpha = nn.Parameter(torch.tensor(0.15))

    def get_raw(self, idx):
        """Return raw embedding without alpha scaling."""
        return self.embed(idx)

    def forward(self, idx):
        return self.mix_alpha * self.embed(idx)


class ByteBoundaryEmbedding(nn.Module):
    """Cross-token byte-boundary hash embedding.
    Captures the byte transition between tokens: last byte of prev token +
    first byte of current token, hashed to a learned embedding table.
    This captures character-level boundary patterns token n-grams miss.
    """
    def __init__(self, tokenizer_enc, vocab_size, embed_dim, table_size=8192):
        super().__init__()
        # Pre-compute first and last byte for each token
        first_bytes = torch.zeros(vocab_size, dtype=torch.long)
        last_bytes = torch.zeros(vocab_size, dtype=torch.long)
        for token_id in range(vocab_size):
            try:
                raw_bytes = tokenizer_enc.decode_single_token_bytes(token_id)
                if len(raw_bytes) > 0:
                    first_bytes[token_id] = raw_bytes[0] + 1  # +1 to avoid 0
                    last_bytes[token_id] = raw_bytes[-1] + 1
            except Exception:
                pass
        self.register_buffer('first_bytes', first_bytes, persistent=False)
        self.register_buffer('last_bytes', last_bytes, persistent=False)
        self.table_size = table_size
        self.embed = nn.Embedding(table_size, embed_dim)
        # Spectral init: orthogonal blocks for hash table
        w = self.embed.weight.data
        n_rows, n_cols = w.shape
        n_blocks = (n_rows + n_cols - 1) // n_cols
        ortho_rows = []
        for _ in range(n_blocks):
            block = torch.randn(n_cols, n_cols)
            q, _ = torch.linalg.qr(block)
            ortho_rows.append(q)
        ortho_mat = torch.cat(ortho_rows, dim=0)[:n_rows]
        w.copy_(ortho_mat * 0.01)
        self.mix_alpha = nn.Parameter(torch.tensor(0.10))
        self.register_buffer('hash_prime', torch.tensor(1000003, dtype=torch.long), persistent=False)

    def _lookup(self, idx):
        B, T = idx.size()
        curr_first = self.first_bytes[idx]
        prev_last = F.pad(self.last_bytes[idx[:, :-1]], (1, 0), value=0)
        h = (prev_last * self.hash_prime + curr_first) % self.table_size
        return self.embed(h)

    def get_raw(self, idx):
        """Return raw embedding without alpha scaling."""
        return self._lookup(idx)

    def forward(self, idx):
        """idx: [B, T]. Hash (last_byte[prev], first_byte[curr]) -> embedding."""
        return self.mix_alpha * self._lookup(idx)


class DualTableNgramEmbedding(nn.Module):
    """Dual-table n-gram hash embedding: two half-size tables per order with
    independent hash functions, summed. Same total params, better collision resistance.
    Compatible with get_raw() interface for adaptive mixing."""
    def __init__(self, table_size, embed_dim, n_orders=3, scale=1.0, narrow_dim=None):
        super().__init__()
        self.half_table = table_size // 2
        self.n_orders = n_orders
        actual_dim = narrow_dim if narrow_dim else embed_dim
        self.narrow_dim = narrow_dim
        self.embed_dim = embed_dim
        # Dual tables for bigram
        self.embed_bi_a = nn.Embedding(self.half_table, actual_dim)
        self.embed_bi_b = nn.Embedding(self.half_table, actual_dim)
        # Dual tables for trigram
        self.embed_tri_a = nn.Embedding(self.half_table, actual_dim)
        self.embed_tri_b = nn.Embedding(self.half_table, actual_dim)
        # Spectral init: orthogonal (QR) initialization for hash tables
        # Gives each bucket a maximally distinct direction in embedding space
        for emb in [self.embed_bi_a, self.embed_bi_b, self.embed_tri_a, self.embed_tri_b]:
            w = emb.weight.data
            # For large tables (rows >> cols), use random orthogonal blocks
            n_rows, n_cols = w.shape
            n_blocks = (n_rows + n_cols - 1) // n_cols
            ortho_rows = []
            for _ in range(n_blocks):
                block = torch.randn(n_cols, n_cols)
                q, _ = torch.linalg.qr(block)
                ortho_rows.append(q)
            ortho_mat = torch.cat(ortho_rows, dim=0)[:n_rows]
            w.copy_(ortho_mat * 0.01 * scale)
        if narrow_dim:
            self.proj = nn.Linear(actual_dim, embed_dim, bias=False)
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        # Per-order mixing coefficients (fallback)
        self.alpha_bi = nn.Parameter(torch.tensor(0.15))
        self.alpha_tri = nn.Parameter(torch.tensor(0.15))
        # Independent hash primes
        self.register_buffer('primes_a', torch.tensor([1000003, 999979, 999961], dtype=torch.long), persistent=False)
        self.register_buffer('primes_b', torch.tensor([104729, 104743, 104759], dtype=torch.long), persistent=False)

    def _hash_lookup(self, idx):
        """Compute dual-hash lookups and return summed bi/tri embeddings."""
        pa0, pa1, pa2 = self.primes_a[0], self.primes_a[1], self.primes_a[2]
        pb0, pb1, pb2 = self.primes_b[0], self.primes_b[1], self.primes_b[2]
        prev1 = F.pad(idx[:, :-1], (1, 0), value=0)
        prev2 = F.pad(idx[:, :-2], (2, 0), value=0)
        # Bigram: dual hashes
        h_bi_a = (idx * pa0 + prev1 * pa1) % self.half_table
        h_bi_b = (idx * pb0 + prev1 * pb1) % self.half_table
        bi = self.embed_bi_a(h_bi_a) + self.embed_bi_b(h_bi_b)
        # Trigram: dual hashes
        h_tri_a = (idx * pa0 + prev1 * pa1 + prev2 * pa2) % self.half_table
        h_tri_b = (idx * pb0 + prev1 * pb1 + prev2 * pb2) % self.half_table
        tri = self.embed_tri_a(h_tri_a) + self.embed_tri_b(h_tri_b)
        if self.narrow_dim:
            bi = self.proj(bi)
            tri = self.proj(tri)
        return bi, tri

    def get_raw(self, idx):
        """Return raw (bi, tri) embeddings without alpha scaling."""
        return self._hash_lookup(idx)

    def forward(self, idx):
        bi, tri = self._hash_lookup(idx)
        return self.alpha_bi * bi + self.alpha_tri * tri


class AdaptiveEmbeddingMixer(nn.Module):
    """Learned per-token mixing weights for 4 embedding source channels.
    Two-layer MLP gate with prev-token context, separate gates for
    bigram, trigram, byte_content, byte_boundary."""
    def __init__(self, embed_dim, n_sources=4, hidden_dim=128):
        super().__init__()
        self.gate_w1 = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.gate_w2 = nn.Linear(hidden_dim, n_sources, bias=True)
        nn.init.normal_(self.gate_w1.weight, std=0.01)
        nn.init.zeros_(self.gate_w1.bias)
        nn.init.zeros_(self.gate_w2.weight)
        with torch.no_grad():
            # Init to match original alphas: bi=0.15, tri=0.15, byte=0.15, boundary=0.10
            # sigmoid(-1.73) ~ 0.15, sigmoid(-2.20) ~ 0.10
            self.gate_w2.bias.copy_(torch.tensor([-1.73, -1.73, -1.73, -2.20]))
        self.ctx_beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, base_embed):
        """base_embed: [B, T, D]. Returns [B, T, n_sources] sigmoid gates."""
        prev_embed = F.pad(base_embed[:, :-1, :], (0, 0, 1, 0))
        ctx = base_embed + self.ctx_beta * prev_embed
        h = F.relu(self.gate_w1(ctx))
        return torch.sigmoid(self.gate_w2(h))


class LightSSM(nn.Module):
    """Ultra-light SSM: 3-stride weighted shift with per-channel gate."""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(dim) + 0.15)
        self.w2 = nn.Parameter(torch.zeros(dim) + 0.08)
        self.w4 = nn.Parameter(torch.zeros(dim) + 0.04)
        # Per-channel gate: learn which channels benefit from SSM context
        # sigmoid(3.0) ~ 0.95, mostly passes SSM output initially
        self.ssm_gate = nn.Parameter(torch.zeros(dim) + 3.0)

    def forward(self, x):
        B, T, D = x.shape
        s1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        y = self.w1.unsqueeze(0).unsqueeze(0) * s1
        if T > 2:
            s2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))
            y = y + self.w2.unsqueeze(0).unsqueeze(0) * s2
        if T > 4:
            s4 = F.pad(x[:, :-4, :], (0, 0, 4, 0))
            y = y + self.w4.unsqueeze(0).unsqueeze(0) * s4
        # Gate the SSM output per-channel
        return torch.sigmoid(self.ssm_gate) * y


class DivergentCorrectionPath(nn.Module):
    """Per-channel affine transform (gain + bias) as divergent correction.
    Nearly zero compute overhead. Init: gain=1.0, bias=0.0 (identity).
    """
    def __init__(self, dim, rank=128, **kwargs):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, **kwargs):
        """x: [B, T, D] -> x * gain + bias"""
        return x * self.gain + self.bias


class InputTokenHiddenGate(nn.Module):
    """Per-input-token hidden state modulation via factored embedding lookup.

    A separate small embedding table (vocab x rank) + linear projection (rank x D)
    produces per-channel gain that modulates the final hidden state before lm_head.
    Embedding lookup + small matmul + element-wise = minimal compute overhead.

    hidden_out = hidden * (1 + scale * tanh(proj(embed[idx])))
    """
    def __init__(self, vocab_size, hidden_dim, rank=64):
        super().__init__()
        # Factored: vocab x rank -> rank x D (much smaller than vocab x D)
        self.embed = nn.Embedding(vocab_size, rank)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.proj = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.proj.weight)  # Start as identity (zero gate)
        self.scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, hidden, idx):
        """
        hidden: [B, T, D] -- final normalized hidden state
        idx: [B, T] -- input token IDs
        Returns: modulated hidden [B, T, D]
        """
        gate = torch.tanh(self.proj(self.embed(idx)))  # [B, T, D]
        return hidden * (1.0 + self.scale * gate)


class HourglassGPT(nn.Module):
    """
    Hourglass transformer: wide -> narrow -> wide.
    With adaptive per-token embedding mixing and divergent prediction corrections.
    With input-token-routed mixture of output projections.
    """
    def __init__(self, vocab_size, full_dim, narrow_dim, head_dim,
                 n_input_layers, n_middle_layers, n_output_layers,
                 n_local_layers, window_size, seq_len,
                 mid_attn_positions=None,
                 narrow_head_dim=64, narrow_window_size=256, narrow_mlp_expansion=6,
                 rope_base_local=10000.0, rope_base_global=10000.0,
                 bigram_table_size=32768, ngram_orders=3, ngram_narrow_dim=None,
                 tokenizer_enc=None,
                 n_div_corrections=0, div_rank=128,
                 output_mlp_hidden_dim=None,
                 n_output_routes=3, output_route_rank=192):
        super().__init__()
        self.full_dim = full_dim
        self.narrow_dim = narrow_dim
        self.head_dim = head_dim
        n_total = n_input_layers + n_middle_layers + n_output_layers
        self.n_layer = n_total
        self.n_input = n_input_layers
        self.n_middle = n_middle_layers
        self.n_output = n_output_layers
        self.seq_len = seq_len
        self.mid_attn_positions = set(mid_attn_positions or [])
        self.narrow_head_dim = narrow_head_dim

        full_heads = full_dim // head_dim
        narrow_heads = narrow_dim // narrow_head_dim

        # Embedding
        self.wte = nn.Embedding(vocab_size, full_dim)
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)

        # N-gram hash embedding (no internal alpha -- mixer handles it)
        self.ngram_embed = DualTableNgramEmbedding(bigram_table_size, full_dim, n_orders=ngram_orders, scale=1.0, narrow_dim=ngram_narrow_dim)

        # Byte-level sub-token features
        self.byte_embed = ByteFeatureEmbedding(tokenizer_enc, vocab_size, full_dim) if tokenizer_enc is not None else None
        # Byte-boundary cross-token features
        self.byte_boundary = ByteBoundaryEmbedding(tokenizer_enc, vocab_size, full_dim) if tokenizer_enc is not None else None

        # Adaptive per-token embedding mixer (4 sources: bi, tri, byte, boundary)
        self.embed_mixer = AdaptiveEmbeddingMixer(full_dim, n_sources=4)

        self.ssm_light = LightSSM(full_dim)

        # Context-conditioned norms at hourglass boundaries
        self.embed_ctx_norm = ContextNorm(full_dim, init_scale=0.12)

        # Weight-tied lm_head (standard)
        self.lm_head = nn.Linear(full_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Per-input-token output hidden additive correction
        # Each token gets a small vector added to the hidden state before lm_head
        self.token_output_emb = nn.Embedding(vocab_size, full_dim)
        nn.init.zeros_(self.token_output_emb.weight)  # Start as zero (identity)
        self.output_emb_scale = nn.Parameter(torch.tensor(0.05))

        # V-Res projection at full dim
        self.vres_proj = nn.Linear(full_dim, full_heads * head_dim, bias=False)
        # V-Res for narrow attention layers (all middle layers now have attention)
        self.vres_proj_narrow = nn.Linear(full_dim, narrow_heads * narrow_head_dim, bias=False)

        # Input layers (full dim)
        self.input_layers = nn.ModuleList()
        for i in range(n_input_layers):
            use_win = i < n_local_layers
            self.input_layers.append(
                Block(full_dim, full_heads, head_dim, i, n_total, use_win, window_size)
            )

        # Down projection: full -> narrow
        self.down_proj = nn.Linear(full_dim, narrow_dim, bias=False)

        # Middle layers: ALL attention-primary (attention + wide MLP)
        self.middle_layers = nn.ModuleList()
        for j in range(n_middle_layers):
            layer_idx = n_input_layers + j
            use_win = True  # All use narrow window
            self.middle_layers.append(
                AttentionPrimaryBlock(narrow_dim, narrow_heads, narrow_head_dim, layer_idx, n_total, use_win, narrow_window_size, mlp_expansion=narrow_mlp_expansion)
            )

        # Affine corrections for bottleneck blocks (narrow dim) -- disabled for now
        self.middle_corrections = nn.ModuleList()

        # Up projection: narrow -> full (with residual skip)
        self.up_proj = nn.Linear(narrow_dim, full_dim, bias=False)
        nn.init.zeros_(self.up_proj.weight)  # Zero-init: start as pure skip
        self.upproj_ctx_norm = ContextNorm(narrow_dim, init_scale=0.12)

        # Token-adaptive routing: per-token scalar gates at multiple points
        # Gate 1: modulates bottleneck (up_proj) contribution
        self.route_gate_bn = nn.Linear(full_dim, 1, bias=True)
        nn.init.zeros_(self.route_gate_bn.weight)
        nn.init.constant_(self.route_gate_bn.bias, 0.0)  # sigmoid(0)=0.5 centered
        # Gate 2: modulates skip connection contribution to output blocks
        self.route_gate_skip = nn.Linear(full_dim, 1, bias=True)
        nn.init.zeros_(self.route_gate_skip.weight)
        nn.init.constant_(self.route_gate_skip.bias, 0.0)

        # (no merge-point affine -- output block corrections only)

        # Output layers (full dim) -- with cross-stage skip
        self.output_layers = nn.ModuleList()
        for k in range(n_output_layers):
            layer_idx = n_input_layers + n_middle_layers + k
            use_win = layer_idx < n_local_layers
            self.output_layers.append(
                Block(full_dim, full_heads, head_dim, layer_idx, n_total, use_win, window_size,
                      use_skip=True, mlp_hidden_dim=output_mlp_hidden_dim)
            )

        # Divergent correction paths interleaved with output blocks (independent per block)
        self.n_div_corrections = n_div_corrections
        self.div_corrections = nn.ModuleList([
            DivergentCorrectionPath(full_dim, rank=div_rank, n_groups=6, vocab_size=vocab_size)
            for _ in range(n_div_corrections)
        ])

        # Narrow RoPE for bottleneck attention
        cos_narrow, sin_narrow = self._precompute_rotary_embeddings(seq_len, narrow_head_dim, base=rope_base_local)
        self.register_buffer("cos_narrow", cos_narrow, persistent=False)
        self.register_buffer("sin_narrow", sin_narrow, persistent=False)
        # Multi-resolution RoPE: different bases for local vs global layers
        cos_local, sin_local = self._precompute_rotary_embeddings(seq_len, head_dim, base=rope_base_local)
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        if rope_base_global != rope_base_local:
            cos_global, sin_global = self._precompute_rotary_embeddings(seq_len, head_dim, base=rope_base_global)
            self.register_buffer("cos_global", cos_global, persistent=False)
            self.register_buffer("sin_global", sin_global, persistent=False)
        else:
            self.register_buffer("cos_global", cos_local, persistent=False)
            self.register_buffer("sin_global", sin_local, persistent=False)
        self.n_local_layers = n_local_layers

        # (no final affine -- output block corrections only)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device="cpu"):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.wte.weight.numel()
        return 6 * (nparams - nparams_exclude)

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        rest = sum(p.numel() for p in self.parameters()) - wte
        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_blocks": rest - wte,
            "total": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos_local.size(1)
        cos_sin_local = self.cos_local[:, :T], self.sin_local[:, :T]
        cos_sin_global = self.cos_global[:, :T], self.sin_global[:, :T]
        cos_sin_narrow = self.cos_narrow[:, :T], self.sin_narrow[:, :T]

        # Base unigram embedding
        x_base = self.wte(idx)

        # Get per-token adaptive mixing weights [B, T, 4]
        # Gates: 0=bigram, 1=trigram, 2=byte_content, 3=byte_boundary
        gates = self.embed_mixer(x_base)

        # Use separate gates for each embedding source
        bi_raw, tri_raw = self.ngram_embed.get_raw(idx)
        x = x_base
        x = x + gates[:, :, 0:1] * bi_raw
        x = x + gates[:, :, 1:2] * tri_raw
        if self.byte_embed is not None:
            x = x + gates[:, :, 2:3] * self.byte_embed.get_raw(idx)
        if self.byte_boundary is not None:
            x = x + gates[:, :, 3:4] * self.byte_boundary.get_raw(idx)

        x = x + self.ssm_light(x)

        x = self.embed_ctx_norm(x)

        # V-Res embeddings
        full_heads = self.full_dim // self.head_dim
        v_embed_full = self.vres_proj(x).view(B, T, full_heads, self.head_dim)
        narrow_heads = self.narrow_dim // self.narrow_head_dim
        v_embed_narrow = self.vres_proj_narrow(x).view(B, T, narrow_heads, self.narrow_head_dim)

        # Input stage (full dim) -- all local
        for block in self.input_layers:
            x = block(x, cos_sin_local, v_embed=v_embed_full)
        # Center residual stream at stage boundary
        x = center(x)

        # Save residual for skip connection
        x_skip = x

        # Per-token route gates from pre-bottleneck state
        x_pre = norm(x)
        bn_scale = 2.0 * torch.sigmoid(self.route_gate_bn(x_pre))  # [0, 2]
        skip_scale = 2.0 * torch.sigmoid(self.route_gate_skip(x_pre))  # [0, 2]

        # Down project to narrow
        x = self.down_proj(x_pre)

        # Middle stage: ALL attention-primary blocks with affine corrections
        for i, block in enumerate(self.middle_layers):
            x = block(x, cos_sin_narrow, v_embed=v_embed_narrow)
            if i < len(self.middle_corrections):
                x = self.middle_corrections[i](x)

        # Up project + scaled bottleneck + skip
        x = bn_scale * self.up_proj(norm(x)) + x_skip

        # Output stage with post-block affine corrections
        skip_for_output = skip_scale * x_skip
        corr_idx = 0
        for block in self.output_layers:
            x = block(x, cos_sin_global, v_embed=v_embed_full, skip_signal=skip_for_output)
            skip_for_output = x
            if corr_idx < self.n_div_corrections:
                x = self.div_corrections[corr_idx](x, idx=idx)
                corr_idx += 1

        x = norm(x)
        # Per-input-token additive correction to output hidden
        x = x + self.output_emb_scale * self.token_output_emb(idx)
        logits = self.lm_head(x)
        logits = 15.0 * torch.tanh(logits / 15.0)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

FULL_DIM = 768
NARROW_DIM = 512
HEAD_DIM = 128
NARROW_HEAD_DIM = 64  # 8 heads in bottleneck for attention diversity
NARROW_WINDOW_SIZE = 256  # Reduced window for throughput
OUTPUT_MLP_HIDDEN = 7 * 768  # 5376: 7x wider output MLP (decoupled working dimension)
NARROW_MLP_EXPANSION = 6  # Wide MLP (v49 best)
N_OUTPUT_ROUTES = 1  # Not used by logit correction approach
OUTPUT_ROUTE_RANK = 64  # Rank of input-conditioned output modulation (ultra-lean)
N_INPUT_LAYERS = 1
N_MIDDLE_LAYERS = 3
N_OUTPUT_LAYERS = 2  # Fewer output blocks for higher throughput
N_DIV_CORRECTIONS = 2  # One per output block
DIV_RANK = 96  # Balanced bottleneck size
MID_ATTN_POSITIONS = list(range(3))  # All attention-primary
TOTAL_LAYERS = N_INPUT_LAYERS + N_MIDDLE_LAYERS + N_OUTPUT_LAYERS  # = 7
N_LOCAL_LAYERS = 5   # input + middle layers use SWA
WINDOW_SIZE = 512

# Gradient accumulation annealing: start with no accumulation (fast steps),
# add accumulation in late training for larger effective batch and smoother convergence
DEVICE_BATCH_SIZE = 64
EARLY_GRAD_ACCUM = 1    # Fast steps early -- 131K tokens/update for rapid exploration
LATE_GRAD_ACCUM = 2     # Large batch late -- 262K tokens/update for smooth convergence
ACCUM_SWITCH_PROGRESS = 0.50  # Switch at 50% of training
LATE_LR_BOOST = 1.2     # Mild LR boost when batch doubles
TOTAL_BATCH_SIZE_EARLY = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * EARLY_GRAD_ACCUM  # 131072
TOTAL_BATCH_SIZE_LATE = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * LATE_GRAD_ACCUM    # 262144
TOTAL_BATCH_SIZE = TOTAL_BATCH_SIZE_EARLY

# Multi-resolution RoPE -- NTK-scaled base for better position resolution
ROPE_BASE_LOCAL = 40000.0    # 4x default base
ROPE_BASE_GLOBAL = 40000.0   # Same base for all layers

MUON_LR = 0.10  # Standard LR
MUON_MOMENTUM = 0.95
DEPTH_LR_BASE = 1.15
DEPTH_LR_TOP = 0.85
DEPTH_MOM_BASE = 0.90
DEPTH_MOM_TOP = 0.97

ADAM_LR = 8e-3  # Standard Adam LR
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_WD = 0.0

WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.5
MUON_WARMDOWN = 0.40
ADAM_WARMDOWN = 0.55
FINAL_LR_FRAC = 0.05  # Focal: keep 5% of peak LR at end (prevent complete decay)

# SWA parameters
SWA_START_FRAC = 0.50
SWA_DECAY = 0.98

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
_SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
B200_BF16_PEAK_FLOPS = 2.25e15

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

BIGRAM_TABLE_SIZE = 1048576  # 1M entries for minimal collisions
NGRAM_ORDERS = 3  # bigram + trigram
NGRAM_NARROW_DIM = 192  # Factored: 1M x 192 -> 192 x 768 per order

model = HourglassGPT(
    vocab_size=vocab_size,
    full_dim=FULL_DIM,
    narrow_dim=NARROW_DIM,
    head_dim=HEAD_DIM,
    n_input_layers=N_INPUT_LAYERS,
    n_middle_layers=N_MIDDLE_LAYERS,
    n_output_layers=N_OUTPUT_LAYERS,
    n_local_layers=N_LOCAL_LAYERS,
    window_size=WINDOW_SIZE,
    seq_len=MAX_SEQ_LEN,
    mid_attn_positions=MID_ATTN_POSITIONS,
    narrow_head_dim=NARROW_HEAD_DIM,
    narrow_window_size=NARROW_WINDOW_SIZE,
    narrow_mlp_expansion=NARROW_MLP_EXPANSION,
    rope_base_local=ROPE_BASE_LOCAL,
    rope_base_global=ROPE_BASE_GLOBAL,
    bigram_table_size=BIGRAM_TABLE_SIZE,
    ngram_orders=NGRAM_ORDERS,
    ngram_narrow_dim=NGRAM_NARROW_DIM,
    tokenizer_enc=tokenizer.enc,
    n_div_corrections=N_DIV_CORRECTIONS,
    div_rank=DIV_RANK,
    output_mlp_hidden_dim=OUTPUT_MLP_HIDDEN,
    n_output_routes=N_OUTPUT_ROUTES,
    output_route_rank=OUTPUT_ROUTE_RANK,
).to(device)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
print(f"Architecture: {N_INPUT_LAYERS}x{FULL_DIM} + {N_MIDDLE_LAYERS}x{NARROW_DIM} + {N_OUTPUT_LAYERS}x{FULL_DIM}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
grad_accum_steps = EARLY_GRAD_ACCUM  # Starts at 1, may increase

# Depth-aware Muon + Adam for embeddings
# Separate hash table params for decoupled warmdown
muon_param_groups = []
adam_params = []
hash_table_params = []
seen_data_ptrs = set()

# Collect all named parameters and assign them
all_layer_params = {}  # layer_idx -> [params]
proj_params = []  # down/up projection params

for name, p in model.named_parameters():
    dp = p.data_ptr()
    if dp in seen_data_ptrs:
        continue
    seen_data_ptrs.add(dp)

    # Hash table embedding weights -> dedicated Adam group
    if p.ndim >= 2 and ("ngram_embed.embed_" in name or "byte_embed.embed" in name or "byte_boundary.embed" in name):
        hash_table_params.append(p)
        continue

    if p.ndim >= 2 and "wte" not in name and "lm_head" not in name and "ngram_embed" not in name and "byte_embed" not in name and "byte_boundary" not in name and "embed_mixer" not in name and "ssm_light" not in name and "ctx_norm" not in name:
        if "vres_proj" in name:
            proj_params.append(p)
            continue
        if "down_proj" in name or "up_proj" in name:
            proj_params.append(p)
            continue
        # Find layer index
        layer_idx = None
        for stage_name, offset in [("input_layers", 0),
                                    ("middle_layers", N_INPUT_LAYERS),
                                    ("output_layers", N_INPUT_LAYERS + N_MIDDLE_LAYERS)]:
            if stage_name in name:
                for i in range(20):
                    if f"{stage_name}.{i}." in name:
                        layer_idx = offset + i
                        break
                break
        if layer_idx is not None:
            if layer_idx not in all_layer_params:
                all_layer_params[layer_idx] = []
            all_layer_params[layer_idx].append(p)
        else:
            adam_params.append(p)
    else:
        adam_params.append(p)

for layer_idx in sorted(all_layer_params.keys()):
    if not all_layer_params[layer_idx]:
        continue
    depth_frac = layer_idx / max(TOTAL_LAYERS - 1, 1)
    lr_mult = DEPTH_LR_BASE + (DEPTH_LR_TOP - DEPTH_LR_BASE) * depth_frac
    mom = DEPTH_MOM_BASE + (DEPTH_MOM_TOP - DEPTH_MOM_BASE) * depth_frac
    muon_param_groups.append({
        "params": all_layer_params[layer_idx],
        "lr": MUON_LR * lr_mult,
        "momentum": mom,
        "ns_steps": 5,
        "lr_mult": lr_mult,
        "weight_decay": 0.005,
        "layer_idx": layer_idx,  # for gradient-informed skip
    })

# Projection params (vres, down, up) get base Muon
if proj_params:
    muon_param_groups.append({
        "params": proj_params,
        "lr": MUON_LR,
        "momentum": MUON_MOMENTUM,
        "ns_steps": 5,
        "lr_mult": 1.0,
        "weight_decay": 0.005,
    })

total_muon = sum(sum(p.numel() for p in g["params"]) for g in muon_param_groups)
print(f"Muon params: {total_muon:,} ({len(muon_param_groups)} groups)")
print(f"Adam params (default): {sum(p.numel() for p in adam_params):,}")
print(f"Adam params (hash tables): {sum(p.numel() for p in hash_table_params):,}")

muon_optimizer = Muon(muon_param_groups, lr=MUON_LR, momentum=MUON_MOMENTUM)

# Decoupled warmdown -- hash tables get shorter warmdown
HASH_WARMDOWN = 0.08
HASH_FINAL_LR_FRAC = 0.05  # Same as default
adam_base_lrs = [ADAM_LR, ADAM_LR]
adam_warmdowns = [ADAM_WARMDOWN, HASH_WARMDOWN]
adam_optimizer = torch.optim.AdamW([
    {"params": adam_params, "lr": ADAM_LR, "betas": (ADAM_BETA1, ADAM_BETA2), "weight_decay": ADAM_WD},
    {"params": hash_table_params, "lr": ADAM_LR, "betas": (ADAM_BETA1, ADAM_BETA2), "weight_decay": 0.0},
])

# ---------------------------------------------------------------------------
# Gradient-informed layer activity schedule
# Per-layer LR multiplier that changes with training progress
# Based on gradient distribution: lower layers converge first, output layers last
# ---------------------------------------------------------------------------
LAYER_ACTIVITY = {
    0: (1.0, 0.5),   # Input: dampened late (converges first)
    1: (1.0, 0.85),  # Middle: slight dampen
    2: (1.0, 0.9),   # Middle: near-full
    3: (1.0, 0.95),  # Middle: near-full
    4: (0.95, 1.3),  # Output: amplified late
    5: (0.9, 1.4),   # Output: strongly amplified late
}
ACTIVITY_START = 0.10  # Start differentiating after warmup

layer_to_muon_group = {}
for gi, group in enumerate(muon_optimizer.param_groups):
    if "layer_idx" in group:
        layer_to_muon_group[group["layer_idx"]] = gi
print(f"Activity schedule for {len(layer_to_muon_group)} layers")

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
current_grad_accum = EARLY_GRAD_ACCUM
current_total_batch = TOTAL_BATCH_SIZE_EARLY
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Grad accum annealing: {EARLY_GRAD_ACCUM} -> {LATE_GRAD_ACCUM} at progress={ACCUM_SWITCH_PROGRESS}")


def get_lr_multiplier(progress, warmdown_ratio=None, final_lr_frac=None):
    """WSD with per-optimizer warmdown (original schedule)."""
    if warmdown_ratio is None:
        warmdown_ratio = WARMDOWN_RATIO
    if final_lr_frac is None:
        final_lr_frac = FINAL_LR_FRAC
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif warmdown_ratio <= 0.0 or progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
total_tokens_processed = 0  # Track actual tokens for final reporting
batch_switched = False

# Post-warmup momentum reset
MOMENTUM_RESET_AT = [0.08]
momentum_reset_done = set()

# SWA: EMA weight averaging -- vectorized
raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
swa_param_list = list(raw_model.parameters())
swa_ema_tensors = None
swa_active = False
swa_steps = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for _micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)

    # Gradient accumulation annealing: increase accum in late training
    if not batch_switched and progress >= ACCUM_SWITCH_PROGRESS:
        batch_switched = True
        current_grad_accum = LATE_GRAD_ACCUM
        grad_accum_steps = LATE_GRAD_ACCUM
        current_total_batch = TOTAL_BATCH_SIZE_LATE
        print(f"\n[ACCUM SWITCH] Switched to grad_accum={LATE_GRAD_ACCUM} at progress={progress:.3f}")

    # Post-warmup momentum reset
    for rp in MOMENTUM_RESET_AT:
        if rp not in momentum_reset_done and progress >= rp:
            muon_optimizer.reset_momentum(decay_factor=0.0)
            momentum_reset_done.add(rp)
            print(f"\n[MOMENTUM RESET at progress={progress:.3f}]")

    lrm = get_lr_multiplier(progress)
    # Apply LR boost when using larger batch in late phase
    lr_boost = LATE_LR_BOOST if batch_switched else 1.0
    # Compute per-layer activity multiplier
    activity_mults = {}
    if progress >= ACTIVITY_START:
        interp = min((progress - ACTIVITY_START) / (1.0 - ACTIVITY_START), 1.0)
        for layer_idx, (early_m, late_m) in LAYER_ACTIVITY.items():
            activity_mults[layer_idx] = early_m + interp * (late_m - early_m)
    for group in muon_optimizer.param_groups:
        base_lr = MUON_LR * group["lr_mult"] * lrm * lr_boost
        # Apply gradient-informed activity scale if this is a layer group
        li = group.get("layer_idx", None)
        if li is not None and li in activity_mults:
            base_lr *= activity_mults[li]
        group["lr"] = base_lr
    # per-group warmdown for Adam (hash tables get shorter warmdown + higher final LR)
    adam_final_lrs = [FINAL_LR_FRAC, HASH_FINAL_LR_FRAC]
    for i, group in enumerate(adam_optimizer.param_groups):
        warmdown = adam_warmdowns[i] if i < len(adam_warmdowns) else WARMDOWN_RATIO
        flr = adam_final_lrs[i] if i < len(adam_final_lrs) else FINAL_LR_FRAC
        group_lrm = get_lr_multiplier(progress, warmdown_ratio=warmdown, final_lr_frac=flr)
        group["lr"] = adam_base_lrs[i] * group_lrm * lr_boost
    adam_lr = adam_optimizer.param_groups[0]["lr"]  # For logging

    # Block-coordinate: pure momentum oscillation
    MOM_HIGH = 0.97
    MOM_LOW = 0.88
    if step % 2 == 0:
        for group in muon_optimizer.param_groups:
            group["momentum"] = MOM_HIGH
    else:
        for group in muon_optimizer.param_groups:
            group["momentum"] = MOM_LOW

    muon_optimizer.step()
    adam_optimizer.step()
    muon_optimizer.zero_grad(set_to_none=True)
    adam_optimizer.zero_grad(set_to_none=True)

    # SWA: vectorized EMA
    if progress >= SWA_START_FRAC:
        if not swa_active:
            swa_active = True
            swa_ema_tensors = [p.data.clone() for p in swa_param_list]
            print(f"\n[SWA] Started EMA averaging at progress={progress:.3f}", flush=True)
        else:
            torch._foreach_mul_(swa_ema_tensors, SWA_DECAY)
            torch._foreach_add_(swa_ema_tensors, [p.data for p in swa_param_list], alpha=1.0 - SWA_DECAY)
        swa_steps += 1

    train_loss_f = train_loss.item()

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    total_tokens_processed += current_total_batch

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(current_total_batch / dt)
    mfu = 100 * num_flops_per_token * current_total_batch / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 10 == 0:
        muon_lr_min = min(g["lr"] for g in muon_optimizer.param_groups)
        muon_lr_max = max(g["lr"] for g in muon_optimizer.param_groups)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | muon_lr: {muon_lr_min:.4f}-{muon_lr_max:.4f} | adam_lr: {adam_lr:.2e} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = total_tokens_processed

# Apply SWA-averaged weights for evaluation
if swa_active and swa_ema_tensors is not None:
    for p, ema in zip(swa_param_list, swa_ema_tensors):
        p.data.copy_(ema)
    print(f"[SWA] Applied EMA-averaged weights ({swa_steps} averaging steps)")

model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
startup_time = t_start_training - t_start
avg_batch = total_tokens_processed / max(step, 1)
steady_state_mfu = (
    100
    * num_flops_per_token
    * avg_batch
    * (step - 10)
    / total_training_time
    / B200_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {TOTAL_LAYERS}")
