"""A small trainable Transformer plus a synthetic toxicity-like task.

The task is built to reproduce, in miniature, the structural situation that
makes real toxicity probing hard: a *causal* content variable that determines
the label, and a *spurious* surface marker that correlates with it but has no
causal path to the output. The correlation strength is a dial.

That dial is the point. Real toxicity data has exactly this structure --
identity terms, dialect markers and profanity co-occur with toxicity labels
without being what the model routes through -- and the theory in
docs/01_THEORY.md predicts the probe/steer gap should widen as the spurious
correlation strengthens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------

@dataclass
class TaskSpec:
    seq_len: int = 12
    n_filler: int = 16          # filler vocabulary size
    n_content: int = 4          # content tokens per class
    n_marker: int = 4           # marker tokens per class
    spurious: float = 0.9       # P(marker class == content class)
    seed: int = 0

    @property
    def vocab(self) -> int:
        # filler | content_benign | content_toxic | marker_a | marker_b | OUT_benign | OUT_toxic
        return self.n_filler + 2 * self.n_content + 2 * self.n_marker + 2

    # token id blocks
    @property
    def content0(self): return self.n_filler
    @property
    def content1(self): return self.n_filler + self.n_content
    @property
    def marker0(self): return self.n_filler + 2 * self.n_content
    @property
    def marker1(self): return self.n_filler + 2 * self.n_content + self.n_marker
    @property
    def out0(self): return self.vocab - 2
    @property
    def out1(self): return self.vocab - 1


def make_dataset(spec: TaskSpec, n: int, seed: int | None = None):
    """Return (tokens (n,T) int64, label (n,) int64, marker_class (n,) int64).

    Layout: filler everywhere, one content token at a random position in the
    first half, one marker token at a random position in the second half, and
    the final position is a fixed QUERY slot (filler id 0) where the model must
    emit OUT_toxic / OUT_benign.
    """
    rng = np.random.default_rng(spec.seed if seed is None else seed)
    T = spec.seq_len
    toks = rng.integers(1, spec.n_filler, size=(n, T))
    toks[:, -1] = 0                                    # query slot

    y = rng.integers(0, 2, size=n)                     # causal content class
    flip = rng.random(n) > spec.spurious
    m = np.where(flip, 1 - y, y)                       # spurious marker class

    cpos = rng.integers(0, T // 2, size=n)
    mpos = rng.integers(T // 2, T - 1, size=n)

    coff = rng.integers(0, spec.n_content, size=n)
    moff = rng.integers(0, spec.n_marker, size=n)
    ctok = np.where(y == 1, spec.content1 + coff, spec.content0 + coff)
    mtok = np.where(m == 1, spec.marker1 + moff, spec.marker0 + moff)

    toks[np.arange(n), cpos] = ctok
    toks[np.arange(n), mpos] = mtok
    return (torch.from_numpy(toks).long(),
            torch.from_numpy(y).long(),
            torch.from_numpy(m).long())


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ModelSpec:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_mlp: int = 256
    seq_len: int = 12
    vocab: int = 32


class Block(nn.Module):
    def __init__(self, s: ModelSpec):
        super().__init__()
        self.ln1 = nn.LayerNorm(s.d_model)
        self.ln2 = nn.LayerNorm(s.d_model)
        self.attn = nn.MultiheadAttention(s.d_model, s.n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(s.d_model, s.d_mlp), nn.GELU(), nn.Linear(s.d_mlp, s.d_model)
        )
        mask = torch.triu(torch.ones(s.seq_len, s.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.causal_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    """Pre-LN decoder-only Transformer with explicit per-layer residual access."""

    def __init__(self, s: ModelSpec):
        super().__init__()
        self.s = s
        self.embed = nn.Embedding(s.vocab, s.d_model)
        self.pos = nn.Parameter(torch.zeros(1, s.seq_len, s.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(s) for _ in range(s.n_layers)])
        self.ln_f = nn.LayerNorm(s.d_model)
        self.unembed = nn.Linear(s.d_model, s.vocab, bias=False)

    # -- staged forward, so a residual state can be read or written ---------

    def resid_at(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual stream entering block `layer` (layer in 0..n_layers)."""
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        for b in self.blocks[:layer]:
            x = b(x)
        return x

    def from_resid(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Continue from residual state entering block `layer`; return logits."""
        for b in self.blocks[layer:]:
            x = b(x)
        return self.unembed(self.ln_f(x))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.from_resid(self.resid_at(tokens, 0), 0)


# --------------------------------------------------------------------------
# Behaviour functional and adjoints
# --------------------------------------------------------------------------

def phi_logit_diff(logits: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    """Behaviour: logit(OUT_toxic) - logit(OUT_benign) at the query position."""
    last = logits[:, -1, :]
    return last[:, spec.out1] - last[:, spec.out0]


def adjoint_covectors(model, tokens, spec, layer, pos=-1, batch=256):
    """g^{(i)} = d phi / d x_layer[pos]  -- one reverse pass per batch.

    Returns (X, G) with X the residual states at `pos` (n,d) and G the
    covectors (n,d), both numpy float64.
    """
    Xs, Gs = [], []
    for i in range(0, len(tokens), batch):
        tb = tokens[i:i + batch]
        with torch.no_grad():
            x = model.resid_at(tb, layer)
        x = x.detach().clone().requires_grad_(True)
        phi = phi_logit_diff(model.from_resid(x, layer), spec)
        g, = torch.autograd.grad(phi.sum(), x)
        Xs.append(x[:, pos, :].detach().numpy().astype(np.float64))
        Gs.append(g[:, pos, :].detach().numpy().astype(np.float64))
    return np.concatenate(Xs), np.concatenate(Gs)


@torch.no_grad()
def measure_steering(model, tokens, spec, layer, w, alpha, pos=-1, batch=256):
    """Measured causal effect D_i(alpha, w) of writing alpha*w at (layer,pos)."""
    w_t = torch.as_tensor(np.asarray(w, dtype=np.float32))
    w_t = w_t / (w_t.norm() + 1e-12)
    out = []
    for i in range(0, len(tokens), batch):
        tb = tokens[i:i + batch]
        x = model.resid_at(tb, layer)
        base = phi_logit_diff(model.from_resid(x, layer), spec)
        xp = x.clone()
        xp[:, pos, :] = xp[:, pos, :] + alpha * w_t
        pert = phi_logit_diff(model.from_resid(xp, layer), spec)
        out.append((pert - base).numpy().astype(np.float64))
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_toy(spec: TaskSpec, mspec: ModelSpec, n_train=24000, steps=1500,
              bs=256, lr=3e-3, seed=0, verbose=False):
    torch.manual_seed(seed)
    model = ToyTransformer(mspec)
    toks, y, m = make_dataset(spec, n_train, seed=seed + 1)
    target = torch.where(y == 1, torch.tensor(spec.out1), torch.tensor(spec.out0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed + 2)
    for t in range(steps):
        idx = torch.randint(0, n_train, (bs,), generator=g)
        logits = model(toks[idx])[:, -1, :]
        loss = F.cross_entropy(logits, target[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if verbose and (t % 250 == 0 or t == steps - 1):
            print(f"    step {t:>5}  loss {loss.item():.4f}")
    model.eval()
    return model


@torch.no_grad()
def accuracy(model, tokens, y, spec, batch=512):
    correct = 0
    for i in range(0, len(tokens), batch):
        logits = model(tokens[i:i + batch])[:, -1, :]
        pred = (logits[:, spec.out1] > logits[:, spec.out0]).long()
        correct += int((pred == y[i:i + batch]).sum())
    return correct / len(tokens)
