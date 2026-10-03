"""Adapter that exposes a HuggingFace causal LM as the controlled system of
docs/01_THEORY.md Section 1.

Provides, for every layer l:
  x_l^{(i)}  residual state entering block l at the final real token
  g_l^{(i)}  adjoint covector d phi / d x_l   (one backward pass gives ALL layers)
  D_i(a,w)   measured causal effect of writing a*w into (l, final token)

The read (x) and write (g) sides are obtained from a single forward/backward,
so the certificate costs one extra backward pass over the probe-training data
and no interventions at all.
"""

from __future__ import annotations

import numpy as np
import torch


class LMAdapter:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu", dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"          # final position == last real token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        base = None
        for attr in ("transformer", "gpt_neox", "model"):          # GPT-2 / NeoX / Llama-style
            base = getattr(self.model, attr, None)
            if base is not None:
                break
        if base is None:
            raise RuntimeError(f"cannot locate transformer body of {model_name}")
        self.blocks = getattr(base, "h", None) or getattr(base, "layers")
        self.n_layers = len(self.blocks)
        cfg = self.model.config
        self.d_model = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size")

    # ------------------------------------------------------------------
    def encode(self, prompts: list[str]):
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=64)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def token_ids(self, words: list[str], leading_space: bool = True) -> list[int]:
        """Single-token ids for the given words (skips words that don't fit one token)."""
        ids = []
        for w in words:
            s = (" " + w) if leading_space else w
            t = self.tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
        return ids

    # ------------------------------------------------------------------
    def _phi(self, logits, pos_ids: list[int], neg_ids: list[int]) -> torch.Tensor:
        """Behaviour functional: mean logit over pos_ids minus mean over neg_ids,
        at the final position."""
        last = logits[:, -1, :]
        return last[:, pos_ids].mean(-1) - last[:, neg_ids].mean(-1)

    # ------------------------------------------------------------------
    def states_and_adjoints(self, prompts: list[str], pos_ids, neg_ids, batch=16):
        """Return X, G: dicts layer -> (N, d) float64, plus phi (N,).

        One forward + one backward per batch yields adjoints at every layer.
        """
        L = self.n_layers
        Xs = {l: [] for l in range(L)}
        Gs = {l: [] for l in range(L)}
        phis = []
        for i in range(0, len(prompts), batch):
            ids, mask = self.encode(prompts[i:i + batch])
            caps: dict[int, torch.Tensor] = {}

            def mk(l):
                def hook(mod, args, kwargs):
                    t = args[0] if args else kwargs.get("hidden_states")
                    t.requires_grad_(True)
                    t.retain_grad()
                    caps[l] = t
                    return None
                return hook

            handles = [self.blocks[l].register_forward_pre_hook(mk(l), with_kwargs=True)
                       for l in range(L)]
            try:
                with torch.enable_grad():
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
                    phi = self._phi(logits, pos_ids, neg_ids)
                    phi.sum().backward()
            finally:
                for h in handles:
                    h.remove()

            phis.append(phi.detach().cpu().numpy().astype(np.float64))
            for l in range(L):
                t = caps[l]
                Xs[l].append(t[:, -1, :].detach().cpu().numpy().astype(np.float64))
                Gs[l].append(t.grad[:, -1, :].detach().cpu().numpy().astype(np.float64))
            self.model.zero_grad(set_to_none=True)

        X = {l: np.concatenate(Xs[l]) for l in range(L)}
        G = {l: np.concatenate(Gs[l]) for l in range(L)}
        return X, G, np.concatenate(phis)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def baseline_phi(self, prompts: list[str], pos_ids, neg_ids, batch=16) -> np.ndarray:
        """Unperturbed phi, cached and reused across every steering measurement."""
        out = []
        for i in range(0, len(prompts), batch):
            ids, mask = self.encode(prompts[i:i + batch])
            out.append(self._phi(self.model(input_ids=ids, attention_mask=mask).logits,
                                 pos_ids, neg_ids).cpu().numpy().astype(np.float64))
        return np.concatenate(out)

    @torch.no_grad()
    def measure_steering(self, prompts: list[str], layer: int, w, alpha: float,
                         pos_ids=None, neg_ids=None, batch=16,
                         base: np.ndarray | None = None) -> np.ndarray:
        """Measured D_i(alpha, w) writing into block `layer` at the final token.

        Pass `base` (from baseline_phi) to skip the unperturbed forward pass.
        """
        w_t = torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.device)
        w_t = w_t / (w_t.norm() + 1e-12)
        out = []
        for i in range(0, len(prompts), batch):
            ids, mask = self.encode(prompts[i:i + batch])
            if base is None:
                b = self._phi(self.model(input_ids=ids, attention_mask=mask).logits,
                              pos_ids, neg_ids).cpu().numpy().astype(np.float64)
            else:
                b = base[i:i + len(ids)]

            def hook(mod, args, kwargs):
                t = args[0] if args else kwargs.get("hidden_states")
                t = t.clone()
                t[:, -1, :] = t[:, -1, :] + alpha * w_t
                if args:
                    return (t,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = t
                return args, kwargs

            h = self.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
            try:
                pert = self._phi(self.model(input_ids=ids, attention_mask=mask).logits,
                                 pos_ids, neg_ids).cpu().numpy().astype(np.float64)
            finally:
                h.remove()
            out.append(pert - b)
        return np.concatenate(out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def next_token_kl(self, prompts: list[str], layer: int, w, alpha: float,
                      batch=16) -> float:
        """Collateral-damage audit: mean KL(perturbed || clean) of the next-token
        distribution.

        This is the right measure for a final-position write. Measuring the
        prompt's own perplexity instead returns exactly zero by causal masking --
        a write at the last position cannot influence predictions made at earlier
        positions -- so that metric is vacuous here, not merely small.
        """
        import torch.nn.functional as F
        w_t = torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.device)
        w_t = w_t / (w_t.norm() + 1e-12)
        kls = []
        for i in range(0, len(prompts), batch):
            ids, mask = self.encode(prompts[i:i + batch])
            clean = F.log_softmax(
                self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :], dim=-1)

            def hook(mod, args, kwargs):
                t = args[0] if args else kwargs.get("hidden_states")
                t = t.clone()
                t[:, -1, :] = t[:, -1, :] + alpha * w_t
                if args:
                    return (t,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = t
                return args, kwargs

            h = self.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
            try:
                pert = F.log_softmax(
                    self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :], dim=-1)
            finally:
                h.remove()
            kl = (pert.exp() * (pert - clean)).sum(-1)
            kls.append(kl.cpu().numpy().astype(np.float64))
        return float(np.concatenate(kls).mean())

    @torch.no_grad()
    def perplexity_delta(self, prompts: list[str], layer: int, w, alpha: float,
                         batch=16) -> float:
        """Prompt-perplexity delta under a final-position write.

        Retained for the record: this is identically zero by causal masking. Use
        next_token_kl for collateral damage at the final position.
        """
        w_t = torch.as_tensor(np.asarray(w, dtype=np.float32), device=self.device)
        w_t = w_t / (w_t.norm() + 1e-12)
        import torch.nn.functional as F
        deltas = []
        for i in range(0, len(prompts), batch):
            ids, mask = self.encode(prompts[i:i + batch])
            tgt = ids[:, 1:]
            m2 = mask[:, 1:].float()

            def nll(logits):
                lp = F.log_softmax(logits[:, :-1, :], dim=-1)
                g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                return (-(g * m2).sum(1) / m2.sum(1).clamp(min=1))

            base = nll(self.model(input_ids=ids, attention_mask=mask).logits)

            def hook(mod, args, kwargs):
                t = args[0] if args else kwargs.get("hidden_states")
                t = t.clone()
                t[:, -1, :] = t[:, -1, :] + alpha * w_t
                if args:
                    return (t,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = t
                return args, kwargs

            h = self.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
            try:
                pert = nll(self.model(input_ids=ids, attention_mask=mask).logits)
            finally:
                h.remove()
            deltas.append((pert - base).cpu().numpy())
        return float(np.concatenate(deltas).mean())
