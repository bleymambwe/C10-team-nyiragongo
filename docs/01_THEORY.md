# Seeing Is Not Steering: Effect-Gramian Certificates for Causally Usable Latent Directions

Formal development. All claims are stated with their assumptions; proofs follow
each statement. Numerical verification lives in `experiments/`.

---

## 1. Forward system, sensor, actuator

Let a frozen decoder-only Transformer have residual stream `x_l` in `R^d`,
`l = 0..L`, evolving by

    x_{l+1} = x_l + F_l(x_l ; c)                                    (1)

with `c` the (fixed) context and `F_l` the block update. Equation (1) is exact:
no continuous-time, autonomy, invertibility, or energy-descent assumption is
made or needed.

**Behaviour functional.** Fix a scalar readout of interest `phi : R^d -> R`
(in experiments: a logit difference between a target and a contrast token).
Define the *layer-l tail map*

    Phi_l := phi . T_{L-1} . ... . T_l ,    T_l(x) = x + F_l(x;c)   (2)

so `Phi_l(x_l)` is the behaviour produced by state `x_l`.

**Sensor (probe).** A linear probe at layer `l` is a covector `w`; its readout
is `w^T x_l`. This is exactly an observation operator `H_l = w^T`.

**Actuator (steering).** An additive intervention at layer `l` is
`x_l -> x_l + a w`, giving the measured causal effect

    D_i(a, w) := Phi_l(x_l^{(i)} + a w) - Phi_l(x_l^{(i)})          (3)

The field's standard practice fits `w` from (sensor) data and then uses the
same `w` as (actuator). Sections 2-5 show why that is a category error with a
measurable cost.

**Assumption A1 (smoothness).** `Phi_l` is `C^2` on a neighbourhood of the
segment `[x_l^{(i)}, x_l^{(i)} + a w]`, with `||Hess Phi_l||_op <= M_l` there.

*Remark.* A1 holds for standard blocks away from the measure-zero kink set of
ReLU-family nonlinearities; GELU/SiLU networks are smooth. `M_l` is estimated
empirically in Section 8, not assumed.

---

## 2. The two geometries

### 2.1 Effect (write) geometry

**Definition 1 (adjoint covector).** For example `i`,

    g_l^{(i)} := grad_x Phi_l (x_l^{(i)}) = J_l^{(i)T} grad phi(x_L^{(i)})

where `J_l^{(i)} = d x_L / d x_l`. One reverse-mode pass per example yields it.

**Definition 2 (effect Gramian).**

    W_g(l) := (1/N) sum_i g_l^{(i)} g_l^{(i)T}     (d x d, PSD)
    gbar_l := (1/N) sum_i g_l^{(i)}

`W_g` is the finite-horizon, behaviour-restricted analogue of a
controllability-to-output Gramian for system (1): it measures how much
*behaviour* a unit-norm write at layer `l` can produce.

**Definition 3 (behavioural null space).**
`N(l) := ker W_g(l) = { w : g_l^{(i)T} w = 0 for all i }`.
Writes into `N(l)` are inert to first order for every example in the sample.

### 2.2 Representation (read) geometry

Given binary concept labels `y_i` in `{0,1}`, let `mu_0, mu_1` be class means
at layer `l`, `delta_l := mu_1 - mu_0`, and `Sigma_l` the pooled within-class
covariance.

**Definition 4 (probe direction).** The Fisher/LDA-optimal linear probe is
`w_probe ~ (Sigma_l + gI)^{-1} delta_l`. Logistic regression converges to the
same direction under class-conditional Gaussianity with shared covariance.

`W_g` depends on `Sigma_l` **not at all**; `w_probe` depends on `W_g` **not at
all**. They are computed from disjoint information. Nothing in training ties
them together. That is the entire source of the phenomenon.

---

## 3. The certificate

**Definition 5 (steerability certificate).** For a direction `w != 0`,

    Sbar(l, w) := | gbar_l^T w | / ||w||          (coherent effect per unit write)
    S2(l, w)   := sqrt( w^T W_g(l) w ) / ||w||    (RMS effect per unit write)

`Sbar` predicts the *population-average* behaviour shift that steering
produces; `S2` bounds the per-example magnitude and is insensitive to sign
cancellation. By Cauchy-Schwarz `Sbar <= S2 <= sqrt(lambda_1(W_g))`, with `S2`
maximised by the leading eigenvector of `W_g` and `Sbar` maximised by `gbar_l`.

Both are computable from `N` reverse-mode passes -- **no intervention is run**.

---

## 4. Decomposition and the three regions

**Theorem 1 (three-way decomposition).**
Let `P_C` be the orthogonal projector onto `C_r(l)`, the span of the top-`r`
eigenvectors of `W_g(l)`, with eigenvalues `lambda_1 >= ... >= lambda_d >= 0`.
Let `P_O` project onto the discriminative subspace
`O(l) := span{ (Sigma_l + gI)^{-1} delta_l }` (more generally the top-`s`
Fisher directions). Then for every `w` and every example `i`,

    D_i(a, w) = a [ g^{(i)T} P_C w + g^{(i)T} (I - P_C) w ] + R_i
    RMS_i | g^{(i)T} (I - P_C) w |  <=  sqrt(lambda_{r+1}) ||w||
    | R_i |                         <=  (M_l / 2) a^2 ||w||^2

Consequently `R^d` partitions, relative to `(C_r, O)`, into

| region | probe sees it | steering moves behaviour | name |
|---|---|---|---|
| `C_r and O`        | yes | yes | **causal** |
| `O minus C_r`      | yes | no  | **epiphenomenal** |
| `C_r minus O`      | no  | yes | **dark** |
| `(C_r or O)^perp`  | no  | no  | inert |

*Proof.* Taylor's theorem with Lagrange remainder on the segment gives
`D_i = a g^{(i)T} w + R_i` with
`|R_i| <= (1/2) a^2 ||w||^2 sup ||Hess Phi_l|| <= (M_l/2) a^2 ||w||^2` under
A1. Splitting `w = P_C w + (I - P_C) w` is an identity. For the RMS bound,
`(1/N) sum_i (g^{(i)T}(I-P_C)w)^2 = w^T (I-P_C) W_g (I-P_C) w
<= lambda_{r+1} ||w||^2`, since `(I-P_C) W_g (I-P_C)` has spectral norm
`lambda_{r+1}` by construction of `P_C`. QED

The table is not a metaphor: membership of each region is decided by two
computable quantities, `Sbar(l,w)` and the probe AUC of `w`.

---

## 5. Exact decoupling: probe quality carries no causal information

**Theorem 2 (constructive decoupling).**
For every `d >= 2`, every `eps` in `(0, 1/2)` there is a system satisfying A1
-- indeed with `Phi_l` linear, so `M_l = 0` and the first-order expansion is
*exact* -- and a labelled representation, such that

  (a) the optimal linear probe attains `AUC >= 1 - eps`;
  (b) steering along the probe direction has **exactly zero** effect,
      `D_i(a, w_probe) = 0` for all `a` and all `i`;
  (c) there is a unit direction `v` with `AUC(v) = 1/2` exactly and
      `|gbar^T v| = ||gbar||`, the maximum attainable -- so `v` is dark and
      `D_i(a, v) = a ||gbar||`.

*Proof (construction).* Take `Phi_l(x) = e_1^T x`, so `g^{(i)} = gbar = e_1`
for all `i`. Draw the representation as `x | y ~ N(y * Dm * e_2, s^2 I_d)`.
Then `delta = Dm e_2`, `Sigma = s^2 I`, and `w_probe ~ e_2`.

(b) `D_i(a, w_probe) = Phi_l(x + a e_2) - Phi_l(x) = a e_1^T e_2 = 0`, exactly,
for all `a` and all `i`.

(a) The probe score `e_2^T x` is `N(y Dm, s^2)`, so `AUC = Ncdf(Dm/(s sqrt2))`;
choosing `Dm/s >= sqrt2 * Nquantile(1 - eps)` gives `AUC >= 1 - eps`.

(c) Take `v = e_1`. Its score `e_1^T x` is `N(0, s^2)` **independent of `y`**,
so its AUC is exactly `1/2`. And `|gbar^T v| = 1 = ||gbar||`, while
`D_i(a, v) = a`. QED

**Corollary 2.1.** No function of probe accuracy alone is a consistent
estimator of steering efficacy. Probe AUC and `Sbar` can be independently set
to any admissible pair of values.

This is the formal content of "seeing is not steering". It also refutes the
converse assumption -- that a probe-invisible direction is causally irrelevant
-- by construction.

### 5.1 Finite-sample probes manufacture causal effect

Theorem 2 concerns the *population* probe. Practitioners fit probes on `N`
samples, and the estimate leaks into the effect direction.

**Proposition 7 (spurious causal leakage).** Let the population probe
direction `w_pop` be exactly behaviourally inert, `gbar^T w_pop = 0`. Let
`what` be the Fisher probe estimated from `N` i.i.d. samples. Then

    E[ rhohat ] = E[ |cos angle(what, gbar)| ]  =  Theta( N^{-1/2} )

with a constant that does **not** grow with `d`. Consequently steering along
an empirical probe produces a measured effect of size `~ ||gbar|| / sqrt(N)`
that is pure estimation noise.

*Proof sketch.* Write `what ~ w_pop + e`, where the estimation error `e` is
asymptotically Gaussian and, for the isotropic-`Sigma` construction, isotropic
within the complement of `w_pop`, with `E||e||^2 = C d / N`. The leakage is
the component of `e` along the *single fixed* unit vector `gbar`, i.e. one
coordinate of an isotropic vector spread over `d - 1` dimensions:
`E[(e^T gbar)^2] = E||e||^2 / (d-1) = C d / (N (d-1)) -> C/N`. Taking square
roots gives `rhohat = Theta(N^{-1/2})`, and the `d` factors cancel. QED

The dimension-cancellation is the practically important part. It says the
artefact is **not** mitigated by working in a wider residual stream, and that
a probe fitted on a typical interpretability dataset (`N ~ 10^3`) carries a
spurious alignment of order `3 x 10^-2` before any real effect exists.

*Measured (E1):* the log-log slope against `N^{-1/2}` is `1.006` (predicted
`1.0`) with `R^2 = 0.96`, versus slope `0.90`, `R^2 = 0.72` for the
dimension-dependent alternative `sqrt(d/N)`. The dimension-free law wins.

---

## 6. How much of the residual stream is inert?

**Definition 6 (effect participation rank).**
`r_eff(l) := tr W_g(l) / lambda_1(W_g(l))`, which lies in `[1, d]`.

**Proposition 3 (random-direction inertness).**
For `w` uniform on the unit sphere `S^{d-1}`,

    E[ S2(l,w)^2 ] = tr W_g(l) / d
    E[ S2(l,w)^2 ] / max_w S2(l,w)^2 = r_eff(l) / d

*Proof.* `E[w w^T] = I/d`, so `E[w^T W_g w] = tr(W_g)/d`. The maximum of
`w^T W_g w` over the unit sphere is `lambda_1`. Divide. QED

So `r_eff/d` is *the* fraction of steering power retained by a direction chosen
without causal information. Measuring `r_eff << d` (prediction **P1**) makes
"we found a direction and it changed behaviour" a claim about `r_eff/d`-scale
luck unless the direction was chosen using write-side information.

**Proposition 4 (probe suboptimality).**
`Sbar(l, w_probe) = ||gbar_l|| * |cos angle( (Sigma_l+gI)^{-1} delta_l , gbar_l )|`,
so the probe direction attains the optimum `||gbar_l||` **iff**
`(Sigma_l+gI)^{-1} delta_l` is parallel to `gbar_l`.

*Proof.* Immediate from Definition 5 and `max_{||w||=1} |gbar^T w| = ||gbar||`,
attained at `w = gbar/||gbar||`. QED

The alignment `rho(l) := |cos angle(w_probe, gbar_l)|` is therefore the single
scalar saying how much causal control a probe-derived steering vector
inherits. It is reported for every layer in every experiment below.

---

## 7. Constructing dark directions

**Definition 7 (dark direction).** `v` is `(eps, kappa)`-dark at layer `l` if
`AUC(v) <= 1/2 + eps` and `Sbar(l,v) >= kappa * Sbar(l, w_probe)`.

**Proposition 5 (closed form).** Let `P_O` project onto the discriminative
subspace `O(l)`. Then

    vstar := (I - P_O) gbar_l / || (I - P_O) gbar_l ||

maximises `Sbar(l, v)` over all `v` orthogonal to `O(l)`, with value
`Sbar(l, vstar) = || (I - P_O) gbar_l ||`. Under the class-conditional
Gaussian model with shared covariance, any `v` orthogonal to `O(l)` in the
whitened metric has probe AUC exactly `1/2`.

*Proof.* Maximising `|gbar^T v|` subject to `||v||=1`, `P_O v = 0` is
maximising `|((I-P_O) gbar)^T v|` over unit `v` in `O^perp`, attained at the
normalised `(I-P_O) gbar`. For the AUC claim: the score `v^T x` has
class-conditional means `v^T mu_y`; `v^T delta` vanishes when `v` is orthogonal
to `(Sigma+gI)^{-1} delta` in the `(Sigma+gI)`-metric. Equal means with shared
variance implies AUC = 1/2. QED

Proposition 5 upgrades "dark directions" from an existence claim to a
**construction**: one projection of one adjoint mean vector.

---

## 8. Validity range of the certificate

**Proposition 6 (usable steering radius).** Under A1, the sign and leading
behaviour of `D_i(a,w)` are determined by the certificate whenever

    |a| < astar(l,w) := 2 |gbar_l^T w| / ( M_l ||w||^2 )

*Proof.* From Theorem 1, `|D - a gbar^T w| <= (M_l/2) a^2 ||w||^2` in
population form; requiring that bound to be smaller than `|a gbar^T w|` gives
the stated radius. QED

`M_l` is estimated by second differences along the steering ray, so `astar` is
reported, not assumed. Beyond `astar` the linear certificate is void and
measured effects must be used -- exactly the regime where large-`a` steering
studies operate, and Proposition 6 says so quantitatively.

---

## 9. Active intervention design

Ranking `M` candidate directions by true effect costs `M` intervention sweeps.
The certificate supplies a free prior. Model measured effects as
`Dhat(w) = D(w) + eta`, `eta ~ N(0, sm^2)`, with prior
`D(w) ~ N(a * Sbar(l,w), tau^2)`. Sequentially select the direction maximising
expected reduction in the entropy of the top-`k` set. Expected budget saving
grows with the prior's rank correlation to truth -- precisely prediction
**P3**. We report measured budget curves rather than a bound, since the
Gaussian prior is an approximation.

---

## 10. Specialisation to toxicity

Let `phi` be the logit difference between a toxic and a matched non-toxic
continuation token; labels `y` mark toxic vs. benign prompts. Then:

- `w_probe(l)` is a standard toxicity probe;
- `gbar_l` is the *toxicity effect covector*;
- `rho(l)` measures how much of a detox-steering vector's effect is real;
- `vstar(l)` from Prop. 5 is a **dark toxicity direction**: it controls toxic
  output while being invisible to the toxicity probe.

The safety consequence is direct and falsifiable: a monitor built from
`w_probe` is blind on `vstar`, so *toxicity-probe-based monitoring is not
evidence of control*. Section 5 makes this a theorem, not a worry.

---

## 11. What would refute this

- `r_eff(l) ~ d` on real models: Proposition 3 has no bite; the stream is
  effectively isotropic for control.
- `rho(l) ~ 1` on real models: probes happen to align with adjoints; the gap
  is empirically empty even though Theorem 2 makes it logically possible.
- `Sbar` failing to rank-predict measured effects: the first-order certificate
  is void at usable `a`, and Section 9 collapses with it.
- `vstar` failing to steer, or being detectable by a probe: Proposition 5's
  Gaussian premise is too strong for real representations.

Each is measured directly in `experiments/`.

---

## 12. What the measurements changed

Written after E1/E3 rather than before, so the record shows which parts of the
theory survived contact with data and which did not.

### 12.1 Proposition 5's AUC claim is only approximately true in practice

Prop. 5 predicts that a direction orthogonal to `O(l)` in the whitened metric
has probe AUC exactly `1/2`. That is exact under shared-covariance Gaussian
class conditionals, and E1 confirms it there (measured `0.5006`, `0.5017`).

In GPT-2 it does **not** hold exactly. The constructed direction `vstar` has
probe AUC ≈ `0.69` across layers, not `0.50` — real residual representations
are not shared-covariance Gaussian, so orthogonality in the whitened metric
does not force equal class means. The *steering* half of the proposition holds
completely: `vstar` retains **99.7 %** of the adjoint's causal effect.

The honest statement is therefore weaker than "invisible" and still strong:
constraining a direction to be exactly orthogonal to the probe costs
essentially none of its control. Symmetrically, the best steering direction
`ḡ` is itself a poor detector (AUC ≈ `0.70` against the probe's `1.000`). The
dissociation is two-way; neither side is total in a real model.

### 12.2 The gap is not caused by dataset confounding

The pre-registered expectation was that the probe/steer gap grows with the
spurious correlation between a surface confound and the label. On GPT-2 this
is **false**: sweeping `P(identity term matches label)` over
`{0.50, 0.70, 0.85, 0.95}` moves mean `ρ` only over `{0.066, 0.051, 0.057,
0.047}` — no trend, and the gap is already near-total at zero confound.

The surviving explanation is Proposition 3, not confounding. With
`r_eff/d ≈ 0.0018`, the effect geometry occupies roughly two thousandths of
the residual stream, so *any* read-side direction — however it was
supervised — is near-orthogonal to it with overwhelming probability. Extreme
low-rankness of `W_g`, not label contamination, is the mechanism.

This makes the result stronger and more general: it does not depend on a
flawed dataset, and cleaning the labels will not fix it.

### 12.3 Prompt perplexity is a vacuous collateral-damage metric here

The first audit measured the change in the prompt's own next-token perplexity
under the steer. It returned identically zero at every layer. That is
structural, not a small effect: the write lands on the final token, and causal
masking forbids a final-position state from influencing predictions made at
earlier positions.

The correct measure for a final-position write is the KL divergence of the
next-token distribution, which the steer does move. Recorded here because a
metric that cannot be nonzero is worse than no metric — it reads as evidence
of safety.

### 12.4 What held up unchanged

- Theorem 2 and Corollary 2.1: verified to floating-point zero.
- Proposition 3: `E[S²] = tr W_g/d` matched Monte-Carlo to `<1.3 %`, with the
  isotropic control at `0.975` against an exact `1.0`.
- Proposition 6: `α*` computed per layer; the write magnitude used sits inside
  the validity radius for layers 2–11 and outside it for layers 0–1, which is
  reported rather than smoothed over.
- Proposition 7: slope `1.006` against the predicted `1.0`, `R² = 0.96`.
- The certificate rank-predicts measured steering at Spearman `0.82–0.96` per
  layer, while probe AUC sits in `[-0.34, +0.28]`.
