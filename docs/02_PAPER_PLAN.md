# Paper plan: NeurIPS / ICML submission

Working title: **Seeing Is Not Steering: Effect-Gramian Certificates for
Causally Usable Latent Directions**

Fallback venue ladder: NeurIPS main → ICML → ICLR → TMLR (the theory alone is a
clean TMLR paper if the empirical scale is judged insufficient).

---

## 1. The one-sentence claim

Reading a residual stream and writing to it are governed by two different,
separately computable subspaces; their misalignment is large, predictable from
cheap adjoint information, and it means probe accuracy carries essentially no
information about steering efficacy.

## 2. Why this is publishable in 2026

The novelty scan (docs/00_GAP_ANALYSIS.md) established that the adjacent claims
are taken:

- circuit non-identifiability — claimed repeatedly since ICLR 2025;
- control-theoretic interpretability — Gramians and Hankel modes have arrived;
- "decodability is not control" — *observed*, qualitatively, in 2026 work;
- toxicity steering / detox — crowded.

What is **not** taken is a *predictive* theory of the gap. Every prior paper
reports the phenomenon after running the interventions. This paper predicts it
before running any, with a closed form for the failure modes and a validity
radius for the prediction. That is the contribution, and it is why the toxicity
application is framed as a consequence rather than the headline.

## 3. Contributions as they should be listed

1. **Formalisation.** A frozen Transformer as an exact discrete controlled
   system; probes as observation operators, steering as actuation. The *effect
   Gramian* `W_g(l) = E[g g^T]` over adjoint covectors, and the certificate
   `S̄(l,w) = |ḡᵀw|/‖w‖`, computable from one backward pass and zero
   interventions.
2. **Theorem 2 (exact decoupling).** A system where the optimal probe attains
   AUC ≥ 1−ε and steers *exactly nothing*, while a chance-level direction steers
   maximally. Corollary: no function of probe accuracy is a consistent estimator
   of steering efficacy. Verified numerically to floating-point zero.
3. **Proposition 3 (inertness).** `E[S²]/max S² = r_eff/d` exactly. Measured
   `r_eff/d ≈ 0.2%` in GPT-2, so a causally uninformed direction retains a
   0.2%-scale share of available control.
4. **Proposition 5 (dark directions, closed form).** `v* = (I−P_O)ḡ / ‖·‖`
   maximises the certificate subject to probe-invisibility. Not an existence
   claim — a construction.
5. **Proposition 6 (validity radius).** `α* = 2|ḡᵀw|/(M‖w‖²)`, with `M`
   estimated by second differences, so the paper states where its own
   first-order prediction expires.
6. **Proposition 7 (finite-sample leakage).** An empirical probe manufactures
   spurious causal alignment at `Θ(N^{-1/2})`, **independent of width**.
   Measured slope 1.006, R² = 0.96.
7. **Empirics.** Constructed systems, transformers trained from scratch with a
   confound dial, and three pretrained LMs on RealToxicityPrompts — with
   difference-of-means (ActAdd/CAA) and logistic probes as first-class
   baselines, not strawmen.
8. **Active design.** Certificate-ordered intervention budgets against random
   and AUC-ordered baselines.

## 4. Structure

| § | Content | Figures |
|---|---|---|
| 1 | Intro: the read/write conflation, stated as a category error | Fig 1: the four regions |
| 2 | Setup: forward system, sensor, actuator; assumption A1 | — |
| 3 | Two geometries; effect Gramian; the certificate | — |
| 4 | Theorem 1–2, Corollary 2.1; Props 3–7 | Fig 2: (AUC, S̄) plane fully occupied |
| 5 | E1 exact verification | Fig 3: P7 scaling; Table 1 |
| 6 | E2 trained transformers + confound dial | Fig 4: ρ vs confound |
| 7 | E3/E5 real models, real data, all baselines | Fig 5: layerwise effects; Table 2 |
| 8 | E4 active design | Fig 6: budget curves |
| 9 | Safety consequence: probe-based monitoring is not evidence of control | Fig 7: identity false positives |
| 10 | Limitations and what would refute it | — |

## 5. The reviewer objections, and the answers

**"Steering along the gradient obviously beats a probe. Trivial."**
The trivial part is that ḡ moves φ. The non-trivial parts are (a) *how far*
the deployed probe falls short — unmeasured before this paper; (b) that the
certificate rank-predicts effects across a *pool* of directions including
random draws and mixtures, which is a genuine predictive claim over the space;
(c) that the best steering direction sits at near-chance probe AUC, which is a
finding rather than an assumption. §5's decoupling theorem makes (a) sharp: the
relationship is not "weaker", it is *unconstrained*.

**"You only beat a Fisher probe; nobody steers with those."**
E5 includes difference-of-means (what ActAdd/CAA actually deploy) and logistic
regression. All three read well and write poorly, because all three are
read-side constructions and none consults `W_g`.

**"First-order certificates fail at the α people use."**
Proposition 6 gives α*, E3 measures `M` and reports α*/α per layer. Where
α*/α < 1 the paper says the prediction is out of warranty. This is stated as a
boundary, not hidden.

**"Only small models."**
True and stated. The theory is scale-free; the measurements are not. Listed
first under next steps. Mitigation in hand: two model families, three
checkpoints, real data, and an exactly-solvable construction anchoring the
mechanism.

**"Synthetic prompts."**
E5 uses RealToxicityPrompts. E3's templates are retained because matched
minimal pairs are the only way to isolate the causal factor from the confound
— that is a feature of the design, and the real-data run confirms the pattern
survives without them.

**"Probe AUC is saturated at 1.0, so of course it's uninformative."**
The confound dial (E3-B) and the layer sweep produce unsaturated AUCs, and the
E1 grid sweeps AUC continuously from 0.5 to 1.0 while holding the certificate
fixed. The decoupling is not an artefact of saturation.

## 6. What would sink the paper

- `r_eff/d → 1` at scale. Then Prop. 3 is vacuous for real models.
- `ρ → 1` at scale. Then the gap is a small-model artefact.
- The certificate failing to rank-predict at usable α on a larger model.

All three are cheap to test given a GPU, and are the first three next steps.

## 7. Artefacts to ship

- `src/egc/` — certificate library, model adapters, toy transformer, data.
- `experiments/e1..e5` — one script per claim, fixed seeds, JSON outputs.
- `results/data/*.json` — every number in every figure.
- `dashboard/` — the reproducible report.
- Theory note with full proofs (docs/01_THEORY.md).
