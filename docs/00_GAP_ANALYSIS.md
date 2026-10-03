# Gap Analysis and Research Goals

Date: 2026-08-24. Author: Blessings Mambwe.

## 1. What the existing research package already establishes

`Research/` contains a 273-source deduplicated corpus, four workstream reviews,
taxonomies, a concept graph, ranked opportunities, and a benchmark plan. Its
central verdict is sound and I adopt it:

- A residual Transformer is an exact **discrete, non-autonomous, controlled
  dynamical system**; continuous-flow (ODE/PDE/energy) readings are heuristic
  for generic pretrained checkpoints.
- The productive transfer from classical inverse problems is not "invert the
  heat equation over layers". It is the *discipline*: declared forward and
  observation operators, observability analysis, designed interventions,
  adjoint sensitivities, explicit model discrepancy, and null-space
  certificates.

That analysis is correct but it is **entirely un-executed**. There is no code,
no data, no result, no figure. For NeurIPS/ICML that is the binding gap, not
the ideas.

## 2. What the 2026 literature has already taken

A novelty scan (2026-08-24) shows several of the package's headline
opportunities have been substantially claimed since the review was written:

| Package opportunity | Status in 2026 literature | Verdict |
|---|---|---|
| Set-valued / non-unique circuits (rank 5) | *Everything, Everywhere, All at Once* (ICLR 2025); *Characterizing Mechanistic Uniqueness and Identifiability* (ICLR 2026); *Certified Circuits* (2602.22968); *Many Circuits, One Mechanism* (2606.06267); *How Much Do Circuits Tell Us?* (2605.08348) | **Crowded.** Non-identifiability is now a stated premise, not a contribution. |
| Control-theoretic interpretability (rank 4/8) | *From Black-Box to White-Box: Control-Theoretic NN Interpretability* (2511.12852); *Controllability Analysis of State-Space LMs* (2511.17970); Hankel-singular-value mode ranking | **Partially taken.** Gramians have arrived; probe-placement has not. |
| Decodability vs. causal control | *Steering the Language Axis: From Linear Decodability to Causal Control* (2608.12334) | **Observed qualitatively.** No predictive quantitative theory. |
| Toxicity circuits / detox steering (rank 5 application) | *CausalDetox* (2604.14602); *Local Linearity Enables Steering via Linear Optimal Control* (2604.19018); *Activation Steering with a Feedback Controller* (2510.04309) | **Crowded as an application.** Weak as a headline claim. |
| Active experimental design **for transformer circuits** | ABCD-Strategy, active causal-model learning exist in *causal discovery*; CD-T/EAP-GP improve circuit search but all fix a hand-designed clean/corrupt set | **Open.** The transfer has not been made. |

**Consequence.** Papers 4 and 5 of the package roadmap (set-valued circuits,
toxicity application) are no longer safe flagships. Paper 2 in its stated form
("Observability-Aware Circuit Tomography", beat ACDC/EAP at equal budget) is a
benchmark-race paper that this hardware cannot win.

## 3. The gap that actually survives

Every 2026 paper above *reports* the decodability/controllability gap. None
**predicts** it. The field's operative practice is still: train a probe, get
high AUC, steer along the probe direction, and report whatever happens.

The surviving gap is therefore sharply stated:

> Given only cheap forward/adjoint information at a layer, can we **certify in
> advance** whether a decoded latent direction will causally control behaviour
> — and quantify, with an explicit null-space, the directions a probe can
> never find but that fully control the output?

This is open, it is falsifiable, it needs no GPU, and it inherits the package's
strongest machinery (observability, adjoints, null-space certificates) while
avoiding every crowded claim.

## 4. Thesis

**Reading and writing a residual stream are governed by two different
subspaces, and their misalignment is measurable, predictable, and large.**

The probe direction is determined by the *representation* geometry
(class-conditional covariance at layer ℓ). The steering-effective directions
are determined by the *effect* geometry — the span of adjoint covectors
g^{(i)} = ∇_{x_ℓ} φ carrying layer ℓ to the behaviour functional φ. These are
distinct objects. Their overlap is the whole story, and it is computable.

This yields a three-way partition of any layer's residual stream:

| Region | Probe finds it | Steering works | Name |
|---|---|---|---|
| observable ∩ controllable | yes | yes | **causal** |
| observable \ controllable | yes | no | **epiphenomenal** |
| controllable \ observable | no | yes | **dark** |

"Dark directions" are the novel object: directions that fully control the
behaviour while being invisible to any linear probe trained on the concept.
Their existence is a testable prediction, not a metaphor.

## 5. Falsifiable predictions

- **P1 (low effect rank).** The effect Gramian `W_g(ℓ) = E[g g^T]` has
  effective rank `r ≪ d`. Hence most of the residual stream is behaviourally
  inert at first order, and a random direction's steering power scales with
  its overlap with the top-`r` eigenspace.
- **P2 (dark directions exist).** The leading eigenvector of `W_g(ℓ)` is not
  the probe direction; it delivers strictly greater behaviour change per unit
  norm while achieving near-chance probe AUC.
- **P3 (the certificate transfers).** `S(ℓ,w) = |ḡ_ℓᵀ w| / ‖w‖`, computed from
  adjoints alone, rank-predicts the *measured nonlinear* steering effect
  across (layer, direction) pairs.
- **P4 (probe AUC does not).** Probe AUC is a poor predictor of the same
  quantity, and the AUC↔steering correlation is near zero once layer is
  controlled for.
- **P5 (active design pays).** Selecting interventions by certificate-weighted
  information gain recovers the causal ranking at a fraction of the uniform
  sweep budget.

Each has a clean negation. **P1–P5 all failing is itself a publishable
result**: it would say the residual stream is effectively isotropic for
control, which contradicts the entire steering literature.

## 6. Goals

| ID | Goal | Done when |
|---|---|---|
| G1 | Formal theory: forward/observation operators, effect Gramian, certificate, decomposition theorem, remainder bound | Theorems stated and proved; assumptions explicit |
| G2 | Exact toy: analytically constructed probe/steer decoupling | Numerical agreement with theory to ~machine precision |
| G3 | Trained toy transformers with planted structure | Certificate predicts steering on models nobody hand-built |
| G4 | Real LM validation (GPT-2 class, CPU) | P1–P4 measured on a real checkpoint |
| G5 | Toxicity application | Probe/steer gap measured on a toxicity behaviour; dark toxicity directions recovered |
| G6 | Active design | P5 measured against uniform and random baselines |
| G7 | HTML dashboard | All results rendered, reproducible from `results/data/*.json` |

## 7. Hardware constraint and its consequence

4 CPU cores, ~8.5 GB RAM (~2.3 GB free), no CUDA. This forbids a
benchmark-race paper and forces a **theory-plus-controlled-experiment** paper.
That is the correct genre for this contribution anyway: the claim is a
mathematical relationship between two subspaces, and the strongest evidence is
an exactly-constructed decoupling plus faithful measurement on a real model —
not a leaderboard.
