# Video transcript - *Seeing Is Not Steering*

Runtime **12.45 min** - 19 scenes - 1920x1080, 30 fps.
Narration synthesised locally with Kokoro-82M (`af_heart`, speed 0.98); no network, no API key.
Rendered with Remotion. Every on-screen figure reads `paper-video/src/facts.json`, generated from `results/data/`.

| # | Time | Scene |
|---|---|---|
| 00 | `00:00` | Opening |
| 01 | `00:09` | The habit: one vector, two jobs |
| 02 | `00:36` | Two geometries, one residual stream |
| 03 | `01:24` | The steerability certificate |
| 04 | `01:45` | Four regions of a residual stream |
| 05 | `02:15` | Theorem 2 - exact decoupling |
| 06 | `03:02` | Proposition 7 - finite-sample leakage |
| 07 | `03:44` | Proposition 3 - effective rank |
| 08 | `04:20` | Experiment E3 - toxicity in GPT-2 |
| 09 | `05:02` | The "gradients obviously steer better" objection |
| 10 | `05:48` | Experiment E6 - structural independence |
| 11 | `06:32` | Experiment E5 - real data, three models |
| 12 | `07:14` | Experiment E2 - trained from scratch |
| 13 | `07:44` | Experiment E4 - active intervention design |
| 14 | `08:21` | What did not work |
| 15 | `09:50` | Limitations |
| 16 | `10:48` | Consequence for monitoring |
| 17 | `11:08` | Next steps |
| 18 | `12:04` | The claim in one sentence |

---

## 00 - 00:00 - Opening

Seeing is not steering. A study of why linear probes find concepts that steering vectors cannot control, and what to measure instead.

## 01 - 00:09 - The habit: one vector, two jobs

Here is a habit almost every interpretability paper shares. Train a linear probe on a model's hidden states until it detects some concept reliably. Then take that same probe direction, add it back into the residual stream, and expect the model's behaviour to move. The probe is being used twice: once as a sensor, and once as an actuator. Those are different jobs, and nothing guarantees one vector is good at both.

## 02 - 00:36 - Two geometries, one residual stream

Write the frozen transformer as what it exactly is: a discrete controlled system. The residual stream evolves layer by layer, and some scalar behaviour is read off at the end. Now look at what determines each side. The probe direction is fixed by the representation geometry: the difference between class means, whitened by the within-class covariance. The steering-effective directions are fixed by the effect geometry: the span of adjoint covectors, the gradients of the behaviour with respect to that layer, summarised by what we call the effect Gramian. The covariance never enters the effect Gramian. The effect Gramian never enters the probe. They are computed from disjoint information, and nothing in training couples them. Their alignment is a free parameter.

## 03 - 01:24 - The steerability certificate

That free parameter is measurable, and cheaply. Define the steerability certificate: the projection of the mean adjoint covector onto a candidate direction, per unit norm. It costs one backward pass over data the probe already needed, and it runs no interventions at all. It predicts what steering will do before you steer.

## 04 - 01:45 - Four regions of a residual stream

The certificate and the probe's accuracy together partition the residual stream into four regions. Causal directions, which the probe sees and which move behaviour. Epiphenomenal directions, which the probe sees but which move nothing. Dark directions, which the probe cannot see but which control the output. And inert directions, invisible and inconsequential. Standard practice assumes everything lives in the first region. The rest of this work measures where things actually live.

## 05 - 02:15 - Theorem 2 - exact decoupling

First, can the two be decoupled in principle? Yes, and completely. There is a construction where the optimal linear probe classifies at an area under the curve of nearly one, and steering along that exact probe direction produces exactly zero effect. Not small: zero. In the same system, a direction that classifies at chance, an area under the curve of one half, produces the maximum behavioural change available. Verified numerically to two times ten to the minus fifteen, which is floating point zero. The consequence is a corollary: no function of probe accuracy alone can estimate steering efficacy. Across a swept family of systems, the correlation between probe accuracy and the certificate is zero point zero six.

## 06 - 03:02 - Proposition 7 - finite-sample leakage

That theorem concerns the ideal probe. Practitioners fit probes on finite data, and the estimate leaks. Even when the true alignment is exactly zero, an estimated probe picks up spurious alignment at a rate proportional to one over the square root of the sample size. Measured slope: one point zero zero six against a predicted one, with an r squared of zero point nine six. The important part is what is absent. The rate does not depend on model width. Working in a wider residual stream does not dilute the artefact. A probe fitted on a thousand examples carries about three percent of manufactured alignment before any real effect exists at all.

## 07 - 03:44 - Proposition 3 - effective rank

Why should the gap be large in practice, rather than merely possible? Because the effect geometry is extraordinarily low rank. The effect Gramian's participation rank, divided by the model width, tells you the fraction of steering power that a direction chosen without causal information retains. In GPT-2, that fraction is zero point one eight percent. Under two thousandths of a seven hundred and sixty eight dimensional residual stream carries any first-order behavioural effect. Almost every direction is inert. A read-side vector has to be lucky, and it usually is not.

## 08 - 04:20 - Experiment E3 - toxicity in GPT-2

So, to a real model. GPT-2, a toxicity behaviour, matched minimal pairs where only the hostile or friendly framing changes. The probe is excellent. From layer one onward it separates held-out prompts perfectly, an area under the curve of one point zero zero zero. Its alignment with the mechanism averages zero point zero five six. And the measured causal effect, at equal write norm, is six percent of what the certificate-optimal direction achieves. Now force a direction to be exactly orthogonal to that probe. It recovers ninety nine point seven percent of the control. The probe is not a weak handle on the mechanism. It is very nearly disjoint from it.

## 09 - 05:02 - The "gradients obviously steer better" objection

An objection worth taking seriously: of course the gradient steers better than a probe. That much is true by construction and carries no information. Three things here do. One: the probe direction is what practitioners actually deploy, and the size of its shortfall had not been measured. Two: the certificate rank-predicts measured effect across a pool of directions including random draws, mixtures, and eigenvectors, at a Spearman correlation of zero point nine two, while probe accuracy sits at minus zero point zero six. It is a genuine predictor over the space, not a restatement of one special case. Three: the best steering direction is itself a poor detector. The dissociation runs both ways.

## 10 - 05:48 - Experiment E6 - structural independence

The sharpest test is this. Hold the model and the prompts fixed. Vary the label definition, which moves only the read side. Vary the behaviour functional, which moves only the write side. Some pairs are matched by construction: the probe is trained on exactly the concept the behaviour measures. If the two geometries were the same object, matched pairs would stand out. Matched alignment: zero point zero three one. Unmatched alignment, for a completely unrelated concept: zero point zero three two. The ratio is zero point nine seven, and matched beats unmatched in six of twelve layers, which is chance. Knowing that a probe was trained on the right concept tells you nothing about whether it points anywhere causally useful.

## 11 - 06:32 - Experiment E5 - real data, three models

Does this survive real data and the vectors people actually ship? On RealToxicityPrompts, across three models spanning two architecture families, with probe accuracy at zero point seven eight rather than saturated at one, so the gap is not an artefact of a maxed-out probe. The Fisher probe recovers nine percent of available control. Difference of means, the vector the activation-steering literature deploys, recovers seventeen percent. A direction built orthogonal to both recovers ninety eight percent, while being nearly undetectable. The pattern holds on distil G P T two, and on Pythia seventy million, which is a different architecture family entirely.

## 12 - 07:14 - Experiment E2 - trained from scratch

Everything so far used models somebody else trained. So we trained twelve transformers from scratch, on a task with a causal content variable and a spurious surface marker, four confound levels by three seeds. The gap appears immediately. At zero confounding, alignment is zero point one five, and the probe recovers twelve percent of available control. Gradient descent produces this on its own. It is not inherited from pretraining, and it is not specific to G P T two.

## 13 - 07:44 - Experiment E4 - active intervention design

If the certificate predicts effect, it should be usable to choose which interventions to run. Given sixty four candidate directions and a budget of measurements, which ordering finds the strongest direction fastest? Ordering by the certificate reaches ninety percent of the best attainable effect after one point three measurements on average. Ordering by probe accuracy needs fifteen. Random selection needs twenty nine. The oracle needs one. That is a twenty two fold reduction in intervention budget, from a quantity that costs nothing extra, because it reuses adjoints the probe already required.

## 14 - 08:21 - What did not work

Three things did not go as predicted, and they are reported as such. First, a registered prediction failed, in two independent designs. Strengthening the confound between a surface marker and the label was supposed to widen the gap. In the pretrained model it did not move it at all. One might object that those representations were fixed long before our confound existed. So we controlled training itself, and swept the confound from a half to zero point nine eight across twelve trained models. Alignment moved by four thousandths, which is one fifth of the variation between random seeds. The manipulation demonstrably worked, because accuracy against the marker tracked the dial exactly. Two designs, one inheriting representations and one creating them, both refute the confound hypothesis. That makes the result stronger: the cause is low rank structure, not a flawed dataset. Cleaning your labels will not fix this. Second, one proposition is only approximately true in practice. Probe-orthogonal directions should classify at exactly one half; in GPT-2 they classify around zero point five six. The Gaussian premise does not hold exactly in a real model. Third, a collateral damage metric had to be thrown away. Prompt perplexity returned identically zero, because causal masking means a write at the final token cannot affect earlier predictions. A metric that cannot be non-zero is worse than no metric, because it reads as evidence of safety.

## 15 - 09:50 - Limitations

The limits, stated plainly. Everything ran on four CPU cores with no GPU. The largest model is GPT-2 small. The theory is scale-free; the measurements are not. There is one weak signal on how this scales. Across the four models measured, spanning a twelve fold range of width, the effect rank in absolute terms stays a small constant, between one point six and four point six, rather than growing with width. If that holds, the live fraction shrinks with scale and the gap widens rather than closing. Four points is far too few to fit a trend, and we say so, but it points the opposite way to the failure we consider most dangerous. The certificate is first order, and its validity radius is computed and reported per layer. Where the write magnitude exceeds it, the prediction is out of warranty and we say so. And dark means invisible to this probe family: linear, this concept, this data. It is not a claim of invisibility in principle.

## 16 - 10:48 - Consequence for monitoring

The safety consequence is direct. If a monitor is built from a toxicity probe, it is close to blind on the directions that most strongly drive toxic output. Probe-based monitoring is therefore not, by itself, evidence of control. That is not a worry. Given the decoupling theorem, it is a consequence.

## 17 - 11:08 - Next steps

What comes next, ordered by what would most change the conclusions. One: replicate the alignment and the rank fraction at one to seven billion parameters. If the live fraction grows with width, the argument loses its force. This is the biggest threat to external validity. Two: extend the certificate to nonlinear and sparse-autoencoder readouts, which tests whether dark directions survive a stronger detector. That is the real safety question. Three: test whether non-normal composition explains the small effect rank. The layer-to-layer Jacobian is a product of generically non-normal factors, and products like that concentrate their singular values. That is a plausible structural cause, and it is measurable with pseudospectral diagnostics we have not computed. Four: a second-order certificate, to extend validity past the first-order radius where most deployed steering actually operates.

## 18 - 12:04 - The claim in one sentence

The claim in one sentence. Reading a residual stream and writing to it are governed by different subspaces, that separation is measurable in advance from one backward pass, and probe accuracy carries essentially no information about causal control. Every number in this walkthrough is reproducible from the repository.
