# Impact Statement Card

## Latent Probing for Toxicity Detection

**Team:** Nyiragongo | **Programme:** AI Saturdays Lagos, Cohort 10

## Intended positive impact

The project tests a resource-efficient way to identify toxic or hostile text from a frozen language model's hidden state. A compact probe may help research teams and platform operators prioritize review queues, study model representations, compare datasets, and prototype safety monitoring without fine-tuning an entire language model.

For users and communities, the hoped-for benefit is faster attention to harmful content and more consistent moderation support. For moderators, triage can reduce queue pressure. For researchers, the open pipeline provides a reproducible baseline and exposes domain-transfer performance rather than presenting one leaderboard number as universal evidence.

## Evidence available

The final CodaBench system achieved 89.76% accuracy on the development split and 92.35% on the testing split. A harder leave-one-source-out evaluation achieved pooled accuracy 0.795946 and AUROC 0.832204. The gap is evidence that domain and source composition materially affect performance.

These results support a claim of promising classification performance on the evaluated task. They do not establish that the system is fair across all groups, understands intent, generalizes to every platform, or is safe for autonomous enforcement.

## People who may benefit

- People exposed to direct threats, harassment, or sustained abuse
- Human moderators and trust-and-safety teams managing high-volume queues
- Community administrators seeking consistent review support
- Researchers studying representation, transfer, and toxicity benchmarks
- Developers testing safety layers for conversational systems

## People who may be harmed

- Speakers whose dialect or community vocabulary is poorly represented
- People discussing identity, discrimination, or abuse in educational or counterspeech contexts
- Users affected by an incorrect sanction or suppressed message
- Moderators exposed to toxic data without adequate wellbeing safeguards
- Individuals whose personal information appears in source data

False positives can silence legitimate participation and disproportionately burden marginalized speakers. False negatives can leave targets exposed. Automation bias can cause reviewers to accept a score without examining context. Attackers may evade a static detector through obfuscation or coded language.

## Mitigations

- Keep a trained human in consequential decisions.
- Use calibrated scores and policy-specific thresholds, not a universal cutoff.
- Audit identity terms, dialects, quotation, counterspeech, threats, and implicit hostility separately.
- Give affected users explanations and a practical appeal route.
- Monitor drift, evasion patterns, and disagreement between reviewers and the model.
- Limit access to harmful raw data and protect moderator wellbeing.
- Log model, data, threshold, and policy versions for accountability.
- Pause or roll back use when subgroup harm or domain drift exceeds agreed limits.

## Environmental and operational impact

Freezing Gemma and fitting only a small probe reduces training demand compared with full model fine-tuning. Activation extraction still requires compute and storage. Teams should batch efficiently, reuse lawful cached activations, record hardware and energy assumptions, and avoid repeated extraction without an experimental reason.

Operationally, the detector adds monitoring, security, review, and maintenance obligations. A cheap model is not a cheap governance process.

## Red lines

Do not use this prototype as the sole basis for account termination, employment or education decisions, law-enforcement action, immigration screening, or other high-impact determinations. Do not infer a person's character or protected attributes from a toxicity score. Do not present the binary output as an objective judgment independent of policy and context.

## Measurement and review plan

Track false-positive and false-negative rates, per-source performance, subgroup and identity-term slices, reviewer overrides, appeals, latency, coverage, and drift. Review metrics with affected users and deploying organizations. Any pilot should define owners, escalation routes, stop conditions, and a date for reassessment before launch.

Stakeholder engagement described in the accompanying plan is proposed governance work; it has not been represented as completed field research.

## Team and contacts

**Mentor:** Moses (guymodscientist@gmail.com). **Team leader:** Blessings Mambwe (bleymambwe@gmail.com). **Team members:** Fafemi Adeola (adeola5678@gmail.com), Musonda Musunga (mamusonda@gmail.com), and research/challenge collaborator Hamna Kaleem (hamnanmah@gmail.com).

All four team contributors receive equal contribution credit. Hamna is also in another team; her role with team Nyiragongo covered shared research and Cohort Challenge work and does not imply exclusive membership.
