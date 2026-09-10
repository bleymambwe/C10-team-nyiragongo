# Stakeholder Engagement

## Latent Probing for Toxicity Detection

**Team:** Latent Lens | **Programme:** AI Saturdays Lagos, Cohort 10

## Purpose

Toxicity is not only a modelling problem. Definitions, examples, thresholds, remedies, and acceptable error trade-offs depend on the people who experience a platform and those responsible for operating it. This plan describes how the project should involve stakeholders before any real deployment.

The activities below are a proposed engagement programme. They should not be read as evidence that interviews, workshops, or field trials have already taken place.

## Stakeholder group 1: users and affected communities

This group includes people targeted by online abuse, members of communities whose identity terms or dialects may be misclassified, ordinary platform users, advocacy groups, and people whose legitimate discussion includes quoted or reclaimed harmful language.

### Questions to explore

- What forms of harm are most urgent in the local setting?
- Which contextual distinctions are lost by a binary label?
- What consequences of false positives and false negatives are acceptable?
- What explanation and appeal process would feel meaningful?
- Which data uses, retention periods, and review practices are unacceptable?

### Engagement format

Use compensated small-group sessions and optional one-to-one interviews, with accessible participation and trauma-aware facilitation. Begin with scenarios rather than technical scores. Include borderline examples, disagreement, and model errors. Allow anonymous feedback and do not require participants to repeat personal experiences of abuse.

A representative community panel should review the working label policy, error slices, interface language, and proposed appeals process. Findings should be recorded as decisions, open questions, and dissenting views rather than reduced to a single consensus score.

## Stakeholder group 2: deploying organizations

This group includes moderators, trust-and-safety leads, community managers, product owners, ML engineers, legal and privacy staff, security teams, and moderator-wellbeing representatives.

### Questions to explore

- Where in the workflow would a model score appear, and who can act on it?
- What evidence must a reviewer see before making a decision?
- How will thresholds reflect policy, capacity, and asymmetric harm?
- Who owns monitoring, incident response, appeals, and rollback?
- What data and model changes trigger revalidation?

### Engagement format

Run a workflow-mapping workshop using realistic queue and escalation scenarios. Follow it with a red-team session covering evasion, prompt injection, data leakage, identity-term bias, drift, and automation bias. Establish a cross-functional oversight group with named owners for policy, engineering, privacy, wellbeing, and community accountability.

## Phased engagement and decision gates

1. **Discovery:** map harms, vocabulary, policies, and existing review practices.
2. **Co-design:** agree label guidance, human-review boundaries, explanations, and appeal routes.
3. **Offline evaluation:** test local data and publish disaggregated errors to the stakeholder groups.
4. **Shadow pilot:** generate scores without allowing them to affect users; compare them with reviewer decisions.
5. **Limited pilot:** use scores only for triage under explicit stop conditions and close monitoring.
6. **Review:** decide jointly whether to revise, expand, pause, or end the deployment.

Progression is conditional, not automatic. A leaderboard score alone cannot satisfy a decision gate.

## Information to share

Provide plain-language documentation of data sources, label mappings, intended use, known limitations, performance by domain, confidence and threshold behavior, retention, security, and the mechanism for contesting decisions. Explain that the model detects statistical patterns and does not determine intent or moral character.

## Feedback and accountability

Maintain a stakeholder decision log linking each material concern to an owner and response. Publish a short change note when labels, thresholds, models, or policies change. Track reviewer overrides, appeals, recurring false positives, missed harms, and differential impact. Offer independent escalation when the operating team cannot resolve a concern.

Stop or roll back the pilot when serious harm occurs, local performance falls below agreed bounds, an unmitigated subgroup disparity appears, data governance is breached, or the human-review capacity needed for safe use is unavailable.

## Team commitment

The project team will treat stakeholder knowledge as evidence that can change the dataset, evaluation, product design, or decision to deploy. Engagement is not a one-time validation exercise and must continue for as long as the system affects people.

## Team and contacts

**Mentor:** Moses (guymodscientist@gmail.com). **Team leader:** Blessings Mambwe (bleymambwe@gmail.com). **Team members:** Fafemi Adeola (adeola5678@gmail.com), Musonda Musunga (mamusonda@gmail.com), and research/challenge collaborator Hamna Kaleem (hamnanmah@gmail.com).

All four team contributors receive equal contribution credit. Hamna is also in another team; her Team Latent Lens role covered shared research and Cohort Challenge work and does not imply exclusive membership.
