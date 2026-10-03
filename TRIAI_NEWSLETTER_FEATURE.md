<!--
EDITORIAL NOTE (delete before sending to triAI, not for publication):

This draft was built only from material that is safe to publish. Left out on
purpose, because it is confidential and would cause real harm if it appeared
in a public newsletter:
  - The UK Global Talent (Tech Nation) readiness assessment: percentage
    scores, named potential referees and the specific reasons each is or
    isn't eligible, and the criterion-by-criterion strategy. That document
    exists to plan a private application, not to be published. Tech Nation's
    own guidance treats AI-assisted personal statements and letters as a
    fraud risk, so publishing AI-drafted material tied to the case would add
    real risk to the application itself.
  - EVE, not mentioned at all, at Blessings's request. No financials, no
    users, no product description.
  - MTN as an employer, removed. Standard Bank is explicitly sanctioned for
    public credibility use elsewhere in this workspace. MTN was not, and the
    private notes indicate any MTN figures still need written employer
    clearance before public use. Only Standard Bank is named below.
  - Any claim not already independently true. For example, this draft does
    not say PRISM's code is public, because the sources indicate the
    repository isn't live yet.

Before sending: confirm you're comfortable with the Speak Life AI mention
(kept minimal, name only, no description or numbers). Say the word and it
comes out too.
-->

# Meet the finalist: Blessings Mambwe is teaching frozen language models to reveal what they already know about toxicity

When TRI AI narrowed 69+ submissions down to 9 finalists, one team's entry stood out for the question it refused to ask. Instead of building a bigger toxicity classifier, they asked whether a language model already contains the answer, if you know where to look.

That question is the core of Latent Probing for Toxicity Detection, the Cohort 10 submission from Team Latent Lens at AI Saturdays Lagos: Blessings Mambwe, Fafemi Adeola, Musonda Musunga, and Hamna Kaleem, mentored by Moses.

## The idea

Modern language models are usually treated as black boxes: you either fine-tune the whole thing for a task, or you don't. Team Latent Lens tested a third option. They froze every one of Gemma-2-2B's 2.6 billion parameters, with no fine-tuning at all, and asked whether a single, simple, auditable linear classifier could read toxicity straight out of the model's internal hidden states at one fixed layer.

It works. Their probe, trained on a harmonised corpus of 108,468 examples pulled from 13 public sources, hit 92.35% accuracy on the competition's held-out test set, landing the team among the top finishers on TRI AI's CodaBench leaderboard.

That result matters beyond the leaderboard. If a model already encodes a concept in a form a linear readout can decode, you don't need to bolt on an expensive, opaque, separately trained moderation system. A small probe you can audit and retrain is enough, and it runs on infrastructure that already exists.

## Why it's not "just" a classifier

The team is explicit that this is a research prototype, not a moderation product. A binary label cannot capture intent, quotation, counterspeech, or cultural context, and they have documented exactly where the probe should and should not be trusted: human review, subgroup audits, and drift monitoring, with no autonomous enforcement. That caution is part of what the submission is actually claiming. They measured, honestly, how much signal a frozen model gives away for free, and where that signal breaks down. That is the real result, more than any single accuracy number.

> "We didn't set out to build the best classifier. We started with an unconventional method: looking at how clinical psychology, neuroscience, and behavioural science think about latent probing, long before we touched a line of model code. That's where the real question came from, not 'how do we score well,' but 'is this signal actually there, the way those fields would recognise it.' Once we had that grounding, the accuracy number stopped being the point. It became confirmation that we'd asked the right question in the first place."

## The person behind it

Blessings' path into this work runs through years of production engineering, not just research. He's spent the last several years building production AI systems at Standard Bank, the kind of environment where a model's failure mode is somebody's real transaction, not a benchmark number. Before that, his training was in mechanical engineering; the move into AI came later, by choice rather than by degree.

Outside the day job, that same instinct drives his independent research: PRISM, his work on permutation-based optimisation, now written up as a preprint, and his involvement with ML Collective, the volunteer-run open research community where he tests ideas with independent researchers around the world. He has also built and shipped Speak Life AI on his own time. Across all of it, the pattern holds. Don't reach for the heaviest possible tool first. Find out what a cheaper, more transparent method can already do, and only add complexity once the evidence earns it.

> "Banking systems don't forgive sloppy thinking. A mistake shows up as someone's money. That's the discipline I bring into the research side: measure honestly, and say plainly where the method breaks."

## Where this goes next

For Blessings, the cohort project turned into something bigger than the competition itself. It sparked a research question he's continuing to pursue independently, beyond Cohort 10: whether safety-relevant concepts in language models, toxicity now, other failure modes later, are better understood as structured, learnable trajectories through a model's internal representations, rather than single fixed directions. The toxicity probe was the first concrete test of that idea, not the end of it. That's early-stage research, the kind that takes years rather than a hackathon weekend to answer properly, and the Cohort 10 result is one honestly measured data point on that longer path.

---

**Quick facts for the newsletter box**

- **Project:** Latent Probing for Toxicity Detection
- **Team:** Latent Lens. Blessings Mambwe (lead), Fafemi Adeola, Musonda Musunga, Hamna Kaleem. Mentor: Moses.
- **Programme:** AI Saturdays Lagos, Cohort 10
- **Result:** 92.35% accuracy on TRI AI's CodaBench held-out test set (89.76% on the development set)
- **Method:** Frozen google/gemma-2-2b, layer-14 hidden states, masked mean pooling, standardised shrinkage-LDA linear probe. No fine-tuning of the base model.
- **Built on:** 108,468 labelled examples harmonised from 13 public sources, validated with leave-one-source-out testing
- **Also building:** PRISM (permutation optimisation research, preprint), Speak Life AI
- **Community:** ML Collective

---

*This is a research prototype for benchmarking and moderation support with human review. It is not a production moderation system, and not the sole basis for any sanction or access decision.*
