---
# Machine-readable fields for downstream agents / dashboards.
stage: early-research   # exploration | early-research | applied-research | productization | shipped
research-area: Models & Foundations
last-updated: 2026-05-12
---

# SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants?

**Team:** Microsoft Research — Deep Learning Group
**Contributors:** Yao Dou, Michel Galley, Baolin Peng, Chris Kedzie, Weixin Cai, Alan Ritter, Chris Quirk, Wei Xu, Jianfeng Gao

---

## What it is

SimulatorArena is a benchmark and methodology for measuring whether LLM-based **user simulators** can serve as reliable, low-cost stand-ins for real users when evaluating AI assistants over multi-turn conversations. It pairs **909 human–LLM conversations** (450 math tutoring + 459 document creation, collected from 107 Amazon Mechanical Turk workers across 9 assistant LLMs) with annotated per-turn and end-outcome ratings, and uses them as ground truth to score simulators along two axes: how closely simulator messages resemble real users, and how well simulator-driven ratings align with human ratings of the same assistants.

🔗 [SimulatorArena overview](https://www.simulatorarena.ai/) · [Paper (arXiv 2510.05444)](https://arxiv.org/abs/2510.05444)

## Core idea

**Claim:** if user simulators are going to drive training and evaluation of multi-turn AI assistants, the field needs a principled way to ask "*is this simulator a good proxy for a real user?*" — and the answer depends as much on **what kind of user** the simulator is asked to play as on the base model behind it. Existing multi-turn benchmarks either use fixed scripted users (no adaptivity) or zero-shot-prompted LLMs (no validation against humans), and silently assume the simulator's verdict transfers to real users. SimulatorArena's contribution is threefold: (1) a **human-anchored evaluation protocol** that scores simulators by Spearman/Pearson/Kendall correlation with human ratings at the instance, intermediate (model × difficulty / doc-type), and system levels, plus a Likert + Turing-Test similarity score on simulator messages themselves; (2) a **user-profile-based simulator** that conditions on three components — *inherent knowledge*, *writing style*, and *interaction style* — covering 25+ fine-grained attributes such as message length, grammar usage, and feedback style; and (3) a public benchmark over five frontier LLMs as simulators (GPT-4o, Gemini 2.0 Flash, Claude 3.7, …) and 18 LLMs as assistants. The combination is what makes the framework load-bearing: the human-anchored data is what lets the profile-conditioning hypothesis be tested at all, and the multi-level correlation analysis is what separates "looks like a user" from "ranks assistants like a user does" — two properties that turn out to come apart in practice.

## Why it matters

**To the field:** Establishes a reproducible standard for evaluating user simulators — moving simulator design from anecdotal prompting to a measurable, comparable property. Demonstrates empirically that profile-conditioned simulators raise Spearman correlation from **0.61 → 0.77** on math tutoring and **0.55 → 0.70** on document creation versus zero-shot CoT, a **~26% gain at roughly 3% of the cost** of recruiting human annotators — which makes simulator-based evaluation a credible alternative to AMT studies for fast iteration on multi-turn assistants.

**Future directions:** Opens research questions in (1) *cross-session and longitudinal simulation* — current evaluation is single-session; how do simulators behave when a user has memory and history?, (2) *distilled, efficient simulators* — can the released conversation data train smaller open-source simulators that match GPT-4o-quality profile fidelity at a fraction of the cost?, (3) *turn-level diagnostics* — moving from conversation-level correlation to fine-grained per-turn attribution of where simulators diverge from humans, and (4) *using SimulatorArena conversations as training data* — beyond evaluation, the annotated corpus is a natural target for alignment, personalization, and reward modeling.

## Current status

**Headline:** Profile-conditioned simulators reach **Spearman ρ = 0.77** (math tutoring) and **ρ = 0.74** (document creation) with human ratings of AI assistants — a 26% lift over zero-shot CoT prompting at ~3% the cost of human evaluation.

- **Benchmark released:** 909 annotated multi-turn conversations across math tutoring and document creation, 107 AMT annotators, 9 assistant LLMs, ~$10,000 annotation cost.
- **Simulator findings:** GPT-4o + interaction-style profile is the strongest simulator for math tutoring (ρ=0.77); Gemini 2.0 Flash + full profile is strongest for document creation (ρ=0.74). All four profile-vs-CoT comparisons statistically significant (three at p<0.01).
- **Assistant leaderboard (18 LLMs):** GPT-5 leads on both tasks (math interaction 8.89, accuracy 90%; doc interaction 9.08, doc rating 8.96), followed by Claude 4.1 Opus and Claude 3.7 Sonnet; Phi-4 leads among open-source models.
- **Caveat documented:** simulators consistently struggle to reproduce specific fine-grained behaviors — math-notation avoidance, intentional grammar mistakes, sentence fragments — and adding more profile attributes reduces faithfulness to any individual attribute.
- **Release:** Annotated conversations, profile templates, simulator and rater prompts, and evaluation code at [simulatorarena.ai](https://www.simulatorarena.ai/).

## Related landscape

- [Wu et al., *CollabLLM: From Passive Responders to Active Collaborators* (arXiv 2502.00640, ICML 2025)](https://arxiv.org/abs/2502.00640) — Builds a training framework on top of an LLM user simulator; SimulatorArena provides the methodology to ask how reliably such simulator-driven training signals transfer to real users.
- [Laban et al., *LLMs Get Lost In Multi-Turn Conversation* (arXiv 2505.06120, May 2025)](https://arxiv.org/abs/2505.06120) — Documents the 39% multi-turn performance gap that motivates the need for trustworthy multi-turn evaluation, which is exactly what SimulatorArena aims to make scalable.
- [Naous, Laban, Xu, Neville — *UserLM-8b* (Hugging Face, Oct 2025)](https://huggingface.co/microsoft/UserLM-8b) — A more advanced, purpose-trained user simulator focused on realism of user turns; SimulatorArena offers the benchmark on which such next-generation simulators can be quantitatively compared against prompted-LLM baselines.

## Publications & links

- [SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants? — arXiv, 2025](https://arxiv.org/abs/2510.05444)
- [Project page: simulatorarena.ai](https://www.simulatorarena.ai/)
