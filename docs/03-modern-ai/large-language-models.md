# Large language models

A large language model (LLM) is a [transformer](../02-deep-learning/transformers.md) trained to predict the next token. That one objective, applied at scale, produces systems that translate, summarize, write code, and reason through problems.

## Pretraining

Given a token sequence $x_1, \dots, x_T$, the model factorizes its probability autoregressively and minimizes the negative log-likelihood:

```math
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
```

Pretraining corpora mix web text, books, code, and scientific writing, usually filtered and deduplicated. Because predicting text well requires modelling grammar, facts, and the reasoning that produced the text, the objective forces the network to learn a great deal about the world. As discussed in [Probability and information](../00-foundations/probability-and-information.md), this is equivalent to learning to **compress** the data.

## In-context learning

**Brown et al. (2020)** showed that GPT-3 (175B parameters) could perform new tasks from a few examples placed in the prompt, without any weight update. Why this works is an active research area; mechanistic studies have connected it to attention circuits such as **induction heads** (Olsson et al. 2022).

## Prompting for reasoning

- **Chain-of-thought prompting** (Wei et al. 2022): showing worked step-by-step examples improves multi-step reasoning in sufficiently large models.
- **Zero-shot chain of thought** (Kojima et al. 2022): simply asking the model to reason step by step helps.
- **Self-consistency** (Wang et al. 2022): sample several reasoning paths and take the majority answer.
- **Tree of Thoughts** (Yao et al. 2023): search over branching reasoning steps with self-evaluation.

## From base model to assistant

A base model continues text; it does not follow instructions reliably. Post-training changes that.

### Supervised fine-tuning (SFT)

Train on curated (instruction, response) pairs so the model learns the assistant format.

### Reinforcement learning from human feedback (RLHF)

Building on Christiano et al. (2017), **Ouyang et al. (2022)** (InstructGPT) used three steps:

1. Collect human rankings of model outputs.
2. Train a **reward model** $r_\phi(x, y)$ to predict those preferences.
3. Optimize the policy with PPO to maximize reward, with a KL penalty keeping it close to the SFT model $\pi_{\text{ref}}$:

```math
\max_{\pi_\theta} \; \mathbb{E}_{x, y \sim \pi_\theta}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \,\Vert\, \pi_{\text{ref}}(\cdot \mid x)\big)
```

Human labellers preferred outputs of the 1.3B-parameter InstructGPT model to those of the 175B GPT-3 base model, showing how much alignment-style post-training matters.

### Direct preference optimization (DPO)

**Rafailov et al. (2023)** showed that the optimal policy of the KL-regularized objective above can be written in closed form in terms of the reward, so preferences can be fit directly with a classification-style loss, no separate reward model or RL loop required:

```math
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
```

where $y_w$ and $y_l$ are the preferred and dispreferred responses.

### Constitutional AI and AI feedback

**Bai et al. (2022)** replaced much of the human harmlessness labelling with feedback from the model itself, guided by a written set of principles (a "constitution").

## Reasoning models

Since 2024, a prominent direction has been training models with reinforcement learning on tasks with checkable answers (math, code) so they produce long chains of reasoning before answering. **Lightman et al. (2023)** showed that rewarding each correct reasoning *step* (process supervision) can outperform rewarding only the final answer. **DeepSeek-R1** (DeepSeek-AI 2025) reported that large-scale RL with rule-based rewards elicits extended reasoning behaviours, including self-verification.

## Efficient adaptation

Full fine-tuning of billions of parameters is expensive. **LoRA** (Hu et al. 2021) freezes the pretrained weights and learns a low-rank update $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$. **QLoRA** (Dettmers et al. 2023) does this on top of a 4-bit quantized base model, allowing fine-tuning of large models on a single GPU.

## Known failure modes

| Failure | Description |
|---|---|
| **Hallucination** | Fluent but unsupported or false statements (survey: Ji et al. 2023) |
| **Sycophancy** | Agreeing with the user rather than the truth |
| **Prompt injection** | Instructions hidden in retrieved content override the intended task |
| **Jailbreaks** | Inputs that bypass safety training; Zou et al. (2023) found transferable adversarial suffixes |
| **Knowledge cutoff** | No knowledge of events after the training data was collected |

Retrieval, tool use, and verification (next page) mitigate several of these.

## Key takeaways

- Next-token prediction at scale yields broad capabilities and in-context learning.
- Post-training (SFT, RLHF, DPO) turns a predictor into an assistant.
- Reasoning can be improved by prompting, by spending more inference compute, and by RL on verifiable tasks.
