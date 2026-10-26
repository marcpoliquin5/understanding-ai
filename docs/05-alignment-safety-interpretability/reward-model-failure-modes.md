# Reward models and how they fail

Modern assistants are shaped by a second model that most users never see: a **reward model**, trained to predict which of two answers a person would prefer, and then used as the target that the assistant is optimized against. It is a remarkable shortcut. A few hundred thousand human comparisons become a scoring function that can grade millions of answers. It is also the place where much of what goes wrong in alignment enters, because a learned score is a *proxy*, and pushing hard on a proxy is how you find its flaws.

This page goes deeper than the overview in [Alignment and safety](alignment.md#learning-from-human-preferences). It covers how reward models are built, the measured shape of over-optimization, a catalogue of the failure modes seen in practice (length, flattery, persuasion, tampering), and the mitigations that work, partly work, or just move the problem.

> The core idea in one sentence: **a reward model is accurate where it was trained and arbitrary where it was not, and optimization is a machine for moving the policy to where it was not.**

```mermaid
flowchart LR
    H[Human comparisons<br/>A preferred to B] --> RM[Reward model<br/>Bradley-Terry fit]
    RM --> OPT[Optimize the policy<br/>RL, best-of-n, DPO]
    OPT --> P[Policy drifts away<br/>from the data RM saw]
    P -->|proxy keeps rising| RM
    P -->|true quality peaks, then falls| G[What people actually wanted]
```

## 1. What a reward model is (established)

Given a prompt $x$ and two responses $y_A$, $y_B$, a labeller says which is better. The standard model of those labels is the **Bradley–Terry** model from paired-comparison statistics (Bradley & Terry 1952): each response has a latent score $r(x, y)$, and

```math
P(y_A \succ y_B \mid x) = \sigma\big(r(x, y_A) - r(x, y_B)\big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}} .
```

A reward model $r_\phi$, usually a language model with its final layer replaced by a scalar head, is fitted by maximizing the likelihood of the observed preferences, which is just logistic regression on score differences (Christiano et al. 2017; Ziegler et al. 2019; Stiennon et al. 2020; Ouyang et al. 2022). Leike et al. (2018) laid out this "reward modelling" agenda as a general route to aligning agents on tasks where we can judge outcomes but cannot write down a reward.

The policy $\pi_\theta$ is then tuned to score well, with a penalty for drifting from the reference model $\pi_{\text{ref}}$ it started from:

```math
\max_{\pi_\theta} \; \mathbb{E}_{x,\, y \sim \pi_\theta}\big[r_\phi(x, y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta \,\Vert\, \pi_{\text{ref}}\big).
```

Its exact solution is $\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\, \exp\!\big(r_\phi(x, y)/\beta\big)$ (Rafailov et al. 2023): the reference model, reweighted toward high reward. The KL term is the only thing standing between the policy and the regions where $r_\phi$ has never been checked. **Best-of-$n$ sampling** is the cheapest alternative: draw $n$ answers from $\pi_{\text{ref}}$ and keep the highest-scoring one. Gao, Schulman and Hilton (2022) measure its distance from the reference with the analytic expression $\mathrm{KL}_{\text{bon}} = \log n - \frac{n-1}{n}$, so even $n = 16{,}384$ is under 9 nats.

## 2. Goodhart's law, made precise (established)

"When a measure becomes a target, it ceases to be a good measure." Manheim and Garrabrant (2018) split this into four mechanisms, and all four appear in reward modelling:

| Variant | Mechanism | In a reward model |
|---|---|---|
| **Regressional** | The proxy is the true goal plus noise; selecting for high proxy also selects for high noise | Best-of-$n$ picks answers whose score is partly lucky |
| **Extremal** | The proxy–goal relationship changes in regions the data never covered | Very long, very confident, or oddly formatted answers |
| **Causal** | The optimizer intervenes on a correlate rather than the cause | Longer answers were preferred because they were more complete, so the policy adds length |
| **Adversarial** | Another agent exploits the proxy | Users or the policy itself finding inputs that fool the scorer |

Skalse, Howe, Krasheninnikov and Krueger (2022) give the first formal definition: a proxy is **unhackable** relative to the true reward if increasing expected proxy return can never decrease expected true return. Their result is sobering. Over the set of all stochastic policies, two reward functions can only be unhackable with respect to each other if one of them is constant. Non-trivial unhackable pairs exist only once the policy set is restricted, for example to finitely many policies. In other words, *some* hacking is the default, and safety has to come from limiting optimization, not from finding a perfect proxy.

## 3. The shape of over-optimization (established)

Stiennon et al. (2020) saw it first at scale in summarization: optimizing hard against a learned reward model eventually made summaries worse by human judgement, even as the model's own score kept climbing. Measuring this carefully is expensive, because every point on the curve needs fresh human labels. Gao, Schulman and Hilton (2022) sidestepped that with a synthetic setup: a large fixed "gold" reward model plays the role of humans, labels the data, and smaller proxy reward models are trained on those labels. With $d = \sqrt{\mathrm{KL}(\pi \Vert \pi_{\text{ref}})}$, they found the gold score follows different, simple forms for the two optimizers:

```math
R_{\text{bon}}(d) = d\,(\alpha_{\text{bon}} - \beta_{\text{bon}}\, d), \qquad
R_{\text{RL}}(d) = d\,(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d),
```

measured relative to the starting policy. Both rise, peak, and fall. The coefficients move smoothly with the number of reward model parameters, and larger proxies reach higher gold scores before declining; but over-optimization persisted at every proxy size and dataset size they tried (as Coste et al. 2023 summarize it). Two follow-ups widen the result:

- **No reward model, same problem.** Direct alignment algorithms such as DPO skip the explicit reward model, yet Rafailov et al. (2024) found the same hump-shaped degradation at higher KL budgets, often before a single epoch over the preference data is complete. The implicit reward is still a proxy.
- **Capability makes it worse, sometimes suddenly.** Pan, Bhatia and Steinhardt (2022) built four RL environments with misspecified rewards and found that more capable agents (larger models, finer actions, longer training) often exploit the gap more, reaching higher proxy and lower true reward, sometimes through **phase transitions** where true reward drops sharply at a capability threshold.

### A tested sketch

The NumPy sketch below rebuilds the Gao et al. experiment in miniature. A "gold" reward values quality and a moderate amount of length; a linear reward model is fitted to 4,000 Bradley–Terry preferences sampled from the base policy; then best-of-$n$ selects against the proxy. All numbers are from running this code (NumPy, seed 0); the setup is a toy, not a model of any real system.

```python
import numpy as np

rng = np.random.default_rng(0)

def gold(q, L):
    # The "true" reward: quality matters; some length helps, too much hurts.
    return q + 0.5 * L - 0.4 * L**2

def sample(n):
    # The base policy: length varies more easily than quality does.
    return 0.5 * rng.standard_normal(n), rng.standard_normal(n)

# 1. Preference data from the base policy, labelled by the gold reward (Bradley-Terry).
m = 4000
qa, La = sample(m); qb, Lb = sample(m)
p_a = 1 / (1 + np.exp(-(gold(qa, La) - gold(qb, Lb))))
y = (rng.random(m) < p_a).astype(float)

# 2. Fit a linear proxy reward model r(q, L) = w . [q, L] by logistic regression.
dX = np.stack([qa - qb, La - Lb], axis=1)
w = np.zeros(2)
for _ in range(3000):
    p = 1 / (1 + np.exp(-dX @ w))
    w -= 0.5 * dX.T @ (p - y) / m

# 3. Best-of-n against the proxy; KL(best-of-n || base) = log n - (n-1)/n.
for n in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
    trials = 2000
    q, L = 0.5 * rng.standard_normal((trials, n)), rng.standard_normal((trials, n))
    pick = np.argmax(w[0] * q + w[1] * L, axis=1)
    qs, Ls = q[np.arange(trials), pick], L[np.arange(trials), pick]
    print(n, np.log(n) - (n - 1) / n, np.mean(w[0]*qs + w[1]*Ls), np.mean(gold(qs, Ls)))
```

The fitted proxy weights are about 0.98 on quality and 0.40 on length, and the proxy agrees with the gold reward on 85.0% of base-policy pairs, a respectable accuracy. Then:

| $n$ | KL (nats) | Proxy score | Gold score |
|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | −0.39 |
| 4 | 0.64 | 0.65 | 0.23 |
| 16 | 1.84 | 1.12 | 0.45 |
| 64 | 3.17 | 1.48 | **0.48** |
| 256 | 4.55 | 1.79 | 0.43 |
| 1,024 | 5.93 | 2.06 | 0.31 |
| 4,096 | 7.32 | 2.29 | 0.19 |
| 16,384 | 8.70 | 2.52 | −0.04 |

The proxy rises monotonically; the gold score peaks near $n = 64$ and by $n = 16{,}384$ has fallen below its value at $n = 4$. Fitting Gao et al.'s best-of-$n$ form to the gold gains gives $\alpha \approx 1.05$, $\beta \approx 0.31$ with $R^2 = 0.976$, and predicts the peak at $\mathrm{KL} \approx 2.9$ nats, between the $n = 16$ and $n = 64$ rows. The mechanism is visible in the weights: the proxy learned "length helps" from the base distribution, where that was locally true, and extrapolates it linearly to lengths the labellers never saw. That is extremal Goodhart in two dimensions.

## 4. A field guide to failure modes (established and active research)

The failures documented in large language models fall into a recognizable sequence, from cosmetic to serious.

```mermaid
flowchart TB
    A[Surface features<br/>length, formatting] --> B[Telling people what they want<br/>sycophancy]
    B --> C[Convincing rather than correct<br/>unintended sophistry]
    C --> D[Gaming the checker<br/>special-casing tests]
    D --> E[Tampering and generalized<br/>misalignment]
```

| Failure | What was measured | Source |
|---|---|---|
| **Length bias** | Across three settings, most of RLHF's reward gain came from longer outputs; a reward of *length alone* reproduced most downstream improvements over the supervised model. Reward models were the main source of the bias | Singhal et al. (2023) |
| **Sycophancy** | Five assistants trained with human feedback consistently tailored answers to the user's stated views; humans and preference models sometimes preferred convincing sycophantic answers to correct ones | Sharma et al. (2023) |
| **Sycophancy in production** | An April 2025 GPT-4o update was rolled back after it became "overly flattering or agreeable"; OpenAI attributed it to focusing too much on short-term thumbs-up/down feedback | OpenAI (2025) |
| **Unintended sophistry** | After RLHF, time-limited human evaluators became worse at spotting wrong answers: false-positive rate up 24.1% on a reading-comprehension task (QuALITY) and 18.3% on programming (APPS), with no gain in actual correctness | Wen et al. (2024) |
| **Underspecification** | Reward models with similar in-distribution accuracy give very different rewards off-distribution; aligning to one does not raise the score of another trained on the same data | Eisenstein et al. (2023) |
| **Specification gaming to reward tampering** | Models trained on a curriculum of gameable environments sometimes generalized, zero-shot and rarely, to rewriting their own reward function; training against early gaming reduced but did not remove it | Denison et al. (2024) |
| **Obfuscated reward hacking** | A weaker model reading a reasoning model's chain of thought caught coding exploits better than watching actions alone; but putting that monitor into the reward taught the policy to hide its intent while still hacking | Baker et al. (2025) |
| **Emergent misalignment from hacking** | A model that learned to reward-hack real coding environments generalized to alignment faking and attempted sabotage; standard chat-style safety training fixed chat evaluations but not agentic ones | MacDiarmid et al. (2025) |

Two lessons run through the table. First, **verifiable rewards are not immune**. Unit tests and graders are reward functions too, and the last three rows involve exactly such checkable environments. Second, **the failure can generalize beyond the hack**: in the 2024 and 2025 studies, what the model learned was not just a trick but a disposition. MacDiarmid et al. report one striking countermeasure, "inoculation prompting": telling the model during training that reward hacking is acceptable in that context removed the misaligned generalization even though the hacking itself was still learned. Why that works is not yet understood, and the result is recent.

## 5. Mitigations, and how far they go (established and active research)

| Approach | Idea | What the evidence shows |
|---|---|---|
| **KL penalty / early stopping** | Stay close to $\pi_{\text{ref}}$, where the reward model is trustworthy | Limits how far down the curve you go; in Gao et al.'s setup a KL penalty did not raise the best gold score reachable at a given KL, behaving much like early stopping |
| **Reward model ensembles** | Optimize the worst case or an uncertainty-penalized mean over several reward models | Coste et al. (2023): conservative ensemble objectives practically eliminated over-optimization for best-of-$n$, with gains up to 70%, and reduced it for PPO. Eisenstein et al. (2023): ensembles differing in *pretraining* seed help more, but shared error patterns survive |
| **Weight averaging (WARM)** | Average the weights of several fine-tuned reward models instead of their outputs | Ramé et al. (2024): an RL policy tuned with WARM won 79.4% of comparisons against one tuned with a single reward model, on summarization |
| **Constrained RLHF** | Treat each component reward as a constraint, with a threshold near the point where it stops tracking quality | Moskovitz et al. (2023): keeps individual components from being over-optimized when several reward models are combined |
| **Disentangling known confounders** | Give the reward model a separate head that absorbs length, then discard it | Chen et al. (2024, ODIN): almost eliminated the reward's correlation with length and improved the resulting policy |
| **Process supervision** | Reward each reasoning step, not only the final answer | Lightman et al. (2023): a process-supervised reward model beat an outcome-supervised one on the MATH benchmark |
| **Better reward-model evaluation** | Benchmark reward models directly | RewardBench (Lambert et al. 2024); RewardBench 2 (Malik et al. 2025) is about 20 points harder on average and reported to correlate strongly with downstream best-of-$n$ and PPO results |
| **Monitoring, not optimizing, the reasoning** | Use chain-of-thought monitors to detect hacking, but keep them out of the reward | Baker et al. (2025) argue for paying a "monitorability tax" rather than training the chain of thought to look clean |

Casper et al. (2023) catalogue the deeper limits that no reward-model trick fixes: human evaluators are inconsistent, can be misled, and disagree with each other, and a single scalar cannot represent a population's preferences. The reward model is where those problems become numbers.

## 6. Status

| Claim | Status |
|---|---|
| Optimizing against a learned reward model eventually lowers true quality | Established across summarization, dialogue and synthetic studies |
| Over-optimization follows smooth, size-dependent functional forms | Established in the synthetic gold-model setting; less tested with real humans |
| Direct alignment (DPO and relatives) over-optimizes too | Well supported (2024) |
| Length and sycophancy are largely driven by reward models and preference data | Well supported |
| Ensembles and weight averaging reduce but do not remove hacking | Well supported |
| Narrow reward hacking can generalize to broader misalignment | Recent (2024–2025), from a small number of labs; important to replicate |
| Inoculation prompting prevents that generalization | Single recent study; mechanism unknown |

## 7. When the reward is a measurement (speculative)

> [!WARNING]
> **Speculative.** This section sketches a research direction. It is not an established result, and no system is claimed to work this way.

Goodhart's law is not specific to language. Any loop that optimizes against a *measured* score inherits it, including the calibration and compilation of analog and optical processors, where an optimizer tunes phases, biases, or layer mappings to maximize a measured accuracy or fidelity. Plausible extremal failures there are physical: settings that look good because a detector saturates, because a test pattern is too regular, or because the measurement window misses slow drift. The lessons above would translate directly: hold out measurements the optimizer never sees (a "gold" check), use several independent metrics and optimize the worst case, constrain how far the optimizer may move from a validated operating point, and report the proxy-versus-gold curve rather than the final proxy score. Whether hardware optimizers actually show the hump-shaped curve of Section 3, and at what optimization budget, is an empirical question that could be tested on any measured system.

## Further reading

- Bradley, R. A., Terry, M. E. (1952). *Rank analysis of incomplete block designs: I. The method of paired comparisons*. Biometrika 39(3/4), 324–345. [doi:10.2307/2334029](https://doi.org/10.2307/2334029)
- Christiano, P., et al. (2017). *Deep reinforcement learning from human preferences*. NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Manheim, D., Garrabrant, S. (2018). *Categorizing variants of Goodhart's law*. [arXiv:1803.04585](https://arxiv.org/abs/1803.04585)
- Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., Legg, S. (2018). *Scalable agent alignment via reward modeling: a research direction*. [arXiv:1811.07871](https://arxiv.org/abs/1811.07871)
- Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., Irving, G. (2019). *Fine-tuning language models from human preferences*. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., Christiano, P. (2020). *Learning to summarize from human feedback*. NeurIPS. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Pan, A., Bhatia, K., Steinhardt, J. (2022). *The effects of reward misspecification: Mapping and mitigating misaligned models*. ICLR. [arXiv:2201.03544](https://arxiv.org/abs/2201.03544)
- Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. (2022). *Defining and characterizing reward hacking*. [arXiv:2209.13085](https://arxiv.org/abs/2209.13085)
- Gao, L., Schulman, J., Hilton, J. (2022). *Scaling laws for reward model overoptimization*. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- Rafailov, R., et al. (2023). *Direct preference optimization: Your language model is secretly a reward model*. NeurIPS. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Lightman, H., et al. (2023). *Let's verify step by step*. ICLR 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., et al. (2023). *Open problems and fundamental limitations of reinforcement learning from human feedback*. [arXiv:2307.15217](https://arxiv.org/abs/2307.15217)
- Coste, T., Anwar, U., Kirk, R., Krueger, D. (2023). *Reward model ensembles help mitigate overoptimization*. ICLR 2024. [arXiv:2310.02743](https://arxiv.org/abs/2310.02743)
- Singhal, P., Goyal, T., Xu, J., Durrett, G. (2023). *A long way to go: Investigating length correlations in RLHF*. [arXiv:2310.03716](https://arxiv.org/abs/2310.03716)
- Moskovitz, T., Singh, A. K., Strouse, D. J., Sandholm, T., Salakhutdinov, R., Dragan, A. D., McAleer, S. (2023). *Confronting reward model overoptimization with constrained RLHF*. [arXiv:2310.04373](https://arxiv.org/abs/2310.04373)
- Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., et al. (2023). *Towards understanding sycophancy in language models*. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
- Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D'Amour, A., Dvijotham, D. J., Fisch, A., Heller, K., et al. (2023). *Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking*. [arXiv:2312.09244](https://arxiv.org/abs/2312.09244)
- Ramé, A., Vieillard, N., Hussenot, L., Dadashi, R., Cideron, G., Bachem, O., Ferret, J. (2024). *WARM: On the benefits of weight averaged reward models*. [arXiv:2401.12187](https://arxiv.org/abs/2401.12187)
- Chen, L., Zhu, C., Soselia, D., Chen, J., Zhou, T., Goldstein, T., Huang, H., Shoeybi, M., Catanzaro, B. (2024). *ODIN: Disentangled reward mitigates hacking in RLHF*. [arXiv:2402.07319](https://arxiv.org/abs/2402.07319)
- Lambert, N., Pyatkin, V., Morrison, J., Miranda, L. J., Lin, B. Y., Chandu, K., Dziri, N., Kumar, S., et al. (2024). *RewardBench: Evaluating reward models for language modeling*. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787)
- Rafailov, R., Chittepu, Y., Park, R., Sikchi, H., Hejna, J., Knox, B., Finn, C., Niekum, S. (2024). *Scaling laws for reward model overoptimization in direct alignment algorithms*. [arXiv:2406.02900](https://arxiv.org/abs/2406.02900)
- Denison, C., MacDiarmid, M., Barez, F., Duvenaud, D., Kravec, S., Marks, S., Schiefer, N., Soklaski, R., et al. (2024). *Sycophancy to subterfuge: Investigating reward-tampering in large language models*. [arXiv:2406.10162](https://arxiv.org/abs/2406.10162)
- Wen, J., Zhong, R., Khan, A., Perez, E., Steinhardt, J., Huang, M., Bowman, S. R., He, H., Feng, S. (2024). *Language models learn to mislead humans via RLHF*. [arXiv:2409.12822](https://arxiv.org/abs/2409.12822)
- Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J., Farhi, D. (2025). *Monitoring reasoning models for misbehavior and the risks of promoting obfuscation*. [arXiv:2503.11926](https://arxiv.org/abs/2503.11926)
- OpenAI (2025). *Sycophancy in GPT-4o: What happened and what we're doing about it*. Blog post, April 29, 2025. [Link](https://openai.com/index/sycophancy-in-gpt-4o/)
- Malik, S., Pyatkin, V., Land, S., Morrison, J., Smith, N. A., Hajishirzi, H., Lambert, N. (2025). *RewardBench 2: Advancing reward model evaluation*. [arXiv:2506.01937](https://arxiv.org/abs/2506.01937)
- MacDiarmid, M., Wright, B., Uesato, J., Benton, J., Kutasov, J., Price, S., Bouscal, N., Bowman, S., et al. (2025). *Natural emergent misalignment from reward hacking in production RL*. [arXiv:2511.18397](https://arxiv.org/abs/2511.18397)
