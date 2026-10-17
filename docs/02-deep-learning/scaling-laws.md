# Scaling laws

One of the most consequential empirical discoveries in modern AI is that, for many model families, **performance improves as a smooth power law** in model size, data, and compute. That predictability turned building large models from a gamble into a planning exercise.

## The empirical pattern

Hestness et al. (2017) found power-law relationships between generalization error and training set size across several domains. **Kaplan et al. (2020)** measured language-model cross-entropy loss $L$ as a function of non-embedding parameters $N$, dataset tokens $D$, and compute $C$, when the other factors are not a bottleneck:

```math
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
```

with fitted exponents of roughly $\alpha_N \approx 0.076$ and $\alpha_D \approx 0.095$ for their setup. On a log–log plot these are straight lines spanning many orders of magnitude.

A useful approximation for dense transformers is that training compute is

```math
C \approx 6\, N D \quad \text{FLOPs}
```

(about 2 FLOPs per parameter per token for the forward pass and 4 for the backward pass).

## Compute-optimal training: Chinchilla

Given a fixed compute budget, how should it be split between parameters and data? **Hoffmann et al. (2022)** fitted

```math
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
```

and concluded that parameters and tokens should grow in roughly **equal proportion**, which in their fits corresponds to about 20 training tokens per parameter. Their 70B-parameter *Chinchilla*, trained on 1.4 trillion tokens, outperformed the 280B-parameter *Gopher* trained with a similar compute budget. Many earlier models had been undertrained relative to their size.

In practice, models intended for heavy deployment are often trained on far more tokens than compute-optimal, because a smaller model that has seen more data is cheaper to run.

## Why power laws?

The origin of scaling laws is not settled. Proposed explanations include:

- **Data-manifold dimension**: if data lies on a manifold of intrinsic dimension $d$, a model that piecewise-approximates it should see loss fall as a power of $N$ with exponent inversely related to $d$ (Sharma & Kaplan 2020).
- **Variance- vs resolution-limited regimes** (Bahri et al. 2021), which derive different scaling behaviours depending on which resource is scarce.
- **Heavy-tailed structure** of natural data, where ever-rarer patterns each contribute a little loss. This connects to [fractal and self-similar structure](../06-frontiers/fractal-and-self-similar-structure.md).

## Emergence, and its critics

**Wei et al. (2022)** catalogued "emergent abilities", tasks on which performance appears to jump suddenly at a certain scale. **Schaeffer, Miranda & Koyejo (2023)** argued that many such jumps are artifacts of discontinuous metrics (for example exact-match accuracy) and that smoother metrics reveal gradual improvement. Both views agree that underlying loss improves smoothly; the debate is about how capabilities measured on specific tasks relate to it.

## Inference-time scaling

A newer axis is compute spent **at inference**: sampling many answers and voting (self-consistency; Wang et al. 2022), searching over reasoning paths, or generating long chains of reasoning before answering. Snell et al. (2024) studied when extra test-time compute can substitute for a larger model. Reasoning-focused models trained with reinforcement learning, such as DeepSeek-R1 (DeepSeek-AI 2025), produce long reasoning traces and improve on math and coding benchmarks as they do so.

## Limits and open questions

- **Data**: high-quality human-written text is finite. Synthetic data helps in some domains (code, math with verifiers) but carries risks such as [model collapse](../06-frontiers/recursion-and-self-improvement.md#model-collapse).
- **Energy and hardware**: training cost grows with $C$; see [Systems](../04-systems/README.md) and [Photonic computing](../06-frontiers/photonic-computing.md).
- **What loss means**: lower next-token loss correlates with capability, but which specific skills appear at which loss is hard to predict.

## Key takeaways

- Loss falls as a power law in parameters, data and compute over many orders of magnitude.
- For a fixed budget, scale parameters and data together (Chinchilla).
- Why it works is still an open scientific question, which makes it a genuine frontier.
