# Mixture of experts and conditional computation

A dense network spends the same arithmetic on every input. Ask it what two plus two is and it fires every weight it owns, just as it would for a proof in number theory. **Conditional computation** breaks that rule: a small learned **router** looks at each token and sends it to a few specialized sub-networks, the **experts**, leaving the rest idle. The payoff is that the number of parameters, which sets how much a model can know, comes apart from the number of operations per token, which sets what it costs to run. Since 2024, most of the largest open-weight language models have been built this way: Mixtral 8x7B, DeepSeek-V3 with 671B parameters but 37B active per token, and Kimi K2 with a trillion parameters and 32B active.

> The core idea in one sentence: **replace one big feed-forward layer with many smaller ones and a router that picks the top $k$ for each token, so capacity grows with the number of experts while compute grows only with $k$.**

```mermaid
flowchart LR
    T[Token state h] --> R[Router<br/>softmax of W·h]
    R -->|top-k scores| S{Select k of E}
    S -->|g₁| E1[Expert 1 FFN]
    S -->|g₃| E3[Expert 3 FFN]
    S -. idle .-> E2[Expert 2 FFN]
    S -. idle .-> EN[Expert E FFN]
    E1 --> Sum[Weighted sum<br/>Σ gᵢ · Eᵢ(h)]
    E3 --> Sum
    SH[Shared expert<br/>always on] --> Sum
    Sum --> O[Output]
```

## 1. From committees of experts to sparse layers (established)

The idea predates deep learning. Jacobs, Jordan, Nowlan and Hinton (1991) trained several small networks together with a **gating network** that learned which one to trust for each input, so the experts divided the input space among themselves instead of all fitting everything. Jordan and Jacobs (1994) stacked the gates into a tree, the **hierarchical mixture of experts**, and trained it with the EM algorithm. In those models the gate produced a soft weighting over all experts, so every expert still ran on every input.

The change that matters for modern AI is **sparsity**. Bengio, Léonard and Courville (2013) framed conditional computation as a goal, activating only parts of a network per example, and studied how to pass gradients through the hard, discrete decisions this requires. Shazeer et al. (2017) made it work at scale with the **sparsely-gated mixture-of-experts layer**: up to thousands of feed-forward experts between stacked LSTM layers, a trainable gate that keeps only the top $k$ scores, and models of up to 137 billion parameters that beat the state of the art in language modeling and translation at lower computational cost. The paper reports more than 1000x improvements in model capacity with only minor losses in computational efficiency.

## 2. The layer, precisely (established)

In a transformer, the MoE layer usually replaces the feed-forward block (see [Transformers](transformers.md)). For a token state $h \in \mathbb{R}^d$, experts $E_1, \dots, E_N$, and router weights $W_r \in \mathbb{R}^{N \times d}$:

```math
s = W_r h, \qquad \mathcal{T} = \operatorname{TopK}(s, k), \qquad g_i = \frac{\exp(s_i)}{\sum_{j \in \mathcal{T}} \exp(s_j)} \;\; (i \in \mathcal{T}), \qquad y = \sum_{i \in \mathcal{T}} g_i \, E_i(h).
```

Each expert is an ordinary feed-forward network. Only the $k$ selected experts run, so the floating-point work per token is roughly $k$ times that of one expert, while the parameter count is $N$ times. The router is trained by ordinary backpropagation through the gate values $g_i$; the choice of *which* experts to use is discrete and gets no gradient of its own, which is why everything in section 3 exists.

The common design points:

| Model | Year | Experts per MoE layer | Active per token | Total / active parameters | Source |
|---|---|---|---|---|---|
| Sparsely-gated MoE | 2017 | up to thousands | top-$k$ | up to 137B total | Shazeer et al. 2017 |
| GShard | 2020 | many, sharded across devices | top-2 | over 600B total | Lepikhin et al. 2020 |
| Switch Transformer | 2021 | many | top-1 | trillion-scale total | Fedus et al. 2021 |
| GLaM | 2021 | many | top-2 | 1.2T total | Du et al. 2021 |
| Mixtral 8x7B | 2024 | 8 | 2 | 47B / 13B | Jiang et al. 2024 |
| OLMoE-1B-7B | 2024 | fine-grained | several | 7B / 1B | Muennighoff et al. 2024 |
| DeepSeek-V3 | 2024 | fine-grained routed + shared | several | 671B / 37B | DeepSeek-AI 2024 |
| Kimi K2 | 2025 | fine-grained | several | 1T / 32B | Kimi Team 2025 |

Rows marked "many" or "several" are deliberately unspecific: the exact counts vary by model size within each paper, and the cited reports give them in full.

**Tested sketch.** The parameter arithmetic for one layer with eight SwiGLU experts (three weight matrices each) and top-2 routing, at illustrative widths of 4,096 and 14,336:

```python
def ffn_params(d_model, d_ff): return 3 * d_model * d_ff
E, k, dm, dff = 8, 2, 4096, 14336
total, active = E * ffn_params(dm, dff), k * ffn_params(dm, dff)
# total 1.41B, active 0.35B (25%) for this one layer
```

Attention, embeddings, and norms are shared by every token, which is why whole-model active fractions (13B of 47B for Mixtral) are higher than the per-layer $k/N$.

## 3. The central problem: keeping the experts busy (established)

A router left alone tends to **collapse**. Early in training a few experts happen to be slightly better, so they receive more tokens, so they get more gradient and improve faster, so they receive still more tokens. The neglected experts never learn, and on hardware where each expert lives on its own device, an overloaded expert makes every other device wait. Four families of fixes are in use.

**Auxiliary balancing losses.** Shazeer et al. (2017) added losses that penalize uneven importance and load. The Switch Transformer simplified this to a single term for $N$ experts,

```math
\mathcal{L}_{\text{aux}} = \alpha \, N \sum_{i=1}^{N} f_i \, P_i ,
```

where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ is the mean router probability for expert $i$ over the batch. The term equals $\alpha$ when routing is perfectly uniform and grows as tokens pile onto a few experts. Its weakness is that it fights the language-modeling loss: a strong balancing weight distorts what the router would otherwise learn.

**Capacity limits and token dropping.** For efficient batched kernels, each expert is given a fixed **capacity**, the expected number of tokens per expert multiplied by a **capacity factor** slightly above one (Fedus et al. 2021). Tokens beyond capacity are dropped from the layer and pass through only on the residual connection. MegaBlocks (Gale et al. 2022) removed the need to choose between dropping tokens and padding: it recasts the expert computation as block-sparse matrix multiplication, never drops tokens, and reports end-to-end training speedups of up to 40% over MoEs trained with the Tutel library.

**Changing who chooses.** In **expert-choice routing** (Zhou et al. 2022) experts pick their top tokens rather than tokens picking experts, so every expert gets a fixed-size bucket by construction and a token may be processed by a variable number of experts; the paper reports more than 2x faster training convergence than Switch top-1 and GShard top-2 gating. **BASE layers** (Lewis et al. 2021) solve a balanced linear assignment problem, and **hash layers** (Roller et al. 2021) skip learning the route altogether and assign tokens by a fixed hash, a useful baseline that learned routers must beat.

**Auxiliary-loss-free balancing.** Wang et al. (2024) add a per-expert bias $b_i$ to the scores *only when selecting* the top $k$, not when computing the gate weights, and after each step nudge $b_i$ up for underloaded experts and down for overloaded ones. Because no balancing gradient touches the model, it does not interfere with the main loss; the authors report both better perplexity and better balance than auxiliary-loss methods on models up to 3B parameters, and DeepSeek-V3 adopted it at 671B parameters.

### 3.1 Tested sketch: collapse and its cure

The script below (NumPy, seed 0) routes 4,096 tokens from eight clusters through a top-2 router that starts biased toward expert 0, then applies the auxiliary-loss-free bias update for 300 steps with step size 0.01.

```python
import numpy as np
rng = np.random.default_rng(0)
T, d, E, k = 4096, 32, 8, 2
centers = rng.normal(size=(E, d))
x = centers[rng.integers(E, size=T)] + 0.5 * rng.normal(size=(T, d))
W = 0.1 * rng.normal(size=(d, E)); W[:, 0] += 0.3 * centers.mean(0)   # skewed router
logits = x @ W

def route(logits, bias=None):
    sel = logits if bias is None else logits + bias        # bias changes WHICH experts...
    top = np.argsort(-sel, axis=1)[:, :k]
    g = np.take_along_axis(logits, top, 1)                 # ...but gates use raw scores
    g = np.exp(g - g.max(1, keepdims=True)); return top, g / g.sum(1, keepdims=True)

def load(top): return np.bincount(top.ravel(), minlength=E) / (T * k)

def dropped(top, cf):
    cap = int(cf * T * k / E)
    return np.maximum(np.bincount(top.ravel(), minlength=E) - cap, 0).sum() / (T * k)

top, _ = route(logits)
print(load(top).max() * E, dropped(top, 1.25))             # 3.11, 0.233

b = np.zeros(E)
for _ in range(300):
    top, _ = route(logits, b)
    b += 0.01 * np.sign(1 / E - load(top))
top, _ = route(logits, b)
print(load(top).max() * E, dropped(top, 1.25))             # 1.03, 0.0
```

Without balancing, expert 0 receives 38.9% of all assignments, 3.11 times its fair share, and with a capacity factor of 1.25 the layer would drop 23.3% of token-expert assignments. After the bias update the busiest expert is at 1.03 times its fair share and nothing is dropped. The Switch balancing sum $N \sum_i f_i P_i$ (computed from each token's first choice) evaluates to 1.74 for the unbalanced router, against 1.0 for perfect balance. This is a toy with a frozen router and synthetic data; it shows the mechanism, not the training dynamics.

## 4. Stability, precision, and fine-tuning (established)

Sparse models were long known as hard to train. The Switch Transformer paper introduced selective precision (computing the router in float32 inside a bfloat16 model), smaller initialization, and expert dropout during fine-tuning, and trained large sparse models in bfloat16 for the first time. Zoph et al. (2022), in a paper written as a design guide, added the **router z-loss**,

```math
\mathcal{L}_z = \frac{1}{B} \sum_{j=1}^{B} \Big( \log \sum_{i=1}^{N} e^{s_{j,i}} \Big)^2 ,
```

which penalizes large router logits so that small round-off errors in the exponential do not blow up. Their ST-MoE-32B, with 269B parameters at the compute cost of a 32B dense encoder-decoder, was the first sparse model to reach state-of-the-art transfer-learning results across a broad set of tasks. Two further tricks recur in production systems. **Sparse upcycling** (Komatsuzaki et al. 2022) initializes every expert as a copy of a trained dense feed-forward layer and continues training, outperforming the dense original for about 50% of its pre-training cost. **Soft MoE** (Puigcerver et al. 2023) avoids discrete routing altogether by feeding each expert slots that are learned weighted averages of all tokens; it is fully differentiable and never drops tokens, but because each slot mixes tokens from the whole sequence, it does not directly fit autoregressive decoding, where a token must not see the ones after it.

## 5. How MoEs scale (established, with open questions)

Clark et al. (2022) fitted scaling laws in which parameter count and compute are two independent axes, across five orders of magnitude of model size and hundreds of experts, and condensed them into an **effective parameter count** along which routed and dense models scale at the same rate. GLaM (Du et al. 2021) put the economics plainly: at 1.2 trillion parameters, about 7x larger than GPT-3, it used one third of the energy to train and half the inference FLOPs while scoring better overall in zero- and one-shot evaluation across 29 NLP tasks.

The design trend since 2024 is **fine-grained experts**. DeepSeekMoE (Dai et al. 2024) splits each expert into $m$ smaller ones and activates $m$ times as many, so the number of possible expert combinations grows combinatorially, and adds a few **shared experts** that every token uses so that common knowledge is not duplicated across routed experts. DeepSeekMoE 16B matched LLaMA2 7B with about 40% of its computation. Krajewski et al. (2024) made expert size a scaling variable, **granularity**, and found that the standard practice of making each expert as large as a dense feed-forward layer is not optimal at almost any compute budget, and that the efficiency gap between MoE and dense models widens as budgets grow.

| Question | Status |
|---|---|
| Sparse MoE beats a dense model at equal training FLOPs | Established across many studies, for language and for vision (V-MoE, Riquelme et al. 2021) |
| Fine-grained experts plus shared experts beat coarse experts | Strong evidence (DeepSeekMoE, granularity scaling laws) |
| Auxiliary-loss-free balancing scales to frontier size | Demonstrated in DeepSeek-V3; independent ablations at that scale are scarce |
| Experts specialize in human-meaningful topics | Mixed: OLMoE reports high routing specialization, while Mixtral's routing analysis found little specialization by subject domain and more by syntax |
| MoE matches dense at equal *total* parameters | Generally no; DeepSeekMoE treats the dense model with equal total parameters as an upper bound |

## 6. Where the cost goes: systems (established)

Sparsity saves arithmetic, not memory. Every expert must be resident somewhere, so an MoE needs the memory of its total parameter count while doing the work of its active count. That shifts the bottleneck from compute to **memory capacity and communication**.

- **Expert parallelism.** GShard (Lepikhin et al. 2020) placed different experts on different accelerators and routed tokens to them with an **all-to-all** exchange, training a translation model of more than 600B parameters on 2,048 TPU v3 accelerators in four days. Two all-to-alls per MoE layer, one to dispatch and one to combine, are now the signature communication pattern of sparse training (see [Hardware and training](../04-systems/hardware-and-training.md)).
- **Inference.** At small batch sizes, decoding is limited by reading weights from memory; an MoE reads only the active experts per token but must hold all of them. DeepSpeed-MoE (Rajbhandari et al. 2022) built inference systems around this trade, and the [inference page](../04-systems/inference-and-efficiency.md#mixture-of-experts) covers how batching changes the picture.
- **Kernels.** Uneven, data-dependent expert loads do not fit fixed-shape matrix multiplications. Block-sparse kernels (MegaBlocks) and grouped GEMMs are the standard answers, and both are natural targets for [ML compilers](../04-systems/ml-compilers.md).

## 7. Routing as computation (speculative)

> [!WARNING]
> **Speculative.** The ideas in this section are hypotheses and open questions, not established results.

Mixture of experts is usually described as a cost-saving trick, but it also gives a network a discrete, input-dependent choice of *which function to apply*, something dense transformers lack. Three open questions follow.

1. **Routing on heterogeneous hardware.** If experts lived on different kinds of devices, for example some on digital accelerators and some on analog or [optical matrix-multiply units](../06-frontiers/photonic-computing.md) with their own noise and precision limits, a router could in principle learn to send tolerant tokens to cheap, noisy experts and sensitive ones to precise experts. Whether routers learn such a split, and whether it survives the [energy accounting](../06-frontiers/optical-energy-and-nonlinearity.md) of moving activations between devices, is untested as far as we know.
2. **Self-similar structure.** Hierarchical mixtures of experts (Jordan & Jacobs 1994) are trees of gates, and fine-grained MoE layers stacked through depth form a branching structure of possible computation paths whose number grows combinatorially. Whether the paths tokens actually take show any [self-similar statistics](../06-frontiers/fractal-and-self-similar-structure.md), such as power-law expert usage, is an empirical question worth measuring rather than assuming.
3. **Routers as a lever for [self-improvement](../06-frontiers/recursion-and-self-improvement.md).** Because experts can be added, copied (upcycled), or retrained independently, MoE models are a natural substrate for systems that grow new capacity for new skills. Whether that can be done without degrading old skills is an open problem shared with continual learning.

## Further reading

- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., Hinton, G. E. (1991). *Adaptive mixtures of local experts*. Neural Computation 3(1), 79–87. [doi:10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79)
- Jordan, M. I., Jacobs, R. A. (1994). *Hierarchical mixtures of experts and the EM algorithm*. Neural Computation 6(2), 181–214. [doi:10.1162/neco.1994.6.2.181](https://doi.org/10.1162/neco.1994.6.2.181)
- Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or propagating gradients through stochastic neurons for conditional computation*. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J. (2017). *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer*. ICLR. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., et al. (2020). *GShard: Scaling giant models with conditional computation and automatic sharding*. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Fedus, W., Zoph, B., Shazeer, N. (2021). *Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity*. JMLR 23 (2022). [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., Zettlemoyer, L. (2021). *BASE layers: Simplifying training of large, sparse models*. ICML. [arXiv:2103.16716](https://arxiv.org/abs/2103.16716)
- Roller, S., Sukhbaatar, S., Szlam, A., Weston, J. (2021). *Hash layers for large sparse models*. NeurIPS. [arXiv:2106.04426](https://arxiv.org/abs/2106.04426)
- Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Pinto, A. S., et al. (2021). *Scaling vision with sparse mixture of experts*. NeurIPS. [arXiv:2106.05974](https://arxiv.org/abs/2106.05974)
- Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., et al. (2021). *GLaM: Efficient scaling of language models with mixture-of-experts*. ICML 2022. [arXiv:2112.06905](https://arxiv.org/abs/2112.06905)
- Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., et al. (2022). *DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale*. ICML. [arXiv:2201.05596](https://arxiv.org/abs/2201.05596)
- Clark, A., de las Casas, D., Guy, A., Mensch, A., Paganini, M., Hoffmann, J., et al. (2022). *Unified scaling laws for routed language models*. ICML. [arXiv:2202.01169](https://arxiv.org/abs/2202.01169)
- Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., et al. (2022). *ST-MoE: Designing stable and transferable sparse expert models*. [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
- Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., et al. (2022). *Mixture-of-experts with expert choice routing*. NeurIPS. [arXiv:2202.09368](https://arxiv.org/abs/2202.09368)
- Gale, T., Narayanan, D., Young, C., Zaharia, M. (2022). *MegaBlocks: Efficient sparse training with mixture-of-experts*. MLSys 2023. [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
- Komatsuzaki, A., Puigcerver, J., Lee-Thorp, J., Riquelme Ruiz, C., Mustafa, B., Ainslie, J., et al. (2022). *Sparse upcycling: Training mixture-of-experts from dense checkpoints*. ICLR 2023. [arXiv:2212.05055](https://arxiv.org/abs/2212.05055)
- Puigcerver, J., Riquelme, C., Mustafa, B., Houlsby, N. (2023). *From sparse to soft mixtures of experts*. ICLR 2024. [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)
- Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., et al. (2024). *Mixtral of experts*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., et al. (2024). *DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models*. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Krajewski, J., Ludziejewski, J., Adamczewski, K., Pióro, M., Krutul, M., Antoniak, S., et al. (2024). *Scaling laws for fine-grained mixture of experts*. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)
- Wang, L., Gao, H., Zhao, C., Sun, X., Dai, D. (2024). *Auxiliary-loss-free load balancing strategy for mixture-of-experts*. [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
- Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., et al. (2024). *OLMoE: Open mixture-of-experts language models*. [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)
- DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., et al. (2024). *DeepSeek-V3 technical report*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Kimi Team, Bai, Y., Bao, Y., et al. (2025). *Kimi K2: Open agentic intelligence*. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
