# Changelog

Understanding AI grows daily. Each entry records what was added or changed, newest first. See [docs/DAILY_UPDATES.md](docs/DAILY_UPDATES.md) for how updates work.

## 2026-10-28: Attention gradients, derived by hand

- **Deep learning**: new page [Attention gradients, derived by hand](docs/02-deep-learning/attention-gradients.md): the forward pass of scaled dot-product attention and the reverse-mode convention, $dV = P^\top dO$ and $dP = dO\,V^\top$, the softmax Jacobian $\operatorname{diag}(p) - pp^\top$ with a one-line proof that its eigenvalues lie in $[0, 1/2]$ and that it vanishes for one-hot rows, $dS = P \odot (dP - D)$ and $dQ$, $dK$, a matmul count giving about 2.5 times the forward arithmetic, the identity $D_i = dO_i \cdot O_i$ that, together with the saved per-row log-sum-exp, lets FlashAttention rebuild the attention matrix tile by tile instead of storing it, a tested NumPy sketch on a causal head ($n = 64$, $d = 16$) where the hand-derived gradients match central finite differences to a maximum relative error of $4.3 \times 10^{-9}$ and a tiled backward pass that stores 64 numbers instead of 4,096 matches the naive one to round-off, why the $1/\sqrt{d}$ scale exists (largest Jacobian eigenvalue $0.28$ at logit spread 4 against $1.1 \times 10^{-3}$ at spread 16), attention-logit growth and QK-norm, entropy collapse and σReparam, the missing global Lipschitz bound, a table of variants (masks, additive biases, sparsemax, linear attention, grouped-query), an honest status table, and a clearly labelled speculative section on recomputing attention when the forward pass runs on noisy analog or optical hardware.
- **References**: new *Attention gradients* section with 10 entries, each checked against its arXiv abstract page or Crossref; seven further sources were already listed.
- **Glossary**: added *FlashAttention*, *Log-sum-exp*, *Online softmax*, and *QK-norm*.
- **Linked** from the Deep learning index, the README contents table, and the efficient-attention paragraph of the transformers page.

## 2026-10-27 (evening): SGD as a stochastic differential equation

- **Frontiers**: new page [SGD as a stochastic differential equation](docs/06-frontiers/sgd-as-a-stochastic-differential-equation.md): minibatch SGD written as an Itô SDE with time $t = k\eta$, weak-order stochastic modified equations and the Fokker–Planck equation, temperature $\eta/B$ and the linear scaling rule with its limits and the gradient noise scale, the Ornstein–Uhlenbeck stationary covariance from the Lyapunov equation and its exact discrete-time correction, a tested NumPy sketch on a two-dimensional least-squares problem whose curvatures differ 37-fold, where SGD's spread matches the SDE prediction at small steps along both directions and the sharp direction's spread nearly doubles (0.0248 against 0.0126) at $\eta = 0.16$, close to the stability limit of 0.219, as the exact discrete formula predicts (0.0243), Kramers escape and why anisotropic noise favours flat minima, implicit gradient regularization and the slow SDE along a manifold of minima, the square-root scaling rule for Adam and critical batch size in modern pre-training, where the picture breaks (edge of stability, heavy-tailed noise, large learning rates), stochastic gradient Langevin dynamics, an honest status table, and a clearly labelled speculative section on physical noise in analog hardware as part of the SDE.
- **References**: new *Frontiers: SGD as a stochastic differential equation* section with 27 entries, each checked against its arXiv abstract page or the publisher's page; four further sources were already listed.
- **Glossary**: added *Fokker–Planck equation*, *Gradient noise scale*, *Implicit gradient regularization*, *Linear scaling rule*, *Stochastic differential equation (SDE)*, and *Stochastic gradient Langevin dynamics (SGLD)*.
- **Linked** from the Frontiers index and mindmap, the README contents table and field map, the SGD section of the optimization page, and the non-linearity and dynamical systems page (removed from its next-topics list).

## 2026-10-27: Sparse autoencoders and the dictionary of features

- **Interpretability**: new page [Sparse autoencoders and the dictionary of features](docs/05-alignment-safety-interpretability/sparse-autoencoders.md): superposition and why individual neurons are hard to read, the sparse-coding roots of the idea (Olshausen and Field, 1996), the SAE objective with reconstruction, sparsity and the L0 and fraction-of-variance-unexplained metrics, scaling from a one-layer model to roughly 34 million features in Claude 3 Sonnet and 16 million latents on GPT-4, the move from L1 to Gated, TopK, JumpReLU, BatchTopK, Matryoshka and end-to-end SAEs, a tested NumPy sketch where no neuron matches any of 128 planted features but a TopK SAE recovers all of them while L1 shrinks their activations to about a third of their true size or less, Golden Gate Claude, sparse feature circuits, transcoders, crosscoders and attribution graphs, automated feature labelling, the 2025 negative results (seed dependence, feature absorption, multi-dimensional features, SAE dark matter, random-transformer baselines, SAEBench, and linear probes beating SAE probes), an honest status table, and a clearly labelled speculative section on sparse codes as a hardware interface.
- **References**: new *Sparse autoencoders and dictionary learning* section with 29 entries, each checked against its arXiv abstract page, Crossref or the publisher's page; two further sources were already listed.
- **Glossary**: added *Dictionary learning*, *Feature absorption*, *Sparse autoencoder (SAE)*, *TopK SAE*, and *Transcoder*.
- **Linked** from the Alignment index, the README contents table and field map, and the superposition section of the interpretability page.

## 2026-10-26 (evening): Reward models and how they fail

- **Alignment**: new page [Reward models and how they fail](docs/05-alignment-safety-interpretability/reward-model-failure-modes.md): Bradley–Terry reward models and the KL-regularized objective with its closed-form optimum, best-of-$n$ and its KL, Goodhart's four variants mapped onto reward modelling, the Skalse et al. result that almost no proxy is unhackable over all stochastic policies, the Gao et al. over-optimization laws for best-of-$n$ and RL and their extension to direct alignment algorithms, capability-driven phase transitions in reward hacking, a tested NumPy sketch where a linear proxy that agrees with the gold reward on 85.0% of pairs drives the gold score from a peak of 0.48 at $n = 64$ to −0.04 at $n = 16{,}384$ (the Gao et al. form fits with $R^2 = 0.976$), a field guide from length bias and sycophancy (including the April 2025 GPT-4o rollback) through unintended sophistry, underspecification, reward tampering and obfuscated reward hacking to 2025 evidence of emergent misalignment from production reward hacking, a table of mitigations (KL control, ensembles, WARM, constrained RLHF, ODIN, process supervision, RewardBench 2, chain-of-thought monitoring), an honest status table, and a clearly labelled speculative section on Goodhart effects when the reward is a physical measurement.
- **References**: new *Reward models and their failure modes* section with 22 entries, each checked against its arXiv abstract page or the publisher's page; six further sources were already listed.
- **Glossary**: added *Bradley–Terry model*, *Reward hacking*, *Reward model*, *Reward over-optimization*, and *Sycophancy*.
- **Linked** from the Alignment index, the README contents table and field map, and the learning-from-human-preferences section of the alignment page.

## 2026-10-26: Gaussian processes and Bayesian optimization

- **Classical ML**: new page [Gaussian processes and Bayesian optimization](docs/01-classical-ml/gaussian-processes-bayesian-optimization.md): Gaussian processes as distributions over functions, RBF and Matérn kernels, kriging (Matheron 1963) and EGO (Jones et al. 1998) as the origins, closed-form posterior mean and variance, the marginal likelihood as an automatic Occam's razor, acquisition functions (probability of improvement, expected improvement, GP-UCB with its regret bounds, Thompson sampling, predictive entropy search), a tested NumPy sketch where the marginal likelihood picks the length-scale and expected-improvement BO finds the best grid point in 20 of 20 runs against 2 of 20 for random search at 15 evaluations, applications from the NeurIPS 2020 black-box optimization challenge to AlphaGo and reaction chemistry, sparse and GPU-scale GPs, TuRBO and the 2024–2025 papers tracing high-dimensional failures to length-scale priors, LLM-assisted BO, the infinite-width link to neural networks (NNGP and the neural tangent kernel), an honest status table, and a clearly labelled speculative section on calibrating analog hardware with few measurements.
- **References**: new *Gaussian processes and Bayesian optimization* section with 26 entries, each checked against its arXiv abstract page, Crossref, the PMLR proceedings page, or the book's webpage; one further source was already listed.
- **Glossary**: added *Acquisition function*, *Bayesian optimization*, *Expected improvement (EI)*, and *Gaussian process (GP)*.
- **Linked** from the Classical ML index, the README contents table and field map, and the kernels section of the supervised learning page.

## 2026-10-25 (evening): Mixture of experts and conditional computation

- **Deep learning**: new page [Mixture of experts and conditional computation](docs/02-deep-learning/mixture-of-experts.md): from adaptive mixtures of local experts (1991) and hierarchical mixtures (1994) to the sparsely-gated MoE layer, the top-$k$ router written out, a design-point table from GShard and Switch to Mixtral, OLMoE, DeepSeek-V3 and Kimi K2, router collapse and four families of fixes (auxiliary losses, capacity limits and dropless block-sparse kernels, expert-choice, BASE and hash routing, and auxiliary-loss-free bias balancing), a tested NumPy sketch where the bias update takes the busiest expert from 3.11x to 1.03x its fair share and token drops from 23.3% to zero, the router z-loss, sparse upcycling and Soft MoE, fine-grained and shared experts and MoE scaling laws, an honest status table, the systems cost of sparsity (expert parallelism, all-to-all, memory), and a clearly labelled speculative section on routing across heterogeneous and optical hardware.
- **References**: new *Mixture of experts* section with 21 entries, each checked against its arXiv abstract page or Crossref; three further sources were already listed.
- **Glossary**: added *Capacity factor*, *Load balancing (MoE)*, and *Router (gating network)*; *Mixture of experts (MoE)* now points to the new page.
- **Linked** from the Deep learning index, the README contents table and field map, and the mixture-of-experts section of the inference page.

## 2026-10-25: Information geometry and the natural gradient

- **Foundations**: new page [Information geometry and the natural gradient](docs/00-foundations/information-geometry-natural-gradient.md): the Fisher information metric as the second-order expansion of KL divergence and Chentsov's uniqueness theorem, the natural gradient as steepest descent under that metric (Amari 1998) with its reparameterization invariance and link to Gauss–Newton, a tested NumPy sketch (natural gradient fits a narrow Gaussian in 21 steps against 1,046 for the best stable plain step, the empirical Fisher diverging from the Fisher, and Newton–Schulz orthogonalization), the empirical-Fisher trap, K-FAC, Shampoo and its AlgoPerf win, SOAP, Muon and steepest descent under the spectral norm with its scaling to Moonlight and Kimi K2, the same geometry in natural policy gradients, TRPO, mirror descent and elastic weight consolidation, an honest status table, and a clearly labelled speculative section on parameterization-free training of physical interferometer meshes.
- **References**: new *Information geometry and natural gradients* section with 19 entries, each checked against its arXiv abstract page, Crossref, the NeurIPS proceedings page, or the author's post.
- **Glossary**: added *Empirical Fisher*, *Fisher information matrix*, *K-FAC*, *Muon*, and *Natural gradient*.
- **Linked** from the Foundations index, the README field map, and the second-order section of the optimization page.

## 2026-10-24 (evening): Fast-weight programmers and self-referential networks

- **Frontiers**: new page [Fast-weight programmers and self-referential networks](docs/06-frontiers/fast-weight-programmers.md): slow networks that program fast weights (Hinton & Plaut 1987, Schmidhuber 1992, Ba et al. 2016), the exact equivalence between causal linear attention and additive outer-product memories and the capacity limit it implies, the delta rule as one online gradient step and its hardware-efficient parallelization (DeltaNet), a tested NumPy sketch of equivalence, rebinding and crosstalk, a family table of editing rules (gated delta rule, Kimi Delta Attention, test-time training, Titans) and their use in production hybrids (Qwen3-Next, Kimi Linear), expressiveness results on parity and negative eigenvalues, the modern self-referential weight matrix, an honest status table, and a clearly labelled speculative section on outer-product updates as a hardware primitive.
- **References**: new *Frontiers: fast-weight programmers* section with 20 entries, each checked against its arXiv abstract page, Crossref, or the official model card (one 1987 conference paper without a DOI is listed without a link).
- **Glossary**: added *Delta rule (fast weights)*, *Fast weights*, and *Self-referential weight matrix (SRWM)*.
- **Linked** from the Frontiers index, the README field map and highlights, and the learning-to-learn section of the recursion page.

## 2026-10-24: Energy per MAC and the nonlinearity problem

- **Frontiers**: new page [Energy per MAC: the accounting of optical neural networks, and the nonlinearity problem](docs/06-frontiers/optical-energy-and-nonlinearity.md): Horowitz's digital energy table as the baseline, the four stages of an optical MAC, the $1/N$ scaling argument and why memory traffic sets the floor, the shot-noise standard quantum limit and the sub-photon-per-multiplication experiments (2022), a tested NumPy energy-budget calculator with clearly illustrative parameters, O-E-O versus electro-optic versus all-optical activations (cold-atom EIT, lithium-niobate ReLU, single-chip in situ training), the two April 2025 Nature photonic accelerators, a six-question checklist for reading energy claims, and a clearly labelled speculative section on energy-aware compilation and per-layer precision.
- **References**: new *Frontiers: optical energy accounting and nonlinearity* section with 15 entries, each checked against Crossref, its arXiv abstract page, or the publisher's page.
- **Glossary**: added *Energy per MAC*, *Shot noise*, and *Standard quantum limit (optical neural networks)*.
- **Linked** from the Frontiers index, the README field map, and the open-challenges section of the photonic computing page.

## 2026-10-23 (evening): Evaluation and benchmarks, and their pitfalls

- **Modern AI**: new page [Evaluation and benchmarks, and their pitfalls](docs/03-modern-ai/evaluation-and-benchmarks.md): evaluations as experiments on a super-population of questions, four families of benchmarks, standard errors and paired differences, the unbiased pass@k estimator with a tested NumPy sketch, and six ways scores mislead (contamination and the GSM1k replication, saturation from MMLU to Humanity's Last Exam, MMLU label errors, metric-driven "emergence", LLM-judge biases, and the 2025 *Leaderboard Illusion* findings on Chatbot Arena), all tied to the four variants of Goodhart's law. Also covers SWE-bench and METR's 50% task-completion time horizon, a reader's checklist, and a clearly labelled speculative section on evaluating models on noisy analog and optical hardware.
- **References**: new *Evaluation and benchmarks* section with 22 entries, each checked against its arXiv abstract page or Crossref.
- **Glossary**: added *Benchmark contamination*, *LLM-as-a-judge*, and *pass@k*.
- **Linked** from the Modern AI index, the README field map, and the evaluation section of the alignment page.

## 2026-10-23: Heavy-tailed weight spectra and generalization

- **Frontiers**: new page [Heavy-tailed weight spectra and generalization](docs/06-frontiers/heavy-tailed-weight-spectra.md): the Marchenko–Pastur baseline, the 5+1 phases of training and Heavy-Tailed Self-Regularization, maximum-likelihood power-law fitting with a tested NumPy sketch, α-based quality metrics and the large correlational studies behind them (with the caveats from controlled studies of generalization measures), competing explanations for where heavy tails come from (α-stable gradient noise, multiplicative noise, large learning rates without noise), the Hausdorff-dimension generalization bound, recent tools that use layer spectra (TempBalance, AlphaPruning, model balancing) and the 2025 SETOL theory, and a clearly labelled speculative section on using spectra to budget precision on analog and optical hardware.
- **References**: 19 new Frontiers entries, each checked against its arXiv abstract page or Crossref, plus the WeightWatcher tool.
- **Glossary**: added *Empirical spectral density (ESD)*, *Heavy-Tailed Self-Regularization (HT-SR)*, and *Marchenko–Pastur law*.
- **Linked** from the Frontiers index and the fractal-structure page.

## 2026-10-22: Normalization, initialization and μP

- **Deep learning**: new page [Normalization, initialization and μP](docs/02-deep-learning/normalization-initialization-mup.md): variance propagation and Glorot/He/orthogonal initialization, dynamical isometry and the edge of chaos, LayerNorm and RMSNorm, why Pre-LN trains without warm-up, Fixup and DeepNet, QK-norm and small-scale proxies for instability, normalization-free (Dynamic Tanh, 2025) and all-normalized (nGPT, 2024) transformers, a worked argument for why the best learning rate drifts with width, the μP rules and μTransfer results, depth-wise transfer, the 2024 debate over whether μP is special, u-μP for FP8, and a clearly labelled speculative section on scale control for analog and optical hardware.
- **References**: 13 new Deep learning entries, each checked against its arXiv abstract page.
- **Glossary**: added *Dynamical isometry*, *Maximal Update Parametrization (μP)*, and *Pre-LN / Post-LN*.

## 2026-10-18: Compilers for machine learning

- **Systems**: new page [Compilers for machine learning](docs/04-systems/ml-compilers.md): fusion and tiling with worked memory-traffic arithmetic, IRs and progressive lowering, a timeline of landmark systems (Halide, TVM, AutoTVM, Relay, TASO, Triton, MLIR, Ansor, PyTorch 2), XLA/OpenXLA, a Triton kernel example, learned schedule search, recent work on hand-tuned and LLM-written kernels, and a clearly labelled speculative section on compiling for analog and optical hardware.
- **References**: 14 new Systems entries, each checked against arXiv, Crossref, or the official project page.
- **Glossary**: added *Operator fusion*, *Intermediate representation (IR)*, and *Triton*.

## 2026-10-17: First edition

- **Foundations**: history of AI, linear algebra, probability and information theory, optimization, learning theory.
- **Classical ML**: supervised learning, unsupervised learning, reinforcement learning.
- **Deep learning**: neural networks and backpropagation, architectures (CNN, RNN/LSTM, ResNet, GNN, state-space models), transformers, scaling laws.
- **Modern AI**: large language models (pretraining, RLHF, DPO, reasoning), generative models, retrieval/tools/agents, multimodal models.
- **Systems**: hardware and training at scale, inference and efficiency.
- **Alignment, safety and interpretability**: alignment and safety, interpretability.
- **Frontiers**: recursion and self-improvement, fractal and self-similar structure, non-linearity and dynamical systems, photonic and optical computing, neuromorphic computing, quantum machine learning, world models.
- **Reference material**: glossary, 220+ checked references, contribution guide, CC BY 4.0 license.

## Backlog

Planned topics, roughly in rotation order:

- [x] Foundations: information geometry and natural gradients (2026-10-25)
- [x] Classical ML: Gaussian processes and Bayesian optimization (2026-10-26)
- [x] Deep learning: worked derivation of attention gradients (2026-10-28)
- [x] Deep learning: normalization, initialization and μP (hyperparameter transfer) (2026-10-22)
- [x] Modern AI: evaluation and benchmarks, and their pitfalls (2026-10-23)
- [ ] Modern AI: code generation and program synthesis
- [ ] Modern AI: AI for science (protein design, materials, weather)
- [x] Systems: compilers for ML (XLA, MLIR, Triton) (2026-10-18)
- [x] Alignment: reward modelling failure modes in depth (2026-10-26)
- [x] Frontiers / recursion: fast-weight programmers and self-referential networks (2026-10-24)
- [x] Frontiers / fractals: heavy-tailed weight spectra and generalization (2026-10-23)
- [x] Frontiers / dynamics: SGD as a stochastic differential equation (2026-10-27)
- [x] Frontiers / photonics: energy-per-MAC accounting and optical nonlinearities (2026-10-24)
- [ ] Frontiers / neuromorphic: memristive crossbars and in-memory computing
- [ ] Frontiers / quantum: tensor networks as ML models
- [ ] Frontiers / world models: causal representation learning
- [ ] Frontiers: continual learning and catastrophic forgetting
- [ ] Frontiers: AI and the theory of computation (what transformers can and cannot compute)
