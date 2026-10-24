# Changelog

Understanding AI grows daily. Each entry records what was added or changed, newest first. See [docs/DAILY_UPDATES.md](docs/DAILY_UPDATES.md) for how updates work.

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

- [ ] Foundations: information geometry and natural gradients
- [ ] Classical ML: Gaussian processes and Bayesian optimization
- [ ] Deep learning: worked derivation of attention gradients
- [x] Deep learning: normalization, initialization and μP (hyperparameter transfer) (2026-10-22)
- [x] Modern AI: evaluation and benchmarks, and their pitfalls (2026-10-23)
- [ ] Modern AI: code generation and program synthesis
- [ ] Modern AI: AI for science (protein design, materials, weather)
- [x] Systems: compilers for ML (XLA, MLIR, Triton) (2026-10-18)
- [ ] Alignment: reward modelling failure modes in depth
- [x] Frontiers / recursion: fast-weight programmers and self-referential networks (2026-10-24)
- [x] Frontiers / fractals: heavy-tailed weight spectra and generalization (2026-10-23)
- [ ] Frontiers / dynamics: SGD as a stochastic differential equation
- [x] Frontiers / photonics: energy-per-MAC accounting and optical nonlinearities (2026-10-24)
- [ ] Frontiers / neuromorphic: memristive crossbars and in-memory computing
- [ ] Frontiers / quantum: tensor networks as ML models
- [ ] Frontiers / world models: causal representation learning
- [ ] Frontiers: continual learning and catastrophic forgetting
- [ ] Frontiers: AI and the theory of computation (what transformers can and cannot compute)
