# Glossary

Short definitions of terms used across Understanding AI, with a link to where each is explained in depth.

| Term | Definition | Read more |
|---|---|---|
| **Activation function** | Elementwise nonlinearity applied after a linear layer (ReLU, GELU, tanh) | [Backprop](docs/02-deep-learning/neural-networks-and-backprop.md#activation-functions) |
| **Adam / AdamW** | Adaptive optimizers that scale each parameter's step by running gradient statistics; AdamW decouples weight decay | [Optimization](docs/00-foundations/optimization.md#momentum-and-adaptive-methods) |
| **Agent** | A system that repeatedly observes, decides, and acts toward a goal, often an LLM with tools | [Agents](docs/03-modern-ai/retrieval-tools-and-agents.md) |
| **Attention** | Weighted averaging of value vectors, with weights from query–key similarity | [Transformers](docs/02-deep-learning/transformers.md) |
| **Backpropagation** | Reverse-mode application of the chain rule to compute all gradients efficiently | [Backprop](docs/02-deep-learning/neural-networks-and-backprop.md#backpropagation) |
| **Barren plateau** | Exponentially vanishing gradients in large random quantum circuits | [Quantum ML](docs/06-frontiers/quantum-machine-learning.md) |
| **Bellman equation** | Recursive relation defining the value of a state through the value of successor states | [RL](docs/01-classical-ml/reinforcement-learning.md) |
| **Chain of thought** | Prompting or training a model to write intermediate reasoning steps | [LLMs](docs/03-modern-ai/large-language-models.md#prompting-for-reasoning) |
| **Cross-entropy** | Expected negative log-probability a model assigns to the true outcomes; the standard classification and language-model loss | [Probability](docs/00-foundations/probability-and-information.md) |
| **Diffusion model** | Generative model that learns to reverse a gradual noising process | [Generative](docs/03-modern-ai/generative-models.md#diffusion-models) |
| **Double descent** | Test error falling, rising, then falling again as capacity grows past the interpolation threshold | [Learning theory](docs/00-foundations/learning-theory.md#double-descent) |
| **DPO** | Direct preference optimization: fit preferences with a classification-style loss instead of RL | [LLMs](docs/03-modern-ai/large-language-models.md#direct-preference-optimization-dpo) |
| **Edge of chaos** | Boundary between ordered and chaotic dynamics where deep signal propagation is possible | [Dynamics](docs/06-frontiers/nonlinearity-and-dynamical-systems.md) |
| **Edge of stability** | Regime where the loss curvature hovers near $2/\eta$ during gradient descent | [Optimization](docs/00-foundations/optimization.md#second-order-ideas) |
| **Embedding** | A learned vector representing a token, image patch, or other item | [Linear algebra](docs/00-foundations/linear-algebra.md) |
| **Foundation model** | Large model pretrained on broad data and adapted to many tasks | [Modern AI](docs/03-modern-ai/README.md) |
| **Fractal dimension** | Non-integer measure of how detail scales with resolution | [Fractals](docs/06-frontiers/fractal-and-self-similar-structure.md) |
| **Grokking** | Delayed, abrupt generalization long after the training set is memorized | [Learning theory](docs/00-foundations/learning-theory.md#grokking) |
| **Hallucination** | Fluent output not supported by inputs or facts | [LLMs](docs/03-modern-ai/large-language-models.md#known-failure-modes) |
| **In-context learning** | Performing a new task from examples in the prompt without weight updates | [LLMs](docs/03-modern-ai/large-language-models.md#in-context-learning) |
| **Induction head** | Attention circuit that completes repeated patterns (`A B … A → B`) | [Interpretability](docs/05-alignment-safety-interpretability/interpretability.md) |
| **KL divergence** | Asymmetric measure of how one distribution differs from another | [Probability](docs/00-foundations/probability-and-information.md) |
| **KV cache** | Stored keys and values of past tokens reused during generation | [Inference](docs/04-systems/inference-and-efficiency.md) |
| **LoRA** | Fine-tuning by learning a low-rank update to frozen weights | [LLMs](docs/03-modern-ai/large-language-models.md#efficient-adaptation) |
| **Mach–Zehnder interferometer (MZI)** | Programmable 2×2 optical element; meshes of MZIs implement matrices in photonic chips | [Photonics](docs/06-frontiers/photonic-computing.md) |
| **Mixture of experts (MoE)** | Layer that routes each token to a few of many expert sub-networks | [Inference](docs/04-systems/inference-and-efficiency.md#mixture-of-experts) |
| **Model collapse** | Degradation from training recursively on model-generated data | [Recursion](docs/06-frontiers/recursion-and-self-improvement.md#model-collapse) |
| **Neural ODE** | Network whose hidden state evolves by a learned differential equation | [Dynamics](docs/06-frontiers/nonlinearity-and-dynamical-systems.md) |
| **NISQ** | Noisy intermediate-scale quantum: today's uncorrected quantum devices | [Quantum ML](docs/06-frontiers/quantum-machine-learning.md) |
| **Perplexity** | Exponentiated per-token cross-entropy; effective branching factor of a language model | [Probability](docs/00-foundations/probability-and-information.md) |
| **Quantization** | Representing weights or activations with fewer bits | [Inference](docs/04-systems/inference-and-efficiency.md#quantization) |
| **RAG** | Retrieval-augmented generation: retrieve documents and condition the answer on them | [Agents](docs/03-modern-ai/retrieval-tools-and-agents.md) |
| **Reservoir computing** | Using a fixed nonlinear dynamical system plus a trained linear readout | [Dynamics](docs/06-frontiers/nonlinearity-and-dynamical-systems.md) |
| **Residual connection** | Skip path that adds a block's input to its output | [Architectures](docs/02-deep-learning/architectures.md#residual-networks) |
| **RLHF** | Reinforcement learning from human feedback via a learned reward model | [LLMs](docs/03-modern-ai/large-language-models.md) |
| **Scaling law** | Power-law relation between loss and parameters, data, or compute | [Scaling laws](docs/02-deep-learning/scaling-laws.md) |
| **Self-supervised learning** | Learning from targets constructed from the data itself | [Unsupervised](docs/01-classical-ml/unsupervised-learning.md) |
| **Spiking neural network** | Network of neurons that communicate with discrete spikes in time | [Neuromorphic](docs/06-frontiers/neuromorphic-computing.md) |
| **Superposition** | Representing more features than dimensions using nearly orthogonal directions | [Interpretability](docs/05-alignment-safety-interpretability/interpretability.md#superposition-and-sparse-features) |
| **SVD** | Factorization $W = U\Sigma V^\top$ into rotations and scalings | [Linear algebra](docs/00-foundations/linear-algebra.md) |
| **Token** | Unit of text (often a sub-word) processed by a language model | [Transformers](docs/02-deep-learning/transformers.md) |
| **Transformer** | Architecture built from attention and MLP blocks with residual connections | [Transformers](docs/02-deep-learning/transformers.md) |
| **World model** | Learned model of environment dynamics used for prediction and planning | [World models](docs/06-frontiers/world-models.md) |
