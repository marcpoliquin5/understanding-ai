# Linear algebra for AI

Almost every computation inside a modern AI model is a **matrix multiplication**. Images, words, and sounds become vectors; layers become matrices; and learning becomes the search for good matrices. This page covers the ideas you need to read the rest of the map.

## Vectors as meaning

A vector $\mathbf{x} \in \mathbb{R}^d$ is a list of $d$ numbers, and also a point or direction in $d$-dimensional space. In AI, vectors are **representations**: an *embedding* of a word, a patch of an image, or the internal state of a network.

Two operations dominate:

- **Dot product**: $\mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i = \lVert \mathbf{x} \rVert \lVert \mathbf{y} \rVert \cos\theta$. It measures alignment and is the core of attention.
- **Norm**: $\lVert \mathbf{x} \rVert_2 = \sqrt{\mathbf{x}^\top \mathbf{x}}$, the length of the vector.

**Cosine similarity**, $\cos\theta = \frac{\mathbf{x}^\top \mathbf{y}}{\lVert \mathbf{x} \rVert \lVert \mathbf{y} \rVert}$, is the standard way to compare embeddings in retrieval systems.

## Matrices as transformations

A matrix $W \in \mathbb{R}^{m \times n}$ maps $\mathbb{R}^n \to \mathbb{R}^m$ by $\mathbf{y} = W\mathbf{x}$. Geometrically it rotates, stretches, projects, and shears space. A neural network layer is

```math
\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})
```

an **affine map** followed by a **nonlinearity** $\sigma$. Without $\sigma$, stacking layers would collapse into a single matrix, because $W_2(W_1 \mathbf{x}) = (W_2 W_1)\mathbf{x}$. Nonlinearity is what gives depth its power (see [Non-linearity and dynamical systems](../06-frontiers/nonlinearity-and-dynamical-systems.md)).

### Cost of a matrix multiply

Multiplying an $m \times k$ matrix by a $k \times n$ matrix takes $m \cdot n \cdot k$ multiply–accumulate operations, about $2mnk$ floating-point operations. This single fact explains a great deal about AI hardware: GPUs, TPUs, and photonic accelerators (see [Photonic computing](../06-frontiers/photonic-computing.md)) are all, at heart, machines for doing this quickly.

## Eigenvalues and eigenvectors

For a square matrix $A$, an eigenvector $\mathbf{v}$ satisfies $A\mathbf{v} = \lambda \mathbf{v}$: the matrix only scales it. Eigenvalues govern what happens when a transformation is applied repeatedly:

```math
A^t \mathbf{v} = \lambda^t \mathbf{v}
```

If $|\lambda| > 1$ the direction explodes; if $|\lambda| < 1$ it vanishes. This is exactly the **exploding and vanishing gradient** problem in deep and recurrent networks (Pascanu, Mikolov & Bengio 2013).

## Singular value decomposition (SVD)

Every matrix factors as

```math
W = U \Sigma V^\top
```

where $U$ and $V$ are orthogonal (rotations/reflections) and $\Sigma$ is diagonal with non-negative **singular values** $\sigma_1 \ge \sigma_2 \ge \dots$. Any linear map is therefore *rotate → scale along axes → rotate*.

Why it matters:

- **Low-rank approximation.** Keeping the top $r$ singular values gives the best rank-$r$ approximation in Frobenius and spectral norm (Eckart–Young theorem). This idea underlies **PCA** and **LoRA** fine-tuning (Hu et al. 2021), which learns weight updates of the form $\Delta W = BA$ with small inner dimension.
- **Conditioning.** The ratio $\sigma_{\max}/\sigma_{\min}$ measures how sensitive a system is to noise, which matters for numerical stability and for analog hardware.
- **Photonic hardware.** A unitary matrix can be realized as a mesh of interferometers (Reck et al. 1994; Clements et al. 2016), so the SVD is literally how some optical chips implement arbitrary weight matrices (Shen et al. 2017).

## Tensors

A **tensor** here simply means a multi-dimensional array. A batch of colour images is a 4-D tensor (batch × channels × height × width). Transformer activations are typically (batch × sequence length × model dimension). Frameworks such as PyTorch and JAX are tensor-algebra engines with automatic differentiation.

## Calculus with vectors

Training requires derivatives of a scalar loss $L$ with respect to vectors and matrices:

- The **gradient** $\nabla_{\mathbf{x}} L$ has the same shape as $\mathbf{x}$ and points in the direction of steepest ascent.
- The **Jacobian** $J = \partial \mathbf{y} / \partial \mathbf{x}$ collects derivatives of a vector-valued function.
- The **chain rule** multiplies Jacobians: $\frac{\partial L}{\partial \mathbf{x}} = J^\top \frac{\partial L}{\partial \mathbf{y}}$.

Backpropagation is the chain rule evaluated from the output backwards, reusing intermediate results (see [Neural networks and backpropagation](../02-deep-learning/neural-networks-and-backprop.md)).

## Key takeaways

- Representations are vectors; layers are matrices; similarity is a dot product.
- Depth requires nonlinearity; otherwise matrices simply compose.
- Eigenvalues explain stability; the SVD explains structure, compression, and even optical hardware.
- The cost of AI is dominated by matrix multiplication, which is why hardware is designed around it.
