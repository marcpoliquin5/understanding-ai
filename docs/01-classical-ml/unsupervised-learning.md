# Unsupervised learning

Without labels, the goal is to discover **structure**: groups, low-dimensional shape, or the distribution that generated the data.

## Clustering

**k-means** (Lloyd 1982) alternates two steps until convergence:

1. Assign each point to its nearest centroid.
2. Move each centroid to the mean of its assigned points.

It minimizes within-cluster squared distance, finds only a local optimum, and assumes roughly spherical clusters. Hierarchical clustering and density-based methods (such as DBSCAN) relax those assumptions.

## Mixture models and EM

A **Gaussian mixture model** assumes each point comes from one of $K$ Gaussians:

```math
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}; \boldsymbol\mu_k, \Sigma_k)
```

The **Expectation–Maximization** algorithm (Dempster, Laird & Rubin 1977) alternates between computing soft assignments (E-step) and re-estimating parameters (M-step), increasing the likelihood at every iteration. k-means is a hard-assignment limit of EM for GMMs. The idea of latent variables optimized through a lower bound reappears in [variational autoencoders](../03-modern-ai/generative-models.md).

## Principal component analysis

PCA (Pearson 1901) finds orthogonal directions of maximum variance. Equivalently, it takes the top singular vectors of the centred data matrix (see [SVD](../00-foundations/linear-algebra.md#singular-value-decomposition-svd)). Projecting onto the first few components compresses data while keeping as much variance as possible.

## Manifold learning and visualization

Real data often lies near a low-dimensional, curved **manifold** inside high-dimensional space. Nonlinear methods such as **t-SNE** (van der Maaten & Hinton 2008) and **UMAP** (McInnes, Healy & Melville 2018) produce 2-D maps that preserve local neighbourhoods. They are excellent for visualizing embeddings but distort global distances, so cluster sizes and gaps in their plots should not be over-interpreted.

The **manifold hypothesis**, that natural data concentrates near low-dimensional manifolds, is also one proposed explanation for [neural scaling laws](../06-frontiers/fractal-and-self-similar-structure.md).

## Density estimation and anomaly detection

Modelling $p(\mathbf{x})$ directly lets you flag inputs with low probability as anomalies. Classical tools include kernel density estimation; modern ones include normalizing flows, autoregressive models, and diffusion models.

## From unsupervised to self-supervised

Modern representation learning creates supervised-looking tasks from unlabelled data:

| Pretext task | Example |
|---|---|
| Predict the next token | GPT-style language models |
| Predict masked tokens | BERT (Devlin et al. 2018) |
| Match two augmented views | Contrastive learning such as SimCLR (Chen et al. 2020) |
| Match images and captions | CLIP (Radford et al. 2021) |
| Predict masked representations | JEPA-style models (Assran et al. 2023) |

This shift, from hand-labelled targets to targets made from the data itself, is what allowed models to train on internet-scale corpora.
