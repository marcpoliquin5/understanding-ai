# Probability and information

Learning is inference under uncertainty. Probability gives the language for it, and information theory gives the yardstick. Nearly every loss function used to train modern models is a probabilistic quantity in disguise.

## Random variables and distributions

A random variable $X$ has a distribution $p(x)$. For a discrete variable $\sum_x p(x) = 1$; for a continuous one $\int p(x)\,dx = 1$.

- **Expectation**: $\mathbb{E}[f(X)] = \sum_x p(x) f(x)$
- **Variance**: $\mathrm{Var}[X] = \mathbb{E}[(X - \mathbb{E}X)^2]$
- **Gaussian**: $\mathcal{N}(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

## Bayes' rule

```math
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}
```

The **posterior** over parameters $\theta$ combines the **likelihood** of the data with a **prior**. Bayesian machine learning keeps the whole posterior; most deep learning keeps a single point estimate.

## Maximum likelihood

Given data $\mathcal{D} = \{x_1, \dots, x_N\}$ and a model $p_\theta$, **maximum likelihood estimation (MLE)** chooses

```math
\hat\theta = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(x_i)
```

Two familiar losses fall out immediately:

| Assumed noise model | Negative log-likelihood becomes |
|---|---|
| Gaussian output with fixed variance | **Mean squared error** |
| Categorical output (softmax) | **Cross-entropy loss** |

Adding a Gaussian prior on $\theta$ and maximizing the posterior (**MAP**) yields **L2 regularization / weight decay**; a Laplace prior yields L1 (the lasso, Tibshirani 1996).

## The softmax

To turn a vector of scores (logits) $\mathbf{z}$ into probabilities:

```math
\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

A language model outputs a softmax over its vocabulary at each step, and attention uses a softmax over positions.

## Entropy, cross-entropy, and KL divergence

Claude Shannon (1948) defined the **entropy** of a distribution, the average number of bits needed to encode outcomes:

```math
H(p) = -\sum_x p(x) \log_2 p(x)
```

The **cross-entropy** of a model $q$ against the true distribution $p$ is

```math
H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q)
```

where the **Kullback–Leibler divergence** (Kullback & Leibler 1951)

```math
D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \;\ge\; 0
```

measures how many extra bits you pay for using $q$ instead of $p$. Because $H(p)$ is fixed, **minimizing cross-entropy is the same as minimizing KL divergence to the data distribution**, which is the same as maximum likelihood.

This is the deepest idea in language modelling: training a model to predict the next token *is* training it to compress text. A language model's **perplexity** is $2^{H}$ (with $H$ the per-token cross-entropy in bits), the effective number of choices the model is uncertain between.

KL divergence also appears as a regularizer in **VAEs** (Kingma & Welling 2013) and as a leash keeping RLHF-tuned models close to their pretrained base (Ouyang et al. 2022).

## Mutual information

```math
I(X; Y) = D_{\mathrm{KL}}\big(p(x, y) \,\Vert\, p(x)p(y)\big)
```

How much knowing $Y$ reduces uncertainty about $X$. Contrastive representation learning methods are often motivated as maximizing (bounds on) mutual information between views of the same data.

## Sampling

Generative models produce outputs by sampling. For language models, common strategies modify the softmax:

- **Temperature** $T$: use $\mathrm{softmax}(\mathbf{z}/T)$. $T<1$ sharpens, $T>1$ flattens.
- **Top-k**: sample only from the $k$ most likely tokens.
- **Nucleus (top-p)** sampling (Holtzman et al. 2019): sample from the smallest set whose cumulative probability exceeds $p$.

## Key takeaways

- Most training losses are negative log-likelihoods.
- Cross-entropy = entropy + KL; predicting well means compressing well.
- Regularization can be read as a prior.
- Sampling strategy changes the character of generated output without retraining.
