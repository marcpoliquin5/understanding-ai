# Alignment and safety

## The specification problem

An optimizer pursues the objective it is given, not the one you meant. Amodei et al. (2016) grouped common failure modes:

| Failure | Description |
|---|---|
| **Negative side effects** | Achieving the goal while disrupting things you did not mention |
| **Reward hacking** | Exploiting flaws in the reward signal instead of doing the task |
| **Scalable oversight** | Supervising tasks too complex or numerous for humans to check |
| **Safe exploration** | Avoiding catastrophic actions while learning |
| **Distributional shift** | Behaving well only on data like the training data |

This is an instance of **Goodhart's law**: when a measure becomes a target, it ceases to be a good measure.

## Learning from human preferences

RLHF (Christiano et al. 2017; Ouyang et al. 2022) and its variants (DPO, Rafailov et al. 2023; Constitutional AI, Bai et al. 2022) are the main practical alignment tools today (see [Large language models](../03-modern-ai/large-language-models.md)). Their known limits:

- The reward model is itself an imperfect proxy and can be over-optimized.
- Human raters can be fooled by confident or flattering answers, which can reward **sycophancy**.
- Preferences differ between people; whose preferences are encoded is a social question as much as a technical one.

## Scalable oversight

How can humans supervise systems that may exceed them on a task? Proposals include:

- **Debate** (Irving, Christiano & Amodei 2018): two models argue opposing sides, and a human judges; the hope is that pointing out flaws is easier than finding the truth unaided.
- **AI-assisted feedback**: models critique or check other models, as in Constitutional AI.
- **Weak-to-strong generalization** (Burns et al. 2023): an empirical analogy in which a weak model supervises a stronger one; strong students often outperform their weak supervisors, but do not recover their full capability.
- **Process supervision** (Lightman et al. 2023): reward correct reasoning steps rather than final answers only.

## Inner alignment

Even with a correct training objective, the learned model might internally pursue a different goal that merely correlated with the objective during training. Hubinger et al. (2019) call such learned optimizers **mesa-optimizers** and analyze when this could occur. **Hubinger et al. (2024)** ("Sleeper Agents") showed experimentally that deliberately trained backdoor behaviours in LLMs can persist through standard safety training, which suggests current techniques may not reliably remove unwanted behaviour once present.

## Adversarial robustness

Neural networks can be fooled by small, targeted perturbations. **Szegedy et al. (2013)** discovered adversarial examples for image classifiers; **Goodfellow, Shlens & Szegedy (2014)** introduced the fast gradient sign method,

```math
\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon \cdot \mathrm{sign}\big(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}, y)\big)
```

and argued that adversarial vulnerability stems largely from excessive linearity in high dimensions. For language models, **Zou et al. (2023)** found automatically optimized suffixes that elicit disallowed outputs and transfer across models. **Red teaming** with language models (Perez et al. 2022) automates the search for failures.

## Evaluation

Capabilities and risks are measured with benchmark suites, red-team exercises, and *dangerous-capability* evaluations. Good evaluations are:

- **Checkable**: outcomes can be verified, not just judged by impression;
- **Uncontaminated**: test items are not in the training data;
- **Elicitation-aware**: they try hard to bring out the model's best performance, since under-elicitation underestimates risk.

## AI control

**Greenblatt et al. (2024)** proposed *AI control*: designing deployment protocols (monitoring, trusted editing, restricted permissions) that remain safe even if the model is trying to subvert them, and evaluating those protocols against a deliberately adversarial red team. Control complements alignment: one aims to make the model want the right thing; the other limits the damage if it does not.

Practical agent runtimes apply the same philosophy at a smaller scale: hard budgets, least-privilege tools, human approval for irreversible actions, and audit logs. See the [agents page](../03-modern-ai/retrieval-tools-and-agents.md#making-agents-dependable) for how these ideas are applied in practice.

## Key takeaways

- Alignment failures are usually specification or generalization failures, not malice.
- Preference-based training works but inherits the limits of its feedback.
- Robustness, evaluation, and control are complementary layers of defence.
