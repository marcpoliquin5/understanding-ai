# Daily updates

Understanding AI is a living document. It is extended on a **daily cadence**: each day adds or deepens one focused piece of the map and records it in the [changelog](../CHANGELOG.md).

## What a daily update looks like

A daily update is small, sourced, and self-contained. Typical forms:

| Type | Example |
|---|---|
| **New page** | A new frontier topic such as optical nonlinearities for photonic neural networks |
| **Deepening** | A worked derivation added to an existing page (e.g. the softmax-attention gradient) |
| **Paper note** | A concise summary of an important paper, linked from the relevant section |
| **Correction** | Fixing an error, an outdated statement, or a broken link |
| **Diagram** | A Mermaid figure that makes a hard idea visual |

## Rules for every update

1. **Sourced.** New claims cite real papers or textbooks, added to [`REFERENCES.md`](../REFERENCES.md) with a working arXiv or DOI link.
2. **Honest about uncertainty.** Speculative ideas carry the *Speculative* callout.
3. **Logged.** One line in [`CHANGELOG.md`](../CHANGELOG.md) under the date (`YYYY-MM-DD`), newest first.
4. **Linked.** The new material is reachable from its section's `README.md` and from the contents table on the main [`README`](../README.md).
5. **Front page stays short.** The main README's *What's new* table holds the six most recent entries, one line each: the linked page title and a description of at most 20 words. Adding an entry removes the oldest. Numbers, findings and reference counts belong in the changelog, not the README. Each learning-path cell holds at most five links.

## Rotation

Updates rotate through the map so that no area goes stale:

```mermaid
flowchart LR
    F[00 Foundations] --> C[01 Classical ML] --> D[02 Deep learning] --> M[03 Modern AI]
    M --> S[04 Systems] --> A[05 Alignment & interpretability] --> X[06 Frontiers]
    X --> F
```

Frontier topics (recursion, self-similarity, nonlinear dynamics, photonic and neuromorphic computing, quantum ML, world models) receive extra attention because they change fastest.

## Backlog

The open backlog of planned topics lives at the bottom of the [changelog](../CHANGELOG.md#backlog).
