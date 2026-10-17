# Contributing to Understanding AI

Thank you for helping make this a trustworthy map of artificial intelligence. The project has one non-negotiable rule:

> **Every claim must be traceable.** If a sentence states a result, a number, a date, or attributes an idea to someone, it must be backed by a real, checkable source listed in [`REFERENCES.md`](REFERENCES.md).

## Editorial standards

1. **Accuracy over coverage.** A short, correct explanation beats a long, vague one.
2. **Cite primary sources.** Prefer the original paper (with arXiv ID or DOI). Use surveys or textbooks for background.
3. **Never invent.** No fabricated references, benchmark numbers, quotes, or dates. If you cannot find a source, leave the claim out.
4. **Label speculation.** Frontier ideas that are not established are marked with a callout:

   > [!WARNING]
   > **Speculative.** This section discusses open hypotheses, not established results.

5. **Separate "what is known" from "what is believed".** Use phrases such as *"empirically observed"*, *"conjectured"*, or *"disputed"* precisely.
6. **Explain intuition, then formalism.** Lead with a plain-language picture, then give the equation.
7. **Date-sensitive content.** Anything that changes quickly (model releases, records, hardware) should state the year it refers to.

## Style

- Markdown, rendered by GitHub. Display math uses fenced ` ```math ` blocks; inline math uses `$...$`.
- Diagrams use [Mermaid](https://mermaid.js.org/) fenced blocks.
- One topic per file; each folder has a `README.md` that indexes it.
- Cross-link generously with relative links.
- Citation format in text: *Author et al. (Year)* linked to the entry in `REFERENCES.md` or directly to the arXiv/DOI page.

## Adding a reference

Add the entry to the right section of [`REFERENCES.md`](REFERENCES.md) in this form:

```text
- Surname, A., Surname, B. (Year). *Title*. Venue. [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
```

Before opening a pull request, open the link and confirm the title and authors match.

## Workflow

1. Fork and branch (`git checkout -b topic/short-name`).
2. Make the change and add a line to [`CHANGELOG.md`](CHANGELOG.md) under today's date.
3. Open a pull request describing what changed and which sources support it.

## License

By contributing, you agree that your contributions are licensed under [CC BY 4.0](LICENSE).
