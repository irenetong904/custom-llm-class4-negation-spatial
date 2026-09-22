# Building a Custom LLM with nanoGPT

Class 4, Fall 26 · From Zero to AI Agents — **Irene Tong's completed submission**

This repository is a fork of the course's
[custom-llm starter template](https://github.com/pepealonso95/custom-llm), which
trains Karpathy's nanoGPT transformer from scratch on a small word-token corpus.

> **All graded content — executed experiments, evidence, and explanations — is in
> [STUDENT_README.md](STUDENT_README.md). Start there.**

The rest of this page is just a map of the repository; it intentionally does not
repeat the starter template's own generic instructions or example numbers, so nothing
here should be read as part of the submission itself.

## Repository layout

- [`STUDENT_README.md`](STUDENT_README.md) — the graded write-up (start here)
- [`experiment1_starter.ipynb`](experiment1_starter.ipynb),
  [`experiment2_expanded.ipynb`](experiment2_expanded.ipynb) — the two executed
  notebooks (starter corpus only; starter + negation + spatial-relations extension)
- `corpus/` — my original teaching sentences:
  [`negation.txt`](corpus/negation.txt), [`spatial_relations.txt`](corpus/spatial_relations.txt)
  (see [`corpus/README.md`](corpus/README.md) for how the corpus loader works)
- `llm_runs/` — full evidence for both experiments: configs, loss history, samples,
  token/embedding/gradient inspection, eval results, chat transcripts, model weights
- `evals/` — the unchanged 48-case fixed eval suite and its
  [runner/scoring guide](evals/README.md)
- [`chat.py`](chat.py) — standalone terminal chat interface for a saved model
- [`ASSIGNMENT.md`](ASSIGNMENT.md) — the original assignment text
- [`embedding-viewer.html`](embedding-viewer.html) — offline 3D viewer for a run's
  `checkpoint.json`

## Credit

The starter template, nanoGPT integration, corpus/eval tooling, and classroom
scaffolding are from [pepealonso95/custom-llm](https://github.com/pepealonso95/custom-llm) —
see that repository for the generic setup/usage instructions this fork doesn't repeat.
[`nanogpt_model.py`](nanogpt_model.py) is an unchanged copy of Karpathy's
[nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py)
([MIT license](NANOGPT_LICENSE)).
