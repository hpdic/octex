# Contributing to octex

Thank you for helping. Every change keeps two things: the recall of the tests in `tests/` at their parameters, and
the index directory format of `docs/design.md` (a format change gets a new `format` number in `meta.json`).

## Development setup

```bash
git clone https://github.com/hpdic/octex.git && cd octex
python -m pip install -e ".[dev]"   # compiles the CUDA library (see docs/guide.md)
pre-commit install                  # ruff and clang-format on every commit
```

A GPU of compute capability 8.0 or newer is needed for the tests (`pytest -m gpu`); `pytest -m "not gpu"` runs the
checks that need no GPU. The recall test reads a SIFT1M index directory from `OCTEX_SIFT1M` and is skipped without
it. The C++ library and its example build with CMake from `cpp/`.

## Making a change

1. Open an issue first for anything beyond a small fix, so that the design can be agreed on before the code.
2. Keep the tests in `tests/` green, and add a test that would fail without your change.
3. A change to the search or the build states its effect on recall and speed, with the command that measured it and
   the GPU it ran on.
4. Add an entry under `[Unreleased]` in `CHANGELOG.md`.
5. Every source file starts with the copyright and license header of the existing files.

## Sign your commits (DCO)

octex uses the [Developer Certificate of Origin](https://developercertificate.org/). Sign every commit with
`git commit -s`, which adds a line `Signed-off-by: Your Name <your@email>` and certifies that you have the right to
submit the change under the Apache License 2.0.

## Pull requests

- One topic per pull request, with a description of what changed and how it was tested.
- CI must pass: lint and the checks without a GPU.

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
