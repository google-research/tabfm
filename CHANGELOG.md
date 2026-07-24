# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/tabfm/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

### Added

* `TabFMDataGenerator`: synthetic tabular data generation. Samples new rows
  mimicking a reference dataset via chain-rule factorization over columns,
  with `TabFMClassifier` providing each conditional distribution. Numerical
  columns use hierarchical quantile refinement (the fine-bin index is sampled
  digit by digit, `n_bins ** n_levels` effective bins — 100 by default);
  sampling is temperature-controlled with seeded or user-supplied column
  orders.

## [1.0.1] - 2026-07-09

### Fixed

* PyTorch weight loading: load `model.safetensors` via `PyTorchModelHubMixin`.
  The `1.0.0` loader looked for `pytorch_model.bin`, which the Hugging Face
  checkpoint no longer provides, so `load()` raised `FileNotFoundError`.
* `EnsembleGenerator` no longer re-transforms the full training set on every
  `predict` call (prediction cost now scales with query size, not context size).
* Query-axis mask/bias collapse in the memory-efficient (FLASH) JAX attention.
* `predict` on multi-device hosts no longer crashes (IndivisibleError / device
  mismatch).
* `TabFMRegressor.predict` before `fit` now raises `NotFittedError`.
* `TabFMClassifier.predict` no longer returns object-dtype labels.
* README regression example now loads the regression checkpoint.
* Loading a checkpoint whose type does not match the estimator now fails fast
  with an actionable error, instead of a cryptic squeeze error (classification
  weights in `TabFMRegressor`) or silently wrong predictions (regression
  weights in `TabFMClassifier`).
* The sklearn layer handles duplicate and non-string column names: duplicates
  fail fast with a clear message, and datetime columns with integer labels no
  longer crash.
* Fitted estimators and the PyTorch model are now picklable, which AutoGluon /
  TabArena require to save a fitted model. The PyTorch gelu activation is a
  module-level function, and the JAX estimators drop their memoized compiled
  predict functions on pickle and rebuild them lazily.

### Changed

* PyTorch model runs in bfloat16 by default, matching the JAX compute dtype.
* Activation chunking is enabled by default to bound peak memory on large tasks.
* JAX and PyTorch models gained Hugging Face Hub support (`from_pretrained` /
  `save_pretrained`); weight downloads are narrowed to the requested model type.

## [1.0.0] - 2026-06-29

* Initial release

[1.0.1]: https://github.com/google-research/tabfm/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/google-research/tabfm/releases/tag/v1.0.0
