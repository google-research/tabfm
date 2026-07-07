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

* Added an MLX backend (`tabfm/src/mlx/`, `pip install -e .[mlx]`) for native
  Apple-silicon inference. It reuses the PyTorch v1.0.0 weight release
  (identical parameter names/layouts) and is parity-tested against the PyTorch
  port to < 1e-4 max abs diff in float32.
* Fixed the `pytorch` extra missing `safetensors`: with a bare `torch`
  install, `tabfm_v1_0_0_pytorch.load()` raised `NameError` inside
  `PyTorchModelHubMixin` when loading the safetensors release.

## [1.0.0] - 2026-06-29

* Initial release

[Unreleased]: https://github.com/google-research/tabfm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/google-research/tabfm/releases/tag/v1.0.0
