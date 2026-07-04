# CLAUDE.md

Guidance for AI coding agents working in this repository.

## Project overview

TabFM is a scikit-learn compatible tabular foundation model that performs
zero-shot classification and regression via in-context learning. The reference
implementation is JAX/Flax (nnx); a numerically parity-verified PyTorch port
exists alongside it. Pre-trained v1.0.0 weights are downloaded from Hugging
Face Hub at load time.

## Repository layout

```
tabfm/__init__.py                      # Public API; guarded backend imports
tabfm/src/classifier_and_regressor.py  # sklearn wrappers + preprocessing + backend dispatch
tabfm/src/jax/                         # Reference backend (model.py, tabfm_v1_0_0.py, ...)
tabfm/src/pytorch/                     # PyTorch port (model.py, tabfm_v1_0_0.py)
tabfm/src/hugging_face/                # Weight conversion / upload utilities
examples/                              # Runnable end-to-end examples
conftest.py                            # Skips backend tests when the backend is not installed
```

## Coding style

Google Python style, enforced by `pyink` (see `[tool.pyink]` in
`pyproject.toml`):

- **2-space indentation**, 80-column lines, majority double quotes.
- Every `.py` file starts with the Apache 2.0 license header
  (`# Copyright 2026 Google LLC ...`) — copy it verbatim from any existing file.
- Module docstring after the header describing the module's purpose.
- Google-style docstrings with `Args:` / `Returns:` sections on public
  functions and methods. Classes document public attributes under
  `Attributes:`; sklearn fitted attributes use the trailing-underscore
  convention (`categories_`, `tfm_`) and are also declared as class-level type
  annotations.
- Type hints use `typing` (`Optional`, `List`, `Dict`, `Union`, `Any`).
  Shape-checked signatures in `classifier_and_regressor.py` use `jaxtyping`
  annotations with the `@jt.typed` decorator.
- Logging via `absl.logging`, not `print` (except `verbose=True` user-facing
  output).
- sklearn-style `X` / `y` capitalization is allowed
  (`# pylint: disable=invalid-name`).
- Comments explain **why**, not what — especially numerical-precision
  rationale (fp32 upcasts, JAX parity). Keep that density when editing model
  code.

## Backend pattern

Each compute backend lives in `tabfm/src/<backend>/` with the same two files:

- `model.py` — the architecture. Module and parameter names **mirror the JAX
  model** so weight conversion is mechanical (`cell_embedder`, `col_embedder`,
  `row_interactor`, `icl_predictor`, `q_proj`, `per_dim_scale`, ...).
- `tabfm_v1_0_0.py` — a `load(model_type, checkpoint_path, ...)` function that
  downloads pre-trained weights from Hugging Face Hub, with a process-wide
  cache keyed on the load arguments.

Integration points when adding a backend:

1. `tabfm/__init__.py`: `try/except ImportError` guarded import exposing
   `tabfm_v1_0_0_<backend>`.
2. `classifier_and_regressor.py`: `HAS_<BACKEND>` flag from a guarded import;
   an `isinstance` check in both `_batch_forward` methods (classifier and
   regressor); a `_predict_step_<backend>` helper that takes numpy in and
   returns numpy out.
3. `pyproject.toml`: an optional-dependency extra named after the backend.
4. `conftest.py`: add the backend's test files to `collect_ignore` when the
   backend (or a parity-test dependency) is not installed.
5. `README.md`: installation + quick-start option for the backend.

## Numerical fidelity rules

The checkpoint is float32; the model is designed to run in **bfloat16** with
targeted fp32 upcasts. When porting or editing model code, preserve:

- RMSNorm: normalize entirely in float32, cast back at the end.
- Fourier feature expansion: `sin`/`cos` computed in float32.
- PerDimScale: softplus in float32, then cast to compute dtype.
- Attention: SDPA with `scale=1.0` (scaling is folded into PerDimScale);
  q/k RMSNorm after RoPE.
- RoPE inverse frequencies are **loaded from the checkpoint**, not recomputed.
- Model entry: `nan_to_num(x, nan=-100.0)` then cast to compute dtype.

New-backend outputs must match the reference within `1e-4` max abs diff in
float32 (see the parity tests in `tabfm/src/pytorch/model_test.py`).

## Tests

- `unittest.TestCase` classes in `*_test.py` files colocated with the source,
  ending with `if __name__ == "__main__": unittest.main()`.
- Parity tests instantiate small random-init configs of two backends, convert
  weights, and assert max abs diff < 1e-4.
- sklearn integration tests (`classifier_and_regressor_<backend>_test.py`)
  run `fit`/`predict`/`predict_proba` on tiny random data with a small
  random-init model.
- Run: `pytest -vv -n auto` from the repo root (CI uses Python 3.11 with
  `pip install -e .[dev,jax,pytorch]`).
- Tests must not require network access or pre-trained weights.

## Releases / docs

- `CHANGELOG.md` follows keepachangelog.com; bumping `__version__` in
  `tabfm/__init__.py` triggers the PyPI auto-publish workflow — do not bump it
  in feature PRs.
- `requirements.txt` is a pip-compile lock for the full reproducible
  environment; `pyproject.toml` constrains only known incompatibilities.
