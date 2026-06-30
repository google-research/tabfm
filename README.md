# TabFM: Tabular Foundation Models

TabFM (Tabular Foundation Model) is a scikit-learn compatible tabular foundation model. It allows you to perform zero-shot classification and regression on tabular datasets with mixed column types out-of-the-box.

At inference time, TabFM does not require training parameters on your dataset; instead, it leverages in-context learning by reading your training data as "context" to make instant predictions on new test samples.

*This is not an officially supported Google product.*

---

## Installation

To install TabFM, clone the repository and install it locally with the backend of your choice:

**JAX (CPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[jax]
```

**JAX (GPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[jax,cuda]
```

**PyTorch (CPU/GPU):**
```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[pytorch]
```
*Note: For PyTorch with GPU support, ensure you have the appropriate PyTorch version installed for your CUDA version before installing TabFM.*

### Requirements
For a complete list of pinned dependencies and versions, please see [requirements.txt](requirements.txt). The core requirements depend on the backend you choose:
*   Python >= 3.11
*   Hugging Face Hub (for downloading pre-trained weights)
*   **JAX Backend:**
    *   JAX (specifically `jax==0.10.1`)
    *   Flax (specifically `flax==0.12.7`, using the modern `flax.nnx` API)
*   **PyTorch Backend:**
    *   PyTorch (specifically `torch==2.12.1+cpu` or a GPU version)

---

## Quick Start (TabFM v1.0.0)

We provide pre-trained weights for the **TabFM v1.0.0** release. The library handles downloading and loading these weights automatically. You can choose to load the model using either the JAX or PyTorch backend.

### 1. Classification Example

```python
import numpy as np
import pandas as pd
from tabfm import TabFMClassifier

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# Initialize scikit-learn compatible classifier (works with either backend model)
clf = TabFMClassifier(model=model)

# Prepare your dataset (supports mixed numerical and categorical features)
X_train = pd.DataFrame({
    "age": [25.0, 45.0, 35.0, 50.0],
    "job": ["engineer", "manager", "engineer", "manager"],
    "income": [80000, 120000, 90000, 130000]
})
y_train = np.array(["low_risk", "high_risk", "low_risk", "high_risk"])

X_test = pd.DataFrame({
    "age": [30.0, 48.0],
    "job": ["engineer", "manager"],
    "income": [85000, 125000]
})

# Fit classifier (prepares ordinal encoders and numerical scalers)
clf.fit(X_train, y_train)

# Predict classes and probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Predictions:", predictions)
print("Class Probabilities:\n", probabilities)
```

### 2. Regression Example

```python
import numpy as np
import pandas as pd
from tabfm import TabFMRegressor

# Choose your backend:

# OPTION A: JAX Backend
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
# from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# Initialize scikit-learn compatible regressor (works with either backend model)
reg = TabFMRegressor(model=model)

# Prepare your dataset
X_train = pd.DataFrame({
    "sqft": [1200, 2500, 1500, 3000],
    "neighborhood": ["A", "B", "A", "C"]
})
y_train = np.array([250000, 550000, 310000, 620000])

X_test = pd.DataFrame({
    "sqft": [1800, 2800],
    "neighborhood": ["A", "B"]
})

# Fit and Predict
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print("Predicted Prices:", predictions)
```

---

## Examples Directory

You can find runnable scripts for both classification and regression under the [examples/](examples/) folder:
*   [classification_example.py](examples/classification_example.py)
*   [regression_example.py](examples/regression_example.py)

To run them, simply execute:
```bash
python examples/classification_example.py
```
*(You can edit these files to switch between JAX and PyTorch backends as shown in the comments inside them).*

---

## Evaluation Results

Our model evaluation results can be found in [results/](results/).

---

## Running Tests

You can run the unit tests directly using Python's `unittest` module:

```bash
# Run all tests (requires both JAX and PyTorch installed)
PYTHONPATH=. python3 -m unittest discover -s tabfm/src/ -p "*_test.py"

# Or run specific test files:
PYTHONPATH=. python3 -m unittest tabfm/src/pytorch/model_test.py
PYTHONPATH=. python3 -m unittest tabfm/src/classifier_and_regressor_pytorch_test.py
```

Alternatively, if you have Bazel installed, you can run tests with:
```bash
bazel test //...
```
