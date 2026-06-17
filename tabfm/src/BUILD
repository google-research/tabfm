load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tabfm:__subpackages__"])

py_library(
    name = "model",
    srcs = [
        "__init__.py",
        "model.py",
    ],
    deps = [
        ":memory_efficient_attention",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//einops",
        "@pip//flax",
        "@pip//jax",
        "@pip//jaxtyping",
    ],
)

py_library(
    name = "checkpointing",
    srcs = ["checkpointing.py"],
    deps = [
        ":model",
        "@pip//flax",
        "@pip//importlib_resources",
        "@pip//orbax_checkpoint",
    ],
)

py_library(
    name = "classifier_and_regressor",
    srcs = ["classifier_and_regressor.py"],
    visibility = ["//tabfm:__subpackages__"],
    deps = [
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//jaxtyping",
        "@pip//numpy",
        "@pip//pandas",
        "@pip//scikit_learn",
    ],
)

py_library(
    name = "memory_efficient_attention",
    srcs = ["memory_efficient_attention.py"],
    deps = [
        "@pip//jax",
        "@pip//jaxtyping",
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    shard_count = 50,
    deps = [
        ":model",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//flax",
        "@pip//jax",
        "@pip//numpy",
        "@pip//parameterized",
    ],
)

py_test(
    name = "checkpointing_test",
    srcs = ["checkpointing_test.py"],
    deps = [
        ":checkpointing",
        ":model",
        "@pip//absl_py",
        "@pip//chex",
        "@pip//flax",
        "@pip//importlib_resources",
        "@pip//jax",
        "@pip//optax",
    ],
)

py_test(
    name = "classifier_and_regressor_test",
    srcs = ["classifier_and_regressor_test.py"],
    deps = [
        ":classifier_and_regressor",
        ":model",
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pandas",
    ],
)

py_library(
    name = "tabfm_v1_0_0",
    srcs = ["tabfm_v1_0_0.py"],
    visibility = ["//tabfm:__subpackages__"],
    deps = [
        ":checkpointing",
        ":model",
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//orbax_checkpoint",
    ],
)
