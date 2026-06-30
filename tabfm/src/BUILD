load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tabfm:__subpackages__"])

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

py_test(
    name = "classifier_and_regressor_test",
    srcs = ["classifier_and_regressor_test.py"],
    deps = [
        ":classifier_and_regressor",
        "//tabfm/src/jax:model",
        "@pip//absl_py",
        "@pip//flax",
        "@pip//jax",
        "@pip//numpy",
        "@pip//pandas",
    ],
)

py_library(
    name = "torch_convert",
    srcs = ["hugging_face/torch_convert.py"],
    deps = [
        "//tabfm/src/jax:model",
        "//tabfm/src/pytorch:model",
        "@pip//numpy",
        "@pip//torch",
        "@pip//flax",
        "@pip//jax",
    ],
)
