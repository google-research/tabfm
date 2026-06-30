load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tabfm:__subpackages__"])

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "@pip//torch",
    ],
)

py_library(
    name = "tabfm_v1_0_0",
    srcs = ["tabfm_v1_0_0.py"],
    deps = [
        ":model",
        "@pip//absl_py",
        "@pip//torch",
        "@pip//huggingface_hub",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":model",
        "//tabfm/src/jax:model",
        "//tabfm/src:torch_convert",
        "@pip//numpy",
        "@pip//torch",
        "@pip//flax",
        "@pip//jax",
    ],
)
