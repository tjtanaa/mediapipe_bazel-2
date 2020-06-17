exports_files(["_nearest_neighbor_ops.so"])

load(
    "//tensorflow/core/platform:default/build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_compatible_with",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")
tf_custom_op_library(
    name = "_nearest_neighbor_ops.so", 
    srcs = ["//mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/nearest_neighbor_kernels.cc",
        "//mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/ops/nearest_neighbor_ops.cc",
        "//mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/knn_.h",
        "//mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/knn_.cxx",],
)