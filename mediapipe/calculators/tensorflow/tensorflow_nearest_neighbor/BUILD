licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"], features = ["-layering_check"],)

cc_library(
    name = "_nearest_neighbor_ops.so",
    visibility = ["//visibility:public"],
    srcs = [
        "cc/kernels/nearest_neighbor_kernels.cc",
        "cc/ops/nearest_neighbor_ops.cc",
        "cc/kernels/knn_.h",
        "cc/kernels/knn_.cxx",
           ] ,
    copts = ["-std=c++11"],
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib",
            "@org_tensorflow//tensorflow/core/util/ctc",
            "@org_tensorflow//third_party/eigen3",
    ],
)