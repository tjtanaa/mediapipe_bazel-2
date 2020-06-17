
exports_files(["lidar_scan.h"])
package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "jsoncpp",
    srcs = [
        "include/json/assertions.h",
        "src/lib_json/json_reader.cpp",
        "src/lib_json/json_tool.h",
        "src/lib_json/json_value.cpp",
        "src/lib_json/json_writer.cpp",
    ],
    hdrs = [
        "include/json/autolink.h",
        "include/json/config.h",
        "include/json/features.h",
        "include/json/forwards.h",
        "include/json/json.h",
        "include/json/reader.h",
        "include/json/value.h",
        "include/json/version.h",
        "include/json/writer.h",
    ],
    copts = [
        "-DJSON_USE_EXCEPTION=0",
        "-DJSON_HAS_INT64",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":private"],
)

cc_library(
    name = "private",
    textual_hdrs = ["src/lib_json/json_valueiterator.inl"],
)

cc_library(
    name = "lidar_scan",
    srcs = ["lidar_scan.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "os1",
    srcs = ["os1.cc",
            "os1.h",
            "os1_packet.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_library(
    name = "os1_util",
    srcs = ["os1_util.cc",
            "os1_packet.h",
            "os1_util.h"
            ],
    visibility = ["//visibility:public"],  
    deps = [
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ouster_client",
    srcs = ["ouster_client.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/ouster:os1",
        "//mediapipe/calculators/ouster:os1_util",
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ]
)
cc_test(
    name = "ouster_lidar_packet_generator_test",
    srcs = ["ouster_lidar_packet_generator_test.cc",
            "lidar_scan.h"
            ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/ouster:os1",
        "//mediapipe/calculators/ouster:os1_util",
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
        "@eigen",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    linkopts = [ "-lpthread" ],
)
