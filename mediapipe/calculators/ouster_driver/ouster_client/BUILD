cc_library(
    name = "ouster_plugin",
    srcs = ["src/compat.cpp",
            "src/lidar_scan.cpp",
            "src/types.cpp",
            "src/client.cpp",
            "include/ouster/client.h",
            "include/ouster/compat.h",
            "include/ouster/lidar_scan.h",
            "include/ouster/packet.h",
            "include/ouster/types.h",
            "include/ouster/version.h",
            "include/ouster/impl/client_impl.h",
            "include/ouster/impl/packet_impl.h",
            ],
    visibility = ["//visibility:public"],  
    deps = [
        "@eigen",
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ],
    alwayslink = 1,
)


cc_binary(
    name = "ouster_client",
    srcs = ["src/main.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/ouster_driver/ouster_client:ouster_plugin",
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ]
)