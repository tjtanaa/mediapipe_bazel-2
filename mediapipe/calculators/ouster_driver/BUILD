cc_library(
    name = "ouster_viz_plugin",
    srcs = ["src/point_viz.cpp",
            "src/cloud.cpp",
            "src/camera.cpp",
            "src/image.cpp",
            "include/ouster/autoexposure.h",
            "include/ouster/beam_uniformity.h",
            "include/ouster/colormaps.h",
            "include/ouster/lidar_scan_viz.h",
            "include/ouster/point_viz.h"
            ],
    visibility = ["//visibility:public"],  
    deps = [
        "@eigen",
    ],
    alwayslink = 1,
)
 
                              

cc_binary(
    name = "ouster_viz",
    srcs = ["src/main.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/ouster_driver:ouster_viz_plugin",
        "//mediapipe/calculators/ouster_driver/ouster_client:ouster_plugin",
        "@com_github_open_source_parsers_jsoncpp//:jsoncpp",
    ],
    linkopts = [ "-lpthread" ]
)