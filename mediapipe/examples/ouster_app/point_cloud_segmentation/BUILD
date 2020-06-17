# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "point_cloud_segmentation",
    deps = [
        ":point_cloud_segmentation_cpu",
        "//mediapipe/examples/ouster_app/point_cloud_segmentation/graphs:point_cloud_segmentation_cpu_calculators",
    ],
)

cc_binary(
    name = "point_cloud_segmentation_cpu",
    srcs = ["point_cloud_segmentation_cpu.cc"],
    data = ["//mediapipe/calculators/tensorflow:test_saved_model_randlanet_v2"],
    linkstatic = 1,
    deps = [
        "//mediapipe/calculators/ouster:os1",
        "//mediapipe/calculators/ouster:os1_util",
        "@eigen",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/calculators/ouster:lidar_scan",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_features2d",
        "//mediapipe/framework/port:opencv_calib3d",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/examples/ouster_app/point_cloud_segmentation/graphs:point_cloud_segmentation_cpu_calculators",
    ],
)

