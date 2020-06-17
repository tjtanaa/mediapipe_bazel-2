// Copyright 2018 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_saved_model_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"
#include "mediapipe/framework/tool/validate_type.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_features2d_inc.h"
#include "mediapipe/framework/port/opencv_calib3d_inc.h"

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include "mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/knn_.h"
#include <iterator>
// #include "tensorflow/c/c_api.h"
#include "mediapipe/calculators/tensorflow/point_cloud_to_randlanet_format_calculator_options.pb.h"

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/hal/interface.h>

// #include <opencv2/core/version.hpp>
// #include <opencv2/opencv.hpp>
// #ifdef CV_VERSION_EPOCH  // for OpenCV 2.x
// #include <opencv2/core/core.hpp>
// // #else
// #include <opencv2/cvconfig.h>

// #include <opencv2/core.hpp>
// #endif
using namespace cv;

// filegroup(
//    name = "test_saved_model",
//    srcs = [
//        "testdata/tensorflow_saved_model/00000000/saved_model.pb",
//        "testdata/tensorflow_saved_model/00000000/variables/variables.data-00000-of-00001",
//        "testdata/tensorflow_saved_model/00000000/variables/variables.index",
//    ],
//)

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

std::string GetSavedModelDir() {
  std::string out_path =
      file::JoinPath("./", "mediapipe/calculators/tensorflow/testdata/",
                     "tensorflow_saved_model/RandLA-Net_builder_v2");
      // file::JoinPath("home/tan/tjtanaa/", "cifar10_eval_builder");
  return out_path;
}

class TensorFlowSessionFromSavedModelCalculatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    extendable_options_.Clear();
    options_ = extendable_options_.MutableExtension(
        TensorFlowSessionFromSavedModelCalculatorOptions::ext);
    options_->set_saved_model_path(GetSavedModelDir());
  }

  CalculatorOptions extendable_options_;
  TensorFlowSessionFromSavedModelCalculatorOptions* options_;
};

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       CreatesPacketWithGraphAndBindings) {


  // std::ifstream ifile; 
  // ifile.open("/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/python/ops/_nearest_neighbor_ops.so");
  // if(ifile) {
  //     std::cout<<"file exists"<< std::endl;
  // } else {
  //     std::cout<<"file doesn't exist" << std::endl;
  // }
	// /* A list of possible environment variables*/
	//  const char *env_var[6] = {"PUBLIC","HOME","SESSIONNAME","LIB","SystemDrive", "LD_LIBRARY_PATH"};
	//  char *env_val[6];

	// for(int i=0; i<6; i++)
	// {
	//  	/* Getting environment value if exists */
	//  	env_val[i] = std::getenv(env_var[i]);
	//  	if (env_val[i] != NULL)
	//  		std::cout << "Variable = " << env_var[i] << ", Value= " << env_val[i] << std::endl;
	//  	else
	//  		std::cout << env_var[i] << " doesn't exist" << std::endl;
	// }
  // TF_Status* status = TF_NewStatus();
  // const char* cname = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/python/ops/_nearest_neighbor_ops.so";
  // TF_Library* h = TF_LoadLibrary(cname, status);
  // std::cout << "Lib loaded: " <<  (TF_GetCode(status) == TF_OK) << std::endl;
  // if (TF_GetCode(status) != TF_OK) {
  //   // Log/fail, details in 
  //   std::cout << TF_Message(status) << std::endl;
  // }
  


  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        })",
                                           options_->DebugString()));
  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  // Session must be set.
  ASSERT_NE(session.session, nullptr);

  // Bindings are inserted.
  EXPECT_EQ(session.tag_to_tensor_map.size(), 22);

  // Display the tag that the serve session has:
  // Essential for debugging
  for(auto it = session.tag_to_tensor_map.cbegin(); it != session.tag_to_tensor_map.cend(); ++it)
  {
      std::cout << it->first << " " << it->second << "\n";
  }
  // For some reason, EXPECT_EQ and EXPECT_NE are not working with iterators.
  EXPECT_FALSE(session.tag_to_tensor_map.find("NEIGHBOR_INDEX_0") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("NEIGHBOR_INDEX_1") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("NEIGHBOR_INDEX_2") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("NEIGHBOR_INDEX_3") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("NEIGHBOR_INDEX_4") ==
               session.tag_to_tensor_map.end());
  // EXPECT_FALSE(session.tag_to_tensor_map.find("SUBPOINTS_0") ==
  //              session.tag_to_tensor_map.end());
  // EXPECT_FALSE(session.tag_to_tensor_map.find("SUBPOINTS_1") ==
  //              session.tag_to_tensor_map.end());
  // EXPECT_FALSE(session.tag_to_tensor_map.find("SUBPOINTS_2") ==
  //              session.tag_to_tensor_map.end());
  // EXPECT_FALSE(session.tag_to_tensor_map.find("SUBPOINTS_3") ==
  //              session.tag_to_tensor_map.end());
  // EXPECT_FALSE(session.tag_to_tensor_map.find("SUBPOINTS_4") ==
  //              session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("POOL_I_0") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("POOL_I_1") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("POOL_I_2") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("POOL_I_3") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("POOL_I_4") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("UP_I_0") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("UP_I_1") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("UP_I_2") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("UP_I_3") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("UP_I_4") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("BATCH_XYZ_0") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("BATCH_XYZ_1") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("BATCH_XYZ_2") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("BATCH_XYZ_3") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("BATCH_XYZ_4") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("BATCH_FEATURE") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("PROB_LOGITS") ==
               session.tag_to_tensor_map.end());
  // Sanity: find() actually returns a reference to end() if element not
  // found.
  EXPECT_TRUE(session.tag_to_tensor_map.find("Z") ==
              session.tag_to_tensor_map.end());

  // Use these saved_model_cli command to check for the name of the input and output of the saved_model
  // e.g. /home/tan/anaconda3/envs/saved_model_cli/bin/saved_model_cli <dir of the saved_model>
  // --tag_set serve --signature_def serving_default
  EXPECT_EQ(session.tag_to_tensor_map.at("BATCH_XYZ_0"), "Batch_XYZ_0:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("BATCH_XYZ_1"), "Batch_XYZ_1:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("BATCH_XYZ_2"), "Batch_XYZ_2:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("BATCH_XYZ_3"), "Batch_XYZ_3:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("BATCH_XYZ_4"), "Batch_XYZ_4:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("NEIGHBOR_INDEX_0"), "Neighbor_Index_0:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("NEIGHBOR_INDEX_1"), "Neighbor_Index_1:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("NEIGHBOR_INDEX_2"), "Neighbor_Index_2:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("NEIGHBOR_INDEX_3"), "Neighbor_Index_3:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("NEIGHBOR_INDEX_4"), "Neighbor_Index_4:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("POOL_I_0"), "Pool_I_0:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("POOL_I_1"), "Pool_I_1:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("POOL_I_2"), "Pool_I_2:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("POOL_I_3"), "Pool_I_3:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("POOL_I_4"), "Pool_I_4:0");
  // EXPECT_EQ(session.tag_to_tensor_map.at("SUBPOINTS_0"), "Subpoints_0:0");
  // EXPECT_EQ(session.tag_to_tensor_map.at("SUBPOINTS_1"), "Subpoints_1:0");
  // EXPECT_EQ(session.tag_to_tensor_map.at("SUBPOINTS_2"), "Subpoints_2:0");
  // EXPECT_EQ(session.tag_to_tensor_map.at("SUBPOINTS_3"), "Subpoints_3:0");
  // EXPECT_EQ(session.tag_to_tensor_map.at("SUBPOINTS_4"), "Subpoints_4:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("PROB_LOGITS"), "results/Softmax:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("BATCH_FEATURE"), "Batch_Feature:0");
}

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       CreateSessionFromSidePacket) {




  options_->clear_saved_model_path();
  CalculatorRunner runner(absl::Substitute(R"(
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        input_side_packet: "STRING_SAVED_MODEL_PATH:saved_model_dir"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        })",
                                           options_->DebugString()));
  runner.MutableSidePackets()->Tag("STRING_SAVED_MODEL_PATH") =
      MakePacket<std::string>(GetSavedModelDir());
  MP_ASSERT_OK(runner.Run());
  const TensorFlowSession& session =
      runner.OutputSidePackets().Tag("SESSION").Get<TensorFlowSession>();
  // Session must be set.
  ASSERT_NE(session.session, nullptr);
}

// Integration test. Verifies that TensorFlowInferenceCalculator correctly
// consumes the Packet emitted by this factory.
TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       ProducesPacketUsableByTensorFlowInferenceCalculator) {
  // try{
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(R"(
          input_stream: "POINT_CLOUD:point_cloud_tensor"
          output_stream: "PROB_LOGITS:softmax_linear"
        node {
          calculator: "PointCloudToRandlanetFormatCalculator"
          input_stream: "POINT_CLOUD:point_cloud_tensor"
          output_stream: "NEIGHBOR_INDEX_0:neighbor_index_0_tensor"
          output_stream: "NEIGHBOR_INDEX_1:neighbor_index_1_tensor"
          output_stream: "NEIGHBOR_INDEX_2:neighbor_index_2_tensor"
          output_stream: "NEIGHBOR_INDEX_3:neighbor_index_3_tensor"
          output_stream: "NEIGHBOR_INDEX_4:neighbor_index_4_tensor"
          output_stream: "POOL_I_0:pool_i_0_tensor"
          output_stream: "POOL_I_1:pool_i_1_tensor"
          output_stream: "POOL_I_2:pool_i_2_tensor"
          output_stream: "POOL_I_3:pool_i_3_tensor"
          output_stream: "POOL_I_4:pool_i_4_tensor"
          output_stream: "UP_I_0:up_i_0_tensor"
          output_stream: "UP_I_1:up_i_1_tensor"
          output_stream: "UP_I_2:up_i_2_tensor"
          output_stream: "UP_I_3:up_i_3_tensor"
          output_stream: "UP_I_4:up_i_4_tensor"
          output_stream: "BATCH_XYZ_0:batch_xyz_0_tensor"
          output_stream: "BATCH_XYZ_1:batch_xyz_1_tensor"
          output_stream: "BATCH_XYZ_2:batch_xyz_2_tensor"
          output_stream: "BATCH_XYZ_3:batch_xyz_3_tensor"
          output_stream: "BATCH_XYZ_4:batch_xyz_4_tensor"
          output_stream: "BATCH_FEATURE:batch_feature_tensor"
        }
      node {
        calculator: "TensorFlowInferenceCalculator"
        input_side_packet: "SESSION:tf_model"
        input_stream: "NEIGHBOR_INDEX_0:neighbor_index_0_tensor"
        input_stream: "NEIGHBOR_INDEX_1:neighbor_index_1_tensor"
        input_stream: "NEIGHBOR_INDEX_2:neighbor_index_2_tensor"
        input_stream: "NEIGHBOR_INDEX_3:neighbor_index_3_tensor"
        input_stream: "NEIGHBOR_INDEX_4:neighbor_index_4_tensor"
        input_stream: "POOL_I_0:pool_i_0_tensor"
        input_stream: "POOL_I_1:pool_i_1_tensor"
        input_stream: "POOL_I_2:pool_i_2_tensor"
        input_stream: "POOL_I_3:pool_i_3_tensor"
        input_stream: "POOL_I_4:pool_i_4_tensor"
        input_stream: "UP_I_0:up_i_0_tensor"
        input_stream: "UP_I_1:up_i_1_tensor"
        input_stream: "UP_I_2:up_i_2_tensor"
        input_stream: "UP_I_3:up_i_3_tensor"
        input_stream: "UP_I_4:up_i_4_tensor"
        input_stream: "BATCH_XYZ_0:batch_xyz_0_tensor"
        input_stream: "BATCH_XYZ_1:batch_xyz_1_tensor"
        input_stream: "BATCH_XYZ_2:batch_xyz_2_tensor"
        input_stream: "BATCH_XYZ_3:batch_xyz_3_tensor"
        input_stream: "BATCH_XYZ_4:batch_xyz_4_tensor"
        input_stream: "BATCH_FEATURE:batch_feature_tensor"
        output_stream: "PROB_LOGITS:softmax_linear"
        options {
          [mediapipe.TensorFlowInferenceCalculatorOptions.ext] {
            batch_size: 1
            add_batch_dim_to_tensors: false
          }
        }
      }
      node {
        calculator: "TensorFlowSessionFromSavedModelCalculator"
        output_side_packet: "SESSION:tf_model"
        options {
          [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
            $0
          }
        }
      }
  )"
, options_->DebugString()));
  // }
  // catch (int e){
  //   std::cout << "An exception occurred. Exception Nr. " << e << '\n';
  // }
  CalculatorGraph graph;
  // auto graph_i = graph.Initialize(graph_config);
  // std::cout << "================Message error===============" << std::endl;
  // std::cout << graph_i.ToString() << std::endl;
  
  MP_ASSERT_OK(graph.Initialize(graph_config));
  StatusOrPoller status_or_poller =
      graph.AddOutputStreamPoller("softmax_linear");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));


  // # user knn to preprocess the point cloud
  // INPUT: Point cloud with the schema (x,y,z, f1, f2, f3, ...) [N, F]
  // batch_feature: (1, N, F) store (f1,f2,f3, ...)
  // input_points     = vector<vector<float>[N, 3]> store x y z
  // input_neighbors  = vector<vector<float>[N, N, K]> store x y z
  // input_pools      = vector<vector<float>[N, 3]> store x y z
  // input_up_samples = vector<vector<float>[N, 3]> store x y z
  
  // Intermediate variables
  // neigh_idx = vector<float>[n, n, k]
  // neigh_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n) [1, M, K]
  // sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :] [1, N//r , F]
  // pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :] [1, N//r , K]
  // up_i = DP.knn_search(sub_points, batch_xyz, 1) [1, N, 1]

  // The input point cloud has been stored in a tensor object
  const int init_batch_size = 1;
  const int init_n_pts = 65536/4;
  const int init_n_features = 3;
  const int init_n_layers = 5;
  const int K_cpp = 16; // hardcode parameter
  const int sub_sampling_ratio[init_n_layers] = {4,4,4,4,2};

  tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});
  auto point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);

  std::ifstream ifile; 
  ifile.open("/home/tan/tjtanaa/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/testdata/tensorflow_saved_model/RandLA-Net_builder_v2/sample.txt");
  if(ifile) {
      std::cout<<"file exists"<< std::endl;
  } else {
      std::cout<<"file doesn't exist" << std::endl;
  }
  std::string line;
  size_t pos = 0;
  std::string delimiter = " ";
  if (ifile.is_open())
  {
    std::cout << "start reading" << std::endl;
    int r = 0;
    while ( getline (ifile,line) )
    {
      std::string token;
      int c = 0;
      while ((pos = line.find(delimiter)) != std::string::npos) {
          token = line.substr(0, pos);
          // std::cout << token << " ";
          point_tensor->tensor<float, 3>()(0, r, c) = std::stod(token);
          bool flag = (std::abs(point_tensor->tensor<float, 3>()(0, r, c) - std::stod(token)) < 1e-5);
          if (!flag){
            std::cout << "Values not equal: " << std::to_string(point_tensor->tensor<float, 3>()(0, r, c)) << std::endl;
          }
          line.erase(0, pos + delimiter.length());
          c++;
      }
      // std::cout << std::endl;
      r++;
    }
    ifile.close();
  }


  // for (int r = 0; r < init_n_pts ; ++r) {
  //   for (int c = 0; c < init_n_features; ++c) {
  //     // point_tensor->tensor<float, 3>()(0, r, c) = rand() % 100 /100.0;
  //     point_tensor->tensor<float, 3>()(0, r, c) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
  //     // if(r ==0 & c == 0){
  //     //   std::cout << "point tensor: " <<std::endl;
  //     // }
  //     // if (r < 5){
  //     //   std::cout << std::to_string(point_tensor->tensor<float, 3>()(0, r, c)) << "\t";
  //     //   if(c ==init_n_features-1 ){
  //     //     std::cout << std::endl;
  //     //   }
  //     // }
  //   }
  // }   

  std::string point_cloud_tensor_name = "point_cloud_tensor";
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      point_cloud_tensor_name,
      Adopt(point_tensor.release()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream(point_cloud_tensor_name));

  Packet packet;
  ASSERT_TRUE(poller.Next(&packet));


  // input tensor gets multiplied by [[3, 2, 1]]. Expected output:
  // tf::Tensor expected_multiplication = TensorMatrix1x3(3, -2, 10);
  // EXPECT_EQ(expected_multiplication.DebugString(),
  //           packet.Get<tf::Tensor>().DebugString());
  
  std::cout << packet.Get<tf::Tensor>().DebugString() << std::endl;
  auto matrix = packet.Get<tf::Tensor>().tensor<float, 2>();
  std::cout << "matrix size: " << matrix.size() << std::endl;
  // std::cout << matrix(0,0) <<std::endl;
  // std::cout << matrix(0,1) <<std::endl;
  // std::cout << matrix(2) <<std::endl;

  std::ifstream resultfile; 
  resultfile.open("/home/tan/tjtanaa/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/testdata/tensorflow_saved_model/RandLA-Net_builder_v2/results.txt");
  if(resultfile) {
      std::cout<<"file exists"<< std::endl;
  } else {
      std::cout<<"file doesn't exist" << std::endl;
  }
  // std::string line;
  pos = 0;
  // std::string delimiter = " ";
  if (resultfile.is_open())
  {
    std::cout << "start reading" << std::endl;
    int r = 0;
    while ( getline (resultfile,line) )
    {
      std::string token;
      int c = 0;
      while ((pos = line.find(delimiter)) != std::string::npos) {
          token = line.substr(0, pos);
          // std::cout << token << " ";
          bool flag = (std::abs(matrix(r,c) - std::stof(token)) < 1e-5);
          if (!flag){
            std::cout << "Values not equal: " << std::to_string(r) << ", " << std::to_string(c) << std::endl;
          }
          line.erase(0, pos + delimiter.length());
          c++;
      }
      // std::cout << std::endl;
      r++;
    }
    resultfile.close();
  }


  ASSERT_FALSE(poller.Next(&packet));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace mediapipe
