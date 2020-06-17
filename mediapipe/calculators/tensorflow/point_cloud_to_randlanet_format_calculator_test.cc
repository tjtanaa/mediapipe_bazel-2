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

#include "mediapipe/calculators/tensorflow/point_cloud_to_randlanet_format_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
// #include <iostream>
#include <fstream>


namespace mediapipe {

namespace tf = ::tensorflow;
namespace {

const std::string InputTag[] = {"POINT_CLOUD"};
const std::string OutputTag[] = {"NEIGHBOR_INDEX_0",
                                "NEIGHBOR_INDEX_1",
                                "NEIGHBOR_INDEX_2",
                                "NEIGHBOR_INDEX_3",
                                "NEIGHBOR_INDEX_4",
                                "POOL_I_0",
                                "POOL_I_1",
                                "POOL_I_2",
                                "POOL_I_3",
                                "POOL_I_4",
                                "UP_I_0",
                                "UP_I_1",
                                "UP_I_2",
                                "UP_I_3",
                                "UP_I_4",
                                "BATCH_XYZ_0",
                                "BATCH_XYZ_1",
                                "BATCH_XYZ_2",
                                "BATCH_XYZ_3",
                                "BATCH_XYZ_4",
                                "BATCH_FEATURE"};
}  // namespace

class PointCloudToRandlanetFormatCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner() {
    CalculatorGraphConfig::Node config;
    config.set_calculator("PointCloudToRandlanetFormatCalculator");
    config.add_input_stream("POINT_CLOUD:input_tensor");
    config.add_output_stream("NEIGHBOR_INDEX_0:neighbor_index_0");
    config.add_output_stream("NEIGHBOR_INDEX_1:neighbor_index_1");
    config.add_output_stream("NEIGHBOR_INDEX_2:neighbor_index_2");
    config.add_output_stream("NEIGHBOR_INDEX_3:neighbor_index_3");
    config.add_output_stream("NEIGHBOR_INDEX_4:neighbor_index_4");
    config.add_output_stream("POOL_I_0:pool_i_0");
    config.add_output_stream("POOL_I_1:pool_i_1");
    config.add_output_stream("POOL_I_2:pool_i_2");
    config.add_output_stream("POOL_I_3:pool_i_3");
    config.add_output_stream("POOL_I_4:pool_i_4");
    config.add_output_stream("UP_I_0:up_i_0");
    config.add_output_stream("UP_I_1:up_i_1");
    config.add_output_stream("UP_I_2:up_i_2");
    config.add_output_stream("UP_I_3:up_i_3");
    config.add_output_stream("UP_I_4:up_i_4");
    config.add_output_stream("BATCH_XYZ_0:batch_xyz_0");
    config.add_output_stream("BATCH_XYZ_1:batch_xyz_1");
    config.add_output_stream("BATCH_XYZ_2:batch_xyz_2");
    config.add_output_stream("BATCH_XYZ_3:batch_xyz_3");
    config.add_output_stream("BATCH_XYZ_4:batch_xyz_4");
    config.add_output_stream("BATCH_FEATURE:batch_feature");
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(PointCloudToRandlanetFormatCalculatorTest, ConvertPointCloudToRandlanetFormatTensor) {
  // This test converts a 1 Dimensional Tensor of length M to a Matrix of Mx1.
  SetUpRunner();
  const int init_batch_size = 1;
  const int init_n_pts = 65536/4;
  const int init_n_features = 3;
  const int init_n_layers = 5;
  const int K_cpp = 16; // hardcode parameter
  const int sub_sampling_ratio[init_n_layers] = {4,4,4,4,2};

  tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});
  auto point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);


  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      // point_tensor->tensor<float, 3>()(0, r, c) = rand() % 100 /100.0;
      point_tensor->tensor<float, 3>()(0, r, c) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
    }
  }  

  const int64 time = 1234;
  runner_->MutableInputs()->Tag("POINT_CLOUD").packets.push_back(
      Adopt(point_tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
//   auto output_pack = runner_->Outputs();
//   std::cout << output_pack << std::endl;
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag("NEIGHBOR_INDEX_1").packets;
  EXPECT_EQ(1, output_packets.size());
  
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(3, output_tensor.dims());
  auto output_tensor_tensor = output_tensor.tensor<long long int, 3>();

  for (int r = 0; r < 1 ; ++r) {
      for (int c = 0; c < K_cpp; ++c) {
      std::cout << std::to_string(output_tensor_tensor(0,r,c)) << " " ;
      }
  }
  std::cout << std::endl;
}

TEST_F(PointCloudToRandlanetFormatCalculatorTest, ConvertSamplePointCloudToRandlanetFormatTensor) {
  // This test converts a 1 Dimensional Tensor of length M to a Matrix of Mx1.
  SetUpRunner();
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
          std::cout << token << " ";
          point_tensor->tensor<float, 3>()(0, r, c) = std::stod(token);
          bool flag = (std::abs(point_tensor->tensor<float, 3>()(0, r, c) - std::stod(token)) < 1e-5);
          if (!flag){
            std::cout << "Values not equal: " << std::to_string(point_tensor->tensor<float, 3>()(0, r, c)) << std::endl;
          }
          line.erase(0, pos + delimiter.length());
          c++;
      }
      std::cout << std::endl;
      r++;
    }
    ifile.close();
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag("POINT_CLOUD").packets.push_back(
      Adopt(point_tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
//   auto output_pack = runner_->Outputs();
//   std::cout << output_pack << std::endl;
  // const std::vector<Packet>& output_packets =
  //     runner_->Outputs().Tag("NEIGHBOR_INDEX_4").packets;
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag("UP_I_0").packets;
  EXPECT_EQ(1, output_packets.size());
  
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(3, output_tensor.dims());
  auto output_tensor_tensor = output_tensor.tensor<long long int, 3>();

  // Read the sample output from file and verify with the indices computed by this calculator
  std::ifstream outputfile; 
  // outputfile.open("/home/tan/tjtanaa/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/testdata/tensorflow_saved_model/RandLA-Net_builder_v2/encoder_sub_indices_0.txt");
  outputfile.open("/home/tan/tjtanaa/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/testdata/tensorflow_saved_model/RandLA-Net_builder_v2/decoder_sub_indices_4.txt");
  if(outputfile) {
      std::cout<<"file exists"<< std::endl;
  } else {
      std::cout<<"file doesn't exist" << std::endl;
  }
  // std::string line;
  pos = 0;
  // std::string delimiter = " ";
  if (outputfile.is_open())
  {
    std::cout << "start reading" << std::endl;
    int r = 0;
    int valid_r = 0;
    while ( getline (outputfile,line) )
    {
      std::string token;
      int c = 0;
      while ((pos = line.find(delimiter)) != std::string::npos) {
          token = line.substr(0, pos);
          std::cout << token << "-" << std::to_string(output_tensor_tensor(0,r,c)) << " ";
          bool flag = (std::abs(output_tensor_tensor(0,r,c) - std::stoll(token)) < 1e-7);
          if (flag){
            // std::cout << "Values not equal: " << std::endl;
            std::cout << "Values equal: " ; //<< std::endl;
            valid_r++;
          }
          line.erase(0, pos + delimiter.length());
          c++;
      }
      std::cout << std::endl;
      r++;
    }
    std::cout << "valid_r: " << std::to_string(valid_r) << std::endl;
    outputfile.close();
  }



  for (int r = 0; r < 1 ; ++r) {
      for (int c = 0; c < K_cpp; ++c) {
      std::cout << std::to_string(output_tensor_tensor(0,r,c)) << " " ;
      }
  }
  std::cout << std::endl;
}





}  // namespace mediapipe