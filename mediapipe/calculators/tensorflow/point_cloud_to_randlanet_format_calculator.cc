// ================= Custom calculator =====================
// Copyright 2019 The MediaPipe Authors.
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
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/knn_.h"
#include <string>
#include <iostream>
namespace mediapipe {



namespace tf = ::tensorflow;

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

class PointCloudToRandlanetFormatCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  PointCloudToRandlanetFormatCalculatorOptions options_;
};
REGISTER_CALCULATOR(PointCloudToRandlanetFormatCalculator);

::mediapipe::Status PointCloudToRandlanetFormatCalculator::GetContract(
    CalculatorContract* cc) {
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
    cc->Inputs().Tag(InputTag[0]).Set<tf::Tensor>(
        /* "Input vector<vector<float>>." */);
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 21)
      << "Must have 21 output streams.";
    for(int i = 0 ; i < 21;  i ++){
      // std::cout << OutputTag[i] << std::endl;
      cc->Outputs().Tag(OutputTag[i]).Set<tf::Tensor>(
          // Output stream with data as tf::Tensor and the same TimeSeriesHeader.
      );        
    }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PointCloudToRandlanetFormatCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<PointCloudToRandlanetFormatCalculatorOptions>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PointCloudToRandlanetFormatCalculator::Process(
    CalculatorContext* cc) {

// std::cout << "PROCESS POINT CLOUD" << std::endl;

// The input point cloud has been stored in a tensor object
  const int init_batch_size = options_.batch_size();
  const int init_n_pts = options_.npts();
  const int init_n_features = options_.n_features();
  const int init_n_layers = options_.n_layers();
  const int K_cpp = options_.k_cpp(); // hardcode parameter
  const int sub_sampling_ratio[init_n_layers] = {4,4,4,4,2}; // hardcode parameter
  // std::cout << "Options Parameters: " << std::to_string(init_batch_size) << "\t" <<
  //           std::to_string(init_n_pts) << "\t" <<
  //           std::to_string(init_n_features) << "\t" <<
  //           std::to_string(init_n_layers) << "\t" <<
  //           std::to_string(K_cpp) << "\t" << std::endl;

  tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});

  const tf::Tensor& point_tensor =
      cc->Inputs().Tag(InputTag[0]).Value().Get<tf::Tensor>();

  auto temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      temp_point_tensor->tensor<float, 3>()(0, r, c) = point_tensor.tensor<float, 3>()(0, r, c);
      // if(r == 0){
      //   std::cout << "r " << r <<  std::to_string(point_tensor.tensor<float, 3>()(0, r, c)) << std::endl;
      // }
    }
  }

  tf::TensorShape batch_feature_tensor_shape({init_batch_size, init_n_pts, init_n_features});
  auto temp_batch_feature_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      temp_batch_feature_tensor->tensor<float, 3>()(0, r, c) = point_tensor.tensor<float, 3>()(0, r, c);
      // temp_batch_feature_tensor->tensor<float, 3>()(0, r, c+3) = point_tensor.tensor<float, 3>()(0, r, c);
    }
  }    
//   std::string batch_feature_tensor_name = "BATCH_FEATURE_tensor";

    // std::cout << "temp_batch_feature_tensor.release()" << OutputTag[26-1] << std::endl;
    cc->Outputs().Tag(OutputTag[21-1]).Add(temp_batch_feature_tensor.release(), cc->InputTimestamp());
// =======
  for(int layer = 0; layer < init_n_layers; layer++ ){
    // std::cout << "Layer: " << layer << std::endl;

    const int batch_size = temp_point_tensor->dim_size(0);
    const int npts = temp_point_tensor->dim_size(1);
    const int dim = temp_point_tensor->dim_size(2);
    const int nqueries = temp_point_tensor->dim_size(1);

    // std::cout << "layer " << layer  << "npts " << npts << "dim " << dim << "nqueries " << nqueries << std::endl;
    // std::cout << "npts/sub_sampling_ratio[layer] " << npts/sub_sampling_ratio[layer] << std::endl;
    // create intermediate variables
    tf::TensorShape neigh_idx_tensor_shape({batch_size, nqueries, K_cpp});
    auto neigh_idx_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, neigh_idx_tensor_shape);
    // auto neigh_idx_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, neigh_idx_tensor_shape);

    tf::TensorShape sub_points_tensor_shape({batch_size, npts/sub_sampling_ratio[layer], dim});
    auto sub_points_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);

    tf::TensorShape pool_i_tensor_shape({batch_size, nqueries/sub_sampling_ratio[layer], K_cpp});
    auto pool_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, pool_i_tensor_shape);
    // auto pool_i_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, pool_i_tensor_shape);
    
    tf::TensorShape up_i_tensor_shape({batch_size, npts, 1});
    auto up_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, up_i_tensor_shape);
    // auto up_i_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, up_i_tensor_shape);

    // std::cout << neigh_idx_tensor_shape << std::endl;
    // std::cout << sub_points_tensor_shape << std::endl;
    // std::cout << pool_i_tensor_shape << std::endl;
    // std::cout << up_i_tensor_shape << std::endl;
    // start to compute
    auto pt_tensor = temp_point_tensor->flat<float>().data();
    auto q_tensor = temp_point_tensor->flat<float>().data();
    auto neigh_idx_flat = neigh_idx_tensor->flat<long long int>().data();
    cpp_knn_batch_omp(pt_tensor, batch_size, npts, dim, q_tensor, nqueries, K_cpp, neigh_idx_flat);
    
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < K_cpp; ++c) {
        pool_i_tensor->tensor<long long int, 3>()(0, r, c) = neigh_idx_tensor->tensor<long long int, 3>()(0, r, c);
        //   if(r ==0 & c == 0){
        //     std::cout << "pool_i_tensor: " <<std::endl;
        //   }
        // // if(r == 0){
        // //   std::cout << "pool_i_tensor: " << std::to_string(pool_i_tensor->tensor<long long int, 3>()(0, r, c)) << std::endl;
        // // }

        //   if (r < 5){
        //     std::cout << std::to_string(pool_i_tensor->tensor<long long int, 3>()(0, c, r)) << "\t";
        //   }
        //   if(c == K_cpp - 1){
        //     std::cout << std::endl;
        // }
      }
    }    

    // std::cout << "subpoint "  << npts/sub_sampling_ratio[layer] << std::endl;
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < dim; ++c) {
        // sub_points_tensor.tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
        sub_points_tensor->tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
        // std::cout << "r " << r << " c " << c << std::endl;
        // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
      }
    }   

    auto sub_points_flat = sub_points_tensor->flat<float>().data();
    auto q_tensor_2 = temp_point_tensor->flat<float>().data();
    auto up_i_flat = up_i_tensor->flat<long long int>().data();
    cpp_knn_batch_omp(sub_points_flat, batch_size, npts/sub_sampling_ratio[layer], dim, q_tensor_2, nqueries, 1, up_i_flat);


    // std::string temp_point_tensor_name = "BATCH_XYZ_" + std::to_string(layer) + "_tensor";

    // std::cout << "temp_point_tensor.release()" << OutputTag[15+layer] << std::endl;
    cc->Outputs().Tag(OutputTag[15+layer]).Add(temp_point_tensor.release(), cc->InputTimestamp());

    // std::cout << "delete temp_point_tensor " << std::endl;
    // temp_point_tensor.release();
    // std::cout << "after deleting temp_point_tensor " << std::endl;
    if(layer < 4){
        temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);
        for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
            for (int c = 0; c < dim; ++c) {
                // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
                temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor->tensor<float, 3>()(0, r, c);
                // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
            }
        }
    } 


    // std::string neigh_idx_tensor_name = "NEIGHBOR_INDEX_" + std::to_string(layer) + "_tensor";

    // std::cout << "neigh_idx_tensor.release()" << OutputTag[layer] << std::endl;
    cc->Outputs().Tag(OutputTag[layer]).Add(neigh_idx_tensor.release(), cc->InputTimestamp());

    // std::string sub_points_tensor_name = "SUBPOINTS_" + std::to_string(layer) + "_tensor";

    // std::cout << "sub_points_tensor.release()" << OutputTag[5 +layer]<< std::endl;
    // cc->Outputs().Tag(OutputTag[5 +layer]).Add(sub_points_tensor.release(), cc->InputTimestamp());

    sub_points_tensor.release();

    // std::string pool_i_tensor_name = "POOL_I_" + std::to_string(layer) + "_tensor";

    // std::cout << "pool_i_tensor.release()" << OutputTag[5 +layer] << std::endl;
    cc->Outputs().Tag(OutputTag[5 + layer]).Add(pool_i_tensor.release(), cc->InputTimestamp());

    // std::string up_i_tensor_name = "UP_I_" + std::to_string(layer) + "_tensor";

    // std::cout << "up_i_tensor.release()" << OutputTag[10 +layer] << std::endl;
    cc->Outputs().Tag(OutputTag[10 +layer]).Add(up_i_tensor.release(), cc->InputTimestamp());

  }
  temp_point_tensor.release();

  // std::cout << "DONE POINT CLOUD TO RANDLANET FORMAT CALCULATOR" << std::endl;
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
