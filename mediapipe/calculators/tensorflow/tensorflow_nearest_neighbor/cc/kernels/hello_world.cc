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
//
// A simple example to print out "Hello World!" from a MediaPipe graph.

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

#include "knn_.h"
#include <stdlib.h> 

namespace mediapipe {

namespace tf = tensorflow;

::mediapipe::Status PrintHelloWorld() {
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    input_stream: "in"
    output_stream: "out"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "in"
      output_stream: "out1"
    }
    node {
      calculator: "PassThroughCalculator"
      input_stream: "out1"
      output_stream: "out"
    }
  )");

  CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));


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
  const int init_n_pts = 65536;
  const int init_n_features = 3;
  const int init_n_layers = 5;
  const int K_cpp = 16; // hardcode parameter
  const int sub_sampling_ratio[init_n_layers] = {4,4,4,4,2};

  // tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});
  // tf::Tensor* point_tensor = new tf::Tensor(tf::DT_FLOAT, point_tensor_shape);
  

  // for (int r = 0; r < init_n_pts ; ++r) {
  //   for (int c = 0; c < init_n_features; ++c) {
  //     point_tensor->tensor<float, 3>()(0, r, c) = rand() % 10000;
  //   }
  // }   

  // tf::Tensor* temp_point_tensor = new tf::Tensor(tf::DT_FLOAT, point_tensor_shape);;

  // for (int r = 0; r < init_n_pts ; ++r) {
  //   for (int c = 0; c < init_n_features; ++c) {
  //     temp_point_tensor->tensor<float, 3>()(0, r, c) = point_tensor->tensor<float, 3>()(0, r, c);
  //   }
  // }   

  tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});
  auto point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      point_tensor->tensor<float, 3>()(0, r, c) = rand() % 10000;
    }
  }   


  auto temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      temp_point_tensor->tensor<float, 3>()(0, r, c) = point_tensor->tensor<float, 3>()(0, r, c);
    }
  }   
// =======
  for(int layer = 0; layer < init_n_layers; layer++ ){
    // const int batch_size = temp_point_tensor->dim_size(0);
    // const int npts = temp_point_tensor->dim_size(1);
    // const int dim = temp_point_tensor->dim_size(2);
    // const int nqueries = temp_point_tensor->dim_size(1);

    const int batch_size = temp_point_tensor->dim_size(0);
    const int npts = temp_point_tensor->dim_size(1);
    const int dim = temp_point_tensor->dim_size(2);
    const int nqueries = temp_point_tensor->dim_size(1);

    std::cout << "layer " << layer  << "npts " << npts << "dim " << dim << "nqueries " << nqueries << std::endl;
    std::cout << "npts/sub_sampling_ratio[layer] " << npts/sub_sampling_ratio[layer] << std::endl;
    // create intermediate variables
    tf::TensorShape neigh_idx_tensor_shape({batch_size, nqueries, K_cpp});
    auto neigh_idx_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, neigh_idx_tensor_shape);

    tf::TensorShape sub_points_tensor_shape({batch_size, npts/sub_sampling_ratio[layer], dim});
    auto sub_points_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);

    tf::TensorShape pool_i_tensor_shape({batch_size, nqueries/sub_sampling_ratio[layer], K_cpp});
    auto pool_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, pool_i_tensor_shape);
    
    tf::TensorShape up_i_tensor_shape({batch_size, npts, 1});
    auto up_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, up_i_tensor_shape);

    // start to compute
    // auto pt_tensor = temp_point_tensor->flat<float>().data();
    // auto q_tensor = temp_point_tensor->flat<float>().data();
    auto pt_tensor = temp_point_tensor->flat<float>().data();
    auto q_tensor = temp_point_tensor->flat<float>().data();
    auto neigh_idx_flat = neigh_idx_tensor->flat<long long int>().data();
    cpp_knn_batch_omp(pt_tensor, batch_size, npts, dim, q_tensor, nqueries, K_cpp, neigh_idx_flat);
    
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < K_cpp; ++c) {
        pool_i_tensor->tensor<long long int, 3>()(0, r, c) = neigh_idx_tensor->tensor<long long int, 3>()(0, r, c);
        // std::cout << pool_i_tensor.tensor<long long int, 3>()(0, r, c) << std::endl;
      }
    }    

    std::cout << "subpoint "  << npts/sub_sampling_ratio[layer] << std::endl;
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < dim; ++c) {
        // sub_points_tensor.tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
        sub_points_tensor->tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
        // std::cout << "r " << r << " c " << c << std::endl;
        // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
      }
    }   

    // std::cout << "delete temp_point_tensor " << std::endl;
    temp_point_tensor.release();
    // std::cout << "after deleting temp_point_tensor " << std::endl;
    
    temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < dim; ++c) {
        // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
        temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor->tensor<float, 3>()(0, r, c);
        // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
      }
    } 

    auto sub_points_flat = sub_points_tensor->flat<float>().data();

    auto up_i_flat = up_i_tensor->flat<long long int>().data();
    cpp_knn_batch_omp(sub_points_flat, batch_size, npts, dim, q_tensor, nqueries, 1, up_i_flat);

    neigh_idx_tensor.release();
    sub_points_tensor.release();
    pool_i_tensor.release();
    up_i_tensor.release();
  }



  // Give 10 input packets that contains the same std::string "Hello World!".
  for (int i = 0; i < 10; ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in", MakePacket<std::string>("Hello World!").At(Timestamp(i))));
  }

  // Close the input stream "in".
  MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
  mediapipe::Packet packet;
  // Get the output packets std::string.
  while (poller.Next(&packet)) {
    LOG(INFO) << packet.Get<std::string>();
  }
  return graph.WaitUntilDone();
}
}  // namespace mediapipe

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK(mediapipe::PrintHelloWorld().ok());
  return 0;
}
