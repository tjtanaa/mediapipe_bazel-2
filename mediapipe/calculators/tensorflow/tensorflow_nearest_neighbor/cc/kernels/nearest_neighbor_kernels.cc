/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include "knn_.h"

using namespace tensorflow;

class KNNOp : public OpKernel {
 public:
  explicit KNNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& point_tensor = context->input(0);
    const Tensor& query_tensor = context->input(1);
    const Tensor& K = context->input(2);
    // auto _K = K.flat<int32>();
    // Create an output tensor
    
    // std::cout << "pt_tensor input" << std::endl;
    auto pt_tensor = point_tensor.flat<float>().data();
    auto q_tensor = query_tensor.flat<float>().data();
    const int batch_size = point_tensor.dim_size(0);
    const int npts = point_tensor.dim_size(1);
    const int dim = point_tensor.dim_size(2);
    const int nqueries = query_tensor.dim_size(1);
    const int K_cpp = K.flat<int>().data()[0];

    // std::cout << "Operation statistics" << std::endl;
    // std::cout << "Batch size: \t" << std::to_string(batch_size) << std::endl;
    // std::cout << "Number Points: \t" << std::to_string(npts) << std::endl;
    // std::cout << "Total dimension: \t" << std::to_string(dim) << std::endl;
    // std::cout << "nqueries: \t" << std::to_string(nqueries) << std::endl;
    // std::cout << "K_cpp: \t" << std::to_string(K_cpp) << std::endl;



    // const int N = batch_size * npts * dim;
    // for (int n = 0; n < N; n++){
    //   std::cout << ((int)pt_tensor[n]) << std::endl;
    // }
    // std::cout << "First dimension of pointer tensor: " << N <<std::endl;
    Tensor* output_tensor = NULL;
    TensorShape output_shape({batch_size, nqueries, K_cpp});
    
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    auto output_flat = output_tensor->flat<long long int>().data();
    
    cpp_knn_batch_omp(pt_tensor, batch_size, npts, dim, q_tensor, nqueries, K_cpp, output_flat);

  }
};

REGISTER_KERNEL_BUILDER(Name("KNN").Device(DEVICE_CPU), KNNOp);
