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
#include <cstdlib>
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
                     "tensorflow_saved_model/00000001");
      // file::JoinPath("home/tan/tjtanaa/", "cifar10_eval_builder");
  return out_path;
}

// Helper function that creates Tensor INT32 matrix with size 1x3.
tf::Tensor TensorMatrix1x3(const int v1, const int v2, const int v3) {
  tf::Tensor tensor(tf::DT_INT32,
                    tf::TensorShape(std::vector<tf::int64>({1, 3})));
  auto matrix = tensor.matrix<int32>();
  matrix(0, 0) = v1;
  matrix(0, 1) = v2;
  matrix(0, 2) = v3;
  return tensor;
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
  EXPECT_EQ(session.tag_to_tensor_map.size(), 2);

  // Display the tag that the serve session has:
  // Essential for debugging
  for(auto it = session.tag_to_tensor_map.cbegin(); it != session.tag_to_tensor_map.cend(); ++it)
  {
      std::cout << it->first << " " << it->second << "\n";
  }
  // For some reason, EXPECT_EQ and EXPECT_NE are not working with iterators.
  EXPECT_FALSE(session.tag_to_tensor_map.find("IMAGES") ==
               session.tag_to_tensor_map.end());
  EXPECT_FALSE(session.tag_to_tensor_map.find("SCORES") ==
               session.tag_to_tensor_map.end());
  // Sanity: find() actually returns a reference to end() if element not
  // found.
  EXPECT_TRUE(session.tag_to_tensor_map.find("Z") ==
              session.tag_to_tensor_map.end());

  // Use these saved_model_cli command to check for the name of the input and output of the saved_model
  // e.g. /home/tan/anaconda3/envs/saved_model_cli/bin/saved_model_cli <dir of the saved_model>
  // --tag_set serve --signature_def serving_default
  EXPECT_EQ(session.tag_to_tensor_map.at("IMAGES"), "Input_Placeholder:0");
  EXPECT_EQ(session.tag_to_tensor_map.at("SCORES"), "softmax_linear/softmax_linear:0");
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
  CalculatorGraphConfig graph_config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(R"(
      node {
        calculator: "TensorFlowInferenceCalculator"
        input_side_packet: "SESSION:tf_model"
        input_stream: "IMAGES:a_tensor"
        output_stream: "SCORES:softmax_linear"
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
      input_stream: "a_tensor"
  )",
                           options_->DebugString()));

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  StatusOrPoller status_or_poller =
      graph.AddOutputStreamPoller("softmax_linear");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));


	const char *env_var[6] = {"PUBLIC","HOME","SESSIONNAME","LIB","SystemDrive", "LD_LIBRARY_PATH"};
	char *env_val[6];

	for(int i=0; i<6; i++)
	{
		/* Getting environment value if exists */
		env_val[i] = std::getenv(env_var[i]);
		if (env_val[i] != NULL)
			std::cout << "Variable = " << env_var[i] << ", Value= " << env_val[i] << std::endl;
		else
			std::cout << env_var[i] << " doesn't exist" << std::endl;
	}

  // code to read an image file using opencv
  // std::string image_path = file::JoinPath("./", "mediapipe/calculators/tensorflow/testdata/car.jpg");
  cv::Mat image;
  // , cv::CV_LOAD_IMAGE_COLOR
  // std::string image_path = "./mediapipe/calculators/tensorflow/testdata/car.jpg";

  /* try to open file to read */
  std::ifstream ifile;
  
  ifile.open("/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/testdata/car.jpg");
  if(ifile) {
      std::cout<<"file exists"<< std::endl;
  } else {
      std::cout<<"file doesn't exist" << std::endl;
  }
  
  image = cv::imread("/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/testdata/car.jpg", cv::IMREAD_UNCHANGED);
  if(! image.data )                              // Check for invalid input
  {
      std::cout <<  "!!!!!!!!!!!! Could not open or find the image !!!!!!!!!!!" << std::endl ;
      // return -1;
  }
  tf::Tensor input_tensor(tf::DT_FLOAT, tf::TensorShape(std::vector<tf::int64>({1, 24,24,3})));
  std::cout << "created input_tensor" << std::endl;
  // get pointer to memory for that Tensor
  float* ptr = input_tensor.flat<float>().data();
  cv::Mat tensor_image(24, 24, CV_32FC3, ptr);
  image.convertTo(tensor_image, CV_32FC3);
  // auto matrix = input_tensor.matrix<float>();

  // std::cout << "M = " << std::endl << " "  << tensor_image << std::endl << std::endl;
  Packet packet;
  for(int k = 0; k < 1000000; k++){
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "a_tensor",
        Adopt(new auto(input_tensor)).At(Timestamp(k))));
    // MP_ASSERT_OK(graph.CloseInputStream("a_tensor"));

    
    ASSERT_TRUE(poller.Next(&packet));
    // input tensor gets multiplied by [[3, 2, 1]]. Expected output:
    // tf::Tensor expected_multiplication = TensorMatrix1x3(3, -2, 10);
    // EXPECT_EQ(expected_multiplication.DebugString(),
    //           packet.Get<tf::Tensor>().DebugString());
    
    // packet.Get<tf::Tensor>().DebugString());
    // auto matrix = packet.Get<tf::Tensor>().matrix<float>();

    std::cout << k <<  " === The Scores of the GTA car" << std::endl;
    // std::cout << packet.Get<tf::Tensor>().DebugString() << std::endl;
    // std::cout << matrix(0,0) << "\t";
    // std::cout << matrix(0,1) <<"\t";
    // std::cout << matrix(0,2) <<"\t";
    // std::cout << matrix(0,3) <<"\t";
    // std::cout << matrix(0,4) <<"\t";
    // std::cout << matrix(0,5) <<"\t";
    // std::cout << matrix(0,6) <<"\t";
    // std::cout << matrix(0,7) <<"\t";
    // std::cout << matrix(0,8) <<"\t";
    // std::cout << matrix(0,9) <<std::endl;


  }
  ASSERT_FALSE(poller.Next(&packet));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(TensorFlowSessionFromSavedModelCalculatorTest,
       GetsBundleGivenParentDirectory) {
  options_->set_saved_model_path(
      std::string(file::SplitPath(GetSavedModelDir()).first));
  options_->set_load_latest_model(true);

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
}

}  // namespace
}  // namespace mediapipe

