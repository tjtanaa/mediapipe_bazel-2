#include <unistd.h>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>

// randla-net
#include "absl/strings/substitute.h"
// #include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/framework/calculator_framework.h"
// #include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
// #include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/validate_type.h"
// #include "mediapipe/framework/deps/status.h"
// #include "mediapipe/framework/deps/status_builder.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

#include <iostream>
#include <stdlib.h>
// #include "tensorflow/c/c_api.h"

constexpr char kInputStream[] = "point_cloud_tensor";
constexpr char kOutputStream[] = "softmax_linear";

// How to include Eigen
// https://stackoverflow.com/questions/56172620/how-to-build-a-simple-c-demo-using-eigen-with-bazel
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mediapipe/calculators/ouster/os1.h"
#include "mediapipe/calculators/ouster/os1_packet.h"
#include "mediapipe/calculators/ouster/os1_util.h"
#include "mediapipe/calculators/ouster/lidar_scan.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
// #include "ouster/os1.h"
// #include "ouster/os1_packet.h"
// #include "ouster/os1_util.h"
// #include "ouster/lidar_scan.h"
// #include "ouster/viz.h"
namespace tf = ::tensorflow;
namespace OS1 = ouster::OS1;
// namespace viz = ouster::viz;
#include <chrono>

// https://stackoverflow.com/questions/50799510/how-to-run-custom-gpu-tensorflowop-from-c-code

DEFINE_string(
    calculator_graph_config_file, "mediapipe/examples/ouster_app/point_cloud_segmentation/graphs/point_cloud_segmentation_cpu.pbtxt",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(lidar_address, "192.168.5.5",
              "Lidar UDP address. ");
DEFINE_string(udp_address, "192.168.5.1",
              "UDP Destination ");
DEFINE_string(mode, "1024x10",
              "<512x10 | 512x20 | 1024x10 | 1024x20 | 2048x10> : lidar mode, ");

/**
 * Print usage
 */
void print_help()
{
    std::cout
        << "Usage: viz [options] [hostname] [udp_destination]\n"
        << "Options:\n"
        << "  -m <512x10 | 512x20 | 1024x10 | 1024x20 | 2048x10> : lidar mode, "
           "default 1024x10\n"
        << "  -f <path> : use provided metadata file; do not configure via TCP"
        << std::endl;
}

std::string read_metadata(const std::string &meta_file)
{
    std::stringstream buf{};
    std::ifstream ifs{};
    ifs.open(meta_file);
    buf << ifs.rdbuf();
    ifs.close();

    if (!ifs)
    {
        std::cerr << "Failed to read " << meta_file
                  << "; check that the path is valid" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return buf.str();
}

::mediapipe::Status RunMPPGraph()
{
    std::cout << "Start Graphs" << std::endl;
    int W = 1024;
    int H = OS1::pixels_per_column;
    OS1::lidar_mode mode = OS1::MODE_1024x10;
    mode = OS1::lidar_mode_of_string(FLAGS_mode);
    W = OS1::n_cols_of_lidar_mode(mode);
    bool do_config = true; // send tcp commands to configure sensor
    std::string metadata{};
    tf::Tensor tf_tensor(tf::DT_FLOAT, tf::TensorShape({1, 65536, 3}));
    tf::TTypes<float, 3>::Tensor map_tensor = tf_tensor.tensor<float, 3>();
    tf::Tensor tf_tensor_small(tf::DT_FLOAT, tf::TensorShape({1, 16384, 3}));
    tf::TTypes<float, 3>::Tensor map_tensor_small = tf_tensor_small.tensor<float, 3>();
    // W = OS1::n_cols_of_lidar_mode(static_cast<lidar_mode>(3));
    // std::string calculator_graph_config_file = "mediapipe/examples/ouster_app/point_cloud_segmentation/graphs/point_cloud_segmentation_cpu.pbtxt";

    // try {
    //     int c = 0;
    //     while ((c = getopt(argc, argv, "hm:f:")) != -1) {
    //         switch (c) {
    //             case 'h':
    //                 print_help();
    //                 return 1;
    //                 break;
    //             case 'm':
    //                 mode = OS1::lidar_mode_of_string(optarg);
    //                 if (mode) {
    //                     W = OS1::n_cols_of_lidar_mode(mode);
    //                 } else {
    //                     std::cout << "Lidar Mode must be 512x10, 512x20, "
    //                                  "1024x10, 1024x20, or 2048x10"
    //                               << std::endl;
    //                     print_help();
    //                     std::exit(EXIT_FAILURE);
    //                 }
    //                 break;
    //             case 'f':
    //                 do_config = false;
    //                 metadata = read_metadata(optarg);
    //                 break;
    //             case '?':
    //                 std::cout << "Invalid Argument Format" << std::endl;
    //                 print_help();
    //                 std::exit(EXIT_FAILURE);
    //                 break;
    //         }
    //     }
    // } catch (const std::exception& ex) {
    //     std::cout << "Invalid Argument Format: " << ex.what() << std::endl;
    //     print_help();
    //     std::exit(EXIT_FAILURE);
    // }

    // if (do_config && argc != optind + 2) {
    //     std::cerr << "Expected 2 arguments after options" << std::endl;
    //     print_help();
    //     std::exit(EXIT_FAILURE);
    // }

    // std::string lidar_address = "192.168.5.5";
    // std::string udp_address = "192.168.5.1";
    std::shared_ptr<OS1::client> cli;
    if (do_config)
    {
        // std::cout << "Configuring sensor: " << argv[optind]
        //           << " UDP Destination:" << argv[optind + 1] << std::endl;
        // cli = OS1::init_client(argv[optind], argv[optind + 1], mode);
        std::cout << "Configuring sensor: " << FLAGS_lidar_address
                  << " UDP Destination:" << FLAGS_udp_address << std::endl;
        cli = OS1::init_client(FLAGS_lidar_address, FLAGS_udp_address, mode);
    }
    else
    {
        std::cout << "Listening for sensor data" << std::endl;
        cli = OS1::init_client();
    }

    if (!cli)
    {
        std::cerr << "Failed to initialize client" << std::endl;
        print_help();
        std::exit(EXIT_FAILURE);
    }

    // Configure the Mediapipe Graphs
    std::string calculator_graph_config_contents;
    // MP_RETURN_IF_ERROR(
    mediapipe::file::GetContents(
        FLAGS_calculator_graph_config_file, &calculator_graph_config_contents);
    // );
    // LOG(INFO)
    std::cout << "Get calculator graph config contents: "
              << calculator_graph_config_contents << std::endl;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    // LOG(INFO) <<
    std::cout << "Initialize the calculator graph." << std::endl;
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    // LOG(INFO) <<
    std::cout << "Initialize the lidar" << std::endl;

    uint8_t lidar_buf[OS1::lidar_packet_bytes + 1];
    uint8_t imu_buf[OS1::imu_packet_bytes + 1];

    auto ls = std::unique_ptr<ouster::LidarScan>(new ouster::LidarScan(W, H));

    // auto vh = viz::init_viz(W, H);

    if (do_config)
        metadata = OS1::get_metadata(*cli);

    auto info = OS1::parse_metadata(metadata);

    auto xyz_lut = OS1::make_xyz_lut(W, H, info.beam_azimuth_angles,
                                     info.beam_altitude_angles);

    // Use to signal termination
    std::atomic_bool end_program{false};

    // auto it = std::back_inserter(*ls);
    auto it = ls->begin();

    // callback that calls update with filled lidar scan
    auto batch_and_update = OS1::batch_to_iter<ouster::LidarScan::iterator>(
        xyz_lut, W, H, ouster::LidarScan::Point::Zero(),
        &ouster::LidarScan::make_val, [&](uint64_t) {
            // swap lidar scan and point it to new buffer
            // viz::update(*vh, ls);
            it = ls->begin();
            // std::cout << "Not Initialized with the first frame" << std::endl;
        });

    // std::cout << "Start running the calculator graph." << std::endl;
    // mediapipe::OutputStreamPoller poller = graph.AddOutputStreamPoller(kOutputStream);
    // graph.StartRun({});
    // std::cout << "Start grabbing and processing point clouds." << std::endl;

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                     graph.AddOutputStreamPoller(kOutputStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    LOG(INFO) << "Start grabbing and processing point clouds.";

    // Start polling lidar input
    int count = 0;
    int batch = 0;
    bool complete = false;
    auto start = std::chrono::high_resolution_clock::now();
    // Start poll thread
    // std::thread poll([&] {
    while (!end_program)
    {
        if (count > 1)
        {
            // std::cout << "Next Batch count " << batch << std::endl;
            batch += 1;
            count = 0;
        }

        // Poll the client for data and add to our lidar scan
        OS1::client_state st = OS1::poll_client(*cli);
        if (st & OS1::client_state::ERROR)
        {
            std::cerr << "Client returned error state" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (st & OS1::client_state::LIDAR_DATA)
        {
            if (OS1::read_lidar_packet(*cli, lidar_buf))
            {

                // Get starting timepoint
                // auto start = std::chrono::high_resolution_clock::now();

                // Call the function, here sort()
                batch_and_update(lidar_buf, it, map_tensor, complete);
                // Get ending timepoint
                // auto stop = std::chrono::high_resolution_clock::now();

                // // Get duration. Substart timepoints to
                // // get durarion. To cast it to proper unit
                // // use duration cast method
                // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

                // std::cout << "Time taken by function: "
                //     << duration.count() << " microseconds" << std::endl;
                // tf::TTypes<float, 3>::Tensor map_tensor_2 = tf_tensor.tensor<float, 3>();
                for (int k = 15096; k < 16384 + 15096; k++)
                {
                    // if(std::abs(map_tensor(0,k,0)) > 1e-5)
                    // std::cout << "k " << std::to_string(k) << std::endl;
                    // std::cout << map_tensor(0,k,0) << " " <<  map_tensor(0,k,1) << " "   <<  map_tensor(0,k,2) << std::endl;
                    map_tensor_small(0, k, 0) = map_tensor(0, k, 0);
                    map_tensor_small(0, k, 1) = map_tensor(0, k, 1);
                    map_tensor_small(0, k, 2) = map_tensor(0, k, 2);
                }
                // std::cout << tf_tensor.shape() << std::endl;
                // std::cout << map_tensor_2.setZero() << std::endl;
                if (complete)
                {
                    // std::chrono::duration_cast<std::chrono::microseconds>(stop)
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    std::cout << "Time taken by lidar data acquisition: "
                              << duration.count() << " microseconds" << std::endl;
                    size_t frame_timestamp_us =
                        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
                    std::cout << "time stamp: " << frame_timestamp_us << std::endl;
                    // MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(map_tensor).At(mediapipe::Timestamp(count))));

                    std::unique_ptr<tf::Tensor> temp_tensor(new tf::Tensor);
                    tf::Tensor &tf_tensor2 = tf_tensor_small;
                    RET_CHECK(temp_tensor->CopyFrom(tf_tensor2, tf::TensorShape({1, 16384, 3})));
                    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(temp_tensor.release()).At(mediapipe::Timestamp(frame_timestamp_us))));
                    // Get the graph result packet, or stop if that fails.
                    // complete = false;

                    start = std::chrono::high_resolution_clock::now();
                    mediapipe::Packet packet;
                    // poller.Next(&packet);
                    if (!poller.Next(&packet))
                        break; //end_program = true;
                    // std::cout << packet.Get<tf::Tensor>().DebugString() << std::endl;
                    // auto matrix = packet.Get<tf::Tensor>().tensor<float,2>();
                    // std::cout << matrix(15096,0) << "\t";
                    // std::cout << matrix(15096,1) <<"\t";
                    // std::cout << matrix(15096,2) <<"\t";
                    // std::cout << matrix(15096,3) <<"\t";
                    // std::cout << matrix(15096,4) <<"\t";
                    // std::cout << matrix(15096,5) <<"\t";
                    // std::cout << matrix(15096,6) <<"\t";
                    // std::cout << matrix(15096,7) <<std::endl;
                    stop = std::chrono::high_resolution_clock::now();
                    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    std::cout << "Time taken by mediapipe map: "
                              << duration.count() << " microseconds" << std::endl;
                    count += 1;
                    map_tensor.setZero();
                    start = std::chrono::high_resolution_clock::now();
                    complete = false;
                    std::cout << "count " << std::to_string(count) << std::endl;
                }

                // if (batch > 0){
                //     end_program = true;
                // }
            }
        }
        if (st & OS1::client_state::IMU_DATA)
        {
            OS1::read_imu_packet(*cli, imu_buf);
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    // MP_RETURN_IF_ERROR(graph.WaitUntilDone());
    // });

    // print the content of the point cloud
    // while(batch < 1){}
    // if (batch == 1){
    //     auto it2 = ls->data_;
    //     int npoints = 0;
    //     for (;npoints < H * W ; npoints++)
    //     {
    //         std::cout << it2[npoints] << std::endl;
    //     }
    // }
    // while(true){}
    // clean up
    // poll.join();
    // return mediapipe::OkStatus();
    return graph.WaitUntilDone();
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ::mediapipe::Status run_status = RunMPPGraph();
    if (!run_status.ok())
    {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else
    {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
