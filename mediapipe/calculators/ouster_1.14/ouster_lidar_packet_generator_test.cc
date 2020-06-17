#include <unistd.h>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>

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


#include "os1.h"
#include "os1_packet.h"
#include "os1_util.h"
#include "lidar_scan.h"
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


/**
 * Print usage
 */
void print_help() {
    std::cout
        << "Usage: viz [options] [hostname] [udp_destination]\n"
        << "Options:\n"
        << "  -m <512x10 | 512x20 | 1024x10 | 1024x20 | 2048x10> : lidar mode, "
           "default 1024x10\n"
        << "  -f <path> : use provided metadata file; do not configure via TCP"
        << std::endl;
}

std::string read_metadata(const std::string& meta_file) {
    std::stringstream buf{};
    std::ifstream ifs{};
    ifs.open(meta_file);
    buf << ifs.rdbuf();
    ifs.close();

    if (!ifs) {
        std::cerr << "Failed to read " << meta_file
                  << "; check that the path is valid" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return buf.str();
}

int main(int argc, char** argv) {
    int W = 1024;
    int H = OS1::pixels_per_column;
    OS1::lidar_mode mode = OS1::MODE_1024x10;
    bool do_config = true;  // send tcp commands to configure sensor
    std::string metadata{};
    tf::Tensor tf_tensor(tf::DT_FLOAT, tf::TensorShape({1, 65536, 3}));
    tf::TTypes<float, 3>::Tensor map_tensor = tf_tensor.tensor<float, 3>();
    try {
        int c = 0;
        while ((c = getopt(argc, argv, "hm:f:")) != -1) {
            switch (c) {
                case 'h':
                    print_help();
                    return 1;
                    break;
                case 'm':
                    mode = OS1::lidar_mode_of_string(optarg);
                    if (mode) {
                        W = OS1::n_cols_of_lidar_mode(mode);
                    } else {
                        std::cout << "Lidar Mode must be 512x10, 512x20, "
                                     "1024x10, 1024x20, or 2048x10"
                                  << std::endl;
                        print_help();
                        std::exit(EXIT_FAILURE);
                    }
                    break;
                case 'f':
                    do_config = false;
                    metadata = read_metadata(optarg);
                    break;
                case '?':
                    std::cout << "Invalid Argument Format" << std::endl;
                    print_help();
                    std::exit(EXIT_FAILURE);
                    break;
            }
        }
    } catch (const std::exception& ex) {
        std::cout << "Invalid Argument Format: " << ex.what() << std::endl;
        print_help();
        std::exit(EXIT_FAILURE);
    }

    if (do_config && argc != optind + 2) {
        std::cerr << "Expected 2 arguments after options" << std::endl;
        print_help();
        std::exit(EXIT_FAILURE);
    }

    std::shared_ptr<OS1::client> cli;
    if (do_config) {
        std::cout << "Configuring sensor: " << argv[optind]
                  << " UDP Destination:" << argv[optind + 1] << std::endl;
        cli = OS1::init_client(argv[optind], argv[optind + 1], mode);
    } else {
        std::cout << "Listening for sensor data" << std::endl;
        cli = OS1::init_client();
    }

    if (!cli) {
        std::cerr << "Failed to initialize client" << std::endl;
        print_help();
        std::exit(EXIT_FAILURE);
    }

    uint8_t lidar_buf[OS1::lidar_packet_bytes + 1];
    uint8_t imu_buf[OS1::imu_packet_bytes + 1];

    auto ls = std::unique_ptr<ouster::LidarScan>(new ouster::LidarScan(W, H));

    // auto vh = viz::init_viz(W, H);

    if (do_config) metadata = OS1::get_metadata(*cli);

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

    int count = 0;
    int batch = 0;
    bool complete = false;
    auto start = std::chrono::high_resolution_clock::now(); 
    // Start poll thread
    std::thread poll([&] {
        while (!end_program) {
            if (count > 1){
                // std::cout << "Next Batch count " << batch << std::endl;
                batch += 1;
                count = 0;
            }

            // Poll the client for data and add to our lidar scan
            OS1::client_state st = OS1::poll_client(*cli);
            if (st & OS1::client_state::ERROR) {
                std::cerr << "Client returned error state" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (st & OS1::client_state::LIDAR_DATA) {
                if (OS1::read_lidar_packet(*cli, lidar_buf)){

                    // Get starting timepoint 
                    // auto start = std::chrono::high_resolution_clock::now(); 
                
                    // Call the function, here sort() 
                    batch_and_update(lidar_buf, it , map_tensor, complete);
                    // Get ending timepoint 
                    // auto stop = std::chrono::high_resolution_clock::now(); 
                
                    // // Get duration. Substart timepoints to  
                    // // get durarion. To cast it to proper unit 
                    // // use duration cast method 
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
                
                    // std::cout << "Time taken by function: "
                    //     << duration.count() << " microseconds" << std::endl; 
                    // tf::TTypes<float, 3>::Tensor map_tensor_2 = tf_tensor.tensor<float, 3>();
                    // for(int k = 0; k < 65536; k ++){
                    //     std::cout << map_tensor_2(0,k,0) << " " <<  map_tensor_2(0,k,1) << " "   <<  map_tensor_2(0,k,2) << std::endl;
                    
                    // }
                    // std::cout << tf_tensor.shape() << std::endl;
                    // std::cout << map_tensor_2.setZero() << std::endl;
                    if(complete){
                        auto stop = std::chrono::high_resolution_clock::now(); 
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
                        std::cout << "Time taken by function: "
                            << duration.count() << " microseconds" << std::endl; 
                        count += 1;
                        map_tensor.setZero();
                        start = std::chrono::high_resolution_clock::now(); 
                    }

                    // if (batch > 0){
                    //     end_program = true;
                    // }
                }
            }
            if (st & OS1::client_state::IMU_DATA) {
                OS1::read_imu_packet(*cli, imu_buf);
            }
        }
    });

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
    while(true){}
    // clean up
    poll.join();
    return EXIT_SUCCESS;
}
