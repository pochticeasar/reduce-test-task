#pragma once
#include <CL/opencl.hpp>
#include <vector>

struct GPU_answer {
    float sum;
    std::chrono::microseconds elapsed_time;
};

GPU_answer gpu_calculate(cl::Device device, std::vector<cl_float> &array);
