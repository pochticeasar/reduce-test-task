#pragma once
#include <CL/opencl.hpp>
#include <vector>

struct GPU_answer {
    float sum;
    double time_without_memory = 0;
    double time_with_memory = 0;
};

GPU_answer gpu_calculate(cl::Device device, std::vector<cl_float> &array);
