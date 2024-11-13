#pragma once
#include <CL/opencl.hpp>
#include <vector>

struct GPUAnswer {
    float sum = 0;
    double time_without_memory = 0;
    double time_with_memory = 0;
};

GPUAnswer gpu_calculate(const cl::Device &device,
                        const std::vector<cl_float> &array);
