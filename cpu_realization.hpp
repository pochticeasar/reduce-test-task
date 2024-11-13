#pragma once

#include <CL/opencl.hpp>
#include <chrono>

struct CPUAnswer {
    float sum = 0;
    std::chrono::microseconds elapsed_time;
};

CPUAnswer cpu_calculate(const std::vector<cl_float> &array);
