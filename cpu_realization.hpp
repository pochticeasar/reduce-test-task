#pragma once 

#include <CL/opencl.hpp>
#include <chrono>

struct CPU_answer {
    float sum;
    std::chrono::microseconds elapsed_time;
};

CPU_answer cpu_calculate(std::vector<cl_float> &array);
