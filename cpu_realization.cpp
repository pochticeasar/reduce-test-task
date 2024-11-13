#include <CL/opencl.hpp>

#include "cpu_realization.hpp"
#include <chrono>
#include <omp.h>
#include <vector>

CPUAnswer cpu_calculate(const std::vector<cl_float> &array) {
    float sum = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < array.size(); ++i) {
        sum += array[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    CPUAnswer answer;

    answer.sum = sum;
    answer.elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
    return answer;
}
