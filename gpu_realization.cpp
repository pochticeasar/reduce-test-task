#include "gpu_realization.hpp"
#include <CL/opencl.hpp>
#include <cstddef>
#include <iostream>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define CONCAT_(a) "-DLOCAL_WORK_SIZE=" #a
#define CONCAT(a) CONCAT_(a)

#ifndef LOCAL_WORK_SIZE
#define LOCAL_WORK_SIZE 64
#endif

cl_uint round_to(cl_uint n, cl_uint m) {
    cl_uint del = n / m + (n % m == 0 ? 0 : 1);
    return del * m;
}

cl_uint round_to_div(cl_uint n, cl_uint m) { return round_to(n, m) / m; }

cl_float calculate_host(const cl::Context &context,
                        const cl::CommandQueue &command_queue,
                        cl::Kernel &kernel, const cl::Buffer &a, size_t n,
                        std::vector<cl::Event> &events) {
    size_t rounded_n = round_to(n, LOCAL_WORK_SIZE * 2);

    cl::Buffer sums_mem(context, CL_MEM_READ_WRITE,
                        sizeof(cl_float) * (rounded_n / (LOCAL_WORK_SIZE * 2)),
                        nullptr);

    size_t global_work_size[] = {round_to_div(rounded_n, 2)};
    size_t local_work_size[] = {LOCAL_WORK_SIZE};
    cl::NDRange global_nderange{round_to_div(rounded_n, 2)};
    cl::NDRange local_nderange{LOCAL_WORK_SIZE};
    kernel.setArg(0, sizeof(cl_mem), &a);
    kernel.setArg(1, sizeof(cl_mem), &sums_mem);
    kernel.setArg(2, sizeof(cl_uint), &n);
    command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_nderange,
                                       local_nderange, nullptr,
                                       &events.emplace_back());
    if (rounded_n > LOCAL_WORK_SIZE * 2) {
        return calculate_host(context, command_queue, kernel, sums_mem,
                              round_to_div(rounded_n, LOCAL_WORK_SIZE * 2),
                              events);
    } else {
        cl_float ans;
        command_queue.enqueueReadBuffer(sums_mem, CL_TRUE, 0,
                                        sizeof(cl_float) * 1, &ans, NULL,
                                        &events.emplace_back());
        return ans;
    }
};

GPUAnswer gpu_calculate(const cl::Device &device,
                        const std::vector<cl_float> &array) {

    cl::Context context(device);
    cl::CommandQueue command_queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::ifstream kernel_file("reduce.cl");
    if (!kernel_file.is_open()) {
        std::cerr << "Error while reading kernel file 'reduce.cl'\n";
        return GPUAnswer{}; 
    }
    std::ostringstream oss;
    oss << kernel_file.rdbuf();
    std::string source_str = oss.str();
    kernel_file.close();

    cl::Program program(context, source_str);
    try {
        program.build({device}, CONCAT(LOCAL_WORK_SIZE));
    } catch (const cl::Error &err) {
        std::string build_log =
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Error during build: " << build_log << "\n";
        return GPUAnswer{};
    }

    cl::Kernel kernel(program, "reduce");
    size_t n = array.size();
    cl::Buffer buffer_a(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n,
                        nullptr);
    std::vector<cl::Event> events;
    command_queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, sizeof(cl_float) * n,
                                     array.data(), NULL,
                                     &events.emplace_back());
    GPUAnswer answer;
    answer.sum =
        calculate_host(context, command_queue, kernel, buffer_a, n, events);
    for (auto event : events) {
        auto time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        auto time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        answer.time_with_memory += (double)(time_end - time_start) / 1000000.0;

        if (event.getInfo<CL_EVENT_COMMAND_TYPE>() ==
            CL_COMMAND_NDRANGE_KERNEL) {
            answer.time_without_memory +=
                (double)(time_end - time_start) / 1000000.0;
        }
    }
    return answer;
}
