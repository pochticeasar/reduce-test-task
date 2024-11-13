#include "gpu_realization.hpp"
#include <CL/opencl.hpp>
#include <iostream>

#include <fstream>
#include <iostream>
#include <sstream>

#define CONCAT_(a) "-DLOCAL2=" #a
#define CONCAT(a) CONCAT_(a)

#ifndef LOCAL2
#define LOCAL2 64
#endif

cl_uint round_to(cl_uint n, cl_uint m) {
    cl_uint del = n / m + (n % m == 0 ? 0 : 1);
    return del * m;
}

cl_uint round_to_div(cl_uint n, cl_uint m) { return round_to(n, m) / m; }

cl_int calculate_host(cl_context context, cl_command_queue command_queue,
                      cl_kernel up_and_down_sweep, cl_mem a_mem, size_t n,
                      cl_event *events, cl_uint *event_i, cl_float *ans) {
    cl_int error = CL_SUCCESS;
    size_t rounded_n = round_to(n, LOCAL2 * 2);
    cl_mem sums_mem = clCreateBuffer(
        context, CL_MEM_READ_WRITE,
        sizeof(cl_float) * (rounded_n / (LOCAL2 * 2)), NULL, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "ClCreateBuffer is not successful: %d\n";
        return 1;
    }
    size_t global_work_size[] = {round_to_div(rounded_n, 2)};
    size_t local_work_size[] = {LOCAL2};

    clSetKernelArg(up_and_down_sweep, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(up_and_down_sweep, 1, sizeof(cl_mem), &sums_mem);
    clSetKernelArg(up_and_down_sweep, 2, sizeof(cl_uint), &n);

    error = clEnqueueNDRangeKernel(command_queue, up_and_down_sweep, 1, NULL,
                                   global_work_size, local_work_size, 0, NULL,
                                   &events[(*event_i)++]);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel is not successful: %d\n",
                error);
        clReleaseMemObject(sums_mem);
        return 1;
    }

    if (rounded_n > LOCAL2 * 2) {
        error = calculate_host(context, command_queue, up_and_down_sweep,
                               sums_mem, round_to_div(rounded_n, LOCAL2 * 2),
                               events, event_i, ans);
        if (error != CL_SUCCESS) {
            clReleaseMemObject(sums_mem);
            return 1;
        };
    } else {
        error = clEnqueueReadBuffer(command_queue, sums_mem, CL_TRUE, 0,
                                    sizeof(cl_float) * 1, ans, 0, NULL,
                                    &events[(*event_i)++]);
        if (error != CL_SUCCESS) {
            fprintf(stderr, "ClEnqueueReadBuffer is not successful: %d\n",
                    error);
            clReleaseMemObject(sums_mem);
            return 1;
        }
    }
    clReleaseMemObject(sums_mem);
    return CL_SUCCESS;
};

GPU_answer gpu_calculate(cl::Device device, std::vector<cl_float> &array) {

     try {
        cl::Context context(device);
        cl::CommandQueue command_queue(context, device, CL_QUEUE_PROFILING_ENABLE);

        std::ifstream kernel_file("reduce.cl");
        if (!kernel_file.is_open()) {
            std::cerr << "Error while reading kernel file 'reduce.cl'\n";
            return GPU_answer{}; // Return an empty answer or handle accordingly
        }
        std::ostringstream oss;
        oss << kernel_file.rdbuf();
        std::string source_str = oss.str();
        kernel_file.close();

        cl::Program program(context, source_str);
        try {
            program.build({device}, CONCAT(LOCAL2));
        } catch (const cl::Error &err) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "Error during build: " << build_log << "\n";
            return GPU_answer{};
        }
        
        cl::Kernel kernel(program, "reduce");

        // Allocate memory buffers
        size_t n = array.size();
        cl::Buffer buffer_a(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n, array.data());
        std::cout << "ok gpu build\n";
        return GPU_answer{};
    }  catch (cl::Error &err) {
        std::cerr << "OpenCL error: " << err.what() << " returned " << err.err() << "\n";
        return GPU_answer{};
    }
    //     // Set kernel arguments
    // //     kernel.setArg(0, buffer_a);
    // //     kernel.setArg(1, cl::Local(sizeof(cl_float) * /* local size */));

    // //     // Determine global and local work sizes
    // //     size_t local_size = 256; // Adjust based on your kernel
    // //     size_t global_size = ((n + local_size - 1) / local_size) *
    // // // cl::Context context({device});
    // // cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    // // FILE *file = fopen("reduce.cl", "rb");
    // // if (file == NULL) {
    // //     fprintf(stderr, "Error while reading kernel\n");
    // //     return 1;
    // // }
    // // if (fseek(file, 0, SEEK_END) != 0) {
    // //     fprintf(stderr, "Error while reading kernel\n");
    // //     fclose(file);
    // //     return 1;
    // // }
    // // size_t size = ftell(file);
    // // if (fseek(file, 0, SEEK_SET) != 0) {
    // //     fprintf(stderr, "Error while reading kernel\n");
    // //     fclose(file);
    // //     return 1;
    // // }

    // // char *source = malloc(size + 1);
    // // if (source == NULL) {
    // //     fprintf(stderr, "Error while allocating memory for kernel\n");
    // //     fclose(file);
    // //     return 1;
    // // }
    // // if (fread(source, 1, size, file) != size) {
    // //     fprintf(stderr, "Error while reading kernel\n");
    // //     fclose(file);
    // //     free(source);
    // //     return 1;
    // // }
    // // if (fclose(file) == EOF) {
    // //     fprintf(stderr, "Error while reading kernel\n");
    // //     free(source);
    // //     return 1;
    // // }
    // // source[size] = '\0';
    // // cl_program program = clCreateProgramWithSource(
    // //     context, 1, (const char **)&source, (const size_t *)&size, NULL);

    // // cl_int build_err =
    // //     clBuildProgram(program, 1, &device_id, CONCAT(LOCAL2), NULL, NULL);
    // // if (build_err != CL_SUCCESS) {
    // //     size_t log_size = 0;
    // //     clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
    // //                           &log_size);
    // //     char *log = malloc(log_size);
    // //     if (log == NULL) {
    // //         fprintf(stderr, "Failed to allocate memory for build log\n");
    // //         clReleaseProgram(program);
    // //         return 1;
    // //     };
    // //     clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
    // //                           log_size, log, NULL);
    // //     clReleaseProgram(program);
    // //     fprintf(stderr, "%s\n", log);
    // //     free(log);
    // //     return 1;
    // // }
    // // cl_int error = CL_SUCCESS;
    // // cl_command_queue command_queue = clCreateCommandQueue(
    // //     context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    // // if (error != CL_SUCCESS) {
    // //     fprintf(stderr, "Error while creating command queue\n");
    // //     clReleaseProgram(program);
    // //     free(source);
    // //     return 1;
    // // }

    // // cl_kernel up_and_down_sweep = clCreateKernel(program, "reduce", &error);

    // // // biggest possible n is 18446744073709551616
    // // // lowest possble blocksize is 2
    // // // log2(2 ** 64) == 64
    // // // 1 write + 64 kernels + 1 read = 66 events
    // // cl_event events[66];
    // // cl_uint event_i = 0;
    // // cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
    // //                               sizeof(cl_float) * array.size(), NULL, &error);
    // // if (error != CL_SUCCESS) {
    // //     fprintf(stderr, "ClCreateBuffer is not successful: %d\n", error);
    // //     clReleaseKernel(up_and_down_sweep);
    // //     clReleaseCommandQueue(command_queue);
    // //     clReleaseProgram(program);
    // //     return {};
    // // }
    // // error = clEnqueueWriteBuffer(command_queue, a_mem, CL_TRUE, 0,
    // //                              sizeof(cl_float) * n, a, 0, NULL,
    // //                              &events[event_i++]);
    // // if (error != CL_SUCCESS) {
    // //     fprintf(stderr, "ClEnqueueWriteBuffer is not successful: %d\n", error);
    // //     clReleaseMemObject(a_mem);
    // //     clReleaseKernel(up_and_down_sweep);
    // //     clReleaseCommandQueue(command_queue);
    // //     clReleaseProgram(program);
    // //     return {};
    // // }
    // // error = calculate_host(context, command_queue, up_and_down_sweep, a_mem, n,
    // //                        events, &event_i, ans);
    // // if (error != CL_SUCCESS) {
    // //     clReleaseMemObject(a_mem);
    // //     clReleaseKernel(up_and_down_sweep);
    // //     clReleaseCommandQueue(command_queue);
    // //     clReleaseProgram(program);
    // //     clReleaseContext(context);
    // //     return 1;
    // // }

    // // clFinish(command_queue);

    // // cl_ulong time_start, time_end;
    // // cl_double time_with_memory = 0;
    // // cl_double time_without_memory = 0;
    // // for (int i = 0; i < event_i; i++) {
    // //     clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
    // //                             sizeof(time_start), &time_start, NULL);
    // //     clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
    // //                             sizeof(time_end), &time_end, NULL);
    // //     time_with_memory += (double)(time_end - time_start) / 1000000.0;
    // //     if (i != 0 && i != event_i - 1) {
    // //         time_without_memory += (double)(time_end - time_start) / 1000000.0;
    // //     }
    // // }
    // // printf("Time: %g\t%g\n", time_without_memory, time_with_memory);
    // // printf("LOCAL_WORK_SIZE [%i, %i]\n", LOCAL2, 1);
    // // clReleaseMemObject(a_mem);
    // // clReleaseKernel(up_and_down_sweep);
    // // clReleaseCommandQueue(command_queue);
    // // clReleaseProgram(program);
    // // return 0;
}
