#include "cpu_realization.hpp"
#include "gpu_realization.hpp"
#include "iostream"
#include <CL/opencl.hpp>
#include <cstddef>
#include <optional>
#include <ostream>
#include <random>
#include <vector>

enum class type_of_device { GPU = 0, CPU = 1, ANY = 2 };

struct Arguments {
    type_of_device device_type;
    int index_of_device;
};

std::optional<cl::Device> get_devices(Arguments &arguments) {
    std::vector<cl::Platform> plaforms;
    cl::Platform::get(&plaforms);

    std::vector<cl::Device> gpu_devices;
    std::vector<cl::Device> cpu_devices;

    for (auto p : plaforms) {
        std::vector<cl::Device> devices_of_platform;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devices_of_platform);
        gpu_devices.insert(gpu_devices.end(), devices_of_platform.begin(),
                           devices_of_platform.end());
        p.getDevices(CL_DEVICE_TYPE_CPU, &devices_of_platform);
        cpu_devices.insert(cpu_devices.end(), devices_of_platform.begin(),
                           devices_of_platform.end());
    }

    std::vector<cl::Device> target_devices;
    if (arguments.device_type == type_of_device::CPU) {
        target_devices = std::move(cpu_devices);
    } else if (arguments.device_type == type_of_device::GPU) {
        target_devices = std::move(gpu_devices);
    } else {
        target_devices = std::move(gpu_devices);
        target_devices.insert(target_devices.end(), cpu_devices.begin(),
                              cpu_devices.end());
    }
    if (arguments.index_of_device >= target_devices.size()) {
        return std::nullopt;
    }
    return target_devices[arguments.index_of_device];
}

int main(int argc, char **argv) {
    Arguments arguments;
    arguments.index_of_device = 0;
    arguments.device_type = type_of_device::ANY;
    for (int i = 1; i < argc; i++) {
        std::string argument_first = argv[i];
        if (argument_first == "--help") {
            printf("%s", "reduce.exe  [ --device-type { gpu | cpu | any } ]\n"
                         "            [ --device-index index ]\n");
            return 0;
        } else if (argument_first == "--device-type") {
            std::string argument = argv[++i];
            if (argument == "cpu") {
                arguments.device_type = type_of_device::CPU;
            } else if (argument == "gpu") {
                arguments.device_type = type_of_device::GPU;
            } else if (argument == "any") {
                arguments.device_type = type_of_device::ANY;
            } else {
                std::cerr << "Incorrect argument\n";
                return 1;
            }
        } else if (argument_first == "--device-index") {
            std::string argument = argv[++i];
            long long num;
            try {
                size_t idx;
                num = std::stoll(argument, &idx);

                if (idx != argument.size() || num < 0) {
                    std::cerr << "Incorrect argument" << std::endl;
                    return 1;
                }
            } catch (const std::invalid_argument &e) {
                std::cerr << "Incorrect argument" << std::endl;
                return 1;
            } catch (const std::out_of_range &e) {
                std::cerr << "Incorrect argument" << std::endl;
                return 1;
            }
            arguments.index_of_device = num;
        } else {
            std::cerr << "Incorrect argument\n";
            return 1;
        }
    }
    size_t n;
    std::optional<cl::Device> current_device = get_devices(arguments);
    if (current_device) {
        std::string device_name = current_device->getInfo<CL_DEVICE_NAME>();
        cl::Platform current_platform{
            current_device->getInfo<CL_DEVICE_PLATFORM>()};
        std::string platform_name =
            current_platform.getInfo<CL_PLATFORM_NAME>();
        std::cout << "Device: " << device_name
                  << "\tPlatform: " << platform_name << std::endl;
    } else {
        std::cerr << "No devices found\n";
        return 3;
    }
    cl_float ans;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    {
        std::cout << "Validity test" << std::endl;
        std::vector<cl_float> array(107107);
        for (auto &value : array) {
            value = dist(gen);
        }
        float reference_sum = 0;
        for (auto e : array) {
            reference_sum += e;
        }
        std::cout << "Test array size: " << array.size() << std::endl
                  << "Reference sum " << reference_sum << std::endl;
        CPUAnswer cpu_answer = cpu_calculate(array);
        GPUAnswer gpu_answer;
        try {
            gpu_answer = gpu_calculate(current_device.value(), array);
        } catch (cl::Error &err) {
            std::cerr << "OpenCL error: " << err.what() << " returned "
                      << err.err() << std::endl;
            return 5;
        }

        std::cout << "CPU ans: " << cpu_answer.sum << "\nTime in ms: "
                  << cpu_answer.elapsed_time.count() / 1000.0f << std::endl;
        std::cout << "GPU ans: " << gpu_answer.sum
                  << "\nTime in ms without memory: "
                  << gpu_answer.time_without_memory
                  << "\nTime in ms with memory: " << gpu_answer.time_with_memory
                  << std::endl;
    }

    std::cout << "------------------------" << std::endl;

    {
        std::cout << "Large test" << std::endl;
        std::vector<cl_float> array(107107 * 5000);
        for (auto &value : array) {
            value = dist(gen);
        }

        CPUAnswer cpu_answer = cpu_calculate(array);
        GPUAnswer gpu_answer;
        try {
            gpu_answer = gpu_calculate(current_device.value(), array);
        } catch (cl::Error &err) {
            std::cerr << "OpenCL error: " << err.what() << " returned "
                      << err.err() << std::endl;
            return 5;
        }

        std::cout << "CPU Time in ms: "
                  << cpu_answer.elapsed_time.count() / 1000.0f << std::endl;
        std::cout << "GPU Time in ms without memory: "
                  << gpu_answer.time_without_memory << std::endl
                  << "GPU Time in ms with memory: " << gpu_answer.time_with_memory
                  << std::endl;
    }
    return 0;
}
