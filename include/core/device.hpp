#pragma once

#include <string>
#include <iostream>

namespace nirithyaii {

enum class DeviceType {
    CPU,
    CUDA
};

struct Device {
    DeviceType type;
    int index;

    Device(DeviceType type = DeviceType::CPU, int index = 0)
        : type(type), index(index) {}

    bool is_cpu() const { return type == DeviceType::CPU; }
    bool is_cuda() const { return type == DeviceType::CUDA; }

    std::string to_string() const {
        std::string s = (type == DeviceType::CPU) ? "CPU" : "CUDA";
        return s + ":" + std::to_string(index);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.to_string();
    return os;
}

} // namespace nirithyaii
