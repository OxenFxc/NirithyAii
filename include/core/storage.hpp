#pragma once

#include "core/device.hpp"
#include <memory>
#include <cstddef>

namespace nirithyaii {

class Storage {
public:
    Storage(size_t nbytes, Device device = Device(DeviceType::CPU));
    ~Storage();

    void* data();
    const void* data() const;
    size_t size() const;
    Device device() const;

private:
    void* data_;
    size_t nbytes_;
    Device device_;
};

} // namespace nirithyaii
