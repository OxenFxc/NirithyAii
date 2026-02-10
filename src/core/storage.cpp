#include "core/storage.hpp"
#include <cstdlib>
#include <stdexcept>
#include <iostream>

namespace nirithyaii {

Storage::Storage(size_t nbytes, Device device)
    : nbytes_(nbytes), device_(device), data_(nullptr) {
    if (nbytes == 0) {
        return;
    }

    if (device.is_cpu()) {
        data_ = std::malloc(nbytes);
        if (!data_) {
            throw std::runtime_error("Failed to allocate memory on CPU.");
        }
    } else if (device.is_cuda()) {
        throw std::runtime_error("CUDA not yet implemented.");
    } else {
        throw std::runtime_error("Unsupported device type.");
    }
}

Storage::~Storage() {
    if (data_) {
        if (device_.is_cpu()) {
            std::free(data_);
        } else if (device_.is_cuda()) {
            // cudaFree(data_);
        }
    }
}

void* Storage::data() {
    return data_;
}

const void* Storage::data() const {
    return data_;
}

size_t Storage::size() const {
    return nbytes_;
}

Device Storage::device() const {
    return device_;
}

} // namespace nirithyaii
