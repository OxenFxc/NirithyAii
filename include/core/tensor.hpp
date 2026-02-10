#pragma once

#include "core/storage.hpp"
#include <vector>
#include <numeric>
#include <iostream>

namespace nirithyaii {

enum class DataType {
    FLOAT32,
    INT32,
};

class Tensor {
public:
    Tensor(std::vector<int> shape, Device device = Device(DeviceType::CPU), DataType dtype = DataType::FLOAT32);

    // Data access
    void* data();
    const void* data() const;

    template<typename T>
    T* data_ptr() {
        return static_cast<T*>(data());
    }

    template<typename T>
    const T* data_ptr() const {
        return static_cast<const T*>(data());
    }

    // Properties
    const std::vector<int>& shape() const;
    const std::vector<int>& stride() const;
    size_t numel() const;
    size_t element_size() const;
    DataType dtype() const;
    Device device() const;

    // Helpers
    void print() const;
    void fill(float value); // Only for float32 for now

private:
    std::shared_ptr<Storage> storage_;
    size_t storage_offset_;
    std::vector<int> shape_;
    std::vector<int> stride_;
    DataType dtype_;
    Device device_;

    void compute_strides();
};

} // namespace nirithyaii
