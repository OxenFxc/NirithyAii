#include "core/tensor.hpp"
#include <numeric>
#include <stdexcept>
#include <iomanip>

namespace nirithyaii {

Tensor::Tensor(std::vector<int> shape, Device device, DataType dtype)
    : shape_(shape), device_(device), dtype_(dtype), storage_offset_(0) {
    compute_strides();

    size_t size = numel() * element_size();
    storage_ = std::make_shared<Storage>(size, device);
}

void* Tensor::data() {
    return static_cast<char*>(storage_->data()) + storage_offset_;
}

const void* Tensor::data() const {
    return static_cast<const char*>(storage_->data()) + storage_offset_;
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
}

const std::vector<int>& Tensor::stride() const {
    return stride_;
}

size_t Tensor::numel() const {
    size_t n = 1;
    for (int s : shape_) {
        n *= s;
    }
    return n;
}

size_t Tensor::element_size() const {
    if (dtype_ == DataType::FLOAT32) {
        return sizeof(float);
    } else if (dtype_ == DataType::INT32) {
        return sizeof(int);
    }
    return 1;
}

DataType Tensor::dtype() const {
    return dtype_;
}

Device Tensor::device() const {
    return device_;
}

void Tensor::compute_strides() {
    stride_.resize(shape_.size());
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        stride_[i] = stride;
        stride *= shape_[i];
    }
}

void Tensor::fill(float value) {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("fill() only implemented for FLOAT32");
    }

    if (device_.is_cpu()) {
        float* ptr = data_ptr<float>();
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = value;
        }
    } else {
         throw std::runtime_error("Device not supported for fill()");
    }
}

void Tensor::print() const {
    std::cout << "Tensor(";
    std::cout << "shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    std::cout << "]";
    std::cout << ", device=" << device_.to_string();
    std::cout << ", dtype=" << (dtype_ == DataType::FLOAT32 ? "Float32" : "Int32");
    std::cout << ")" << std::endl;

    // Simple print of data (flat for now)
    if (device_.is_cpu() && dtype_ == DataType::FLOAT32) {
        const float* ptr = data_ptr<float>();
        size_t n = numel();
        std::cout << "[";
        for (size_t i = 0; i < std::min(n, (size_t)20); ++i) {
             std::cout << ptr[i] << (i < std::min(n, (size_t)20) - 1 ? ", " : "");
        }
        if (n > 20) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
}

} // namespace nirithyaii
