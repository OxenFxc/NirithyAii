#include "core/tensor.hpp"
#include <iostream>

using namespace nirithyaii;

int main() {
    std::cout << "Creating a Tensor on CPU..." << std::endl;

    // Create a 2x3 tensor
    Tensor t({2, 3}, Device(DeviceType::CPU));

    std::cout << "Tensor created. Shape: [2, 3]" << std::endl;

    // Fill with 1.5
    t.fill(1.5f);

    std::cout << "Filled tensor with 1.5." << std::endl;

    // Print
    t.print();

    return 0;
}
