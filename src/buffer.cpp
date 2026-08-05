#include "buffer.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "metal.h"

Buffer::Buffer(MTL::Device* device, size_t size_bytes) : size_(size_bytes) {
    // newBuffer(0) returns nullptr, which would surface as "failed to allocate".
    // A zero-element shape is legitimate (a degenerate axis in generated code),
    // so round the allocation up to one byte and keep reporting size() == 0.
    buffer_ = device->newBuffer(std::max<size_t>(size_bytes, 1), MTL::ResourceStorageModeShared);
    if (!buffer_) {
        throw AllocationError("failed to allocate Metal buffer of " + std::to_string(size_bytes) +
                              " bytes");
    }
}

Buffer::Buffer(MTL::Device* device, void* external_ptr, size_t size_bytes) : size_(size_bytes) {
    size_t page_size = (size_t)getpagesize();
    if ((uintptr_t)external_ptr % page_size != 0 || size_bytes % page_size != 0) {
        throw std::invalid_argument(
            "Buffer: external_ptr and size_bytes must both be a multiple of the page size (" +
            std::to_string(page_size) + " bytes) to wrap without copying");
    }
    // No deallocator: the caller frees external_ptr, not this Buffer or Metal.
    buffer_ = device->newBuffer(external_ptr, size_bytes, MTL::ResourceStorageModeShared, nullptr);
    if (!buffer_) {
        throw AllocationError("failed to wrap external memory of " + std::to_string(size_bytes) +
                              " bytes as a Metal buffer");
    }
}

Buffer::~Buffer() {
    if (buffer_) buffer_->release();
}

Buffer::Buffer(Buffer&& other) noexcept : buffer_(other.buffer_), size_(other.size_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
}

void* Buffer::contents() const { return buffer_->contents(); }
