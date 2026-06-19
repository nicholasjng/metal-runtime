#include "buffer.h"

#include <stdexcept>

#include "metal.h"

Buffer::Buffer(MTL::Device* device, size_t size_bytes) : size_(size_bytes) {
    buffer_ = device->newBuffer(size_bytes, MTL::ResourceStorageModeShared);
    if (!buffer_) {
        throw std::runtime_error("failed to allocate Metal buffer");
    }
}

Buffer::~Buffer() {
    if (buffer_) buffer_->release();
}

Buffer::Buffer(Buffer&& other) : buffer_(other.buffer_), size_(other.size_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
}

void* Buffer::contents() const { return buffer_->contents(); }
