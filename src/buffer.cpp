#include "buffer.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "metal.h"

Buffer::Buffer(MTL::Device* device, size_t size_bytes) : size_(size_bytes) {
    // newBuffer(0) returns nullptr, which would surface as "failed to allocate" --
    // an out-of-memory message for what is really an empty array. A
    // zero-element shape is legitimate (a degenerate axis in generated code),
    // so round the allocation up to one byte and keep reporting size() == 0.
    // Nothing indexes into it, and contents() stays a valid pointer.
    buffer_ = device->newBuffer(std::max<size_t>(size_bytes, 1), MTL::ResourceStorageModeShared);
    if (!buffer_) {
        throw std::runtime_error("failed to allocate Metal buffer of " +
                                 std::to_string(size_bytes) + " bytes");
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
