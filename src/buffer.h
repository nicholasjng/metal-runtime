#pragma once
#include <cstddef>

namespace MTL {
class Device;
class Buffer;
}  // namespace MTL

// Wraps an MTL::Buffer allocated with ResourceStorageModeShared: on Apple Silicon's
// unified memory, host and GPU read/write the same bytes, so contents() is a plain
// host pointer -- no explicit upload/readback copy once allocated.
class Buffer {
   public:
    Buffer(MTL::Device* device, size_t size_bytes);
    ~Buffer();
    Buffer(Buffer&& other);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* contents() const;
    size_t size() const { return size_; }
    MTL::Buffer* handle() const { return buffer_; }

   private:
    MTL::Buffer* buffer_ = nullptr;
    size_t size_ = 0;
};
