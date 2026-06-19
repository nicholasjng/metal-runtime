#pragma once
#include <cstddef>

namespace MTL {
class Device;
class Buffer;
}  // namespace MTL

// ResourceStorageModeShared: unified memory means host and GPU read/write the
// same bytes, so contents() is a plain pointer, no upload/readback copy needed.
class Buffer {
   public:
    Buffer(MTL::Device* device, size_t size_bytes);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* contents() const;

    // The requested size, which for an empty buffer is 0 even though the
    // underlying allocation isn't -- see the constructor.
    size_t size() const { return size_; }
    MTL::Buffer* handle() const { return buffer_; }

   private:
    MTL::Buffer* buffer_ = nullptr;
    size_t size_ = 0;
};
