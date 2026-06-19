#pragma once
#include <cstddef>

namespace MTL {
class Device;
class Buffer;
}  // namespace MTL

// Unified memory, so host and GPU read/write the same bytes.
// This makes contents() a plain pointer, no upload/readback copy needed.
class Buffer {
   public:
    Buffer(MTL::Device* device, size_t size_bytes);

    // Wraps caller-owned memory with no copy. external_ptr must stay valid
    // and unmoved for this Buffer's lifetime; the caller keeps ownership and
    // frees it only after this Buffer is destroyed. Requires page-aligned
    // external_ptr/size_bytes; throws std::invalid_argument otherwise.
    Buffer(MTL::Device* device, void* external_ptr, size_t size_bytes);

    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* contents() const;

    // The requested size, 0 for an empty buffer (=1 byte allocation).
    size_t size() const { return size_; }
    MTL::Buffer* handle() const { return buffer_; }

   private:
    MTL::Buffer* buffer_ = nullptr;
    size_t size_ = 0;
};
