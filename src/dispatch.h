#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "buffer.h"
#include "library.h"

namespace MTL {
class Device;
class CommandQueue;
class CommandBuffer;
class ComputeCommandEncoder;
class ComputePipelineState;
}  // namespace MTL

// The GPU rejected or aborted a committed command buffer. Distinct from a
// compile error: the kernel built fine, the execution didn't.
struct DispatchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A 1-, 2- or 3-dimensional extent. Metal always works in 3D; unused
// dimensions are 1.
struct Dim3 {
    size_t x = 1, y = 1, z = 1;
    size_t volume() const { return x * y * z; }
};

class ComputePipeline {
   public:
    ComputePipeline(MTL::Device* device, const Library& library, const std::string& function_name);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    MTL::ComputePipelineState* handle() const { return pipeline_; }

    // Hardware limits for this specific kernel, not the device: register
    // pressure can push a kernel's ceiling well below the device maximum.
    size_t max_threads_per_threadgroup() const;
    size_t thread_execution_width() const;

    // A threadgroup that fills whole SIMD groups without exceeding the
    // kernel's own ceiling -- what dispatch() picks when the caller doesn't.
    Dim3 default_threadgroup(Dim3 grid) const;

   private:
    MTL::ComputePipelineState* pipeline_ = nullptr;
};

// One kernel launch. `buffers` bind at indices 0..n-1; `scalars` are copied
// inline with setBytes at the indices that follow, so a kernel can take an
// element count or a stride without the caller allocating a buffer per
// scalar. `threadgroup_memory` sizes the threadgroup address space at
// indices 0..m-1.
struct Launch {
    ComputePipeline* pipeline = nullptr;
    std::vector<Buffer*> buffers;
    std::vector<std::pair<const void*, size_t>> scalars;
    std::vector<size_t> threadgroup_memory;
    Dim3 grid;
    Dim3 threadgroup;
};

// Several launches encoded into one command buffer: one commit and one GPU
// round-trip for the whole sequence instead of one per kernel. The encoder is
// serial, so launches observe each other's writes in submission order without
// explicit barriers.
class CommandBatch {
   public:
    explicit CommandBatch(MTL::CommandQueue* queue);
    ~CommandBatch();
    CommandBatch(CommandBatch&&) = delete;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    // Validates `launch` against the pipeline's limits and the device's
    // threadgroup capabilities, then encodes it. Throws before touching the
    // encoder if the launch is invalid.
    void add(const Launch& launch);

    // Commits and blocks until the GPU is done. Throws DispatchError if the
    // command buffer faulted. Calling it again is a no-op, so the destructor
    // can rely on it.
    void wait();

   private:
    MTL::CommandBuffer* command_buffer_ = nullptr;
    MTL::ComputeCommandEncoder* encoder_ = nullptr;
    bool committed_ = false;
};

// A single-launch batch, committed and waited on immediately.
void dispatch(MTL::CommandQueue* queue, const Launch& launch);
