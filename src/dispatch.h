#pragma once
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "buffer.h"
#include "export.h"
#include "library.h"
#include "ns_ptr.h"

namespace MTL {
class Device;
class CommandBuffer;
class ComputeCommandEncoder;
class ComputePipelineState;
class Function;
}  // namespace MTL

class MetalRuntime;

// The GPU rejected or aborted a committed command buffer.
struct MR_API DispatchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// newComputePipelineState failed: the source compiled, the back end for this GPU did not.
struct MR_API PipelineBuildError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// A 1-, 2- or 3-dimensional extent; unused dimensions are 1.
struct Dim3 {
    size_t x = 1, y = 1, z = 1;
    size_t volume() const { return x * y * z; }
    bool operator==(const Dim3& o) const { return x == o.x && y == o.y && z == o.z; }
};

// A kernel argument the compiler reports as used.
struct BindingInfo {
    size_t index;
    std::string name;
};

class MR_API ComputePipeline {
   public:
    // `label` is for error messages only.
    ComputePipeline(MTL::Device* device, MTL::Function* function, const std::string& label);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&&) = delete;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    MTL::ComputePipelineState* handle() const { return pipeline_.get(); }
    const std::string& label() const { return label_; }

    // This kernel's limits, which register pressure can push below the device's.
    size_t max_threads_per_threadgroup() const { return max_threads_per_threadgroup_; }
    size_t thread_execution_width() const { return thread_execution_width_; }
    size_t static_threadgroup_memory_length() const { return static_threadgroup_memory_length_; }

    // Used buffer and [[threadgroup(i)]] arguments.
    const std::vector<BindingInfo>& buffer_bindings() const { return buffer_bindings_; }
    const std::vector<BindingInfo>& threadgroup_bindings() const { return threadgroup_bindings_; }

    // Whole SIMD groups up to the kernel's ceiling.
    Dim3 default_threadgroup(Dim3 grid) const;

    // Validates binding count, threadgroup dims and threadgroup memory; results are cached.
    void validate_shape(size_t binding_count, const std::vector<size_t>& threadgroup_memory,
                        Dim3 threadgroup, size_t device_max_threadgroup_memory);

   private:
    struct LaunchShape {
        size_t binding_count = 0;
        size_t device_max_threadgroup_memory = 0;
        Dim3 threadgroup;
        std::vector<size_t> threadgroup_memory;
    };

    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
    std::string label_;
    std::vector<BindingInfo> buffer_bindings_;
    std::vector<BindingInfo> threadgroup_bindings_;

    size_t max_threads_per_threadgroup_ = 0;
    size_t thread_execution_width_ = 0;
    size_t static_threadgroup_memory_length_ = 0;

    // Linear scan: a kernel sees a handful of shapes. Full means re-validate.
    static constexpr size_t kMaxValidatedShapes = 16;
    std::mutex shape_cache_mutex_;
    std::vector<LaunchShape> validated_shapes_;
};

// One kernel launch. Buffers bind at indices 0..n-1, scalars (setBytes) at
// the indices after. With indirect_grid set, grid is ignored and the GPU reads
// three uint32 threadgroup counts from that buffer at indirect_offset.
struct Launch {
    ComputePipeline* pipeline = nullptr;
    std::vector<std::pair<Buffer*, size_t>> buffers;  // (buffer, byte offset)
    std::vector<std::pair<const void*, size_t>> scalars;
    std::vector<size_t> threadgroup_memory;
    Dim3 grid;
    Dim3 threadgroup;
    Buffer* indirect_grid = nullptr;
    size_t indirect_offset = 0;
};

// Several launches in one command buffer. A concurrent encoder lets them
// overlap, ordered only across barrier(). Methods are synchronized.
class MR_API CommandBatch {
   public:
    explicit CommandBatch(MetalRuntime& rt, bool concurrent = false);
    ~CommandBatch();
    CommandBatch(CommandBatch&&) = delete;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    // Validates, then encodes. Throws before touching the encoder if invalid.
    void add(const Launch& launch);

    // Orders buffer writes across it; only needed on a concurrent encoder.
    void barrier();

    // Submits without blocking.
    void commit();

    // Commits if needed, then blocks until done. Throws DispatchError on a
    // faulted command buffer. Every caller blocks, however many times.
    void wait();

    // Device-side execution seconds for the whole batch; set by wait().
    std::optional<double> gpu_time() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return gpu_time_;
    }

   private:
    void commit_locked();

    // Declaration order: the encoder is released before its command buffer.
    NS::SharedPtr<MTL::CommandBuffer> command_buffer_;
    NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
    mutable std::mutex state_mutex_;
    bool committed_ = false;
    bool waited_ = false;
    bool non_uniform_ = false;
    size_t max_threadgroup_memory_ = 0;
    std::optional<double> gpu_time_;
};

MR_API void dispatch(MetalRuntime& rt, const Launch& launch);
