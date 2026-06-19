#pragma once
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
class Function;
class BinaryArchive;
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

// A kernel argument the compiler reports as used, so a launch that misses it
// fails host-side instead of faulting on the GPU.
struct BindingInfo {
    size_t index;
    std::string name;
};

class ComputePipeline {
   public:
    // Takes ownership of `function` (releases it after building the pipeline).
    // `label` is the entry-point name, used in error messages only. `archive`
    // (borrowed, may be null) skips recompilation on a matching prior build,
    // even across process runs; a fresh build stages into it.
    ComputePipeline(MTL::Device* device, MTL::Function* function, const std::string& label,
                    MTL::BinaryArchive* archive = nullptr);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    MTL::ComputePipelineState* handle() const { return pipeline_; }
    const std::string& label() const { return label_; }

    // Hardware limits for this specific kernel, not the device: register
    // pressure can push a kernel's ceiling well below the device maximum.
    size_t max_threads_per_threadgroup() const;
    size_t thread_execution_width() const;
    size_t static_threadgroup_memory_length() const;

    // Used buffer and [[threadgroup(i)]] arguments; optimized-out ones are
    // not listed.
    const std::vector<BindingInfo>& buffer_bindings() const { return buffer_bindings_; }
    const std::vector<BindingInfo>& threadgroup_bindings() const { return threadgroup_bindings_; }

    // A threadgroup that fills whole SIMD groups without exceeding the
    // kernel's own ceiling -- what dispatch() picks when the caller doesn't.
    Dim3 default_threadgroup(Dim3 grid) const;

    // Validates binding counts, threadgroup dims, and threadgroup memory
    // budget; not buffer identities or grid size, so a stepping loop
    // relaunching the same kernel at the same shape can cache the result.
    void validate_shape(size_t binding_count, const std::vector<size_t>& threadgroup_memory,
                        Dim3 threadgroup, size_t device_max_threadgroup_memory);

   private:
    MTL::ComputePipelineState* pipeline_ = nullptr;
    std::string label_;
    std::vector<BindingInfo> buffer_bindings_;
    std::vector<BindingInfo> threadgroup_bindings_;

    std::mutex shape_cache_mutex_;
    std::unordered_set<std::string> validated_shapes_;
};

// One kernel launch. Buffers bind at indices 0..n-1, each at a byte offset
// into its allocation; scalars are copied inline with setBytes at the
// indices after; threadgroup_memory sizes the threadgroup address space at
// indices 0..m-1. If indirect_grid is set, grid is ignored and the GPU reads
// three uint32 threadgroup counts from that buffer at indirect_offset when
// it reaches the dispatch.
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

// Several launches in one command buffer: one commit for the sequence. The
// default serial encoder orders launches; a concurrent encoder lets them
// overlap, with ordering only across an explicit barrier().
class CommandBatch {
   public:
    explicit CommandBatch(MTL::CommandQueue* queue, bool concurrent = false);
    ~CommandBatch();
    CommandBatch(CommandBatch&&) = delete;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    // Validates `launch` against the pipeline's limits, its compiler-reported
    // argument list, and the device's threadgroup capabilities, then encodes
    // it. Throws before touching the encoder if the launch is invalid.
    void add(const Launch& launch);

    // Orders buffer writes across it; only needed on a concurrent encoder.
    void barrier();

    // Closes the encoder and submits without blocking, so the next batch can
    // be encoded while this one runs.
    void commit();

    // Commits if commit() hasn't run, then blocks until the GPU is done.
    // Throws DispatchError if the command buffer faulted. Calling it again is
    // a no-op, so the destructor can rely on it.
    void wait();

    // Device-side execution seconds for the whole batch; set by wait().
    std::optional<double> gpu_time() const { return gpu_time_; }

   private:
    MTL::CommandBuffer* command_buffer_ = nullptr;
    MTL::ComputeCommandEncoder* encoder_ = nullptr;
    bool committed_ = false;
    bool waited_ = false;
    // Device capabilities, read once in the constructor.
    bool non_uniform_ = false;
    size_t max_threadgroup_memory_ = 0;
    std::optional<double> gpu_time_;
};

// A single-launch batch, committed and waited on immediately.
void dispatch(MTL::CommandQueue* queue, const Launch& launch);
