#pragma once
#include <cstddef>
#include <mutex>
#include <optional>
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
class Function;
class BinaryArchive;
}  // namespace MTL

// The GPU rejected or aborted a committed command buffer.
// Distinct from a compile error: the kernel built fine, the execution didn't.
struct DispatchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// newComputePipelineState failed: the MSL front-end accepted the source, but
// the back-end compile to a pipeline for this specific GPU did not (e.g. a
// feature the target family lacks). A subclass of MSLCompileError so one
// `except CompileError` in Python catches both halves of compilation.
struct PipelineBuildError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// A 1-, 2- or 3-dimensional extent.
// Metal always works in 3D, unused dimensions are 1.
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

    // Hardware limits for this specific kernel, not the device:
    // register pressure can push a kernel's ceiling well below the device maximum.
    size_t max_threads_per_threadgroup() const;
    size_t thread_execution_width() const;
    size_t static_threadgroup_memory_length() const;

    // Used buffer and [[threadgroup(i)]] arguments, optimized-out ones are not listed.
    const std::vector<BindingInfo>& buffer_bindings() const { return buffer_bindings_; }
    const std::vector<BindingInfo>& threadgroup_bindings() const { return threadgroup_bindings_; }

    // A threadgroup that fills whole SIMD groups without exceeding the
    // kernel's own ceiling -- what dispatch() picks when the caller doesn't.
    Dim3 default_threadgroup(Dim3 grid) const;

    // Validates binding counts, threadgroup dims, and threadgroup memory budget,
    // not buffer identities or grid size, so a stepping loop relaunching the same
    // kernel at the same shape can cache the result.
    void validate_shape(size_t binding_count, const std::vector<size_t>& threadgroup_memory,
                        Dim3 threadgroup, size_t device_max_threadgroup_memory);

   private:
    // The subset of a launch this validation actually depends on.
    struct LaunchShape {
        size_t binding_count = 0;
        size_t device_max_threadgroup_memory = 0;
        Dim3 threadgroup;
        std::vector<size_t> threadgroup_memory;

        bool operator==(const LaunchShape& other) const {
            return binding_count == other.binding_count &&
                   device_max_threadgroup_memory == other.device_max_threadgroup_memory &&
                   threadgroup.x == other.threadgroup.x && threadgroup.y == other.threadgroup.y &&
                   threadgroup.z == other.threadgroup.z &&
                   threadgroup_memory == other.threadgroup_memory;
        }
    };

    MTL::ComputePipelineState* pipeline_ = nullptr;
    std::string label_;
    std::vector<BindingInfo> buffer_bindings_;
    std::vector<BindingInfo> threadgroup_bindings_;

    // Cached in the constructor, fixed for the lifetime of the pipeline.
    size_t max_threads_per_threadgroup_ = 0;
    size_t thread_execution_width_ = 0;
    size_t static_threadgroup_memory_length_ = 0;

    // Linear-scanned rather than hashed: a caller uses a handful of distinct
    // shapes per kernel, so a scan over a short vector beats hashing. Capped
    // so that a caller which does vary its shape every launch degrades to
    // re-validating (the pre-cache cost) instead of growing without bound and
    // turning the scan quadratic.
    static constexpr size_t kMaxValidatedShapes = 16;
    std::mutex shape_cache_mutex_;
    std::vector<LaunchShape> validated_shapes_;
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

// Several launches in one command buffer, one commit for the sequence.
// The default serial encoder orders launches, a concurrent encoder lets them overlap,
// with ordering only across an explicit barrier().
class CommandBatch {
   public:
    explicit CommandBatch(MTL::CommandQueue* queue, bool concurrent = false);
    ~CommandBatch();
    CommandBatch(CommandBatch&&) = delete;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    // Validates `launch` against the pipeline's limits, its compiler-reported
    // argument list, and the device's threadgroup capabilities, then encodes it.
    // Throws before touching the encoder if the launch is invalid.
    void add(const Launch& launch);

    // Orders buffer writes across it; only needed on a concurrent encoder.
    void barrier();

    // Closes the encoder and submits without blocking, so the next batch can
    // be encoded while this one runs.
    void commit();

    // Commits if commit() hasn't run, then blocks until the GPU is done.
    // Throws DispatchError if the command buffer faulted. Safe to call repeatedly
    // and from multiple threads: every caller blocks until completion.
    void wait();

    // Device-side execution seconds for the whole batch; set by wait().
    std::optional<double> gpu_time() const { return gpu_time_; }

    // The four command buffer timestamps, in one common epoch (seconds), so
    // their differences separate driver submission from GPU execution from
    // the host's own wake-up. Set by wait().
    struct Timestamps {
        double kernel_start = 0;  // driver began processing the submission
        double kernel_end = 0;    // driver handed it to the GPU
        double gpu_start = 0;     // GPU began executing
        double gpu_end = 0;       // GPU finished
    };
    std::optional<Timestamps> timestamps() const { return timestamps_; }

   private:
    void commit_locked();

    MTL::CommandBuffer* command_buffer_ = nullptr;
    MTL::ComputeCommandEncoder* encoder_ = nullptr;
    // Guards committed_, waited_, and the one-time timestamp capture.
    std::mutex state_mutex_;
    bool committed_ = false;
    bool waited_ = false;
    // Device capabilities, read once in the constructor.
    bool non_uniform_ = false;
    size_t max_threadgroup_memory_ = 0;
    std::optional<double> gpu_time_;
    std::optional<Timestamps> timestamps_;
};

// A single-launch batch, committed and waited on immediately.
void dispatch(MTL::CommandQueue* queue, const Launch& launch);
