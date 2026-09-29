#include "dispatch.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "metal.h"
#include "runtime.h"

namespace {

// Metal's ceiling on a setBytes argument.
constexpr size_t kMaxInlineScalarBytes = 4096;

// Threadgroup memory allocations are sized in 16-byte units.
constexpr size_t kThreadgroupMemoryAlignment = 16;

// MTLDispatchThreadgroupsIndirectArguments: three uint32 threadgroup counts.
constexpr size_t kIndirectArgumentsSize = 3 * sizeof(uint32_t);

std::string to_string(Dim3 d) {
    return "(" + std::to_string(d.x) + ", " + std::to_string(d.y) + ", " + std::to_string(d.z) +
           ")";
}

size_t rounded_threadgroup_length(size_t length) {
    if (length > std::numeric_limits<size_t>::max() - (kThreadgroupMemoryAlignment - 1)) {
        throw std::invalid_argument("dispatch: threadgroup memory allocation overflows");
    }
    return (length + kThreadgroupMemoryAlignment - 1) / kThreadgroupMemoryAlignment *
           kThreadgroupMemoryAlignment;
}

}  // namespace

ComputePipeline::ComputePipeline(MTL::Device* device, MTL::Function* function,
                                 const std::string& label)
    : label_(label) {
    AutoreleaseScope scope;
    NS::Error* error = nullptr;

    // Binding reflection backs the host-side launch validation in add().
    MTL::ComputePipelineReflection* reflection = nullptr;
    pipeline_ = NS::TransferPtr(device->newComputePipelineState(
        function, MTL::PipelineOptionBindingInfo, &reflection, &error));
    if (!pipeline_) {
        throw PipelineBuildError("failed to build compute pipeline: " + describe(error));
    }
    max_threads_per_threadgroup_ = pipeline_->maxTotalThreadsPerThreadgroup();
    thread_execution_width_ = pipeline_->threadExecutionWidth();
    static_threadgroup_memory_length_ = pipeline_->staticThreadgroupMemoryLength();

    if (reflection) {
        NS::Array* bindings = reflection->bindings();
        for (NS::UInteger i = 0; i < bindings->count(); ++i) {
            auto* binding = (MTL::Binding*)bindings->object(i);
            if (!binding->isUsed()) continue;
            BindingInfo info{binding->index(), binding->name()->utf8String()};
            if (binding->type() == MTL::BindingTypeBuffer) {
                buffer_bindings_.push_back(std::move(info));
            } else if (binding->type() == MTL::BindingTypeThreadgroupMemory) {
                threadgroup_bindings_.push_back(std::move(info));
            }
        }
    }
}

ComputePipeline::~ComputePipeline() = default;

void ComputePipeline::validate_shape(size_t binding_count,
                                     const std::vector<size_t>& threadgroup_memory, Dim3 tg,
                                     size_t device_max_threadgroup_memory) {
    {
        std::lock_guard<std::mutex> lock(shape_cache_mutex_);
        for (const LaunchShape& seen : validated_shapes_) {
            if (seen.binding_count == binding_count &&
                seen.device_max_threadgroup_memory == device_max_threadgroup_memory &&
                seen.threadgroup == tg && seen.threadgroup_memory == threadgroup_memory) {
                return;
            }
        }
    }

    if (tg.x == 0 || tg.y == 0 || tg.z == 0) {
        throw std::invalid_argument("dispatch: threadgroup " + to_string(tg) +
                                    " must be non-zero in every dimension");
    }

    size_t max_total = max_threads_per_threadgroup();
    if (tg.x > max_total || tg.y > max_total || tg.z > max_total || tg.volume() > max_total) {
        throw std::invalid_argument("dispatch: threadgroup " + to_string(tg) + " has " +
                                    std::to_string(tg.volume()) +
                                    " threads, but this kernel supports at most " +
                                    std::to_string(max_total) + " per threadgroup");
    }

    if (binding_count > 31) {
        throw std::invalid_argument("dispatch: " + std::to_string(binding_count) +
                                    " buffer bindings requested, but Metal allows at most 31");
    }

    // A used binding the launch doesn't cover reads unbound memory.
    for (const BindingInfo& binding : buffer_bindings_) {
        if (binding.index >= binding_count) {
            throw std::invalid_argument(
                "dispatch: kernel '" + label_ + "' reads argument '" + binding.name +
                "' at buffer index " + std::to_string(binding.index) + ", but only " +
                std::to_string(binding_count) +
                " bindings were provided (buffers bind first, scalars after)");
        }
    }
    for (const BindingInfo& binding : threadgroup_bindings_) {
        if (binding.index >= threadgroup_memory.size()) {
            throw std::invalid_argument("dispatch: kernel '" + label_ +
                                        "' uses threadgroup memory '" + binding.name +
                                        "' at index " + std::to_string(binding.index) +
                                        "; pass its byte size in threadgroup_memory");
        }
    }

    // Static and dynamic threadgroup memory share one budget, exceeding it downstream
    // is a process abort (Metal API validation), not an error.
    size_t threadgroup_total = static_threadgroup_memory_length();
    for (size_t length : threadgroup_memory) {
        size_t rounded = rounded_threadgroup_length(length);
        if (threadgroup_total > device_max_threadgroup_memory ||
            rounded > device_max_threadgroup_memory - threadgroup_total) {
            throw std::invalid_argument(
                "dispatch: threadgroup memory exceeds this device's budget of " +
                std::to_string(device_max_threadgroup_memory) + " bytes per threadgroup");
        }
        threadgroup_total += rounded;
    }
    if (threadgroup_total > device_max_threadgroup_memory) {
        throw std::invalid_argument(
            "dispatch: " + std::to_string(threadgroup_total) +
            " bytes of threadgroup memory requested (including " +
            std::to_string(static_threadgroup_memory_length()) +
            " bytes of static allocations in the kernel), but this device supports at most " +
            std::to_string(device_max_threadgroup_memory) + " bytes per threadgroup");
    }

    std::lock_guard<std::mutex> lock(shape_cache_mutex_);
    if (validated_shapes_.size() < kMaxValidatedShapes) {
        validated_shapes_.push_back(
            LaunchShape{binding_count, device_max_threadgroup_memory, tg, threadgroup_memory});
    }
}

Dim3 ComputePipeline::default_threadgroup(Dim3 grid) const {
    size_t max_total = std::max<size_t>(max_threads_per_threadgroup(), 1);
    size_t width = std::max<size_t>(thread_execution_width(), 1);

    Dim3 tg;
    if (grid.y <= 1 && grid.z <= 1) {
        // Whole SIMD groups, as many as the kernel's register budget allows,
        // but never more threads than there is work to do.
        size_t total = (max_total / width) * width;
        if (total == 0) total = max_total;
        tg.x = std::max<size_t>(std::min(grid.x, total), 1);
        return tg;
    }
    // 2D/3D: one SIMD group along x, then spend what's left on y and z.
    tg.x = std::max<size_t>(std::min(grid.x, width), 1);
    tg.y = std::max<size_t>(std::min(grid.y, max_total / tg.x), 1);
    tg.z = std::max<size_t>(std::min(grid.z, max_total / (tg.x * tg.y)), 1);
    return tg;
}

CommandBatch::CommandBatch(MetalRuntime& rt, bool concurrent)
    : non_uniform_(rt.supports_non_uniform_threadgroups()),
      max_threadgroup_memory_(rt.max_threadgroup_memory_length()) {
    AutoreleaseScope scope;
    MTL::CommandQueue* queue = rt.queue();

    // Both come back autoreleased and the batch outlives this pool, so retain.
    command_buffer_ = NS::RetainPtr(queue->commandBuffer());
    if (!command_buffer_) {
        throw DispatchError("could not create a Metal command buffer");
    }
    encoder_ = NS::RetainPtr(
        concurrent ? command_buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent)
                   : command_buffer_->computeCommandEncoder());
    if (!encoder_) {
        throw DispatchError("could not create a Metal compute command encoder");
    }
}

CommandBatch::~CommandBatch() {
    // Metal requires an open encoder to be ended before its command buffer is released.
    if (!committed_ && encoder_) encoder_->endEncoding();
}

void CommandBatch::add(const Launch& launch) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (committed_) {
        throw DispatchError("cannot add to a batch that has already been committed");
    }
    if (!launch.pipeline) {
        throw std::invalid_argument("dispatch: launch has no pipeline");
    }

    const Dim3 grid = launch.grid;
    const Dim3 tg = launch.threadgroup;

    if (!launch.indirect_grid && (grid.x == 0 || grid.y == 0 || grid.z == 0)) {
        throw std::invalid_argument("dispatch: grid " + to_string(grid) +
                                    " must be non-zero in every dimension");
    }

    size_t binding_count = launch.buffers.size() + launch.scalars.size();
    launch.pipeline->validate_shape(binding_count, launch.threadgroup_memory, tg,
                                    max_threadgroup_memory_);

    for (const auto& scalar : launch.scalars) {
        if (scalar.size > kMaxInlineScalarBytes) {
            throw std::invalid_argument("dispatch: inline scalar of " +
                                        std::to_string(scalar.size) + " bytes exceeds Metal's " +
                                        std::to_string(kMaxInlineScalarBytes) +
                                        "-byte setBytes limit; pass it as a Buffer instead");
        }
    }
    for (size_t i = 0; i < launch.buffers.size(); ++i) {
        const auto& [buffer, offset] = launch.buffers[i];
        if (offset >= buffer->size() && !(offset == 0 && buffer->size() == 0)) {
            throw std::invalid_argument(
                "dispatch: buffers[" + std::to_string(i) + "] offset " + std::to_string(offset) +
                " is out of bounds for a buffer of " + std::to_string(buffer->size()) + " bytes");
        }
    }

    if (launch.indirect_grid) {
        size_t size = launch.indirect_grid->size();
        if (launch.indirect_offset % 4 != 0 || launch.indirect_offset > size ||
            kIndirectArgumentsSize > size - launch.indirect_offset) {
            throw std::invalid_argument(
                "dispatch: indirect grid arguments need " + std::to_string(kIndirectArgumentsSize) +
                " bytes at a 4-byte-aligned offset, but offset " +
                std::to_string(launch.indirect_offset) + " into a buffer of " +
                std::to_string(launch.indirect_grid->size()) + " bytes doesn't provide that");
        }
    } else if (!non_uniform_ && (grid.x % tg.x || grid.y % tg.y || grid.z % tg.z)) {
        throw std::invalid_argument(
            "dispatch: this GPU does not support non-uniform threadgroups, so grid " +
            to_string(grid) + " must divide evenly by threadgroup " + to_string(tg));
    }

    encoder_->setComputePipelineState(launch.pipeline->handle());
    NS::UInteger index = 0;
    for (const auto& [buffer, offset] : launch.buffers) {
        encoder_->setBuffer(buffer->handle(), offset, index++);
    }
    for (const auto& scalar : launch.scalars) {
        encoder_->setBytes(scalar.data, scalar.size, index++);
    }
    for (size_t i = 0; i < launch.threadgroup_memory.size(); ++i) {
        encoder_->setThreadgroupMemoryLength(
            rounded_threadgroup_length(launch.threadgroup_memory[i]), i);
    }

    MTL::Size mtl_tg = MTL::Size::Make(tg.x, tg.y, tg.z);
    if (launch.indirect_grid) {
        encoder_->dispatchThreadgroups(launch.indirect_grid->handle(), launch.indirect_offset,
                                       mtl_tg);
    } else if (non_uniform_) {
        encoder_->dispatchThreads(MTL::Size::Make(grid.x, grid.y, grid.z), mtl_tg);
    } else {
        encoder_->dispatchThreadgroups(MTL::Size::Make(grid.x / tg.x, grid.y / tg.y, grid.z / tg.z),
                                       mtl_tg);
    }
}

void CommandBatch::barrier() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (committed_) {
        throw DispatchError("cannot add a barrier to a batch that has already been committed");
    }
    encoder_->memoryBarrier(MTL::BarrierScopeBuffers);
}

void CommandBatch::commit() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    commit_locked();
}

void CommandBatch::commit_locked() {
    if (committed_) return;
    committed_ = true;

    AutoreleaseScope scope;
    encoder_->endEncoding();
    command_buffer_->commit();
}

void CommandBatch::wait() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        commit_locked();
    }
    {
        // Every caller blocks; waitUntilCompleted returns at once on a finished buffer.
        AutoreleaseScope scope;
        command_buffer_->waitUntilCompleted();
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!gpu_time_) {
        gpu_time_ = command_buffer_->GPUEndTime() - command_buffer_->GPUStartTime();
    }

    if (command_buffer_->status() == MTL::CommandBufferStatusError) {
        throw DispatchError("kernel execution failed: " + describe(command_buffer_->error()));
    }
}

void dispatch(MetalRuntime& rt, const Launch& launch) {
    CommandBatch batch(rt);
    batch.add(launch);
    batch.wait();
}
