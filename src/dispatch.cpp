#include "dispatch.h"

#include <algorithm>
#include <string>

#include "metal.h"

namespace {

// Metal's own ceiling on a setBytes argument. Past this the data has to live
// in a Buffer, so say that rather than letting Metal fail the encode.
constexpr size_t kMaxInlineScalarBytes = 4096;

// Threadgroup memory allocations are sized in 16-byte units.
constexpr size_t kThreadgroupMemoryAlignment = 16;

// MTLDispatchThreadgroupsIndirectArguments: three uint32 threadgroup counts.
constexpr size_t kIndirectArgumentsSize = 3 * sizeof(uint32_t);

std::string to_string(Dim3 d) {
    return "(" + std::to_string(d.x) + ", " + std::to_string(d.y) + ", " + std::to_string(d.z) +
           ")";
}

// Round up rather than reject: the caller asked for room for N elements, and
// Metal only allocates in 16-byte units.
size_t rounded_threadgroup_length(size_t length) {
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
    pipeline_ = device->newComputePipelineState(function, MTL::PipelineOptionBindingInfo,
                                                &reflection, &error);
    function->release();
    if (!pipeline_) {
        std::string message = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error("failed to build compute pipeline: " + message);
    }
    if (reflection) {
        NS::Array* bindings = reflection->bindings();
        for (NS::UInteger i = 0; i < bindings->count(); ++i) {
            auto* binding = static_cast<MTL::Binding*>(bindings->object(i));
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

ComputePipeline::~ComputePipeline() {
    if (pipeline_) pipeline_->release();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : pipeline_(other.pipeline_),
      label_(std::move(other.label_)),
      buffer_bindings_(std::move(other.buffer_bindings_)),
      threadgroup_bindings_(std::move(other.threadgroup_bindings_)) {
    other.pipeline_ = nullptr;
}

size_t ComputePipeline::max_threads_per_threadgroup() const {
    return pipeline_->maxTotalThreadsPerThreadgroup();
}

size_t ComputePipeline::thread_execution_width() const { return pipeline_->threadExecutionWidth(); }

size_t ComputePipeline::static_threadgroup_memory_length() const {
    return pipeline_->staticThreadgroupMemoryLength();
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

CommandBatch::CommandBatch(MTL::CommandQueue* queue, bool concurrent) {
    AutoreleaseScope scope;
    MTL::Device* device = queue->device();
    non_uniform_ =
        device->supportsFamily(MTL::GPUFamilyApple4) || device->supportsFamily(MTL::GPUFamilyMac2);
    max_threadgroup_memory_ = device->maxThreadgroupMemoryLength();

    // Both come back autoreleased. A batch outlives the pool that created it
    // (it stays open across successive add() calls, which in the Python
    // bindings means across separate calls into the extension), so retain
    // them and hand the references back in the destructor.
    command_buffer_ = queue->commandBuffer();
    if (!command_buffer_) {
        throw DispatchError("could not create a Metal command buffer");
    }
    command_buffer_->retain();
    encoder_ = concurrent ? command_buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent)
                          : command_buffer_->computeCommandEncoder();
    if (!encoder_) {
        command_buffer_->release();
        command_buffer_ = nullptr;
        throw DispatchError("could not create a Metal compute command encoder");
    }
    encoder_->retain();
}

CommandBatch::~CommandBatch() {
    // An abandoned batch (an exception between add() and wait()) still holds a
    // live encoder; Metal insists the encoding be closed before the command
    // buffer is released.
    if (!committed_ && encoder_) encoder_->endEncoding();
    if (encoder_) encoder_->release();
    if (command_buffer_) command_buffer_->release();
}

void CommandBatch::add(const Launch& launch) {
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
    if (tg.x == 0 || tg.y == 0 || tg.z == 0) {
        throw std::invalid_argument("dispatch: threadgroup " + to_string(tg) +
                                    " must be non-zero in every dimension");
    }

    // Not clamped. Silently shrinking the threadgroup changes what a kernel
    // that indexes threadgroup memory by thread_position_in_threadgroup
    // computes, and the caller gets wrong numbers with no indication why.
    // Per-dimension checks first: the product could wrap size_t.
    size_t max_total = launch.pipeline->max_threads_per_threadgroup();
    if (tg.x > max_total || tg.y > max_total || tg.z > max_total || tg.volume() > max_total) {
        throw std::invalid_argument("dispatch: threadgroup " + to_string(tg) + " has " +
                                    std::to_string(tg.volume()) +
                                    " threads, but this kernel supports at most " +
                                    std::to_string(max_total) + " per threadgroup");
    }

    size_t binding_count = launch.buffers.size() + launch.scalars.size();
    if (binding_count > 31) {
        throw std::invalid_argument("dispatch: " + std::to_string(binding_count) +
                                    " buffer bindings requested, but Metal allows at most 31");
    }
    for (const auto& scalar : launch.scalars) {
        if (scalar.second > kMaxInlineScalarBytes) {
            throw std::invalid_argument("dispatch: inline scalar of " +
                                        std::to_string(scalar.second) + " bytes exceeds Metal's " +
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

    // A used binding the launch doesn't cover reads unbound memory.
    for (const BindingInfo& binding : launch.pipeline->buffer_bindings()) {
        if (binding.index >= binding_count) {
            throw std::invalid_argument(
                "dispatch: kernel '" + launch.pipeline->label() + "' reads argument '" +
                binding.name + "' at buffer index " + std::to_string(binding.index) +
                ", but only " + std::to_string(binding_count) +
                " bindings were provided (buffers bind first, scalars after)");
        }
    }
    for (const BindingInfo& binding : launch.pipeline->threadgroup_bindings()) {
        if (binding.index >= launch.threadgroup_memory.size()) {
            throw std::invalid_argument("dispatch: kernel '" + launch.pipeline->label() +
                                        "' uses threadgroup memory '" + binding.name +
                                        "' at index " + std::to_string(binding.index) +
                                        "; pass its byte size in threadgroup_memory");
        }
    }

    // Static and dynamic threadgroup memory share one budget; exceeding it
    // downstream is a process abort (Metal API validation), not an error.
    size_t threadgroup_total = launch.pipeline->static_threadgroup_memory_length();
    for (size_t length : launch.threadgroup_memory) {
        threadgroup_total += rounded_threadgroup_length(length);
    }
    if (threadgroup_total > max_threadgroup_memory_) {
        throw std::invalid_argument(
            "dispatch: " + std::to_string(threadgroup_total) +
            " bytes of threadgroup memory requested (including " +
            std::to_string(launch.pipeline->static_threadgroup_memory_length()) +
            " bytes of static allocations in the kernel), but this device supports at most " +
            std::to_string(max_threadgroup_memory_) + " bytes per threadgroup");
    }

    if (launch.indirect_grid) {
        size_t needed = launch.indirect_offset + kIndirectArgumentsSize;
        if (launch.indirect_offset % 4 != 0 || needed > launch.indirect_grid->size()) {
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
        encoder_->setBytes(scalar.first, scalar.second, index++);
    }
    for (size_t i = 0; i < launch.threadgroup_memory.size(); ++i) {
        encoder_->setThreadgroupMemoryLength(
            rounded_threadgroup_length(launch.threadgroup_memory[i]), i);
    }

    MTL::Size mtl_tg = MTL::Size::Make(tg.x, tg.y, tg.z);
    if (launch.indirect_grid) {
        // Indirect counts are threadgroups; there is no non-uniform variant.
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
    if (committed_) {
        throw DispatchError("cannot add a barrier to a batch that has already been committed");
    }
    encoder_->memoryBarrier(MTL::BarrierScopeBuffers);
}

void CommandBatch::commit() {
    if (committed_) return;
    committed_ = true;

    AutoreleaseScope scope;
    encoder_->endEncoding();
    command_buffer_->commit();
}

void CommandBatch::wait() {
    if (waited_) return;
    commit();
    waited_ = true;

    AutoreleaseScope scope;
    command_buffer_->waitUntilCompleted();
    gpu_time_ = command_buffer_->GPUEndTime() - command_buffer_->GPUStartTime();

    if (command_buffer_->status() == MTL::CommandBufferStatusError) {
        NS::Error* error = command_buffer_->error();
        std::string message =
            error ? error->localizedDescription()->utf8String() : "unknown GPU error";
        throw DispatchError("kernel execution failed: " + message);
    }
}

void dispatch(MTL::CommandQueue* queue, const Launch& launch) {
    CommandBatch batch(queue);
    batch.add(launch);
    batch.wait();
}
