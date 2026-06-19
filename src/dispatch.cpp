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

std::string to_string(Dim3 d) {
    return "(" + std::to_string(d.x) + ", " + std::to_string(d.y) + ", " + std::to_string(d.z) +
           ")";
}

}  // namespace

ComputePipeline::ComputePipeline(MTL::Device* device, const Library& library,
                                 const std::string& function_name) {
    AutoreleaseScope scope;
    MTL::Function* fn = library.function(function_name);
    NS::Error* error = nullptr;
    pipeline_ = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pipeline_) {
        std::string message = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error("failed to build compute pipeline: " + message);
    }
}

ComputePipeline::~ComputePipeline() {
    if (pipeline_) pipeline_->release();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept : pipeline_(other.pipeline_) {
    other.pipeline_ = nullptr;
}

size_t ComputePipeline::max_threads_per_threadgroup() const {
    return pipeline_->maxTotalThreadsPerThreadgroup();
}

size_t ComputePipeline::thread_execution_width() const { return pipeline_->threadExecutionWidth(); }

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

CommandBatch::CommandBatch(MTL::CommandQueue* queue) {
    AutoreleaseScope scope;
    // Both come back autoreleased. A batch outlives the pool that created it
    // (it stays open across successive add() calls, which in the Python
    // bindings means across separate calls into the extension), so retain
    // them and hand the references back in the destructor.
    command_buffer_ = queue->commandBuffer();
    if (!command_buffer_) {
        throw DispatchError("could not create a Metal command buffer");
    }
    command_buffer_->retain();
    encoder_ = command_buffer_->computeCommandEncoder();
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
        throw DispatchError("cannot add to a batch that has already been waited on");
    }
    if (!launch.pipeline) {
        throw std::invalid_argument("dispatch: launch has no pipeline");
    }

    const Dim3 grid = launch.grid;
    const Dim3 tg = launch.threadgroup;

    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
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
    size_t max_total = launch.pipeline->max_threads_per_threadgroup();
    if (tg.volume() > max_total) {
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

    MTL::Device* device = command_buffer_->device();
    bool non_uniform =
        device->supportsFamily(MTL::GPUFamilyApple4) || device->supportsFamily(MTL::GPUFamilyMac2);
    if (!non_uniform && (grid.x % tg.x || grid.y % tg.y || grid.z % tg.z)) {
        throw std::invalid_argument(
            "dispatch: this GPU does not support non-uniform threadgroups, so grid " +
            to_string(grid) + " must divide evenly by threadgroup " + to_string(tg));
    }

    encoder_->setComputePipelineState(launch.pipeline->handle());
    NS::UInteger index = 0;
    for (Buffer* buffer : launch.buffers) {
        encoder_->setBuffer(buffer->handle(), 0, index++);
    }
    for (const auto& scalar : launch.scalars) {
        encoder_->setBytes(scalar.first, scalar.second, index++);
    }
    for (size_t i = 0; i < launch.threadgroup_memory.size(); ++i) {
        size_t length = launch.threadgroup_memory[i];
        // Round up rather than reject: the caller asked for room for N
        // elements, and Metal only allocates in 16-byte units.
        length = (length + kThreadgroupMemoryAlignment - 1) / kThreadgroupMemoryAlignment *
                 kThreadgroupMemoryAlignment;
        encoder_->setThreadgroupMemoryLength(length, i);
    }

    MTL::Size mtl_grid = MTL::Size::Make(grid.x, grid.y, grid.z);
    MTL::Size mtl_tg = MTL::Size::Make(tg.x, tg.y, tg.z);
    if (non_uniform) {
        encoder_->dispatchThreads(mtl_grid, mtl_tg);
    } else {
        encoder_->dispatchThreadgroups(MTL::Size::Make(grid.x / tg.x, grid.y / tg.y, grid.z / tg.z),
                                       mtl_tg);
    }
}

void CommandBatch::wait() {
    if (committed_) return;
    committed_ = true;

    AutoreleaseScope scope;
    encoder_->endEncoding();
    command_buffer_->commit();
    command_buffer_->waitUntilCompleted();

    // Without this a faulted command buffer -- a page fault in a kernel, a
    // device removal, a timeout -- returns from waitUntilCompleted exactly
    // like a successful one, and the caller reads back stale memory believing
    // the kernel ran.
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
