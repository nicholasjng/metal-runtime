#include "dispatch.h"

#include <algorithm>
#include <stdexcept>

#include "metal.h"

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

ComputePipeline::ComputePipeline(ComputePipeline&& other) : pipeline_(other.pipeline_) {
    other.pipeline_ = nullptr;
}

void dispatch(MTL::CommandQueue* queue, ComputePipeline& pipeline,
              const std::vector<Buffer*>& buffers, size_t grid_size, size_t threadgroup_size) {
    if (threadgroup_size == 0) {
        throw std::invalid_argument("dispatch: threadgroup_size must be > 0");
    }

    AutoreleaseScope scope;

    MTL::CommandBuffer* command_buffer = queue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline.handle());
    for (size_t i = 0; i < buffers.size(); ++i) {
        encoder->setBuffer(buffers[i]->handle(), 0, i);
    }

    size_t max_threads = pipeline.handle()->maxTotalThreadsPerThreadgroup();
    size_t group_size = std::min(threadgroup_size, max_threads);

    encoder->dispatchThreads(MTL::Size::Make(grid_size, 1, 1), MTL::Size::Make(group_size, 1, 1));
    encoder->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}
