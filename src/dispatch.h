#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "buffer.h"
#include "library.h"

namespace MTL {
class Device;
class CommandQueue;
class ComputePipelineState;
}  // namespace MTL

// Wraps an MTL::ComputePipelineState built from a named function in a Library.
class ComputePipeline {
   public:
    ComputePipeline(MTL::Device* device, const Library& library, const std::string& function_name);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other);
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    MTL::ComputePipelineState* handle() const { return pipeline_; }

   private:
    MTL::ComputePipelineState* pipeline_ = nullptr;
};

// Dispatches `pipeline` over a 1D grid of `grid_size` threads (grouped in
// `threadgroup_size`-sized threadgroups, clamped to the pipeline's own max),
// binding `buffers` at sequential buffer indices [0, buffers.size()). Blocks
// until the GPU finishes.
void dispatch(MTL::CommandQueue* queue, ComputePipeline& pipeline,
              const std::vector<Buffer*>& buffers, size_t grid_size, size_t threadgroup_size);
