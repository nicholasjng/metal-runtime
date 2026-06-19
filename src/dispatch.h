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

// 1D grid of `grid_size` threads, `threadgroup_size` clamped to the pipeline's
// own max. `buffers` bind at sequential indices starting from 0. Blocks until
// the GPU finishes.
void dispatch(MTL::CommandQueue* queue, ComputePipeline& pipeline,
              const std::vector<Buffer*>& buffers, size_t grid_size, size_t threadgroup_size);
