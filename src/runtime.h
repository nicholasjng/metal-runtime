#pragma once
#include <string>
#include <unordered_map>

#include "library.h"

namespace MTL {
class Device;
class CommandQueue;
}  // namespace MTL

class MetalRuntime {
   public:
    MetalRuntime();
    ~MetalRuntime();
    MetalRuntime(MetalRuntime&&) = delete;
    MetalRuntime(const MetalRuntime&) = delete;
    MetalRuntime& operator=(const MetalRuntime&) = delete;

    MTL::Device* device() const { return device_; }
    MTL::CommandQueue* queue() const { return queue_; }

    // Returns a cached Library for `msl_source`, compiling it on first use so
    // repeated dispatch of an already-seen generated kernel doesn't recompile.
    Library& library_for(const std::string& msl_source);

   private:
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    std::unordered_map<std::string, Library> libraries_;
};

MetalRuntime& runtime();
