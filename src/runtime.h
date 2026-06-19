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

    // Cached by source string: repeated dispatch of an already-seen kernel
    // skips recompilation.
    Library& library_for(const std::string& msl_source);

   private:
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    std::unordered_map<std::string, Library> libraries_;
};

MetalRuntime& runtime();
