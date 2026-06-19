
#include "runtime.h"

#include "metal.h"

MetalRuntime::MetalRuntime() {
    device_ = MTL::CreateSystemDefaultDevice();
    queue_ = device_->newCommandQueue();
}

MetalRuntime::~MetalRuntime() {
    if (queue_) queue_->release();
    if (device_) device_->release();
}

Library& MetalRuntime::library_for(const std::string& msl_source) {
    auto it = libraries_.find(msl_source);
    if (it == libraries_.end()) {
        it = libraries_.emplace(msl_source, Library(device_, msl_source)).first;
    }
    return it->second;
}

MetalRuntime& runtime() {
    static MetalRuntime rt;
    return rt;
}
