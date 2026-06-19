#include "runtime.h"

#include "metal.h"

MetalRuntime::MetalRuntime() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        throw NoDeviceError(
            "no Metal device found: this runtime needs macOS with a "
            "Metal-capable GPU");
    }
    queue_ = device_->newCommandQueue();
    if (!queue_) {
        device_->release();
        device_ = nullptr;
        throw NoDeviceError("could not create a Metal command queue on this device");
    }
    non_uniform_threadgroups_ = device_->supportsFamily(MTL::GPUFamilyApple4) ||
                                device_->supportsFamily(MTL::GPUFamilyMac2);
}

MetalRuntime::~MetalRuntime() {
    if (queue_) queue_->release();
    if (device_) device_->release();
}

std::string MetalRuntime::device_name() const {
    AutoreleaseScope scope;
    return std::string(device_->name()->utf8String());
}

bool MetalRuntime::has_unified_memory() const { return device_->hasUnifiedMemory(); }

size_t MetalRuntime::recommended_max_working_set_size() const {
    return static_cast<size_t>(device_->recommendedMaxWorkingSetSize());
}

size_t MetalRuntime::max_threads_per_threadgroup() const {
    return static_cast<size_t>(device_->maxThreadsPerThreadgroup().width);
}

std::shared_ptr<Library> MetalRuntime::library_for(const std::string& msl_source,
                                                   const CompileOptions& options) {
    // Length-prefixed so the options blob can't be confused with the start of
    // the source text.
    std::string serialized = options.cache_key();
    const std::string key = std::to_string(serialized.size()) + ":" + serialized + msl_source;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = libraries_.find(key);
        if (it != libraries_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.second);
            return it->second.first;
        }
    }

    // Compiled outside the lock: newLibrary runs the whole MSL front end and
    // takes milliseconds, and holding the mutex across it would serialize
    // every thread compiling a *different* kernel. Two threads racing on the
    // same source both compile, and the loser's copy is dropped below --
    // wasted work in a rare case, in exchange for no contention in the common
    // one.
    auto library = std::make_shared<Library>(device_, msl_source, options);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = libraries_.find(key);
    if (it != libraries_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second.second);
        return it->second.first;
    }
    lru_.push_front(key);
    libraries_.emplace(key, std::make_pair(library, lru_.begin()));
    evict_locked();
    return library;
}

void MetalRuntime::evict_locked() {
    if (cache_limit_ == 0) return;
    while (libraries_.size() > cache_limit_) {
        libraries_.erase(lru_.back());
        lru_.pop_back();
    }
}

void MetalRuntime::set_library_cache_limit(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_limit_ = limit;
    evict_locked();
}

size_t MetalRuntime::library_cache_limit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_limit_;
}

size_t MetalRuntime::library_cache_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return libraries_.size();
}

void MetalRuntime::clear_library_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    libraries_.clear();
    lru_.clear();
}

MetalRuntime& runtime() {
    static MetalRuntime rt;
    return rt;
}
