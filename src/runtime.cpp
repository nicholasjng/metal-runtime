#include "runtime.h"

#include <unistd.h>

#include "metal.h"

namespace {

bool file_exists(const std::string& path) { return ::access(path.c_str(), F_OK) == 0; }

}  // namespace

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
    pipeline_archive_.reset();
    if (queue_) queue_->release();
    if (device_) device_->release();
}

std::string MetalRuntime::device_name() const {
    AutoreleaseScope scope;
    return std::string(device_->name()->utf8String());
}

bool MetalRuntime::has_unified_memory() const { return device_->hasUnifiedMemory(); }

namespace {

struct GPUFamilyEntry {
    const char* group;
    int index;
    MTL::GPUFamily family;
};

constexpr GPUFamilyEntry kGPUFamilyTable[] = {
    {"apple", 1, MTL::GPUFamilyApple1},   {"apple", 2, MTL::GPUFamilyApple2},
    {"apple", 3, MTL::GPUFamilyApple3},   {"apple", 4, MTL::GPUFamilyApple4},
    {"apple", 5, MTL::GPUFamilyApple5},   {"apple", 6, MTL::GPUFamilyApple6},
    {"apple", 7, MTL::GPUFamilyApple7},   {"apple", 8, MTL::GPUFamilyApple8},
    {"apple", 9, MTL::GPUFamilyApple9},   {"apple", 10, MTL::GPUFamilyApple10},
    {"mac", 2, MTL::GPUFamilyMac2},       {"common", 1, MTL::GPUFamilyCommon1},
    {"common", 2, MTL::GPUFamilyCommon2}, {"common", 3, MTL::GPUFamilyCommon3},
    {"metal", 3, MTL::GPUFamilyMetal3},   {"metal", 4, MTL::GPUFamilyMetal4},
};

}  // namespace

std::map<std::string, int> MetalRuntime::supported_gpu_families() const {
    std::map<std::string, int> out;
    for (const auto& entry : kGPUFamilyTable) {
        if (!device_->supportsFamily(entry.family)) continue;
        auto [it, inserted] = out.try_emplace(entry.group, entry.index);
        if (!inserted && entry.index > it->second) it->second = entry.index;
    }
    return out;
}

size_t MetalRuntime::recommended_max_working_set_size() const {
    return (size_t)(device_->recommendedMaxWorkingSetSize());
}

size_t MetalRuntime::max_threads_per_threadgroup() const {
    return (size_t)(device_->maxThreadsPerThreadgroup().width);
}

size_t MetalRuntime::max_threadgroup_memory_length() const {
    return (size_t)(device_->maxThreadgroupMemoryLength());
}

size_t MetalRuntime::max_buffer_length() const { return (size_t)(device_->maxBufferLength()); }

std::shared_ptr<Library> MetalRuntime::library_for(const std::string& msl_source,
                                                   const CompileOptions& options) {
    // Length-prefixed so the options blob can't be confused with the start of the source text.
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

void MetalRuntime::set_pipeline_cache_dir(std::optional<std::string> path) {
    AutoreleaseScope scope;
    std::shared_ptr<MTL::BinaryArchive> archive;
    if (path) {
        MTL::BinaryArchiveDescriptor* descriptor =
            MTL::BinaryArchiveDescriptor::alloc()->init()->autorelease();
        if (file_exists(*path)) {
            NS::String* p = NS::String::string(path->c_str(), NS::UTF8StringEncoding);
            descriptor->setUrl(NS::URL::fileURLWithPath(p));
        }
        NS::Error* error = nullptr;
        MTL::BinaryArchive* created = device_->newBinaryArchive(descriptor, &error);
        if (!created) {
            std::string message =
                error ? error->localizedDescription()->utf8String() : "unknown error";
            throw PipelineCacheError("failed to open pipeline cache at " + *path + ": " + message);
        }
        archive = std::shared_ptr<MTL::BinaryArchive>(created, [](MTL::BinaryArchive* value) {
            if (value) value->release();
        });
    }

    std::shared_ptr<MTL::BinaryArchive> replaced;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        replaced = std::move(pipeline_archive_);
        pipeline_archive_ = std::move(archive);
        pipeline_cache_path_ = path.value_or("");
        archive_add_stats_ = {};
    }
}

void MetalRuntime::note_archive_add_failure(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    archive_add_stats_.failures += 1;
    archive_add_stats_.last_error = message;
}

ArchiveAddStats MetalRuntime::archive_add_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return archive_add_stats_;
}

std::optional<std::string> MetalRuntime::pipeline_cache_dir() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pipeline_archive_) return std::nullopt;
    return pipeline_cache_path_;
}

void MetalRuntime::save_pipeline_cache() {
    AutoreleaseScope scope;
    std::shared_ptr<MTL::BinaryArchive> archive;
    std::string path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pipeline_archive_) {
            throw std::invalid_argument(
                "no pipeline cache directory set; call set_pipeline_cache_dir() first");
        }
        archive = pipeline_archive_;
        path = pipeline_cache_path_;
    }
    NS::String* p = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nullptr;
    if (!archive->serializeToURL(NS::URL::fileURLWithPath(p), &error)) {
        std::string message = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw PipelineCacheError("failed to write pipeline cache to " + path + ": " + message);
    }
}

std::shared_ptr<MTL::BinaryArchive> MetalRuntime::pipeline_archive() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pipeline_archive_;
}

void MetalRuntime::start_capture(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("start_capture: output path must not be empty");

    std::lock_guard<std::mutex> lock(capture_mutex_);
    AutoreleaseScope scope;
    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (manager->isCapturing()) {
        throw CaptureError("a Metal capture is already in progress");
    }
    if (!manager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument)) {
        throw CaptureError("this system does not support GPU trace document captures");
    }

    MTL::CaptureDescriptor* descriptor = MTL::CaptureDescriptor::alloc()->init()->autorelease();
    // A device capture includes every queue on this physical device. This is
    // useful when another native binding in the same process owns its queue.
    descriptor->setCaptureObject(device_);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    NS::String* output_path = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    descriptor->setOutputURL(NS::URL::fileURLWithPath(output_path));

    NS::Error* error = nullptr;
    if (!manager->startCapture(descriptor, &error)) {
        std::string message =
            error ? error->localizedDescription()->utf8String() : "unknown Metal capture error";
        throw CaptureError("failed to start Metal capture at " + path + ": " + message);
    }
    capture_started_ = true;
}

void MetalRuntime::stop_capture() {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!capture_started_) throw CaptureError("no capture started by metal_runtime is active");
    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (!manager->isCapturing()) {
        capture_started_ = false;
        throw CaptureError("the Metal capture was stopped outside metal_runtime");
    }
    manager->stopCapture();
    capture_started_ = false;
}

bool MetalRuntime::is_capturing() const {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    return MTL::CaptureManager::sharedCaptureManager()->isCapturing();
}

MetalRuntime& runtime() {
    static MetalRuntime rt;
    return rt;
}
