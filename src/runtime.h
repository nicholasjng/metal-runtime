#pragma once
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "library.h"

namespace MTL {
class Device;
class CommandQueue;
}  // namespace MTL

// No Metal device on this machine, or no command queue on it.
struct NoDeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Default cap on cached libraries.
inline constexpr size_t kDefaultLibraryCacheLimit = 256;

class MetalRuntime {
   public:
    MetalRuntime();
    ~MetalRuntime();
    MetalRuntime(MetalRuntime&&) = delete;
    MetalRuntime(const MetalRuntime&) = delete;
    MetalRuntime& operator=(const MetalRuntime&) = delete;

    MTL::Device* device() const { return device_; }
    MTL::CommandQueue* queue() const { return queue_; }

    std::string device_name() const;
    bool has_unified_memory() const;
    size_t recommended_max_working_set_size() const;
    size_t max_threads_per_threadgroup() const;
    size_t max_threadgroup_memory_length() const;
    size_t max_buffer_length() const;

    // dispatchThreads() -- an arbitrary grid, with partial threadgroups at the
    // edges -- needs Apple family 4+ or Mac family 2. Older GPUs take the
    // dispatchThreadgroups path instead, which needs the grid to divide
    // evenly by the threadgroup.
    bool supports_non_uniform_threadgroups() const { return non_uniform_threadgroups_; }

    // Compiled libraries keyed by source text *and* compile options:
    // dispatching an already-seen kernel skips recompilation, but the same
    // source under a different math mode is a different library, not a cache
    // hit. Returns a shared_ptr rather than a reference into the map because
    // eviction, or another thread's insert, must not pull the Library out from
    // under a caller that is still building a pipeline from it.
    std::shared_ptr<Library> library_for(const std::string& msl_source,
                                         const CompileOptions& options = {});

    // 0 disables eviction. Evicts least-recently-used entries down to `limit`.
    void set_library_cache_limit(size_t limit);
    size_t library_cache_limit() const;
    size_t library_cache_size() const;
    void clear_library_cache();

   private:
    void evict_locked();

    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    bool non_uniform_threadgroups_ = false;

    // Guards every cache member below. The nanobind module is built
    // FREE_THREADED, so on a free-threaded interpreter two threads can be
    // inside library_for() at once with no GIL serializing them, and an
    // unsynchronized rehash of `libraries_` is a corrupted map.
    mutable std::mutex mutex_;
    using LRUList = std::list<std::string>;
    LRUList lru_;  // front = most recently used
    std::unordered_map<std::string, std::pair<std::shared_ptr<Library>, LRUList::iterator>>
        libraries_;
    size_t cache_limit_ = kDefaultLibraryCacheLimit;
};

MetalRuntime& runtime();
