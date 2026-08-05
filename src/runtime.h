#pragma once
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "library.h"

namespace MTL {
class Device;
class CommandQueue;
class BinaryArchive;
}  // namespace MTL

// No Metal device on this machine, or no command queue on it.
struct NoDeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Opening or writing the on-disk pipeline cache (MTL::BinaryArchive) failed.
// An I/O problem, not a compile or dispatch one; registered in Python as OSError.
struct PipelineCacheError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Non-fatal pipeline-cache staging diagnostics; see pipeline_cache_status().
struct ArchiveAddStats {
    size_t failures = 0;
    std::string last_error;
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

    // The highest GPU family index this device supports, per independent GPU
    // family group ("apple" -> 8, "metal" -> 4, ...). Each family index is documented
    // as cumulative, so any existing family version prior to the given index
    // may be assumed as supported.
    std::map<std::string, int> supported_gpu_families() const;

    // Compiled libraries are keyed by source text *and* compile options.
    // Returns a shared_ptr rather than a reference into the map because
    // eviction, or another thread's insert, must not pull the Library out from
    // under a caller that is still building a pipeline from it.
    std::shared_ptr<Library> library_for(const std::string& msl_source,
                                         const CompileOptions& options = {});

    // Evicts least-recently-used entries down to `limit`, 0 disables eviction.
    void set_library_cache_limit(size_t limit);
    size_t library_cache_limit() const;
    size_t library_cache_size() const;
    void clear_library_cache();

    // A path loads an archive if it exists, nullopt disables caching.
    void set_pipeline_cache_dir(std::optional<std::string> path);
    std::optional<std::string> pipeline_cache_dir() const;

    // Explicit, not automatic on process exit.
    void save_pipeline_cache();

    // Records a failed BinaryArchive::addComputePipelineFunctions call.
    // Non-fatal by design (the pipeline itself built), but counted so a
    // cache that never populates is observable instead of silent.
    void note_archive_add_failure(const std::string& message);
    ArchiveAddStats archive_add_stats() const;

    MTL::BinaryArchive* pipeline_archive() const;

   private:
    void evict_locked();

    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    bool non_uniform_threadgroups_ = false;

    mutable std::mutex mutex_;
    using LRUList = std::list<std::string>;
    LRUList lru_;  // front = most recently used
    std::unordered_map<std::string, std::pair<std::shared_ptr<Library>, LRUList::iterator>>
        libraries_;
    size_t cache_limit_ = kDefaultLibraryCacheLimit;

    MTL::BinaryArchive* pipeline_archive_ = nullptr;
    std::string pipeline_cache_path_;
    ArchiveAddStats archive_add_stats_;
};

MetalRuntime& runtime();
