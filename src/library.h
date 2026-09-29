#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "dtype.h"
#include "export.h"
#include "lru.h"

namespace MTL {
class Device;
class Library;
class Function;
}  // namespace MTL

class ComputePipeline;

// `Fast` (Metal's default) permits reassociation, which silently deletes
// compensated-arithmetic error terms; such code needs `Safe`.
enum class MathMode { Safe, Relaxed, Fast };

// Part of the library cache key.
struct MR_API CompileOptions {
    MathMode math_mode = MathMode::Fast;
    std::map<std::string, std::string> defines;  // preprocessor macros

    std::string cache_key() const;
};

// One MSL function constant, baked in at pipeline build without recompiling the library.
struct FunctionConstant {
    enum class Kind : uint8_t { Bool, Int, Float, Exact };

    std::string name;
    Kind kind = Kind::Exact;
    bool bool_value = false;
    long long int_value = 0;
    unsigned long long uint_value = 0;  // used when int_value overflows (> INT64_MAX)
    bool int_is_wide_unsigned = false;
    double float_value = 0.0;
    DType dtype{};                   // Exact only
    std::array<uint8_t, 8> value{};  // Exact only: little-endian bytes
};
using FunctionConstants = std::vector<FunctionConstant>;

struct MR_API MSLCompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The source compiled but has no such entry point.
struct MR_API MSLFunctionNotFoundError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// MSL source compiled at runtime via newLibrary.
class MR_API Library {
   public:
    Library(MTL::Device* device, const std::string& msl_source, const CompileOptions& options = {});
    ~Library();
    Library(Library&&) = delete;
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // Cached per (name, constants as given). Throws MSLFunctionNotFoundError
    // for an unknown name, MSLCompileError for constants that don't match
    // the function's declaration.
    std::shared_ptr<ComputePipeline> pipeline_for(const std::string& name,
                                                  const FunctionConstants& constants = {});

   private:
    bool has_function(const std::string& name) const;

    // Validates `constants` against reflection, then specializes. Caller owns the result.
    MTL::Function* create_specialized(const std::string& name,
                                      const FunctionConstants& constants) const;

    MTL::Device* device_ = nullptr;  // borrowed
    MTL::Library* library_ = nullptr;

    static constexpr size_t kMaxCachedPipelines = 128;
    std::mutex mutex_;
    LruCache<ComputePipeline> pipelines_{kMaxCachedPipelines};
};
