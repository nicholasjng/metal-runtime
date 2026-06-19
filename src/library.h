#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "dtype.h"

namespace MTL {
class Device;
class Library;
class Function;
}  // namespace MTL

class ComputePipeline;

// How much freedom the Metal compiler has to rewrite floating-point
// arithmetic. `Fast` is Metal's own default, and it permits reassociation:
// under it the compensation term of a Kahan summation, `c = (t - s) - y`,
// folds algebraically to zero and is deleted outright, with no diagnostic.
// Anything built on error-free transformations (compensated accumulation,
// double-single arithmetic), has to compile under `Safe` to survive.
enum class MathMode { Safe, Relaxed, Fast };

// Compile-time configuration for one MSL translation unit. Part of the library
// cache key, so the same source compiled in two different ways yields two distinct libraries.
struct CompileOptions {
    MathMode math_mode = MathMode::Fast;  // Metal's default, kept as ours

    // Emitted as preprocessor macros. The lever a code generator uses to
    // specialize one source string by block size or element type instead of
    // emitting a textually distinct kernel per variant.
    std::map<std::string, std::string> defines;

    // Injective: every distinct CompileOptions maps to a distinct string, so
    // it can be concatenated into a cache key without ambiguity.
    std::string cache_key() const;
};

// One MSL function constant, baked in at pipeline build. Unlike `defines`,
// specializing skips recompiling the library. Plain Python bool/int/float
// arrive as their own kinds and coerce to the declared MSL type; numpy
// scalars arrive Exact and must match it.
struct FunctionConstant {
    enum class Kind : uint8_t { Bool, Int, Float, Exact };

    std::string name;
    Kind kind = Kind::Exact;
    bool bool_value = false;
    long long int_value = 0;
    // Set only when a Python int overflows `long long` (i.e. > INT64_MAX),
    // the one case a ulong constant can represent that int_value cannot.
    unsigned long long uint_value = 0;
    bool int_is_wide_unsigned = false;
    double float_value = 0.0;
    DType dtype{};  // Exact only
    // Exact only: raw little-endian bytes, the first dtype.itemsize() meaningful.
    std::array<uint8_t, 8> value{};
};
using FunctionConstants = std::vector<FunctionConstant>;

// Thrown when newLibrary fails to compile MSL source.
struct MSLCompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The source compiled but has no such entry point.
struct MSLFunctionNotFoundError : MSLCompileError {
    using MSLCompileError::MSLCompileError;
};

// Compiles MSL source at runtime via newLibrary, the NVRTC equivalent:
// no offline metal/metallib toolchain needed.
class Library {
   public:
    Library(MTL::Device* device, const std::string& msl_source, const CompileOptions& options = {});
    ~Library();
    Library(Library&& other) noexcept;
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // Caller-owned; throws MSLFunctionNotFoundError if `name` isn't found,
    // MSLCompileError if `constants` doesn't satisfy the function's declared
    // constants (missing required, unknown name, type mismatch).
    MTL::Function* function(const std::string& name, const FunctionConstants& constants = {}) const;

    // Cached: newComputePipelineState costs milliseconds. The cache lives
    // here so evicting a library drops its pipelines with it.
    std::shared_ptr<ComputePipeline> pipeline_for(const std::string& name,
                                                  const FunctionConstants& constants = {});

   private:
    bool has_function(const std::string& name) const;

    // Reflects the *unspecialized* function for its declared constants
    // (functionConstantsDictionary is populated only there), validates and
    // coerces `constants` against them, then creates via the constantValues
    // variant -- the only path whose product survives pipeline creation.
    // Fills `canonical` (all Exact, post-coercion) when non-null, so two
    // spellings of the same value share a pipeline cache entry.
    MTL::Function* create_specialized(const std::string& name, const FunctionConstants& constants,
                                      FunctionConstants* canonical) const;

    MTL::Device* device_ = nullptr;  // borrowed from the runtime singleton
    MTL::Library* library_ = nullptr;

    // Guards pipelines_; the module is built free-threaded.
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ComputePipeline>> pipelines_;
};
