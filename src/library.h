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
class BinaryArchive;
}  // namespace MTL

class ComputePipeline;

// How much freedom the Metal compiler has to rewrite floating-point
// arithmetic. `Fast` is Metal's own default, and it permits reassociation:
// under it the compensation term of a Kahan summation, `c = (t - s) - y`,
// folds algebraically to zero and is deleted outright, with no diagnostic.
// Anything built on error-free transformations (compensated accumulation,
// double-single arithmetic), has to compile under `Safe` to survive.
enum class MathMode { Safe, Relaxed, Fast };

// Compile-time configuration for one MSL translation unit.
// Part of the library cache key, so different options produce different libraries.
struct CompileOptions {
    MathMode math_mode = MathMode::Fast;  // Metal's default, kept as ours

    // Emitted as preprocessor macros.
    std::map<std::string, std::string> defines;

    std::string cache_key() const;
};

// One MSL function constant, baked in at pipeline build. Unlike `defines`,
// specializing skips recompiling the library.
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

// Compiles MSL source at runtime via newLibrary, no metal/metallib toolchain needed.
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

    // Evicting a library drops its pipelines with it.
    // `archive` passes straight through to ComputePipeline on a cache miss.
    std::shared_ptr<ComputePipeline> pipeline_for(const std::string& name,
                                                  const FunctionConstants& constants = {},
                                                  MTL::BinaryArchive* archive = nullptr);

   private:
    bool has_function(const std::string& name) const;

    // Reflects the *unspecialized* function for its declared constants.
    // Validates and coerces `constants` against them, then creates via the constantValues variant.
    MTL::Function* create_specialized(const std::string& name, const FunctionConstants& constants,
                                      FunctionConstants* canonical) const;

    MTL::Device* device_ = nullptr;  // borrowed from the runtime singleton
    MTL::Library* library_ = nullptr;

    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ComputePipeline>> pipelines_;
};
