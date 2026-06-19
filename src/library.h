#pragma once
#include <map>
#include <stdexcept>
#include <string>

namespace MTL {
class Device;
class Library;
class Function;
}  // namespace MTL

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

    // Caller-owned; throws MSLFunctionNotFoundError if `name` isn't found.
    MTL::Function* function(const std::string& name) const;

   private:
    MTL::Library* library_ = nullptr;
};
