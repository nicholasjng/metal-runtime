#pragma once
#include <stdexcept>
#include <string>

namespace MTL {
class Device;
class Library;
class Function;
}  // namespace MTL

// Thrown when MTL::Device::newLibrary fails to compile MSL source text. Bound
// to Python as metal_runtime.CompileError so callers -- codegen pipelines
// especially -- can catch compile failures distinctly from other runtime
// errors instead of a generic RuntimeError.
struct MSLCompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Compiles MSL source text at runtime via MTL::Device::newLibrary (the NVRTC
// equivalent) -- no offline `metal`/`metallib` toolchain involved.
class Library {
   public:
    Library(MTL::Device* device, const std::string& msl_source);
    ~Library();
    Library(Library&& other);
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // Returns a new, caller-owned MTL::Function for `name`; throws if not found.
    MTL::Function* function(const std::string& name) const;

   private:
    MTL::Library* library_ = nullptr;
};
