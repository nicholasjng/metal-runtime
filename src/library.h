#pragma once
#include <stdexcept>
#include <string>

namespace MTL {
class Device;
class Library;
class Function;
}  // namespace MTL

// Thrown when newLibrary fails to compile MSL source. Bound to Python as
// CompileError so codegen callers can catch compile failures specifically,
// not a generic RuntimeError.
struct MSLCompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Compiles MSL source at runtime via newLibrary, the NVRTC equivalent: no
// offline metal/metallib toolchain needed.
class Library {
   public:
    Library(MTL::Device* device, const std::string& msl_source);
    ~Library();
    Library(Library&& other);
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // Caller-owned; throws if `name` isn't found.
    MTL::Function* function(const std::string& name) const;

   private:
    MTL::Library* library_ = nullptr;
};
