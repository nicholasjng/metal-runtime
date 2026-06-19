#include "library.h"

#include <vector>

#include "metal.h"

std::string CompileOptions::cache_key() const {
    // Every variable-length field is length-prefixed, so no two distinct
    // option sets can serialize to the same bytes -- a define named "a=1" and
    // a define "a" valued "1" have to stay distinguishable.
    std::string out = "math=" + std::to_string(static_cast<int>(math_mode));
    for (const auto& [name, value] : defines) {
        out += ";" + std::to_string(name.size()) + ":" + name;
        out += "=" + std::to_string(value.size()) + ":" + value;
    }
    return out;
}

namespace {

MTL::MathMode to_mtl(MathMode mode) {
    switch (mode) {
        case MathMode::Safe:
            return MTL::MathModeSafe;
        case MathMode::Relaxed:
            return MTL::MathModeRelaxed;
        case MathMode::Fast:
            break;
    }
    return MTL::MathModeFast;
}

// Caller owns nothing: everything here is autoreleased into the enclosing
// AutoreleaseScope, including the returned dictionary.
MTL::CompileOptions* build_options(const CompileOptions& options) {
    MTL::CompileOptions* mtl_options = MTL::CompileOptions::alloc()->init()->autorelease();
    mtl_options->setMathMode(to_mtl(options.math_mode));

    if (!options.defines.empty()) {
        std::vector<NS::Object*> keys;
        std::vector<NS::Object*> values;
        keys.reserve(options.defines.size());
        values.reserve(options.defines.size());
        for (const auto& [name, value] : options.defines) {
            keys.push_back(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
            values.push_back(NS::String::string(value.c_str(), NS::UTF8StringEncoding));
        }
        mtl_options->setPreprocessorMacros(
            NS::Dictionary::dictionary(values.data(), keys.data(), values.size()));
    }
    return mtl_options;
}

}  // namespace

Library::Library(MTL::Device* device, const std::string& msl_source,
                 const CompileOptions& options) {
    AutoreleaseScope scope;
    NS::Error* error = nullptr;
    NS::String* source = NS::String::string(msl_source.c_str(), NS::UTF8StringEncoding);
    library_ = device->newLibrary(source, build_options(options), &error);
    if (!library_) {
        std::string message = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw MSLCompileError("MSL compile error: " + message);
    }
}

Library::~Library() {
    if (library_) library_->release();
}

Library::Library(Library&& other) noexcept : library_(other.library_) { other.library_ = nullptr; }

MTL::Function* Library::function(const std::string& name) const {
    AutoreleaseScope scope;
    NS::String* fn_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    MTL::Function* fn = library_->newFunction(fn_name);
    if (!fn) {
        throw MSLFunctionNotFoundError("no such MSL function: " + name);
    }
    return fn;
}
