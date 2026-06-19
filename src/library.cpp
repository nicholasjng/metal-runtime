#include "library.h"

#include <stdexcept>

#include "metal.h"

Library::Library(MTL::Device* device, const std::string& msl_source) {
    AutoreleaseScope scope;
    NS::Error* error = nullptr;
    NS::String* source = NS::String::string(msl_source.c_str(), NS::UTF8StringEncoding);
    library_ = device->newLibrary(source, nullptr, &error);
    if (!library_) {
        std::string message = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw MSLCompileError("MSL compile error: " + message);
    }
}

Library::~Library() {
    if (library_) library_->release();
}

Library::Library(Library&& other) : library_(other.library_) { other.library_ = nullptr; }

MTL::Function* Library::function(const std::string& name) const {
    AutoreleaseScope scope;
    NS::String* fn_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    MTL::Function* fn = library_->newFunction(fn_name);
    if (!fn) {
        throw std::runtime_error("no such MSL function: " + name);
    }
    return fn;
}
