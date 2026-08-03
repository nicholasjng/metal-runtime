#include "library.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "dispatch.h"
#include "metal.h"

std::string CompileOptions::cache_key() const {
    // Every variable-length field is length-prefixed, so no two distinct
    // option sets can serialize to the same bytes. A define named "a=1" and
    // a define "a" valued "1" stay distinguishable.
    std::string out = "math=" + std::to_string((int)(math_mode));
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
    if (__builtin_available(macOS 15.0, *)) {
        mtl_options->setMathMode(to_mtl(options.math_mode));
    } else {
        // mathMode is macOS 15+; the selector crashes on older OSes. The
        // legacy boolean keeps what matters: SAFE preserves EFTs.
        mtl_options->setFastMathEnabled(options.math_mode != MathMode::Safe);
    }

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

// The scalar MSL types a function constant can have, mapped in both
// directions: MTL::DataType for Metal, DType for the canonical cache key,
// the MSL spelling for error messages.
struct TypeEntry {
    MTL::DataType mtl;
    DType dtype;
    const char* msl_name;
};

constexpr TypeEntry kTypeTable[] = {
    {MTL::DataTypeBool, {DType::Bool, 8}, "bool"},
    {MTL::DataTypeChar, {DType::Int, 8}, "char"},
    {MTL::DataTypeShort, {DType::Int, 16}, "short"},
    {MTL::DataTypeInt, {DType::Int, 32}, "int"},
    {MTL::DataTypeLong, {DType::Int, 64}, "long"},
    {MTL::DataTypeUChar, {DType::UInt, 8}, "uchar"},
    {MTL::DataTypeUShort, {DType::UInt, 16}, "ushort"},
    {MTL::DataTypeUInt, {DType::UInt, 32}, "uint"},
    {MTL::DataTypeULong, {DType::UInt, 64}, "ulong"},
    {MTL::DataTypeHalf, {DType::Float, 16}, "half"},
    {MTL::DataTypeFloat, {DType::Float, 32}, "float"},
    {MTL::DataTypeBFloat, {DType::Bfloat, 16}, "bfloat"},
};

const TypeEntry* type_entry(MTL::DataType t) {
    for (const TypeEntry& entry : kTypeTable) {
        if (entry.mtl == t) return &entry;
    }
    return nullptr;
}

// One entry point's declared constants, from reflection on the
// unspecialized function.
struct DeclaredConstant {
    MTL::DataType type;
    bool required;
};

std::map<std::string, DeclaredConstant> read_declared(MTL::Function* fn) {
    std::map<std::string, DeclaredConstant> out;
    NS::Dictionary* dict = fn->functionConstantsDictionary();
    NS::Enumerator<NS::String>* keys = dict->keyEnumerator<NS::String>();
    while (NS::String* key = keys->nextObject()) {
        auto* constant = dict->object<MTL::FunctionConstant>(key);
        out.emplace(key->utf8String(), DeclaredConstant{constant->type(), constant->required()});
    }
    return out;
}

// Coerces one provided constant to its declared MSL type; the result is
// Exact with the declared dtype and the typed value bytes.
FunctionConstant coerce(const FunctionConstant& c, MTL::DataType declared) {
    const TypeEntry* target = type_entry(declared);
    if (!target) {
        throw MSLCompileError(
            "function constant '" + c.name + "' has an unsupported MSL type (MTLDataType " +
            std::to_string((int)declared) + "); only scalar constants are supported");
    }
    FunctionConstant out;
    out.name = c.name;
    out.kind = FunctionConstant::Kind::Exact;
    out.dtype = target->dtype;

    auto put = [&](auto typed) { std::memcpy(out.value.data(), &typed, sizeof(typed)); };
    auto type_error = [&](const char* what, const std::string& hint = "") -> MSLCompileError {
        return MSLCompileError("function constant '" + c.name + "' is declared as '" +
                               target->msl_name + "', but the value is " + what + hint);
    };

    switch (c.kind) {
        case FunctionConstant::Kind::Exact:
            if (c.dtype != target->dtype) {
                throw type_error(("a '" + std::string(dtype_name(c.dtype)) + "' scalar").c_str());
            }
            out.value = c.value;
            return out;

        case FunctionConstant::Kind::Bool:
            if (declared != MTL::DataTypeBool) throw type_error("a Python bool");
            put((uint8_t)(c.bool_value ? 1 : 0));
            return out;

        case FunctionConstant::Kind::Float:
            if (declared != MTL::DataTypeFloat) {
                std::string hint;
                if (declared == MTL::DataTypeHalf) hint = "; pass numpy.float16(value)";
                throw type_error("a Python float", hint);
            }
            put((float)c.float_value);
            return out;

        case FunctionConstant::Kind::Int: {
            long long v = c.int_value;
            auto in_range = [&](long long lo, long long hi) {
                if (v < lo || v > hi) {
                    throw MSLCompileError(
                        "function constant '" + c.name + "' = " + std::to_string(v) +
                        " is out of range for declared type '" + target->msl_name + "'");
                }
            };
            switch (declared) {
                case MTL::DataTypeChar:
                    in_range(-128, 127);
                    put((int8_t)v);
                    return out;
                case MTL::DataTypeShort:
                    in_range(-32768, 32767);
                    put((int16_t)v);
                    return out;
                case MTL::DataTypeInt:
                    in_range(INT32_MIN, INT32_MAX);
                    put((int32_t)v);
                    return out;
                case MTL::DataTypeLong:
                    put((int64_t)v);
                    return out;
                case MTL::DataTypeUChar:
                    in_range(0, 255);
                    put((uint8_t)v);
                    return out;
                case MTL::DataTypeUShort:
                    in_range(0, 65535);
                    put((uint16_t)v);
                    return out;
                case MTL::DataTypeUInt:
                    in_range(0, 4294967295LL);
                    put((uint32_t)v);
                    return out;
                case MTL::DataTypeULong:
                    // Any non-negative long long fits a ulong.
                    in_range(0, INT64_MAX);
                    put((uint64_t)v);
                    return out;
                case MTL::DataTypeFloat:
                    put((float)v);
                    return out;
                default:
                    throw type_error("a Python int");
            }
        }
    }
    throw std::logic_error("unreachable: unhandled FunctionConstant::Kind");
}

// Injective, same contract as CompileOptions::cache_key. Flexible kinds are
// tagged so a pre-coercion key can't collide with a canonical one.
std::string pipeline_key(const std::string& name, const FunctionConstants& constants) {
    std::string key = std::to_string(name.size()) + ":" + name;
    for (const FunctionConstant& c : constants) {
        key += ";" + std::to_string(c.name.size()) + ":" + c.name + "=";
        switch (c.kind) {
            case FunctionConstant::Kind::Bool:
                key += c.bool_value ? "B1" : "B0";
                break;
            case FunctionConstant::Kind::Int:
                key += "I" + std::to_string(c.int_value);
                break;
            case FunctionConstant::Kind::Float: {
                uint64_t bits;
                std::memcpy(&bits, &c.float_value, sizeof(bits));
                key += "F" + std::to_string(bits);
                break;
            }
            case FunctionConstant::Kind::Exact:
                key += "E" + std::to_string((int)c.dtype.code) + "." +
                       std::to_string((int)c.dtype.bits) + ":";
                for (size_t i = 0; i < c.dtype.itemsize(); ++i) {
                    key += std::to_string((int)c.value[i]) + ",";
                }
                break;
        }
    }
    return key;
}

}  // namespace

Library::Library(MTL::Device* device, const std::string& msl_source, const CompileOptions& options)
    : device_(device) {
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

Library::Library(Library&& other) noexcept
    : device_(other.device_), library_(other.library_), pipelines_(std::move(other.pipelines_)) {
    other.library_ = nullptr;
}

bool Library::has_function(const std::string& name) const {
    AutoreleaseScope scope;
    NS::Array* names = library_->functionNames();
    for (NS::UInteger i = 0; i < names->count(); ++i) {
        auto* fn_name = static_cast<NS::String*>(names->object(i));
        if (name == fn_name->utf8String()) return true;
    }
    return false;
}

MTL::Function* Library::function(const std::string& name,
                                 const FunctionConstants& constants) const {
    return create_specialized(name, constants, nullptr);
}

MTL::Function* Library::create_specialized(const std::string& name,
                                           const FunctionConstants& constants,
                                           FunctionConstants* canonical) const {
    AutoreleaseScope scope;
    NS::String* fn_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);

    MTL::Function* probe = library_->newFunction(fn_name);
    if (!probe) {
        if (!has_function(name)) {
            throw MSLFunctionNotFoundError("no such MSL function: " + name);
        }
        throw MSLCompileError("MSL function '" + name +
                              "' exists but could not be created for constants reflection");
    }
    std::map<std::string, DeclaredConstant> declared = read_declared(probe);

    // No constants declared, none provided: the probe *is* the function.
    if (declared.empty() && constants.empty()) return probe;

    // Metal neither rejects an unset required constant (it specializes to an
    // undefined value) nor a misspelled name, so both are checked here.
    // Constants the entry point doesn't use don't appear in its dictionary
    // and are rejected as unknown; optional ones (guarded by
    // is_function_constant_defined) report required=false and may be omitted.
    std::string missing;
    for (const auto& entry : declared) {
        const std::string& declared_name = entry.first;
        bool provided =
            std::any_of(constants.begin(), constants.end(),
                        [&](const FunctionConstant& c) { return c.name == declared_name; });
        if (entry.second.required && !provided) {
            missing += missing.empty() ? "'" : ", '";
            missing += declared_name + "'";
        }
    }
    std::string unknown;
    for (const FunctionConstant& c : constants) {
        if (!declared.count(c.name)) {
            unknown += unknown.empty() ? "'" : ", '";
            unknown += c.name + "'";
        }
    }
    if (!missing.empty() || !unknown.empty()) {
        probe->release();
        std::string message;
        if (!missing.empty()) {
            message += "MSL function '" + name + "' requires function constant(s) " + missing +
                       ", but they were not set; pass values via Kernel(..., constants={...})";
        }
        if (!unknown.empty()) {
            if (!missing.empty()) message += "; additionally, ";
            message += "constant(s) " + unknown + " do not exist in MSL function '" + name + "'";
        }
        throw MSLCompileError(message);
    }
    probe->release();

    FunctionConstants coerced;
    coerced.reserve(constants.size());
    for (const FunctionConstant& c : constants) {
        coerced.push_back(coerce(c, declared.at(c.name).type));
    }

    MTL::FunctionConstantValues* values =
        MTL::FunctionConstantValues::alloc()->init()->autorelease();
    for (const FunctionConstant& c : coerced) {
        MTL::DataType type = declared.at(c.name).type;
        values->setConstantValue(c.value.data(), type,
                                 NS::String::string(c.name.c_str(), NS::UTF8StringEncoding));
    }
    NS::Error* error = nullptr;
    MTL::Function* fn = library_->newFunction(fn_name, values, &error);
    if (!fn) {
        std::string message = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw MSLCompileError("failed to specialize MSL function '" + name +
                              "' with the given constants: " + message);
    }
    if (canonical) *canonical = std::move(coerced);
    return fn;
}

std::shared_ptr<ComputePipeline> Library::pipeline_for(const std::string& name,
                                                       const FunctionConstants& constants) {
    const std::string key = pipeline_key(name, constants);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) return it->second;
    }

    FunctionConstants canonical;
    MTL::Function* fn = create_specialized(name, constants, &canonical);

    // Coercion may map this spelling onto a pipeline that already exists
    // under another one ({N: 8} vs {N: np.uint32(8)}); alias rather than
    // build a duplicate.
    const std::string canonical_key = pipeline_key(name, canonical);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pipelines_.find(canonical_key);
        if (it != pipelines_.end()) {
            fn->release();
            pipelines_.emplace(key, it->second);
            return it->second;
        }
    }

    // Built outside the lock, same tradeoff as the library cache.
    auto pipeline = std::make_shared<ComputePipeline>(device_, fn, name);

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = pipelines_.emplace(canonical_key, std::move(pipeline));
    if (key != canonical_key) pipelines_.emplace(key, it->second);
    return it->second;
}
