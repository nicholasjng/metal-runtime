#include "dtype.h"

#include <stdexcept>
#include <string>

namespace {

// The MSL spelling of each type is in the comment: that's the declaration a
// generated kernel needs for a buffer of this dtype to line up.
struct Entry {
    const char* name;
    DType dtype;
};

constexpr Entry kEntries[] = {
    {"bool", {DType::Bool, 8}},       // bool
    {"int8", {DType::Int, 8}},        // char
    {"int16", {DType::Int, 16}},      // short
    {"int32", {DType::Int, 32}},      // int
    {"int64", {DType::Int, 64}},      // long
    {"uint8", {DType::UInt, 8}},      // uchar
    {"uint16", {DType::UInt, 16}},    // ushort
    {"uint32", {DType::UInt, 32}},    // uint
    {"uint64", {DType::UInt, 64}},    // ulong
    {"float16", {DType::Float, 16}},  // half
    {"float32", {DType::Float, 32}},  // float
    {"bfloat16", {DType::Bfloat, 16}},
};

}  // namespace

DType dtype_from_name(const std::string& name) {
    for (const Entry& entry : kEntries) {
        if (name == entry.name) return entry.dtype;
    }
    if (name == "float64" || name == "double") {
        throw std::invalid_argument(
            "dtype 'float64' is not supported: Metal has no double-precision type. "
            "Convert with .astype(numpy.float32) before uploading.");
    }
    throw std::invalid_argument("unsupported dtype '" + name +
                                "'; supported: " + supported_dtype_names());
}

const char* dtype_name(DType dt) {
    for (const Entry& entry : kEntries) {
        if (dt == entry.dtype) return entry.name;
    }
    return nullptr;
}

std::string supported_dtype_names() {
    std::string out;
    for (const Entry& entry : kEntries) {
        if (!out.empty()) out += ", ";
        out += entry.name;
    }
    return out;
}
