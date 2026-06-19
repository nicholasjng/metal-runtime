#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// Element type of a Buffer, encoded as DLPack's (code, bits) pair for nb::ndarray compat.
struct DType {
    enum Code : uint8_t { Int = 0, UInt = 1, Float = 2, Bfloat = 4, Bool = 6 };

    uint8_t code = Float;
    uint8_t bits = 32;

    size_t itemsize() const { return (size_t)(bits + 7) / 8; }
    bool operator==(const DType& other) const { return code == other.code && bits == other.bits; }
    bool operator!=(const DType& other) const { return !(*this == other); }
};

// Every element type an MSL kernel can address. Throws std::invalid_argument,
// naming the supported set, for anything else.
DType dtype_from_name(const std::string& name);

// nullptr if `dt` isn't a dtype this runtime supports.
const char* dtype_name(DType dt);

// Comma-separated list of names of supported dtypes, for error messages.
std::string supported_dtype_names();
