#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// Element type of a Buffer, encoded as DLPack's (code, bits) pair. That's what
// nanobind's ndarray already carries, so the binding layer compares these
// directly instead of maintaining a translation table.
struct DType {
    enum Code : uint8_t { Int = 0, UInt = 1, Float = 2, Bfloat = 4, Bool = 6 };

    uint8_t code = Float;
    uint8_t bits = 32;

    size_t itemsize() const { return static_cast<size_t>(bits + 7) / 8; }
    bool operator==(const DType& other) const { return code == other.code && bits == other.bits; }
    bool operator!=(const DType& other) const { return !(*this == other); }
};

// Every element type an MSL kernel can address. float64 is deliberately
// absent: Metal has no `double`, so a float64 buffer could only ever be read
// back as garbage, and silently narrowing it to float32 on upload hides a real
// precision loss. Callers pass `.astype(np.float32)` and see the cost.
// Throws std::invalid_argument, naming the supported set, for anything else.
DType dtype_from_name(const std::string& name);

// Nullptr if `dt` isn't a type this runtime supports.
const char* dtype_name(DType dt);

// Comma-separated list of supported names, for error messages.
std::string supported_dtype_names();
