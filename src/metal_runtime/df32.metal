// An emulation of float64 as float32x2.
#include <metal_stdlib>
using namespace metal;

struct df32 {
    float hi, lo;
};

// requires |a| >= |b|.
inline df32 quick_two_sum(float a, float b) {
    float s = a + b;
    return df32{s, b - (s - a)};
}

// No ordering required, 6 flops.
inline df32 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return df32{s, (a - (s - bb)) + (b - bb)};
}

// 2 flops, and the reason this is affordable on Apple GPUs at all.
inline df32 two_prod(float a, float b) {
    float p = a * b;
    return df32{p, fma(a, b, -p)};
}
